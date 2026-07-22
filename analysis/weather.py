"""
weather.py
Fetches winds-aloft and temperature data from the NOAA Aviation
Weather Center API and enriches the flight DataFrame with OAT
(outside air temperature) and wind components per row.

NOAA endpoints used (both free, no API key required):
  /api/data/windtemp  — FD winds/temps aloft forecast text
  /api/data/stationinfo — station coordinates (fetched once per session)

Important limitation:
  NOAA's windtemp endpoint returns the *current* forecast cycle, not
  historical data. Weather applied to past flights is approximate —
  winds-aloft at cruise altitude are stable enough that this is a
  reasonable approximation for short-to-medium haul flights, but it
  should be noted in any analysis output.

FD format reference:
  FT  3000    6000    9000   12000   18000   24000  30000  34000  39000
  ATL 9900 3217+15 3114+11 3116+05 2722-08 2740-17 275732 265742 255754

  Each entry like "3217+15": direction=320°, speed=17kts, temp=+15°C
  "9900"  = calm/variable, no temp
  "////"  = data unavailable
  Above FL240: 6-digit format, temp always negative (e.g. "275732" = 270°/57kts/-32°C)
"""

import math
import re
from datetime import datetime, timezone
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

WINDTEMP_URL    = "https://aviationweather.gov/api/data/windtemp"
STATIONINFO_URL = "https://aviationweather.gov/api/data/stationinfo"
HEADERS         = {"User-Agent": "AircraftPerformanceAnalyzer/1.0 (student project)"}


# ---------------------------------------------------------------------------
# Dynamic station coordinate lookup
# ---------------------------------------------------------------------------

def _fetch_station_coords_for_ids(station_ids: list[str]) -> dict[str, tuple[float, float]]:
    """
    Fetch coordinates from NOAA's stationinfo endpoint for a specific set
    of station IDs. The FD data uses bare 3-letter codes (ATL, JFK) but
    NOAA's stationinfo endpoint requires K-prefixed ICAO IDs (KATL, KJFK).

    This function converts, queries, and converts back.

    Parameters
    ----------
    station_ids : list[str]   3-letter station codes from FD data (e.g. ['ATL', 'JFK', 'RIC'])

    Returns
    -------
    dict
        { station_id_bare -> (lat, lon) } for stations that were successfully looked up.
        Falls back to hardcoded dict for any missing stations.
    """
    if not station_ids:
        # No stations in FD response, use fallback
        return _get_fallback_coords()

    # Convert bare codes to K-prefixed ICAO IDs for the API query
    k_prefixed = [f"K{sid}" for sid in station_ids]

    try:
        # Query stationinfo for the specific stations in this FD response
        response = requests.get(
            STATIONINFO_URL,
            params={"ids": ",".join(k_prefixed), "format": "json"},
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        stations = response.json()

        coords = {}
        for s in stations:
            icao_id = s.get("icaoId") or s.get("id")
            lat     = s.get("lat")
            lon     = s.get("lon")

            if icao_id and lat is not None and lon is not None:
                # Strip K prefix to match FD station codes
                bare_id = icao_id.lstrip("K").upper()
                coords[bare_id] = (float(lat), float(lon))

        # If we got at least some coordinates, return what we found
        # (stations not in stationinfo response won't have coords, which is fine)
        if coords:
            return coords

    except Exception as e:
        print(f"[weather] Station coordinate fetch failed: {e}. Using fallback dict.")

    # Fallback — use hardcoded dict
    return _get_fallback_coords()


def _get_fallback_coords() -> dict[str, tuple[float, float]]:
    """Hardcoded station coordinates for common US routes. Used when stationinfo fetch fails."""
    return {
        "ATL": (33.64, -84.43), "ORD": (41.98, -87.90), "DFW": (32.90, -97.04),
        "DEN": (39.86, -104.67), "LAX": (33.94, -118.41), "JFK": (40.64, -73.78),
        "SFO": (37.62, -122.38), "SEA": (47.45, -122.31), "MIA": (25.80, -80.29),
        "BOS": (42.36, -71.01), "MSP": (44.88, -93.22), "CLT": (35.21, -80.94),
        "RIC": (37.51, -77.32), "RDU": (35.88, -78.79), "GSP": (34.90, -82.22),
        "BNA": (36.12, -86.68), "BHM": (33.56, -86.75), "TYS": (35.81, -83.99),
        "IAH": (29.98, -95.34), "MCO": (28.43, -81.31), "SLC": (40.79, -111.98),
        "PHX": (33.44, -112.01), "LAS": (36.08, -115.15), "PHL": (39.87, -75.24),
        "ORF": (36.90, -76.03), "AVP": (41.34, -75.73), "ROA": (37.32, -79.97),
        "CRW": (38.37, -81.59), "CAE": (33.94, -81.12), "SAV": (32.13, -81.20),
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in km between two lat/lon points.
    More accurate than raw degree difference at mid-latitudes where
    a degree of longitude is shorter than a degree of latitude.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _nearest_station(
    lat: float, lon: float, available_ids: list[str], coords: dict[str, tuple[float, float]]
) -> str | None:
    """
    Return the station ID from available_ids that is geographically
    nearest to (lat, lon), using haversine distance for accuracy.

    Parameters
    ----------
    lat           : float   Aircraft latitude
    lon           : float   Aircraft longitude
    available_ids : list    Station IDs present in the current FD response
    coords        : dict    { station_id -> (lat, lon) }, pre-fetched by the caller
    """
    # Only consider stations that successfully got coordinates
    candidates = {
        sid: c
        for sid, c in coords.items()
        if sid in available_ids
    }

    if not candidates:
        # Last resort — return first available ID with no distance check
        return available_ids[0] if available_ids else None

    nearest = min(
        candidates.items(),
        key=lambda item: _haversine_km(lat, lon, item[1][0], item[1][1])
    )
    return nearest[0]


# ---------------------------------------------------------------------------
# Fetching FD wind/temp data
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _fetch_windtemp_raw(region: str, level: str, fcst: str) -> str:
    """
    Fetch raw FD winds/temps text from NOAA. Cached per session so
    repeated pipeline runs don't re-hit the API.

    Parameters
    ----------
    region : str   "us" for CONUS-wide, or sub-region like "bos", "chi"
    level  : str   "low" (3k–18k ft) or "high" (24k–39k ft)
    fcst   : str   Forecast hour offset: "06", "12", or "24"
    """
    response = requests.get(
        WINDTEMP_URL,
        params={"region": region, "level": level, "fcst": fcst, "layout": "off"},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    return response.text


# ---------------------------------------------------------------------------
# FD text parsing
# ---------------------------------------------------------------------------

def _parse_fd_entry(
    entry: str, altitude_ft: int
) -> tuple[float | None, float | None, float | None]:
    """
    Parse one encoded FD entry into (wind_dir_deg, wind_speed_kts, temp_c).
    Returns (None, None, None) for calm/variable or unavailable data.
    """
    entry = entry.strip()
    if not entry or entry.startswith("/") or entry == "9900":
        return None, None, None

    # Above FL240: 6-digit no-sign format, temp always negative
    if altitude_ft > 24000 and re.fullmatch(r"\d{6}", entry):
        return (
            float(int(entry[0:2]) * 10),
            float(int(entry[2:4])),
            -float(entry[4:6]),
        )

    # Standard: DDSS+TT or DDSS-TT
    m = re.fullmatch(r"(\d{2})(\d{2})([+-])(\d{1,2})", entry)
    if m:
        sign = 1.0 if m.group(3) == "+" else -1.0
        return (
            float(int(m.group(1)) * 10),
            float(int(m.group(2))),
            sign * float(m.group(4)),
        )

    # Low-altitude entries sometimes omit temperature (e.g. at 3000 ft)
    m2 = re.fullmatch(r"(\d{2})(\d{2})", entry)
    if m2:
        return float(int(m2.group(1)) * 10), float(int(m2.group(2))), None

    return None, None, None


def _parse_windtemp_table(raw_text: str) -> dict[str, dict[int, dict]]:
    """
    Parse a full FD text block into:
      { station_id -> { altitude_ft -> { wind_dir, wind_spd, temp_c } } }
    """
    lines = raw_text.strip().splitlines()

    header_idx, altitudes = None, []
    for i, line in enumerate(lines):
        if line.strip().startswith("FT"):
            header_idx = i
            altitudes = [int(x) for x in re.findall(r"\d{4,5}", line)]
            break

    if header_idx is None:
        return {}

    stations = {}
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        sid = parts[0].upper()
        if not re.fullmatch(r"[A-Z]{3}", sid):
            continue

        station_data = {}
        for i, entry in enumerate(parts[1:]):
            if i >= len(altitudes):
                break
            alt = altitudes[i]
            wd, ws, tc = _parse_fd_entry(entry, alt)
            station_data[alt] = {"wind_dir": wd, "wind_spd": ws, "temp_c": tc}

        stations[sid] = station_data

    return stations


# ---------------------------------------------------------------------------
# Altitude interpolation
# ---------------------------------------------------------------------------

def _interpolate_at_altitude(
    station_data: dict[int, dict], target_alt_ft: float
) -> dict[str, float | None]:
    """
    Linearly interpolate wind dir, speed, and temperature between the
    two nearest reporting altitude bands for a given altitude.
    """
    available = sorted(
        alt for alt, v in station_data.items() if v["wind_spd"] is not None
    )
    if not available:
        return {"wind_dir": None, "wind_spd": None, "temp_c": None}

    if target_alt_ft <= available[0]:
        return station_data[available[0]]
    if target_alt_ft >= available[-1]:
        return station_data[available[-1]]

    lower = max(a for a in available if a <= target_alt_ft)
    upper = min(a for a in available if a >= target_alt_ft)
    if lower == upper:
        return station_data[lower]

    ratio = (target_alt_ft - lower) / (upper - lower)
    lo, hi = station_data[lower], station_data[upper]

    def interp(a, b):
        if a is None or b is None:
            return a if b is None else b
        return a + ratio * (b - a)

    return {
        "wind_dir": interp(lo["wind_dir"], hi["wind_dir"]),
        "wind_spd": interp(lo["wind_spd"], hi["wind_spd"]),
        "temp_c":   interp(lo["temp_c"],   hi["temp_c"]),
    }


def _wind_components(
    wind_dir: float | None, wind_spd: float | None, track: float
) -> tuple[float, float]:
    """
    Resolve wind into headwind (+)/tailwind (-) and crosswind components
    relative to the aircraft's track. Returns (headwind_kts, crosswind_kts).
    """
    if wind_dir is None or wind_spd is None:
        return 0.0, 0.0
    angle = np.radians(track - wind_dir)
    return wind_spd * np.cos(angle), wind_spd * np.sin(angle)


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame, region: str = "us") -> pd.DataFrame:
    """
    Fetch NOAA winds-aloft data and add weather columns to the DataFrame.

    Uses per-row nearest-station lookup with haversine distance so that
    a cross-country flight uses geographically appropriate weather data
    at each point along the route, not just a single station for the
    whole flight.

    Added columns
    -------------
    oat_c         : outside air temperature (°C) at flight altitude
    wind_dir      : wind direction (degrees true)
    wind_spd_kts  : wind speed (knots)
    headwind_kts  : headwind component (+= headwind, -= tailwind)
    crosswind_kts : crosswind component (knots)
    wx_station    : which NOAA station was used for each row (useful for debugging)
    wx_time       : row's time_position formatted as "HH:MM:SS UTC"

    Parameters
    ----------
    df     : pd.DataFrame   Output of phase_detector.label_phases()
    region : str            NOAA region code (default "us" for CONUS-wide)
    """
    if df.empty:
        return df

    df = df.copy()

    # Fetch low (3k-18k ft) and high (24k-39k ft) altitude tables
    try:
        raw_low  = _fetch_windtemp_raw(region, "low",  "06")
        raw_high = _fetch_windtemp_raw(region, "high", "06")
    except requests.RequestException as e:
        print(f"[weather] NOAA windtemp fetch failed: {e}. Weather columns will be NaN.")
        for col in ["oat_c", "wind_dir", "wind_spd_kts", "headwind_kts", "crosswind_kts", "wx_station"]:
            df[col] = np.nan
        return df

    # Merge low and high altitude tables
    all_stations = {**_parse_windtemp_table(raw_low)}
    for sid, data in _parse_windtemp_table(raw_high).items():
        if sid in all_stations:
            all_stations[sid].update(data)
        else:
            all_stations[sid] = data

    if not all_stations:
        print("[weather] Could not parse any station data from NOAA response.")
        for col in ["oat_c", "wind_dir", "wind_spd_kts", "headwind_kts", "crosswind_kts", "wx_station"]:
            df[col] = np.nan
        return df

    available_ids = list(all_stations.keys())

    # Per-row enrichment — nearest station selected at each row's actual
    # lat/lon position so cross-country flights get geographically
    # appropriate weather data throughout the route
    oat_list, wdir_list, wspd_list, hw_list, cw_list, stn_list, time_list = [], [], [], [], [], [], []

    # Station coordinates are static for a given available_ids set, so we only
    # need to hit NOAA's stationinfo endpoint periodically rather than once per
    # row. Every STATION_COORD_REFRESH_ROWS rows we refresh the cache; rows in
    # between reuse the last fetched coordinates.
    STATION_COORD_REFRESH_ROWS = 10
    coord_cache = None

    for i, (_, row) in enumerate(df.iterrows()):
        if coord_cache is None or i % STATION_COORD_REFRESH_ROWS == 0:
            coord_cache = _fetch_station_coords_for_ids(available_ids)

        nearest = _nearest_station(row["latitude"], row["longitude"], available_ids, coord_cache)
        stn_list.append(nearest)
        station_data = all_stations.get(nearest, {})

        wx = _interpolate_at_altitude(station_data, row["baro_altitude"])
        hw, cw = _wind_components(wx["wind_dir"], wx["wind_spd"], row.get("true_track", 0))

        oat_list.append(wx["temp_c"])
        wdir_list.append(wx["wind_dir"])
        wspd_list.append(wx["wind_spd"])
        hw_list.append(hw)
        cw_list.append(cw)

        row_time = row.get("time_position")
        time_list.append(
            datetime.fromtimestamp(row_time, tz=timezone.utc).strftime("%H:%M:%S UTC")
            if pd.notna(row_time) else None
        )

    df["oat_c"]         = oat_list
    df["wind_dir"]      = wdir_list
    df["wind_spd_kts"]  = wspd_list
    df["headwind_kts"]  = hw_list
    df["crosswind_kts"] = cw_list
    df["wx_station"]    = stn_list
    df["wx_time"]       = time_list

    return df


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing dynamic station coordinate fetch for specific stations...")
    sample_ids = ["ATL", "RIC", "RDU", "SEA", "ORD"]
    coords = _fetch_station_coords_for_ids(sample_ids)
    print(f"Loaded {len(coords)} station coordinates")
    for sid in sample_ids:
        print(f"  {sid}: {coords.get(sid, 'NOT FOUND (used fallback)')}")

    print("\nTesting nearest station (DCA→ATL midpoint at 37°N, 79°W)...")
    # Simulate stations available in a typical ATL→JFK FD response
    sample_ids = ["ATL", "JFK", "RIC", "RDU", "GSP", "BOS", "ORD", "CLT", "ORF"]
    sample_coords = _fetch_station_coords_for_ids(sample_ids)
    nearest = _nearest_station(37.0, -79.0, sample_ids, sample_coords)
    print(f"  Selected: {nearest}  (expected: RIC or RDU)")

    print("\nTesting FD parser with sample data...")
    sample_fd = """
FT  3000    6000    9000   12000   18000   24000  30000  34000  39000
ATL 9900 3217+15 3114+11 3116+05 2722-08 2740-17 275732 265742 255754
RIC 2510 2615+13 2718+09 2820+04 2935-07 3040-18 304533 295243 286054
"""
    stations = _parse_windtemp_table(sample_fd)
    print(f"  Parsed: {list(stations.keys())}")
    ric_9k = stations.get("RIC", {}).get(9000)
    print(f"  RIC at 9000 ft: {ric_9k}")

    interp = _interpolate_at_altitude(stations["RIC"], 10500)
    print(f"  RIC interpolated at 10,500 ft: {interp}")
    hw, cw = _wind_components(interp["wind_dir"], interp["wind_spd"], 200)
    print(f"  Headwind: {hw:.1f} kts  Crosswind: {cw:.1f} kts (track 200°)")

    #TODO
    # Be able to pull weather data while the plane passes through the airspace, not just at the start and end of the flight
    # This would require a more dynamic approach to fetching and applying weather data, potentially using real-time APIs or more frequent polling of weather conditions along the flight path.
    # Would also mean we would have to pass data to a database as it's ingested to process in real-time
    # Test output and check for accuracy and if it fits desire