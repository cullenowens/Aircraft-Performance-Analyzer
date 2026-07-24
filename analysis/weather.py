"""
weather.py
Fetches winds-aloft and temperature data from the NOAA Aviation
Weather Center API and enriches flight data with OAT (outside air
temperature) and wind components.

Two ways to use this module:
  1. Live, per-point lookup (used by poller.py during active polling):
       cache = build_weather_cache()
       wx = get_weather_for_point(lat, lon, alt_ft, track, cache)
  2. Batch enrichment of an already-collected DataFrame (used for
     reprocessing old JSON-based flight caches):
       df = enrich(df)

NOAA endpoints used (both free, no API key required):
  /api/data/windtemp    — FD winds/temps aloft forecast text
  /api/data/stationinfo — station coordinates (queried per station set)

Important limitation:
  NOAA's windtemp endpoint returns the *current* forecast cycle, not
  historical data. For live polling this is actually a strength — the
  poller looks up weather at the moment it captures each point, using
  whatever forecast cycle is live right then, which is about as close
  to "real" as this free data source gets. For batch reprocessing of
  old flights, the weather applied is only approximate.

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
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

WINDTEMP_URL    = "https://aviationweather.gov/api/data/windtemp"
STATIONINFO_URL = "https://aviationweather.gov/api/data/stationinfo"
HEADERS         = {"User-Agent": "AircraftPerformanceAnalyzer/1.0 (student project)"}

# How long a WeatherCache stays valid before the poller should refresh it.
# NOAA's FD forecast only updates every 6 hours, so refreshing every 15
# minutes is generous — this just protects a long poll session from
# using one stale cycle for its entire duration.
CACHE_TTL_SECONDS = 15 * 60


# ---------------------------------------------------------------------------
# Live weather cache — built once, refreshed periodically, used per-row
# ---------------------------------------------------------------------------

class WeatherCache:
    """
    Holds a fetched-and-parsed snapshot of NOAA winds-aloft data plus
    station coordinates, so a live poller can look up weather for many
    points without hitting the network on every single poll.

    Build with build_weather_cache(), pass to get_weather_for_point()
    for each polled row. Call is_stale() to know when to rebuild.
    """

    def __init__(self, stations: dict, coords: dict, fetched_at: float):
        self.stations = stations      # { station_id -> { alt_ft -> {wind_dir, wind_spd, temp_c} } }
        self.coords = coords          # { station_id -> (lat, lon) }
        self.fetched_at = fetched_at  # time.time() when built

    def is_stale(self) -> bool:
        return (time.time() - self.fetched_at) > CACHE_TTL_SECONDS


def build_weather_cache(region: str = "us") -> "WeatherCache":
    """
    Fetch and parse NOAA winds-aloft data + station coordinates into a
    WeatherCache. This is the only function that hits the network for
    live polling — call once at poller startup and whenever
    WeatherCache.is_stale() is True.
    """
    try:
        raw_low  = _fetch_windtemp_raw(region, "low",  "06")
        raw_high = _fetch_windtemp_raw(region, "high", "06")
    except requests.RequestException as e:
        print(f"[weather] NOAA windtemp fetch failed: {e}. Cache will be empty.")
        return WeatherCache(stations={}, coords={}, fetched_at=time.time())

    stations = {**_parse_windtemp_table(raw_low)}
    for sid, data in _parse_windtemp_table(raw_high).items():
        if sid in stations:
            stations[sid].update(data)
        else:
            stations[sid] = data

    coords = _fetch_station_coords_for_ids(list(stations.keys()))

    return WeatherCache(stations=stations, coords=coords, fetched_at=time.time())


def get_weather_for_point(
    lat: float, lon: float, alt_ft: float, track: float, cache: "WeatherCache"
) -> dict:
    """
    Look up weather for a single point using an already-built WeatherCache.
    Pure computation, no network call — safe to call once per poll.

    Parameters
    ----------
    lat, lon : float          Aircraft position
    alt_ft   : float          Aircraft barometric altitude in feet
    track    : float          Aircraft true track in degrees
    cache    : WeatherCache   Built by build_weather_cache()

    Returns
    -------
    dict with keys: oat_c, wind_dir, wind_spd_kts, headwind_kts, crosswind_kts, wx_station
    """
    available_ids = list(cache.stations.keys())
    if not available_ids:
        return {
            "oat_c": None, "wind_dir": None, "wind_spd_kts": None,
            "headwind_kts": 0.0, "crosswind_kts": 0.0, "wx_station": None,
        }

    nearest = _nearest_station(lat, lon, available_ids, cache.coords)
    station_data = cache.stations.get(nearest, {})

    wx = _interpolate_at_altitude(station_data, alt_ft)
    hw, cw = _wind_components(wx["wind_dir"], wx["wind_spd"], track)

    return {
        "oat_c": wx["temp_c"],
        "wind_dir": wx["wind_dir"],
        "wind_spd_kts": wx["wind_spd"],
        "headwind_kts": hw,
        "crosswind_kts": cw,
        "wx_station": nearest,
    }


# ---------------------------------------------------------------------------
# Dynamic station coordinate lookup
# ---------------------------------------------------------------------------

def _fetch_station_coords_for_ids(station_ids: list[str]) -> dict[str, tuple[float, float]]:
    """
    Fetch coordinates from NOAA's stationinfo endpoint for a specific set
    of station IDs. The FD data uses bare 3-letter codes (ATL, JFK) but
    NOAA's stationinfo endpoint requires K-prefixed ICAO IDs (KATL, KJFK).
    """
    if not station_ids:
        return _get_fallback_coords()

    k_prefixed = [f"K{sid}" for sid in station_ids]

    try:
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
                bare_id = icao_id.lstrip("K").upper()
                coords[bare_id] = (float(lat), float(lon))

        if coords:
            return coords

    except Exception as e:
        print(f"[weather] Station coordinate fetch failed: {e}. Using fallback dict.")

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
    """Great-circle distance in km between two lat/lon points."""
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
    Return the station ID from available_ids nearest to (lat, lon),
    using haversine distance. Requires pre-fetched coords (no network
    call here — see build_weather_cache() / _fetch_station_coords_for_ids()).
    """
    candidates = {sid: c for sid, c in coords.items() if sid in available_ids}

    if not candidates:
        return available_ids[0] if available_ids else None

    nearest = min(
        candidates.items(),
        key=lambda item: _haversine_km(lat, lon, item[1][0], item[1][1])
    )
    return nearest[0]


# ---------------------------------------------------------------------------
# Fetching FD wind/temp data
# ---------------------------------------------------------------------------
_windtemp_cache: dict[tuple[str, str, str], tuple[str, float]] = {}


def _fetch_windtemp_raw(region: str, level: str, fcst: str) -> str:
    """Fetch raw FD winds/temps text from NOAA. Cached until CACHE_TTL_SECONDS elapses."""
    key = (region, level, fcst)
    cached = _windtemp_cache.get(key)
    if cached is not None and (time.time() - cached[1]) <= CACHE_TTL_SECONDS:
        return cached[0]

    response = requests.get(
        WINDTEMP_URL,
        params={"region": region, "level": level, "fcst": fcst, "layout": "off"},
        headers=HEADERS,
        timeout=15,
    )
    response.raise_for_status()
    _windtemp_cache[key] = (response.text, time.time())
    return response.text


# ---------------------------------------------------------------------------
# FD text parsing
# ---------------------------------------------------------------------------

def _parse_fd_entry(
    entry: str, altitude_ft: int
) -> tuple[float | None, float | None, float | None]:
    """Parse one encoded FD entry into (wind_dir_deg, wind_speed_kts, temp_c)."""
    entry = entry.strip()
    if not entry or entry.startswith("/") or entry == "9900":
        return None, None, None

    if altitude_ft > 24000 and re.fullmatch(r"\d{6}", entry):
        return (
            float(int(entry[0:2]) * 10),
            float(int(entry[2:4])),
            -float(entry[4:6]),
        )

    m = re.fullmatch(r"(\d{2})(\d{2})([+-])(\d{1,2})", entry)
    if m:
        sign = 1.0 if m.group(3) == "+" else -1.0
        return (
            float(int(m.group(1)) * 10),
            float(int(m.group(2))),
            sign * float(m.group(4)),
        )

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
    """Linearly interpolate wind dir, speed, temp between nearest altitude bands."""
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
    """Resolve wind into headwind(+)/tailwind(-) and crosswind relative to track."""
    if wind_dir is None or wind_spd is None:
        return 0.0, 0.0
    angle = np.radians(track - wind_dir)
    return wind_spd * np.cos(angle), wind_spd * np.sin(angle)


# ---------------------------------------------------------------------------
# Batch enrichment — for reprocessing old JSON-based flight caches
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame, region: str = "us") -> pd.DataFrame:
    """
    Fetch NOAA winds-aloft data and add weather columns to an entire
    DataFrame at once. Used for reprocessing old flights collected
    before the live-weather poller existed. New flights get weather
    attached row-by-row during polling instead (see get_weather_for_point).

    Added columns: oat_c, wind_dir, wind_spd_kts, headwind_kts,
    crosswind_kts, wx_station, wx_time
    """
    if df.empty:
        return df

    df = df.copy()
    cache = build_weather_cache(region)

    if not cache.stations:
        for col in ["oat_c", "wind_dir", "wind_spd_kts", "headwind_kts", "crosswind_kts", "wx_station"]:
            df[col] = np.nan
        return df

    oat_list, wdir_list, wspd_list, hw_list, cw_list, stn_list, time_list = [], [], [], [], [], [], []

    for _, row in df.iterrows():
        wx = get_weather_for_point(
            row["latitude"], row["longitude"], row["baro_altitude"],
            row.get("true_track", 0), cache,
        )
        oat_list.append(wx["oat_c"])
        wdir_list.append(wx["wind_dir"])
        wspd_list.append(wx["wind_spd_kts"])
        hw_list.append(wx["headwind_kts"])
        cw_list.append(wx["crosswind_kts"])
        stn_list.append(wx["wx_station"])

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
    print("Testing WeatherCache with sample FD data (no network)...")
    sample_fd = """
FT  3000    6000    9000   12000   18000   24000  30000  34000  39000
ATL 9900 3217+15 3114+11 3116+05 2722-08 2740-17 275732 265742 255754
RIC 2510 2615+13 2718+09 2820+04 2935-07 3040-18 304533 295243 286054
"""
    stations = _parse_windtemp_table(sample_fd)
    coords = {"ATL": (33.64, -84.43), "RIC": (37.51, -77.32)}
    cache = WeatherCache(stations=stations, coords=coords, fetched_at=time.time())

    print(f"Cache stations: {list(cache.stations.keys())}")
    print(f"Cache is_stale: {cache.is_stale()}")

    print("\nLookup near ATL at 9000ft, track 90°:")
    wx = get_weather_for_point(33.7, -84.5, 9000, 90, cache)
    print(f"  {wx}")

    print("\nLookup near RIC at 10500ft, track 200° (interpolated altitude):")
    wx2 = get_weather_for_point(37.5, -77.3, 10500, 200, cache)
    print(f"  {wx2}")

    print("\nTesting staleness after TTL:")
    old_cache = WeatherCache(stations=stations, coords=coords, fetched_at=time.time() - CACHE_TTL_SECONDS - 1)
    print(f"  Old cache is_stale: {old_cache.is_stale()} (expected True)")

#TODO
# Add better weather handling for heights