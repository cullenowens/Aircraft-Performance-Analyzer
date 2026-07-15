"""
weather.py
Fetches winds-aloft and temperature data from the NOAA Aviation
Weather Center API and enriches the flight DataFrame with OAT
(outside air temperature) and wind components per row.

NOAA endpoint used:
  https://aviationweather.gov/api/data/windtemp
  ?region=us&level=low&fcst=06&layout=off

The response is a legacy FD (winds/temps aloft forecast) text format:

  FT  3000    6000    9000   12000   18000   24000  30000  34000  39000
  ATL 9900 3217+15 3114+11 3116+05 2722-08 2740-17 275732 265742 255754

Each encoded value like "3217+15" means:
  - Wind direction: 320° (first two digits * 10)
  - Wind speed:      17 knots (next two digits)
  - Temperature:    +15°C (remainder after + or -)

Special cases:
  - "9900" = calm / light and variable winds, no temp
  - Values above 24000 ft omit the +/- sign (temps always negative above FL240)
  - "/////" = data not available
"""

import re
import requests
import pandas as pd
import numpy as np
from functools import lru_cache

WINDTEMP_URL = "https://aviationweather.gov/api/data/windtemp"

# Altitude bands (ft) that NOAA winds-aloft data is reported at
ALTITUDE_BANDS = [3000, 6000, 9000, 12000, 18000, 24000, 30000, 34000, 39000]

# Request headers — NOAA asks for a descriptive user-agent
HEADERS = {"User-Agent": "AircraftPerformanceAnalyzer/1.0 (student project)"}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _fetch_windtemp_raw(region: str, level: str, fcst: str) -> str:
    """
    Fetch raw FD winds/temps text from NOAA. Cached so repeated calls
    for the same region/level/forecast hour don't hit the API again
    during a single session.

    Parameters
    ----------
    region : str   e.g. "us", "bos", "chi", "dfw", "mia", "slc", "sfo"
    level  : str   "low"  (3000–18000 ft) or "high" (24000–39000 ft)
    fcst   : str   forecast hour offset: "06", "12", or "24"
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
# Parsing the FD text format
# ---------------------------------------------------------------------------

def _parse_fd_entry(entry: str, altitude_ft: int) -> tuple[float | None, float | None, float | None]:
    """
    Parse a single FD winds-aloft encoded entry into
    (wind_dir_deg, wind_speed_kts, temp_c).

    Returns (None, None, None) for calm/variable or unavailable entries.

    Examples
    --------
    "3217+15" -> (320, 17, +15.0)
    "2722-08" -> (270, 22,  -8.0)
    "275732"  -> (270, 57, -32.0)  # above 24000 ft, temp always negative
    "9900"    -> (None, None, None) # calm/variable
    "////"    -> (None, None, None) # unavailable
    """
    entry = entry.strip()

    if not entry or entry.startswith("/") or entry == "9900":
        return None, None, None

    # Entries above FL240 are 6 digits with no sign: DDSSTR
    # where DD=direction/10, SS=speed, TR=temp (always negative above 24k)
    if altitude_ft > 24000 and re.fullmatch(r"\d{6}", entry):
        wind_dir = int(entry[0:2]) * 10
        wind_spd = int(entry[2:4])
        temp_c   = -float(entry[4:6])
        return float(wind_dir), float(wind_spd), temp_c

    # Standard format: DDSS+TT or DDSS-TT
    match = re.fullmatch(r"(\d{2})(\d{2})([+-])(\d{1,2})", entry)
    if match:
        wind_dir = int(match.group(1)) * 10
        wind_spd = int(match.group(2))
        sign     = 1.0 if match.group(3) == "+" else -1.0
        temp_c   = sign * float(match.group(4))
        return float(wind_dir), float(wind_spd), temp_c

    # Some low-altitude entries have direction/speed only (no temp at 3000 ft)
    match_notmp = re.fullmatch(r"(\d{2})(\d{2})", entry)
    if match_notmp:
        wind_dir = int(match_notmp.group(1)) * 10
        wind_spd = int(match_notmp.group(2))
        return float(wind_dir), float(wind_spd), None

    return None, None, None


def _parse_windtemp_table(raw_text: str) -> dict[str, dict[int, dict]]:
    """
    Parse the full FD text block into a nested dict:
      { station_id -> { altitude_ft -> { wind_dir, wind_spd, temp_c } } }

    Parameters
    ----------
    raw_text : str
        Raw text from _fetch_windtemp_raw()

    Returns
    -------
    dict
        Nested station -> altitude -> values mapping
    """
    lines = raw_text.strip().splitlines()

    # Find the header line that contains altitude bands
    header_idx = None
    altitudes = []
    for i, line in enumerate(lines):
        if line.strip().startswith("FT"):
            header_idx = i
            # Parse which altitude bands are present in this table
            altitudes = [int(x) for x in re.findall(r"\d{4,5}", line)]
            break

    if header_idx is None:
        return {}

    stations = {}
    for line in lines[header_idx + 1:]:
        # Station lines start with a 3-letter ID followed by data
        parts = line.split()
        if len(parts) < 2:
            continue

        station_id = parts[0].upper()
        if not re.fullmatch(r"[A-Z]{3}", station_id):
            continue

        entries = parts[1:]
        station_data = {}

        for i, entry in enumerate(entries):
            if i >= len(altitudes):
                break
            alt = altitudes[i]
            wind_dir, wind_spd, temp_c = _parse_fd_entry(entry, alt)
            station_data[alt] = {
                "wind_dir": wind_dir,
                "wind_spd": wind_spd,
                "temp_c":   temp_c,
            }

        stations[station_id] = station_data

    return stations


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _interpolate_at_altitude(
    station_data: dict[int, dict], target_alt_ft: float
) -> dict[str, float | None]:
    """
    Linearly interpolate wind direction, speed, and temperature
    between the two nearest altitude bands for a given altitude.
    """
    available_alts = sorted(
        [alt for alt, vals in station_data.items()
         if vals["wind_spd"] is not None]
    )

    if not available_alts:
        return {"wind_dir": None, "wind_spd": None, "temp_c": None}

    # Below lowest band or above highest: use the nearest endpoint
    if target_alt_ft <= available_alts[0]:
        return station_data[available_alts[0]]
    if target_alt_ft >= available_alts[-1]:
        return station_data[available_alts[-1]]

    # Find surrounding bands and interpolate
    lower = max(a for a in available_alts if a <= target_alt_ft)
    upper = min(a for a in available_alts if a >= target_alt_ft)

    if lower == upper:
        return station_data[lower]

    ratio = (target_alt_ft - lower) / (upper - lower)
    lo, hi = station_data[lower], station_data[upper]

    def interp(lo_val, hi_val):
        if lo_val is None or hi_val is None:
            return lo_val if hi_val is None else hi_val
        return lo_val + ratio * (hi_val - lo_val)

    return {
        "wind_dir": interp(lo["wind_dir"], hi["wind_dir"]),
        "wind_spd": interp(lo["wind_spd"], hi["wind_spd"]),
        "temp_c":   interp(lo["temp_c"],   hi["temp_c"]),
    }


def _wind_components(wind_dir_deg: float | None, wind_spd_kts: float | None,
                     aircraft_track_deg: float) -> tuple[float, float]:
    """
    Resolve wind into headwind (+) / tailwind (-) and crosswind components
    relative to the aircraft's track.

    Returns (headwind_kts, crosswind_kts).
    """
    if wind_dir_deg is None or wind_spd_kts is None:
        return 0.0, 0.0

    relative_angle = np.radians(aircraft_track_deg - wind_dir_deg)
    headwind  =  wind_spd_kts * np.cos(relative_angle)
    crosswind =  wind_spd_kts * np.sin(relative_angle)
    return headwind, crosswind


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich(df: pd.DataFrame, region: str = "us") -> pd.DataFrame:
    """
    Fetch NOAA winds-aloft data and add weather columns to the DataFrame.

    Added columns
    -------------
    oat_c        : outside air temperature (°C) interpolated to flight altitude
    wind_dir     : wind direction (degrees true) at flight altitude
    wind_spd_kts : wind speed (knots) at flight altitude
    headwind_kts : headwind component (positive = headwind, negative = tailwind)
    crosswind_kts: crosswind component (knots)

    Parameters
    ----------
    df     : pd.DataFrame   Output of phase_detector.label_phases()
    region : str            NOAA region code (default "us" for CONUS-wide data)
    """
    if df.empty:
        return df

    df = df.copy()

    # Fetch both low (3-18k ft) and high (24-39k ft) altitude tables
    try:
        raw_low  = _fetch_windtemp_raw(region, "low",  "06")
        raw_high = _fetch_windtemp_raw(region, "high", "06")
    except requests.RequestException as e:
        print(f"[weather] NOAA fetch failed: {e}. Weather columns will be NaN.")
        for col in ["oat_c", "wind_dir", "wind_spd_kts", "headwind_kts", "crosswind_kts"]:
            df[col] = np.nan
        return df

    stations_low  = _parse_windtemp_table(raw_low)
    stations_high = _parse_windtemp_table(raw_high)

    # Merge both tables — high-altitude data takes precedence for overlapping stations
    all_stations = {**stations_low}
    for station, data in stations_high.items():
        if station in all_stations:
            all_stations[station].update(data)
        else:
            all_stations[station] = data

    if not all_stations:
        print("[weather] Could not parse any station data from NOAA response.")
        for col in ["oat_c", "wind_dir", "wind_spd_kts", "headwind_kts", "crosswind_kts"]:
            df[col] = np.nan
        return df

    # For simplicity, use the single nearest station to the flight's
    # mean lat/lon. A more advanced version would interpolate spatially
    # between multiple surrounding stations.
    #
    # For a US domestic flight this approximation is reasonable — FD
    # stations are ~200 miles apart and temperature/wind changes
    # gradually at cruise altitudes.
    #
    # Nearest-station lookup uses station IDs as 3-letter airport codes.
    # We pick the closest one geographically using a simple Euclidean
    # approximation (fine for distances under ~500 nm).
    mean_lat = df["latitude"].mean()
    mean_lon = df["longitude"].mean()

    nearest_station = _nearest_station(mean_lat, mean_lon, list(all_stations.keys()))
    station_data = all_stations.get(nearest_station, {})

    # Apply interpolated weather values row by row
    oat_list, wdir_list, wspd_list, hw_list, cw_list = [], [], [], [], []

    for _, row in df.iterrows():
        wx = _interpolate_at_altitude(station_data, row["baro_altitude"])
        hw, cw = _wind_components(wx["wind_dir"], wx["wind_spd"], row.get("true_track", 0))
        oat_list.append(wx["temp_c"])
        wdir_list.append(wx["wind_dir"])
        wspd_list.append(wx["wind_spd"])
        hw_list.append(hw)
        cw_list.append(cw)

    df["oat_c"]         = oat_list
    df["wind_dir"]      = wdir_list
    df["wind_spd_kts"]  = wspd_list
    df["headwind_kts"]  = hw_list
    df["crosswind_kts"] = cw_list

    return df


def _nearest_station(lat: float, lon: float, station_ids: list[str]) -> str | None:
    """
    Return the station ID from station_ids whose 3-letter airport code
    is geographically nearest to (lat, lon).

    Uses aviationweather.gov's airport lookup as a fallback. For now,
    returns the first station as a placeholder — replace with a real
    lookup table or API call for production use.

    In practice, for a first version you can hardcode a small dict of
    common US hub airports and their coordinates.
    """
    # Basic coordinate lookup for common US hubs — expand as needed
    STATION_COORDS = {
        "ATL": (33.64, -84.43), "ORD": (41.98, -87.90), "DFW": (32.90, -97.04),
        "DEN": (39.86, -104.67), "LAX": (33.94, -118.41), "JFK": (40.64, -73.78),
        "SFO": (37.62, -122.38), "SEA": (47.45, -122.31), "MIA": (25.80, -80.29),
        "BOS": (42.36, -71.01), "MSP": (44.88, -93.22), "DTW": (42.21, -83.35),
        "PHL": (39.87, -75.24), "CLT": (35.21, -80.94), "LAS": (36.08, -115.15),
        "PHX": (33.44, -112.01), "IAH": (29.98, -95.34), "MCO": (28.43, -81.31),
        "EWR": (40.69, -74.17), "SLC": (40.79, -111.98),
    }

    known = {sid: coords for sid, coords in STATION_COORDS.items() if sid in station_ids}
    if not known:
        return station_ids[0] if station_ids else None

    nearest = min(
        known.items(),
        key=lambda item: (item[1][0] - lat) ** 2 + (item[1][1] - lon) ** 2
    )
    return nearest[0]


if __name__ == "__main__":
    # Test the parser with a real sample of NOAA FD text (no network call needed)
    sample_fd = """000 FBUS33 KWNO 200800 FD3US3
DATA BASED ON 200600Z
VALID 201800Z   FOR USE 1500-0000Z. TEMPS NEG ABV 24000

FT  3000    6000    9000   12000   18000   24000  30000  34000  39000
ATL 9900 3217+15 3114+11 3116+05 2722-08 2740-17 275732 265742 255754
ORD 2518 2820+08 2931+02 2950-03 2869-14 2977-24 288339 278349 277953
DFW 2016 1911+17 2009+10 2112-07 3024-17 303734 304943 295551"""

    stations = _parse_windtemp_table(sample_fd)
    print("Parsed stations:", list(stations.keys()))

    atl = stations.get("ATL", {})
    print("\nATL at 9000 ft:", atl.get(9000))
    print("ATL at 18000 ft:", atl.get(18000))

    # Test interpolation between bands
    interp = _interpolate_at_altitude(atl, 10500)
    print("\nATL interpolated at 10,500 ft:", interp)

    # Test wind components
    hw, cw = _wind_components(interp["wind_dir"], interp["wind_spd"], 270)
    print(f"Headwind: {hw:.1f} kts, Crosswind: {cw:.1f} kts (for track 270°)")