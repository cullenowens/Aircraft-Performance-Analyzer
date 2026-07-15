"""
fetcher.py
Pulls complete flight data from the OpenSky Network API using a
two-step approach:

  Step 1 — fetch_flights_for_day()
    Given an ICAO24 hex and a date, query the flights endpoint for
    the full day (00:00:00 → 23:59:59). Returns a list of flights
    OpenSky recorded for that aircraft that day.

  Step 2 — fetch_state_vectors()
    Given a flight's firstSeen/lastSeen timestamps, fetch the dense
    (~1 Hz) state vectors for that exact window. Automatically chunks
    into 1-hour requests (OpenSky's max window per call) and
    concatenates the results.

  fetch_flight_track()
    Convenience wrapper that converts the raw state vector response
    into a list of labeled dicts ready for cleaner.py.

Glossary:
  ICAO24    — unique 24-bit address assigned to each aircraft,
              represented as a 6-character hex string (e.g. "a1b2c3")
  firstSeen — Unix timestamp of the flight's first recorded position
  lastSeen  — Unix timestamp of the flight's last recorded position
"""

import json
import os
import time
from datetime import datetime, timezone

import requests

from auth import tokens

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
OPENSKY_BASE_URL = "https://opensky-network.org/api"

# Field order per OpenSky's documented state vector schema.
# Used by both load_polled_track() and any raw state vector conversion.
STATE_VECTOR_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position",
    "last_contact", "longitude", "latitude", "baro_altitude",
    "on_ground", "velocity", "true_track", "vertical_rate",
    "sensors", "geo_altitude", "squawk", "spi", "position_source",
]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> str:
    """Return a path inside the cache directory for the given key."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.json")


def _load_cache(key: str):
    path = _cache_path(key)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_cache(key: str, data) -> None:
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)

# ---------------------------------------------------------------------------
# Primary data source — load from a file written by poller.py
# ---------------------------------------------------------------------------
def load_polled_track(path: str) -> list[dict]:
    """
    Load a flight track collected by poller.py from a local JSON file
    and convert it into labeled dicts ready for cleaner.clean().
 
    This is the primary way flight data enters the analysis pipeline.
    Run poller.py during a live flight to generate the input file,
    then pass the output path here.
 
    Parameters
    ----------
    path : str
        Path to a JSON file written by poller.py.
        Expected format: {"states": [[field0, field1, ...], ...]}
 
    Returns
    -------
    list[dict]
        One dict per state vector with named fields, e.g.:
        {"icao24": "ada6ed", "baro_altitude": 10500.0, ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No polled track found at '{path}'. "
            "Run poller.py during a live flight first to generate this file."
        )
 
    with open(path, "r") as f:
        data = json.load(f)
 
    states = data.get("states", [])
    if not states:
        return []
 
    return [dict(zip(STATE_VECTOR_FIELDS, state)) for state in states]

# ---------------------------------------------------------------------------
# Secondary source — find flights for a given aircraft on a given day
# ---------------------------------------------------------------------------
def fetch_flights_for_day(icao24: str, date: datetime, use_cache: bool = True) -> list[dict]:
    """
    Query OpenSky's flights endpoint for all flights recorded for an
    aircraft on a given calendar day (00:00:00 → 23:59:59 UTC).

    Parameters
    ----------
    icao24 : str
        6-character ICAO hex address of the aircraft (e.g. "a1b2c3").
        Case-insensitive — will be lowercased automatically.
    date : datetime
        The calendar day to search. Only the date portion is used;
        time of day is ignored. Treated as UTC.
    use_cache : bool
        Load from local cache if available, to avoid redundant API calls.

    Returns
    -------
    list[dict]
        Each dict represents one recorded flight with keys:
          icao24, callsign, estDepartureAirport, estArrivalAirport,
          firstSeen (Unix ts), lastSeen (Unix ts)
        Returns an empty list if no flights were found for that day.
    """
    icao24 = icao24.lower().strip()

    # Build start/end Unix timestamps for the full calendar day in UTC
    day_start = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc)
    day_end   = datetime(date.year, date.month, date.day, 23, 59, 59, tzinfo=timezone.utc)
    begin_ts  = int(day_start.timestamp())
    end_ts    = int(day_end.timestamp())

    cache_key = f"flights_{icao24}_{date.strftime('%Y%m%d')}"
    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            return cached

    response = requests.get(
        f"{OPENSKY_BASE_URL}/flights/aircraft",
        params={"icao24": icao24, "begin": begin_ts, "end": end_ts},
        headers=tokens.headers(),
        timeout=15,
    )

    if response.status_code == 401:
        response = requests.get(
            f"{OPENSKY_BASE_URL}/flights/aircraft",
            params={"icao24": icao24, "begin": begin_ts, "end": end_ts},
            headers=tokens.headers(force_refresh=True),
            timeout=15,
        )

    response.raise_for_status()
    flights = response.json()

    # OpenSky returns null for airports it couldn't identify — normalize
    # those to a readable string so the UI doesn't have to handle None
    for flight in flights:
        flight["estDepartureAirport"] = flight.get("estDepartureAirport") or "Unknown"
        flight["estArrivalAirport"]   = flight.get("estArrivalAirport")   or "Unknown"

    _save_cache(cache_key, flights)
    return flights


def format_flights_for_display(flights: list[dict]) -> list[dict]:
    """
    Convert raw flight records into human-readable dicts for the
    Streamlit flight picker UI. Adds formatted departure/arrival times
    in UTC alongside the raw Unix timestamps.

    Parameters
    ----------
    flights : list[dict]
        Output of fetch_flights_for_day()

    Returns
    -------
    list[dict]
        Same records with added 'departure_time_utc' and
        'arrival_time_utc' string fields (e.g. "14:32 UTC").
    """
    display = []
    for f in flights:
        dep_ts = f.get("firstSeen")
        arr_ts = f.get("lastSeen")

        dep_str = (
            datetime.fromtimestamp(dep_ts, tz=timezone.utc).strftime("%H:%M UTC")
            if dep_ts else "Unknown"
        )
        arr_str = (
            datetime.fromtimestamp(arr_ts, tz=timezone.utc).strftime("%H:%M UTC")
            if arr_ts else "Unknown"
        )

        display.append({
            **f,
            "departure_time_utc": dep_str,
            "arrival_time_utc":   arr_str,
            "label": (
                f"{f.get('callsign', 'Unknown').strip() or 'Unknown'} — "
                f"{f['estDepartureAirport']} → {f['estArrivalAirport']} "
                f"({dep_str} – {arr_str})"
            ),
        })
    return display


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test load_polled_track() with the most recent polled file in cache/
    import glob
 
    polled_files = sorted(glob.glob(os.path.join(CACHE_DIR, "live_*.json")))
    if not polled_files:
        print("No polled track files found in cache/.")
        print("Run poller.py during a live flight first:")
        print("  .venv/bin/python3 -m data.poller --icao24 YOUR_HEX")
    else:
        latest = polled_files[-1]
        print(f"Loading polled track: {latest}")
        track = load_polled_track(latest)
        print(f"Loaded {len(track)} state vectors")
        if track:
            print("First record:", track[0])
            print("Last record: ", track[-1])