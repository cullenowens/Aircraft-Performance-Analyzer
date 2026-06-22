#pulls raw ADS-B state vector data from OpenSky API
"""
glossary:
ICAO24: unique 24-bit address assigned to each aircraft, represented as a 6-character hexadecimal string
"""

import requests
import json
import os
import time

from data.auth import tokens

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
OPENSKY_BASE_URL = "https://opensky-network.org/api"

#builds cache file name based on ICAO24, start time, and end time
def cache_path(icao24: str, start: int, end: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    filename = f"{icao24}_{start}_{end}.json"
    return os.path.join(CACHE_DIR, filename)

#fetch state vectors for a single aircraft between two timestamps
def fetch_state_vector(icao: str, start: int, end: int, use_cache: bool = True) -> dict:
    path = cache_path(icao, start, end)
    
    #check if cached data exists
    if use_cache and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    params = {
        "icao24": icao,
        "time": start,
        "end_time": end
    }
    response = requests.get(f"{OPENSKY_BASE_URL}/states/all", params=params, headers=tokens.headers(), timeout=15)
    
    #if the token expired (return 401), force a refresh
    if response.status_code == 401:
        response = requests.get(f"{OPENSKY_BASE_URL}/states/all", params=params, headers=tokens.headers(force_refresh=True), timeout=15)
    
    response.raise_for_status()
    data = response.json()
    #cache the response for future use
    with open(path, "w") as f:
        json.dump(data, f)

    return data

#converts the JSON formatted data to a list of dictionaries
def fetch_flight_track(icao24: str, start: int, end: int, use_cache: bool = True) -> list[dict]:
    raw = fetch_state_vector(icao24, start, end, use_cache)
    if not raw or not raw.get("states"):
        return []
    
    fields = [
        "icao24", "callsign", "origin_country", "time_position",
        "last_contact", "longitude", "latitude", "baro_altitude",
        "on_ground", "velocity", "true_track", "vertical_rate",
        "sensors", "geo_altitude", "squawk", "spi", "position_source",
    ]
    
    records = []
    for state in raw["states"]:
        record = dict(zip(fields, state))
        records.append(record)
    
    return records