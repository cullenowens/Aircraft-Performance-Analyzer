"""
poller.py
Collects live ADS-B state vectors from OpenSky for a specific aircraft,
attaches weather at the moment each point is captured, and writes
directly to the SQLite database (see db.py) — no more JSON files.

Auto-stop on landing
---------------------
The poller tracks which flight phases it has observed (via a lightweight
live heuristic, not the smoothed batch phase_detector) and automatically
stops once it has seen a full flight profile — takeoff, climb, cruise,
descent, landing — followed by two consecutive on_ground readings.

This is a heuristic, not a guarantee: coverage gaps near the ground can
occasionally prevent the "landing" phase from ever being observed. A
safety-net fallback stops polling anyway after several consecutive
on_ground readings regardless of phase history, so the poller can't
hang forever. Ctrl+C always works too.

Usage
-----
  .venv/bin/python3 -m data.poller --icao24 ada6ed
  .venv/bin/python3 -m data.poller --icao24 ada6ed --callsign AAL1002 --dep KATL --arr KJFK
"""

import argparse
import time
from datetime import datetime, timezone

import requests

from auth import tokens
from data.db import (
    get_connection, init_db, create_flight, insert_state_vector,
    mark_flight_landed,
)
from analysis.weather import build_weather_cache, get_weather_for_point

OPENSKY_BASE_URL = "https://opensky-network.org/api"

# Poll intervals in seconds by flight phase — dense during transitions,
# sparse during stable cruise to conserve API credits.
CLIMB_INTERVAL  = 10
CRUISE_INTERVAL = 30

# Safety-net fallback: if this many consecutive on_ground polls happen,
# stop regardless of whether the full phase history was observed. This
# protects against ADS-B coverage gaps near the ground preventing the
# primary landing condition from ever being satisfied.
GROUND_SAFETY_LIMIT = 6

# Full set of live phase hints that must all be observed at least once
# before landing can be auto-confirmed by the primary condition.
REQUIRED_PHASES_FOR_LANDING = {"takeoff", "climb", "cruise", "descent", "landing"}


def check_credits(icao24: str) -> int | None:
    """
    Make a single test request to check how many /states/* credits
    remain before starting a poll session. Returns None if the header
    isn't present, 0 if already rate limited.
    """
    response = requests.get(
        f"{OPENSKY_BASE_URL}/states/all",
        params={"icao24": icao24},
        headers=tokens.headers(),
        timeout=10,
    )
    remaining = response.headers.get("X-Rate-Limit-Remaining")

    if response.status_code == 429:
        secs  = int(response.headers.get("X-Rate-Limit-Retry-After-Seconds", 0))
        hours, mins = secs // 3600, (secs % 3600) // 60
        print(f"[credits] Already rate limited — resets in {hours}h {mins}m.")
        return 0

    return int(remaining) if remaining is not None else None


def _live_phase_hint(on_ground: bool, alt_ft: float | None, vs_fpm: float | None, spd_kts: float | None) -> str:
    """
    Lightweight per-row phase classification used only for the live
    status display and the landing-detection hook. Not a substitute
    for phase_detector.label_phases(), which smooths over the full
    trajectory after landing and is the source of truth for analysis.
    """
    if on_ground:
        return "on_ground"
    if alt_ft is None:
        return "takeoff" if (spd_kts and spd_kts > 50) else "unknown"
    # Landing check comes before the generic low-altitude takeoff check —
    # both conditions can be true at low altitude, but a clearly negative
    # vertical rate means descending toward touchdown, not climbing out.
    if alt_ft < 2500 and vs_fpm is not None and vs_fpm < -200:
        return "landing"
    if alt_ft < 1000:
        return "takeoff"
    if vs_fpm is None:
        return "cruise" if alt_ft > 10000 else "unknown"
    if vs_fpm > 200:
        return "climb"
    if vs_fpm < -200:
        return "descent"
    return "cruise"


def poll_and_store(
    icao24: str,
    conn,
    callsign: str | None = None,
    dep_airport: str | None = None,
    arr_airport: str | None = None,
    climb_interval: int = CLIMB_INTERVAL,
    cruise_interval: int = CRUISE_INTERVAL,
) -> int:
    """
    Poll OpenSky for a specific aircraft, attach weather at each point,
    and write directly to the database. Stops automatically once a full
    flight profile has been observed and the aircraft is confirmed on
    the ground (or via Ctrl+C / safety-net fallback).

    Returns
    -------
    int
        The flight_id of the tracked flight, for use by the orchestrator.
    """
    icao24 = icao24.lower().strip()
    flight_id = create_flight(conn, icao24, callsign, dep_airport, arr_airport)

    print(f"\nChecking API credit balance...")
    credits = check_credits(icao24)
    if credits == 0:
        print("Cannot start polling — no credits remaining. Try again after reset.")
        mark_flight_landed(conn, flight_id)
        return flight_id

    print(f"\nTracking aircraft: {icao24.upper()}  (flight_id={flight_id})")
    print(f"Poll interval:     {climb_interval}s (climb/descent)  {cruise_interval}s (cruise)")
    print(f"Started:           {datetime.now(tz=timezone.utc).strftime('%H:%M:%S UTC')}")
    if credits is not None:
        print(f"Credits remaining: {credits - 1:,}")
    print(f"\nAuto-stops on confirmed landing. Press Ctrl+C to stop manually.\n")
    print(f"{'Time (UTC)':<12} {'Alt (ft)':<10} {'Speed':<10} {'VS':<10} {'Phase':<12} {'Wind':<14} {'OAT':<8} {'Next'}")
    print("-" * 92)

    weather_cache = build_weather_cache()
    seen_phases: set[str] = set()
    consecutive_ground = 0
    row_count = 0
    landed_confirmed = False
    has_been_airborne = False

    try:
        while True:
            if weather_cache.is_stale():
                weather_cache = build_weather_cache()

            try:
                response = requests.get(
                    f"{OPENSKY_BASE_URL}/states/all",
                    params={"icao24": icao24},
                    headers=tokens.headers(),
                    timeout=10,
                )
            except requests.exceptions.ConnectionError as e:
                now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                print(f"{now_str:<12} [connection error — retrying next interval]")
                time.sleep(climb_interval)
                continue
            except requests.exceptions.Timeout:
                now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                print(f"{now_str:<12} [timeout — retrying next interval]")
                time.sleep(climb_interval)
                continue

            if response.status_code == 401:
                response = requests.get(
                    f"{OPENSKY_BASE_URL}/states/all",
                    params={"icao24": icao24},
                    headers=tokens.headers(force_refresh=True),
                    timeout=10,
                )

            if response.status_code == 429:
                secs = int(response.headers.get("X-Rate-Limit-Retry-After-Seconds", 0))
                hours, mins = secs // 3600, (secs % 3600) // 60
                print(f"\n[rate limited] Daily credits exhausted. Resets in ~{hours}h {mins}m.")
                print(f"Saving {row_count} collected rows and exiting...")
                break

            if not response.ok:
                print(f"  [HTTP {response.status_code}] retrying next interval...")
                time.sleep(climb_interval)
                continue

            data = response.json()
            sleep_secs = climb_interval  # default; overridden below when cruising

            if data and data.get("states"):
                state = data["states"][0]

                # Raw OpenSky units: altitude in meters, velocity in m/s
                baro_alt_m  = state[7]
                on_ground   = bool(state[8])
                velocity_ms = state[9]
                true_track  = state[10] or 0.0
                vert_rate_ms = state[11]
                lat, lon = state[6], state[5]

                # Convert to aviation units once, here, at write time —
                # everything downstream (db, phase_detector, dashboard)
                # works in feet/knots/fpm consistently from this point on
                alt_ft  = round(baro_alt_m  * 3.28084) if baro_alt_m  is not None else None
                spd_kts = round(velocity_ms * 1.94384) if velocity_ms is not None else None
                vs_fpm  = round(vert_rate_ms * 196.85) if vert_rate_ms is not None else None

                hint = _live_phase_hint(on_ground, alt_ft, vs_fpm, spd_kts)
                if hint not in ("on_ground", "unknown"):
                    seen_phases.add(hint)
                consecutive_ground = consecutive_ground + 1 if on_ground else 0
                if not on_ground:
                    has_been_airborne = True

                # Weather lookup — pure computation against the cached
                # NOAA tables, no network call on this hot path
                wx = {"oat_c": None, "wind_dir": None, "wind_spd_kts": None,
                      "headwind_kts": None, "crosswind_kts": None, "wx_station": None}
                if lat is not None and lon is not None and alt_ft is not None:
                    wx = get_weather_for_point(lat, lon, alt_ft, true_track, weather_cache)

                insert_state_vector(conn, flight_id, {
                    "time_position": state[3],
                    "latitude": lat,
                    "longitude": lon,
                    "baro_altitude": alt_ft,
                    "velocity": spd_kts,
                    "vertical_rate": vs_fpm,
                    "true_track": true_track,
                    "on_ground": on_ground,
                    **wx,
                })
                row_count += 1

                sleep_secs = cruise_interval if hint == "cruise" else climb_interval

                now_str  = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                alt_str  = f"{alt_ft}ft" if alt_ft is not None else "N/A"
                spd_str  = f"{spd_kts}kt" if spd_kts is not None else "N/A"
                vs_str   = f"{vs_fpm}fpm" if vs_fpm is not None else "N/A"
                wind_str = f"{wx['wind_spd_kts']:.0f}kt@{wx['wind_dir']:.0f}°" if wx.get("wind_spd_kts") is not None else "N/A"
                oat_str  = f"{wx['oat_c']:.0f}C" if wx.get("oat_c") is not None else "N/A"
                print(
                    f"{now_str:<12} {alt_str:<10} {spd_str:<10} {vs_str:<10} "
                    f"{hint:<12} {wind_str:<14} {oat_str:<8} [{sleep_secs}s]"
                )

                # Primary landing condition: full phase profile observed,
                # then 2+ consecutive on_ground readings
                if consecutive_ground >= 2 and REQUIRED_PHASES_FOR_LANDING.issubset(seen_phases):
                    print(f"\n[landing] Full flight profile observed, aircraft on ground. Confirmed landed.")
                    landed_confirmed = True
                    break

                # Safety-net fallback: on_ground for a while regardless
                # of phase history (protects against coverage gaps).
                # Gated on has_been_airborne so this can't fire while the
                # aircraft is still sitting at the gate before departure —
                # see the has_been_airborne comment near its initialization.
                if consecutive_ground >= GROUND_SAFETY_LIMIT and has_been_airborne:
                    missing = REQUIRED_PHASES_FOR_LANDING - seen_phases
                    print(f"\n[landing] On ground for {consecutive_ground} consecutive polls "
                          f"(safety fallback — missing phases: {missing or 'none'}). Stopping.")
                    landed_confirmed = True
                    break

            else:
                now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                print(f"{now_str:<12} [no data — coverage gap]")

            time.sleep(sleep_secs)

    except KeyboardInterrupt:
        print(f"\n\nStopped manually at {datetime.now(tz=timezone.utc).strftime('%H:%M:%S UTC')}")

    print(f"Collected {row_count} state vectors for flight_id={flight_id}")
    mark_flight_landed(conn, flight_id)
    return flight_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poll OpenSky for a live flight, attach weather, and store in SQLite."
    )
    parser.add_argument("--icao24", required=True, help="ICAO24 hex address (e.g. ada6ed)")
    parser.add_argument("--callsign", default=None, help="Flight callsign (optional, for reference)")
    parser.add_argument("--dep", default=None, help="Departure airport ICAO code (optional)")
    parser.add_argument("--arr", default=None, help="Arrival airport ICAO code (optional)")
    parser.add_argument("--climb-interval", type=int, default=CLIMB_INTERVAL)
    parser.add_argument("--cruise-interval", type=int, default=CRUISE_INTERVAL)
    parser.add_argument("--db-path", default=None, help="Override default DB path")
    args = parser.parse_args()

    conn = get_connection(args.db_path) if args.db_path else get_connection()
    init_db(conn)

    poll_and_store(
        icao24=args.icao24,
        conn=conn,
        callsign=args.callsign,
        dep_airport=args.dep,
        arr_airport=args.arr,
        climb_interval=args.climb_interval,
        cruise_interval=args.cruise_interval,
    )

#TODO
#Add tokens printing to know how many tokens are left.