"""
poller.py
Collects live ADS-B state vectors from OpenSky for a specific aircraft
while a flight is happening, and saves them to a local JSON file.

Usage
-----
Start this before or shortly after the flight departs. Stop it with
Ctrl+C after the aircraft lands. The saved file is then fed into
the analysis pipeline via fetcher.load_polled_track().

  .venv/bin/python3 -m data.poller --icao24 ada6ed

Or with a custom output path and poll interval:

  .venv/bin/python3 -m data.poller --icao24 ada6ed --interval 15 --output cache/my_flight.json

The output file format matches OpenSky's /states/all response so it's
compatible with the same field mapping used everywhere else in the pipeline.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import requests

from auth import tokens

OPENSKY_BASE_URL = "https://opensky-network.org/api"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")

# Poll intervals in seconds by flight phase.
# Dense during transitions (takeoff/climb/descent) where data changes
# rapidly, sparse during cruise where altitude and speed are stable.
CLIMB_INTERVAL  = 10   # takeoff, climb, descent, landing
CRUISE_INTERVAL = 30   # cruise only


def check_credits(icao24: str) -> int | None:
    """
    Make a single test request to check how many /states/* credits
    remain before starting a poll session.

    Returns the remaining credit count, or None if the header isn't
    present (e.g. anonymous requests don't include it).
    """
    response = requests.get(
        f"{OPENSKY_BASE_URL}/states/all",
        params={"icao24": icao24},
        headers=tokens.headers(),
        timeout=10,
    )
    remaining = response.headers.get("X-Rate-Limit-Remaining")
    retry_after = response.headers.get("X-Rate-Limit-Retry-After-Seconds")

    if response.status_code == 429:
        secs  = int(retry_after or 0)
        hours = secs // 3600
        mins  = (secs % 3600) // 60
        print(f"[credits] Already rate limited — resets in {hours}h {mins}m.")
        return 0

    return int(remaining) if remaining is not None else None


def poll_flight(
    icao24: str,
    output_path: str,
    climb_interval: int = CLIMB_INTERVAL,
    cruise_interval: int = CRUISE_INTERVAL,
) -> list:
    """
    Poll OpenSky for a specific aircraft with dynamic interval adjustment:
    polls frequently during climb/descent and less often during cruise
    to save API credits without losing resolution where it matters.

    Parameters
    ----------
    icao24          : str   ICAO24 hex of the aircraft to track
    output_path     : str   File path to save collected state vectors
    climb_interval  : int   Seconds between polls during climb/descent (default 10)
    cruise_interval : int   Seconds between polls during cruise (default 30)
    """
    icao24 = icao24.lower().strip()
    records = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check credit balance before starting — each serial-only /states/all
    # call costs 1 credit, so we can estimate how long we can poll.
    print(f"\nChecking API credit balance...")
    credits = check_credits(icao24)

    if credits == 0:
        print("Cannot start polling — no credits remaining. Try again after reset.")
        return records

    print(f"\nTracking aircraft: {icao24.upper()}")
    print(f"Poll interval:     {climb_interval}s (climb/descent)  {cruise_interval}s (cruise)")
    print(f"Saving to:         {output_path}")
    print(f"Started:           {datetime.now(tz=timezone.utc).strftime('%H:%M:%S UTC')}")

    if credits is not None:
        credits_available = credits - 1
        # Estimate using a blended rate: assume ~20% of flight is
        # transitions and ~80% is cruise (conservative for a long flight)
        blended_interval = (0.2 * climb_interval) + (0.8 * cruise_interval)
        max_mins = int((credits_available * blended_interval) // 60)
        hours    = max_mins // 60
        mins     = max_mins % 60
        print(f"Credits remaining: {credits_available:,}  "
              f"(~{hours}h {mins}m estimated at blended rate)")
        if credits_available < 200:
            print(f"  ⚠ Low credits — consider increasing cruise interval")
    else:
        print(f"Credits remaining: unknown (header not returned)")

    print(f"\nPress Ctrl+C when the flight lands to stop and save.\n")
    print(f"{'Time (UTC)':<12} {'Alt (ft)':<12} {'Speed (kts)':<14} {'VS (fpm)':<12} {'Phase hint':<16} {'Credits left'}")
    print("-" * 78)

    try:
        while True:
            response = requests.get(
                f"{OPENSKY_BASE_URL}/states/all",
                params={"icao24": icao24},
                headers=tokens.headers(),
                timeout=10,
            )

            # Handle token expiry
            if response.status_code == 401:
                response = requests.get(
                    f"{OPENSKY_BASE_URL}/states/all",
                    params={"icao24": icao24},
                    headers=tokens.headers(force_refresh=True),
                    timeout=10,
                )

            # Handle rate limiting — save data and exit rather than
            # sleeping for hours, which just freezes the terminal
            if response.status_code == 429:
                secs  = int(response.headers.get("X-Rate-Limit-Retry-After-Seconds", 0))
                hours = secs // 3600
                mins  = (secs % 3600) // 60
                print(f"\n[rate limited] Daily /states/* credits exhausted.")
                print(f"Credits reset in approximately {hours}h {mins}m (at next UTC midnight).")
                print(f"Tip: use --interval 30 to use fewer credits per flight.")
                print(f"Saving {len(records)} collected vectors and exiting...")
                break

            if response.ok:
                # Update running credit count from response header
                remaining_hdr = response.headers.get("X-Rate-Limit-Remaining")
                if remaining_hdr is not None:
                    credits = int(remaining_hdr)

                data = response.json()
                if data and data.get("states"):
                    state = data["states"][0]
                    records.append(state)

                    # OpenSky returns altitude in meters, velocity in m/s —
                    # convert to ft and knots for the status display.
                    baro_alt_m  = state[7]   # baro_altitude (meters)
                    on_ground   = state[8]   # on_ground (bool)
                    velocity_ms = state[9]   # velocity (m/s)
                    vert_rate   = state[11]  # vertical_rate (m/s)

                    # Use `is not None` so 0.0 values aren't treated as
                    # missing — a VS of exactly 0 is valid cruise data
                    alt_ft  = round(baro_alt_m  * 3.28084) if baro_alt_m  is not None else None
                    spd_kts = round(velocity_ms * 1.94384) if velocity_ms is not None else None
                    vs_fpm  = round(vert_rate   * 196.85)  if vert_rate   is not None else None

                    # Phase hint uses altitude + speed context so takeoff
                    # roll and cruise with null VS are handled correctly
                    if on_ground:
                        hint = "on ground" if spd_kts and spd_kts < 30 else "↑ takeoff" if spd_kts else "unknown"
                    elif alt_ft is None:
                        hint = "↑ takeoff" if (spd_kts and spd_kts > 50) else "unknown"
                    elif alt_ft < 1000:
                        hint = "↑ takeoff" if (vs_fpm and vs_fpm > 0) else "↓ landing" if (vs_fpm and vs_fpm < 0) else "unknown"
                    elif vs_fpm is None:
                        hint = "→ cruise" if alt_ft > 10000 else "unknown"
                    elif vs_fpm > 200:
                        hint = "↑ climbing"
                    elif vs_fpm < -200:
                        hint = "↓ descending"
                    else:
                        hint = "→ cruise"

                    now_str     = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                    alt_str     = f"{alt_ft} ft"   if alt_ft  is not None else "N/A"
                    spd_str     = f"{spd_kts} kts" if spd_kts is not None else "N/A"
                    vs_str      = f"{vs_fpm} fpm"  if vs_fpm  is not None else "N/A"
                    credits_str = str(credits) if credits is not None else "?"

                    # Dynamic interval: slow down during cruise to save credits
                    sleep_secs = cruise_interval if hint == "→ cruise" or hint == "on ground" else climb_interval

                    print(
                        f"{now_str:<12} "
                        f"{alt_str:<12} "
                        f"{spd_str:<14} "
                        f"{vs_str:<12} "
                        f"{hint:<16} "
                        f"{credits_str:<10} "
                        f"[next: {sleep_secs}s]"
                    )
                else:
                    now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                    sleep_secs = climb_interval  # default to dense when no data
                    print(f"{now_str:<12} [no data — coverage gap or flight ended]")

            else:
                print(f"  [HTTP {response.status_code}] retrying next interval...")
                sleep_secs = climb_interval

            time.sleep(sleep_secs)

    except KeyboardInterrupt:
        print(f"\n\nStopped at {datetime.now(tz=timezone.utc).strftime('%H:%M:%S UTC')}")
        print(f"Collected {len(records)} state vectors")

    # Save everything — even if zero records, write a valid empty file
    # so load_polled_track() doesn't crash on a partial run
    payload = {"states": records, "icao24": icao24}
    with open(output_path, "w") as f:
        json.dump(payload, f)

    print(f"Saved to {output_path}")
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Poll OpenSky for a live flight and save state vectors to disk."
    )
    parser.add_argument(
        "--icao24",
        required=True,
        help="ICAO24 hex address of the aircraft (e.g. ada6ed)",
    )
    parser.add_argument(
        "--climb-interval",
        type=int,
        default=CLIMB_INTERVAL,
        help=f"Poll interval in seconds during climb/descent (default: {CLIMB_INTERVAL})",
    )
    parser.add_argument(
        "--cruise-interval",
        type=int,
        default=CRUISE_INTERVAL,
        help=f"Poll interval in seconds during cruise (default: {CRUISE_INTERVAL})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: cache/live_<icao24>_<timestamp>.json)",
    )
    args = parser.parse_args()

    # Auto-generate output path if not specified, including a timestamp
    # so multiple flights for the same aircraft don't overwrite each other
    if args.output is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(CACHE_DIR, f"live_{args.icao24.lower()}_{ts}.json")

    poll_flight(
        icao24=args.icao24,
        output_path=args.output,
        climb_interval=args.climb_interval,
        cruise_interval=args.cruise_interval,
    )