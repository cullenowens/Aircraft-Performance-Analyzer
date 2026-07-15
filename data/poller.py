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

# Default poll interval in seconds. 10s gives good density (~360
# points/hour) while staying well within the 4,000 daily credit limit
# even for long flights. Each poll = 1 API call.
DEFAULT_INTERVAL = 10


def poll_flight(
    icao24: str,
    output_path: str,
    interval: int = DEFAULT_INTERVAL,
) -> list:
    """
    Poll OpenSky every `interval` seconds for a specific aircraft and
    append each state vector to a local JSON file.

    Prints a live status line each poll so you can confirm data is
    flowing. Press Ctrl+C to stop — the file is saved automatically.

    Parameters
    ----------
    icao24      : str   ICAO24 hex of the aircraft to track
    output_path : str   File path to save collected state vectors
    interval    : int   Seconds between polls (default 10)

    Returns
    -------
    list
        All collected raw state vectors (also saved to output_path).
    """
    icao24 = icao24.lower().strip()
    records = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"\nTracking aircraft: {icao24.upper()}")
    print(f"Poll interval:     every {interval}s")
    print(f"Saving to:         {output_path}")
    print(f"Started:           {datetime.now(tz=timezone.utc).strftime('%H:%M:%S UTC')}")
    print(f"\nPress Ctrl+C when the flight lands to stop and save.\n")
    print(f"{'Time (UTC)':<12} {'Alt (ft)':<12} {'Speed (kts)':<14} {'VS (fpm)':<12} {'Phase hint'}")
    print("-" * 62)

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

            # Handle rate limiting — wait then retry
            if response.status_code == 429:
                retry = int(
                    response.headers.get("X-Rate-Limit-Retry-After-Seconds", 30)
                )
                print(f"  [rate limited] waiting {retry}s...")
                time.sleep(retry)
                continue

            if response.ok:
                data = response.json()
                if data and data.get("states"):
                    state = data["states"][0]
                    records.append(state)

                    # Unpack the fields we care about for the status line.
                    # OpenSky returns altitude in meters and velocity in m/s
                    # so convert to ft and knots for readability.
                    baro_alt_m  = state[7]   # baro_altitude (meters)
                    on_ground   = state[8]   # on_ground (bool)
                    velocity_ms = state[9]   # velocity (m/s)
                    vert_rate   = state[11]  # vertical_rate (m/s)
                    #should handle these to not return as none -> should be 0 and depending on prior phases know phase
                    alt_ft  = round(baro_alt_m * 3.28084) if baro_alt_m  else None
                    spd_kts = round(velocity_ms * 1.94384) if velocity_ms else None
                    vs_fpm  = round(vert_rate * 196.85)   if vert_rate is not None else None

                    # Simple phase hint based on vertical rate
                    # need to adjust to have vs_fpm corrently represented along with phase hint -- eg. plane is shown as cruising during takeoff, plane is shown as unknown while cruising
                    if on_ground:
                        hint = "on ground"
                    elif vs_fpm is None:
                        hint = "unknown"
                    elif vs_fpm < 100 and spd_kts > 0 and spd_kts < 200:
                        hint = "→ takeoff"
                    elif vs_fpm > 200:
                        hint = "↑ climbing"
                    elif vs_fpm < -200:
                        hint = "↓ descending"
                    else:
                        hint = "→ cruise"

                    now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                    print(
                        f"{now_str:<12} "
                        f"{str(alt_ft) + ' ft' if alt_ft is not None else 'N/A':<12} "
                        f"{str(spd_kts) + ' kts' if spd_kts is not None else 'N/A':<14} "
                        f"{str(vs_fpm) + ' fpm' if vs_fpm is not None else 'N/A':<12} "
                        f"{hint}"
                    )
                else:
                    now_str = datetime.now(tz=timezone.utc).strftime("%H:%M:%S")
                    print(f"{now_str:<12} [no data — coverage gap or flight ended]")

            else:
                print(f"  [HTTP {response.status_code}] retrying next interval...")

            time.sleep(interval)

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
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL,
        help=f"Poll interval in seconds (default: {DEFAULT_INTERVAL})",
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
        interval=args.interval,
    )