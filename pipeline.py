#!/usr/bin/env python3
"""
pipeline.py
Top-level orchestrator — the only script you need to run manually.

Ties together:
  1. data.poller.poll_and_store()        — live polling + weather, writes to DB
  2. data.phase_detector.apply_and_store_phases()  — batch phase labeling, writes back to DB

Runs poller.poll_and_store() until the flight is confirmed landed
(automatically, or via Ctrl+C), then immediately runs phase detection
over the completed trajectory and writes the labels back — no manual
steps in between.

Usage
-----
  .venv/bin/python3 pipeline.py --icao24 ada6ed
  .venv/bin/python3 pipeline.py --icao24 ada6ed --callsign AAL1002 --dep KATL --arr KJFK

After it finishes, query the result any time with:
  from data.db import get_connection, get_flight_dataframe
  conn = get_connection()
  df = get_flight_dataframe(conn, flight_id=<printed at the end>)
"""

import argparse
import sys

from data.db import get_connection, init_db, list_flights
from data.poller import poll_and_store, CLIMB_INTERVAL, CRUISE_INTERVAL
from data.phase_detector import apply_and_store_phases, phase_summary


def track_flight(
    icao24: str,
    callsign: str | None = None,
    dep_airport: str | None = None,
    arr_airport: str | None = None,
    climb_interval: int = CLIMB_INTERVAL,
    cruise_interval: int = CRUISE_INTERVAL,
    db_path: str | None = None,
) -> int:
    """
    Run the full autonomous pipeline for one flight: poll until landed,
    then label phases and write them back. This is the single function
    that replaces the old manual chain of separate script invocations.

    Returns
    -------
    int
        The flight_id, for querying the result afterward.
    """
    conn = get_connection(db_path) if db_path else get_connection()
    init_db(conn)

    # Phase 1 — live polling with weather attached per row, until landed
    flight_id = poll_and_store(
        icao24=icao24,
        conn=conn,
        callsign=callsign,
        dep_airport=dep_airport,
        arr_airport=arr_airport,
        climb_interval=climb_interval,
        cruise_interval=cruise_interval,
    )

    # Phase 2 — batch phase detection over the full trajectory, written back
    print(f"\n[pipeline] Running phase detection for flight_id={flight_id}...")
    labeled_df = apply_and_store_phases(conn, flight_id)

    if labeled_df.empty:
        print(f"[pipeline] No data collected for flight_id={flight_id} — nothing to summarize.")
        conn.close()
        return flight_id

    print(f"[pipeline] Phases labeled and written back to database.\n")
    print("=== Flight Summary ===")
    print(phase_summary(labeled_df))
    print()
    print(f"[pipeline] Done. Query this flight anytime with flight_id={flight_id}:")
    print(f"  from data.db import get_connection, get_flight_dataframe")
    print(f"  conn = get_connection()")
    print(f"  df = get_flight_dataframe(conn, flight_id={flight_id})")

    conn.close()
    return flight_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autonomously track a flight from takeoff to landing: poll, "
                    "attach weather live, and label phases — no manual steps."
    )
    parser.add_argument("--icao24", required=True, help="ICAO24 hex address (e.g. ada6ed)")
    parser.add_argument("--callsign", default=None, help="Flight callsign (optional)")
    parser.add_argument("--dep", default=None, help="Departure airport ICAO code (optional)")
    parser.add_argument("--arr", default=None, help="Arrival airport ICAO code (optional)")
    parser.add_argument("--climb-interval", type=int, default=CLIMB_INTERVAL)
    parser.add_argument("--cruise-interval", type=int, default=CRUISE_INTERVAL)
    parser.add_argument("--db-path", default=None, help="Override default DB path (cache/flights.db)")
    parser.add_argument("--list", action="store_true", help="List all tracked flights and exit")
    args = parser.parse_args()

    if args.list:
        conn = get_connection(args.db_path) if args.db_path else get_connection()
        init_db(conn)
        print(list_flights(conn).to_string(index=False))
        sys.exit(0)

    track_flight(
        icao24=args.icao24,
        callsign=args.callsign,
        dep_airport=args.dep,
        arr_airport=args.arr,
        climb_interval=args.climb_interval,
        cruise_interval=args.cruise_interval,
        db_path=args.db_path,
    )