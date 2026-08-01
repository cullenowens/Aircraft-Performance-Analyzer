"""
db.py
SQLite storage layer for flight tracking data.

Replaces the old per-flight JSON cache files with a single queryable
database. One row per polled state vector, with weather already
attached at insert time (see poller.py) and phase labels filled in
after the flight lands (see phase_detector.apply_and_store_phases()).

Schema
------
flights table       — one row per tracked flight (metadata + status)
state_vectors table — one row per poll, FK to flights.flight_id

Usage
-----
    from data.db import get_connection, init_db, create_flight, insert_state_vector

    conn = get_connection()
    init_db(conn)
    flight_id = create_flight(conn, icao24="ada6ed", callsign="AAL1002")
    insert_state_vector(conn, flight_id, {...})
"""

import sqlite3
import os
from datetime import datetime, timezone

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "cache", "flights.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    flight_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    icao24          TEXT NOT NULL,
    callsign        TEXT,
    dep_airport     TEXT,
    arr_airport     TEXT,
    aircraft_type   TEXT,      -- ICAO typecode e.g. "A321", "C172" — see aircraft_lookup.py
    poll_started_at INTEGER,
    poll_ended_at   INTEGER,
    status          TEXT DEFAULT 'polling'   -- polling | landed | processed
);

CREATE TABLE IF NOT EXISTS state_vectors (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_id       INTEGER NOT NULL REFERENCES flights(flight_id),
    time_position   INTEGER,
    latitude        REAL,
    longitude       REAL,
    baro_altitude   REAL,      -- feet
    velocity        REAL,      -- knots
    vertical_rate   REAL,      -- feet per minute
    true_track      REAL,
    on_ground       INTEGER,   -- 0/1
    phase           TEXT,      -- filled in by phase_detector after landing
    oat_c           REAL,
    wind_dir        REAL,
    wind_spd_kts    REAL,
    headwind_kts    REAL,
    crosswind_kts   REAL,
    wx_station      TEXT
);

CREATE INDEX IF NOT EXISTS idx_state_vectors_flight_id ON state_vectors(flight_id);
CREATE INDEX IF NOT EXISTS idx_state_vectors_time ON state_vectors(flight_id, time_position);
"""


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open (creating if needed) a SQLite connection with row access by
    column name, so query results can be used like dicts.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create tables and indexes if they don't already exist. Safe to call
    every run. Also runs a small migration step for databases created
    before the aircraft_type column existed — CREATE TABLE IF NOT EXISTS
    alone won't add a column to an already-existing table.
    """
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate_add_aircraft_type_column(conn)


def _migrate_add_aircraft_type_column(conn: sqlite3.Connection) -> None:
    """
    Add the aircraft_type column to an existing flights table if it's
    missing (i.e. the database was created before this column existed).
    No-op if the column is already present.
    """
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(flights)").fetchall()}
    if "aircraft_type" not in existing_cols:
        conn.execute("ALTER TABLE flights ADD COLUMN aircraft_type TEXT")
        conn.commit()


def create_flight(
    conn: sqlite3.Connection,
    icao24: str,
    callsign: str | None = None,
    dep_airport: str | None = None,
    arr_airport: str | None = None,
    aircraft_type: str | None = None,
) -> int:
    """
    Insert a new flight record and return its flight_id.
    Called once at the start of a poll session.

    aircraft_type is the ICAO typecode (e.g. "A321", "C172"). Pass it
    explicitly if known, or leave None and populate later via
    set_aircraft_type() — see data.aircraft_lookup.lookup_typecode()
    for resolving it automatically from the icao24 hex.
    """
    now = int(datetime.now(tz=timezone.utc).timestamp())
    cur = conn.execute(
        """
        INSERT INTO flights (icao24, callsign, dep_airport, arr_airport, aircraft_type, poll_started_at, status)
        VALUES (?, ?, ?, ?, ?, ?, 'polling')
        """,
        (icao24.lower(), callsign, dep_airport, arr_airport, aircraft_type, now),
    )
    conn.commit()
    return cur.lastrowid


def set_aircraft_type(conn: sqlite3.Connection, flight_id: int, aircraft_type: str) -> None:
    """
    Populate (or update) a flight's aircraft_type after the fact — e.g.
    once data.aircraft_lookup.lookup_typecode() resolves it, which may
    happen lazily rather than at create_flight() time.
    """
    conn.execute(
        "UPDATE flights SET aircraft_type = ? WHERE flight_id = ?",
        (aircraft_type, flight_id),
    )
    conn.commit()


def insert_state_vector(conn: sqlite3.Connection, flight_id: int, row: dict) -> None:
    """
    Insert one polled state vector (with weather already attached) for a flight.

    Parameters
    ----------
    row : dict
        Expected keys: time_position, latitude, longitude, baro_altitude,
        velocity, vertical_rate, true_track, on_ground, oat_c, wind_dir,
        wind_spd_kts, headwind_kts, crosswind_kts, wx_station.
        Missing keys default to None.
    """
    conn.execute(
        """
        INSERT INTO state_vectors (
            flight_id, time_position, latitude, longitude, baro_altitude,
            velocity, vertical_rate, true_track, on_ground,
            oat_c, wind_dir, wind_spd_kts, headwind_kts, crosswind_kts, wx_station
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            flight_id,
            row.get("time_position"),
            row.get("latitude"),
            row.get("longitude"),
            row.get("baro_altitude"),
            row.get("velocity"),
            row.get("vertical_rate"),
            row.get("true_track"),
            int(bool(row.get("on_ground"))) if row.get("on_ground") is not None else None,
            row.get("oat_c"),
            row.get("wind_dir"),
            row.get("wind_spd_kts"),
            row.get("headwind_kts"),
            row.get("crosswind_kts"),
            row.get("wx_station"),
        ),
    )
    conn.commit()


def mark_flight_landed(conn: sqlite3.Connection, flight_id: int) -> None:
    """Mark a flight as landed (polling stopped) with an end timestamp."""
    now = int(datetime.now(tz=timezone.utc).timestamp())
    conn.execute(
        "UPDATE flights SET status = 'landed', poll_ended_at = ? WHERE flight_id = ?",
        (now, flight_id),
    )
    conn.commit()


def mark_flight_processed(conn: sqlite3.Connection, flight_id: int) -> None:
    """Mark a flight as fully processed (phases labeled and written back)."""
    conn.execute(
        "UPDATE flights SET status = 'processed' WHERE flight_id = ?",
        (flight_id,),
    )
    conn.commit()


def delete_flight(conn: sqlite3.Connection, flight_id: int) -> int:
    """
    Delete a flight and all its state vectors. Useful for clearing out
    test data during development.

    Returns
    -------
    int
        Number of state_vectors rows deleted.
    """
    cur = conn.execute("DELETE FROM state_vectors WHERE flight_id = ?", (flight_id,))
    deleted_rows = cur.rowcount
    conn.execute("DELETE FROM flights WHERE flight_id = ?", (flight_id,))
    conn.commit()
    return deleted_rows


def delete_all_flights(conn: sqlite3.Connection) -> None:
    """
    Wipe every flight and state vector from the database — a full reset.
    Equivalent to deleting the .db file, but keeps the file/schema in place.
    """
    conn.execute("DELETE FROM state_vectors")
    conn.execute("DELETE FROM flights")
    conn.commit()


def update_phases(conn: sqlite3.Connection, phase_updates: list[tuple[str, int]]) -> None:
    """
    Bulk-write phase labels back to state_vectors rows.

    Parameters
    ----------
    phase_updates : list of (phase, row_id) tuples — matches sqlite3 executemany order
    """
    conn.executemany(
        "UPDATE state_vectors SET phase = ? WHERE id = ?",
        phase_updates,
    )
    conn.commit()


def get_flight_dataframe(conn: sqlite3.Connection, flight_id: int):
    """
    Load all state vectors for a flight as a pandas DataFrame, ordered
    by time. Includes weather columns and phase (if already labeled).
    """
    import pandas as pd

    return pd.read_sql_query(
        "SELECT * FROM state_vectors WHERE flight_id = ? ORDER BY time_position",
        conn,
        params=(flight_id,),
    )


def list_flights(conn: sqlite3.Connection):
    """Return all flight records as a pandas DataFrame — useful for a dashboard picker."""
    import pandas as pd

    return pd.read_sql_query(
        "SELECT * FROM flights ORDER BY poll_started_at DESC", conn
    )


def get_flight_meta(conn: sqlite3.Connection, flight_id: int) -> dict | None:
    """Return a single flight's metadata row as a dict, or None if not found."""
    row = conn.execute(
        "SELECT * FROM flights WHERE flight_id = ?", (flight_id,)
    ).fetchone()
    return dict(row) if row else None


if __name__ == "__main__":
    # Manual test — create an in-memory DB, insert a flight and a few
    # rows, verify round-trip through get_flight_dataframe()
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    init_db(conn)

    fid = create_flight(conn, icao24="ada6ed", callsign="AAL1002", dep_airport="KATL", arr_airport="KJFK")
    print(f"Created flight_id: {fid}")

    for i in range(5):
        insert_state_vector(conn, fid, {
            "time_position": 1783512680 + i * 10,
            "latitude": 33.64 + i * 0.05,
            "longitude": -84.43 + i * 0.05,
            "baro_altitude": 1000.0 + i * 500,
            "velocity": 150.0 + i * 10,
            "vertical_rate": 1500.0,
            "true_track": 45.0,
            "on_ground": False,
            "oat_c": 15.0 - i,
            "wind_dir": 270.0,
            "wind_spd_kts": 20.0,
            "headwind_kts": 5.0,
            "crosswind_kts": 3.0,
            "wx_station": "ATL",
        })

    mark_flight_landed(conn, fid)

    df = get_flight_dataframe(conn, fid)
    print(f"\nRetrieved {len(df)} rows")
    print(df[["time_position", "baro_altitude", "velocity", "oat_c", "wx_station"]])

    print("\nFlight metadata:")
    print(get_flight_meta(conn, fid))

    print("\nAll flights:")
    print(list_flights(conn))