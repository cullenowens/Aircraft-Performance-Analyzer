"""
aircraft_lookup.py
Resolves an ICAO24 hex address to an aircraft type (ICAO typecode, e.g.
"A321", "C172") using OpenSky's public aircraft metadata database.

OpenSky publishes this as a downloadable CSV aggregating official and
unofficial registry sources (not a documented single-aircraft REST
endpoint — several third-party sites claim one exists, but it isn't
in OpenSky's official REST API docs, so this module deliberately
doesn't depend on it). The CSV is downloaded once and cached locally;
subsequent lookups just read the cache.

IMPORTANT — verify the download URL yourself before relying on this
in anything beyond a portfolio project. AIRCRAFT_DB_CSV_URL below
points at a sample/snapshot URL found via web search. OpenSky's own
data page (https://opensky-network.org/data) is the authoritative
place to get the current full-database download link — check there
if lookups seem to be missing aircraft that should be present.

Coverage caveat: OpenSky states this database updates irregularly and
aggregates from multiple registries, so gaps are expected — especially
for newer registrations. A missing lookup should be treated as "unknown
type", not an error, and the pipeline must degrade gracefully.

Usage
-----
    from data.aircraft_lookup import lookup_typecode

    typecode = lookup_typecode("a1b2c3")   # -> "A321" or None
"""

import csv
import io
import os
import time

import requests

# NOTE: verify against https://opensky-network.org/data before
# depending on this in anything beyond personal/portfolio use.
AIRCRAFT_DB_CSV_URL = "https://s3.opensky-network.org/data-samples/metadata/aircraftDatabase.csv"

CACHE_DIR  = os.path.join(os.path.dirname(__file__), "..", "cache")
CACHE_PATH = os.path.join(CACHE_DIR, "aircraft_database.csv")

# Re-download if the cached copy is older than this. OpenSky states the
# database "updates irregularly" so there's no urgency — this mainly
# guards against ever being stuck with a very stale local copy.
CACHE_MAX_AGE_SECONDS = 30 * 24 * 3600  # 30 days

HEADERS = {"User-Agent": "AircraftPerformanceAnalyzer/1.0 (student project)"}


def _cache_is_stale() -> bool:
    if not os.path.exists(CACHE_PATH):
        return True
    age = time.time() - os.path.getmtime(CACHE_PATH)
    return age > CACHE_MAX_AGE_SECONDS


def download_aircraft_db(force: bool = False) -> bool:
    """
    Download OpenSky's aircraft metadata CSV to the local cache, if it's
    missing or stale. Returns True if a usable cache file exists
    afterward (freshly downloaded or already present), False if the
    download failed and no usable cache exists at all.

    This can be a large file and a slow download on a first run — call
    it explicitly ahead of time (e.g. once during setup) rather than
    relying on it happening transparently inside a time-sensitive path.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not force and not _cache_is_stale():
        return True

    try:
        print("[aircraft_lookup] Downloading OpenSky aircraft database "
              "(one-time, may take a moment)...")
        response = requests.get(AIRCRAFT_DB_CSV_URL, headers=HEADERS, timeout=60, stream=True)
        response.raise_for_status()

        with open(CACHE_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[aircraft_lookup] Cached to {CACHE_PATH}")
        return True

    except requests.RequestException as e:
        print(f"[aircraft_lookup] Download failed: {e}")
        if os.path.exists(CACHE_PATH):
            print("[aircraft_lookup] Falling back to existing (possibly stale) cache.")
            return True
        return False


def lookup_typecode(icao24: str) -> str | None:
    """
    Look up the ICAO typecode (e.g. "A321", "C172") for a given icao24
    hex address. Downloads/caches the aircraft database on first use.

    Returns None if the download failed, the cache doesn't exist, or
    the icao24 isn't found in the database — all of which are normal,
    expected outcomes (coverage gaps are common), not errors. Callers
    must handle None as "type unknown" rather than crashing.
    """
    icao24 = icao24.lower().strip()

    if not download_aircraft_db():
        return None

    try:
        with open(CACHE_PATH, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_icao = (row.get("icao24") or "").strip().lower()
                if row_icao == icao24:
                    typecode = (row.get("typecode") or "").strip()
                    return typecode or None
    except (OSError, csv.Error) as e:
        print(f"[aircraft_lookup] Error reading cached database: {e}")
        return None

    return None


def lookup_full_metadata(icao24: str) -> dict | None:
    """
    Like lookup_typecode() but returns the full metadata row (registration,
    manufacturer, model, typecode, operator, etc.) rather than just the
    typecode. Returns None under the same conditions as lookup_typecode().
    """
    icao24 = icao24.lower().strip()

    if not download_aircraft_db():
        return None

    try:
        with open(CACHE_PATH, "r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_icao = (row.get("icao24") or "").strip().lower()
                if row_icao == icao24:
                    return dict(row)
    except (OSError, csv.Error) as e:
        print(f"[aircraft_lookup] Error reading cached database: {e}")
        return None

    return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        test_icao = sys.argv[1]
    else:
        test_icao = "a1b2c3"
        print(f"No icao24 given, testing with placeholder '{test_icao}' "
              f"(expected to not be found)\n")

    print(f"Looking up: {test_icao}")
    typecode = lookup_typecode(test_icao)
    if typecode:
        print(f"  Typecode: {typecode}")
        meta = lookup_full_metadata(test_icao)
        print(f"  Full metadata: {meta}")
    else:
        print("  Not found (download may have failed, or this icao24 "
              "isn't in OpenSky's database — both are normal outcomes).")