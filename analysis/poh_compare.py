"""
poh_compare.py
Compares observed flight performance against published reference data.

Reference data lives in poh_data/*.json — one file per aircraft type.
This module is deliberately aircraft-agnostic: it knows how to read the
schema, interpolate a table, and compute deltas, but nothing about any
specific aircraft.

Two important honesty constraints are built into this module rather
than left to the user to remember:

  1. Weight is unknown. ADS-B does not transmit aircraft weight, and
     weight is one of the strongest drivers of climb performance. Every
     climb comparison therefore carries a caveat, and large deviations
     should not be read as anomalies without knowing the load.

  2. Not all reference data is equal. A certified POH climb table and a
     "typical airline profile" are very different kinds of claim. Each
     JSON file declares its own source_confidence, and that text is
     surfaced in every report this module generates so the distinction
     never gets lost downstream.

Lookup is done against density altitude rather than pressure altitude,
which folds temperature into a single axis. This is a standard
simplification, not an exact reproduction of a two-axis POH lookup.
"""

import json
import os

import numpy as np
import pandas as pd

from analysis.performance import (
    climb_performance, cruise_performance, descent_performance, flight_summary,
)

POH_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "poh_data")

# Maps an ICAO aircraft typecode (as resolved by data.aircraft_lookup)
# to the poh_data/*.json filename (without extension) that describes it.
# Extend this whenever a new reference file is added — a typecode with
# no entry here (or no reference file at all) is a normal, expected
# case, not an error; generate_report() falls back to a descriptive-
# only report rather than failing.
TYPECODE_TO_REFERENCE = {
    "A321": "a321",
    "A21N": "a21n",
    "A359": "a359",
    "B738": "b738",
    "B739": "b739",
    "B752": "b752",
    "B763": "b763",
    "C172": "c172_s",
    "C172S": "c172_s",
}


def reference_key_for_typecode(typecode: str | None) -> str | None:
    """
    Map an ICAO typecode to a reference file key, or None if there's
    no reference data for that type. Case-insensitive.
    """
    if not typecode:
        return None
    return TYPECODE_TO_REFERENCE.get(typecode.strip().upper())


# ---------------------------------------------------------------------------
# Loading reference data
# ---------------------------------------------------------------------------

def list_available_references() -> list[str]:
    """Return the keys of every reference file in poh_data/ (filename without .json)."""
    if not os.path.isdir(POH_DATA_DIR):
        return []
    return sorted(
        f[:-5] for f in os.listdir(POH_DATA_DIR) if f.endswith(".json")
    )


def load_reference(key: str) -> dict:
    """
    Load a reference performance file by key (filename without .json).

    Raises FileNotFoundError with the available options listed, since
    a typo here is the most likely failure mode.
    """
    path = os.path.join(POH_DATA_DIR, f"{key}.json")
    if not os.path.exists(path):
        available = list_available_references()
        raise FileNotFoundError(
            f"No reference data for '{key}'. Available: {available or '(none)'}"
        )
    with open(path, "r") as f:
        return json.load(f)


def reference_caveat(reference: dict) -> str:
    """
    Build the disclaimer string that should accompany any output derived
    from this reference file. Pulled from the file itself so it can't
    drift out of sync with the data.
    """
    meta = reference.get("aircraft", {})
    lines = [
        f"Aircraft:   {meta.get('type', 'unknown')}",
        f"Source:     {meta.get('source', 'unspecified')}",
        f"Confidence: {meta.get('source_confidence', 'unspecified')}",
    ]
    if not meta.get("verified", False):
        lines.append(
            "STATUS:     UNVERIFIED — these reference numbers have not been "
            "checked against an authoritative document."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table interpolation
# ---------------------------------------------------------------------------

def _interp_table(table: list[dict], x_key: str, y_key: str, x_value: float):
    """
    Linearly interpolate y from a reference table at a given x.

    Returns (value, in_range). in_range is False when x_value falls
    outside the table's span — the value is then clamped to the nearest
    endpoint, and callers should treat it as an extrapolation rather
    than a real reference figure.
    """
    pts = sorted(
        [(float(r[x_key]), float(r[y_key])) for r in table if r.get(y_key) is not None],
        key=lambda p: p[0],
    )
    if not pts:
        return None, False

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    if x_value <= xs[0]:
        return ys[0], x_value >= xs[0]
    if x_value >= xs[-1]:
        return ys[-1], x_value <= xs[-1]

    return float(np.interp(x_value, xs, ys)), True


def expected_climb_rate(reference: dict, density_alt_ft: float):
    """Expected rate of climb (fpm) at a given density altitude."""
    climb = reference.get("climb", {})
    return _interp_table(climb.get("table", []), "density_altitude_ft", "roc_fpm", density_alt_ft)


def expected_climb_speed(reference: dict, density_alt_ft: float):
    """Expected climb speed (KIAS) at a given density altitude."""
    climb = reference.get("climb", {})
    return _interp_table(climb.get("table", []), "density_altitude_ft", "kias", density_alt_ft)


def expected_cruise_speed(reference: dict, density_alt_ft: float):
    """
    Expected cruise speed at a given density altitude, plus the units.
    Returns (value, in_range, speed_type) where speed_type is 'mach'
    or 'ktas' depending on how the reference file expresses it.
    """
    cruise = reference.get("cruise", {})
    speed_type = cruise.get("speed_type", "ktas")
    val, in_range = _interp_table(
        cruise.get("table", []), "density_altitude_ft", speed_type, density_alt_ft
    )
    return val, in_range, speed_type


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------

def compare_climb(df: pd.DataFrame, reference: dict, band_ft: int = 5000) -> pd.DataFrame:
    """
    Compare observed climb rate per altitude band against the reference
    table, using the band's mean density altitude as the lookup key.

    Returns a DataFrame with actual, expected, delta, and percent-of-
    expected columns, plus an in_range flag marking bands that fell
    outside the reference table's altitude span.
    """
    from analysis.performance import prepare

    bands = climb_performance(df, band_ft)
    if bands.empty:
        return pd.DataFrame()

    # Recompute mean density altitude per band for the lookup
    d = prepare(df)
    climb_rows = d[d["phase"].isin(["takeoff", "climb"])].copy()
    climb_rows["band"] = (climb_rows["baro_altitude"] // band_ft).astype(int) * band_ft
    da_by_band = climb_rows.groupby("band")["density_alt_ft"].mean()

    out = []
    for _, row in bands.iterrows():
        band = int(row["band_low_ft"])
        mean_da = float(da_by_band.get(band, band))

        exp_roc, roc_in_range = expected_climb_rate(reference, mean_da)
        exp_kias, _ = expected_climb_speed(reference, mean_da)

        actual = float(row["roc_fpm"])
        delta = actual - exp_roc if exp_roc is not None else None
        pct = (actual / exp_roc * 100.0) if exp_roc else None

        out.append({
            "band_low_ft":     band,
            "band_high_ft":    int(row["band_high_ft"]),
            "mean_density_alt_ft": round(mean_da, 0),
            "actual_roc_fpm":  round(actual, 0),
            "expected_roc_fpm": round(exp_roc, 0) if exp_roc is not None else None,
            "delta_fpm":       round(delta, 0) if delta is not None else None,
            "pct_of_expected": round(pct, 1) if pct is not None else None,
            "actual_eas_kts":  row["avg_eas_kts"],
            "expected_kias":   round(exp_kias, 0) if exp_kias is not None else None,
            "isa_dev_c":       row["avg_isa_dev_c"],
            "in_table_range":  roc_in_range,
        })

    return pd.DataFrame(out)


def compare_cruise(df: pd.DataFrame, reference: dict) -> dict:
    """
    Compare observed cruise performance against the reference table.
    Handles both Mach-based (transport) and KTAS-based (GA) references.
    """
    from analysis.performance import prepare

    cruise = cruise_performance(df)
    if not cruise:
        return {}

    d = prepare(df)
    cruise_rows = d[d["phase"] == "cruise"]
    mean_da = float(cruise_rows["density_alt_ft"].mean()) if not cruise_rows.empty else np.nan

    exp_speed, in_range, speed_type = expected_cruise_speed(reference, mean_da)

    actual = cruise["avg_mach"] if speed_type == "mach" else cruise["avg_tas_kts"]
    delta = (actual - exp_speed) if exp_speed is not None else None
    pct = (actual / exp_speed * 100.0) if exp_speed else None

    result = {
        "speed_type":          speed_type,
        "mean_density_alt_ft": round(mean_da, 0) if not np.isnan(mean_da) else None,
        "primary_level_ft":    cruise["primary_level_ft"],
        "actual_speed":        actual,
        "expected_speed":      round(exp_speed, 3) if exp_speed is not None else None,
        "delta":               round(delta, 3) if delta is not None else None,
        "pct_of_expected":     round(pct, 1) if pct is not None else None,
        "in_table_range":      in_range,
        "avg_isa_dev_c":       cruise["avg_isa_dev_c"],
        "duration_min":        cruise["duration_min"],
    }

    # Flag against published limits where the file provides them
    limits = reference.get("limits", {})
    mmo = limits.get("mmo")
    if mmo and speed_type == "mach":
        result["mmo"] = mmo
        result["max_mach_observed"] = cruise["max_mach"]
        result["exceeded_mmo"] = bool(cruise["max_mach"] > mmo)

    ceiling = limits.get("service_ceiling_ft")
    if ceiling:
        result["service_ceiling_ft"] = ceiling
        result["above_service_ceiling"] = bool(cruise["max_altitude_ft"] > ceiling)

    return result


def compare_descent(df: pd.DataFrame, reference: dict, band_ft: int = 5000) -> dict:
    """
    Compare observed descent against the reference's typical descent
    figures. Descent reference data is usually a single typical value
    rather than a table, so this is a coarser comparison than climb.
    """
    bands = descent_performance(df, band_ft)
    if bands.empty:
        return {}

    ref_desc = reference.get("descent", {})
    typical = ref_desc.get("typical_rod_fpm")

    # Weight each band's rate by its duration to get a fair overall figure
    total_s = bands["duration_s"].sum()
    if total_s > 0:
        mean_rod = float((bands["rod_fpm"] * bands["duration_s"]).sum() / total_s)
    else:
        mean_rod = float(bands["rod_fpm"].mean())

    return {
        "mean_rod_fpm":     round(mean_rod, 0),
        "typical_rod_fpm":  typical,
        "delta_fpm":        round(mean_rod - typical, 0) if typical is not None else None,
        "steepest_band_fpm": round(float(bands["rod_fpm"].min()), 0),
        "notes":            ref_desc.get("notes", ""),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(
    df: pd.DataFrame,
    reference_key: str | None = None,
    aircraft_type: str | None = None,
    band_ft: int = 5000,
) -> dict:
    """
    Run every comparison and bundle the results with the reference
    metadata and caveats.

    This is the entry point the dashboard should call. The returned
    dict always includes a 'caveats' list — display it. The whole point
    of this module is that the comparison is approximate in specific,
    knowable ways, and a report that hides that is worse than no report.

    Parameters
    ----------
    df            : pd.DataFrame   Labeled, weather-enriched flight data
    reference_key : str, optional  Explicit poh_data/*.json key (e.g. "a321").
                                    Takes priority if given.
    aircraft_type : str, optional  ICAO typecode (e.g. "A321"). Used to look
                                    up reference_key automatically via
                                    TYPECODE_TO_REFERENCE if reference_key
                                    isn't given directly. Typically this is
                                    the flight's stored aircraft_type from
                                    db.get_flight_meta().
    band_ft       : int            Altitude band size for climb/descent.

    If neither reference_key nor a resolvable aircraft_type is available,
    returns a descriptive-only report (flight_summary populated, climb/
    cruise/descent comparison sections empty) rather than raising —
    there's no aircraft type known, or no reference file for it, and
    that's a normal outcome worth reporting plainly rather than failing.
    """
    if reference_key is None:
        reference_key = reference_key_for_typecode(aircraft_type)

    summary = flight_summary(df)

    if reference_key is None:
        return {
            "reference_key":  None,
            "reference_meta": {},
            "caveat_text":    (
                f"No performance reference available"
                + (f" for aircraft type '{aircraft_type}'" if aircraft_type else " (aircraft type unknown)")
                + f". Showing descriptive flight data only — no actual-vs-expected comparison possible.\n"
                f"Available references: {list_available_references()}"
            ),
            "caveats": [
                "No reference performance data was available for this flight's aircraft type "
                "(either the type is unknown, or no reference file exists for it yet). "
                "Only descriptive flight_summary is populated below."
            ],
            "flight_summary": summary,
            "climb":          pd.DataFrame(),
            "cruise":         {},
            "descent":        {},
        }

    reference = load_reference(reference_key)

    caveats = [
        "Aircraft weight is unknown (not transmitted via ADS-B). Weight strongly "
        "affects climb performance, so climb deltas should not be read as "
        "performance anomalies without knowing the load.",
        "True airspeed is derived from groundspeed plus forecast wind, not measured. "
        "Wind forecast error propagates directly into TAS and Mach figures.",
        "Reference lookup uses density altitude as a single axis, which is a "
        "simplification of the two-axis (pressure altitude x temperature) lookup "
        "a published table actually specifies.",
    ]

    wx_cov = summary.get("weather_data_coverage")
    if wx_cov is not None and not (isinstance(wx_cov, float) and np.isnan(wx_cov)) and wx_cov < 1.0:
        caveats.append(
            f"Only {wx_cov * 100:.0f}% of samples had real weather data; the remainder "
            "fell back to ISA standard temperature, reducing accuracy of TAS, Mach, "
            "and density altitude for those rows."
        )

    if not reference.get("aircraft", {}).get("verified", False):
        caveats.append(
            "The reference data itself is UNVERIFIED. Check it against an "
            "authoritative source before presenting these results."
        )

    return {
        "reference_key":  reference_key,
        "reference_meta": reference.get("aircraft", {}),
        "caveat_text":    reference_caveat(reference),
        "caveats":        caveats,
        "flight_summary": summary,
        "climb":          compare_climb(df, reference, band_ft),
        "cruise":         compare_cruise(df, reference),
        "descent":        compare_descent(df, reference, band_ft),
    }


def print_report(report: dict) -> None:
    """Human-readable rendering of generate_report() output, for CLI use."""
    print("=" * 74)
    print("PERFORMANCE COMPARISON REPORT")
    print("=" * 74)
    print(report["caveat_text"])
    print()

    print("--- FLIGHT SUMMARY ---")
    for k, v in report["flight_summary"].items():
        print(f"  {k:<24} {v}")

    climb = report["climb"]
    if isinstance(climb, pd.DataFrame) and not climb.empty:
        print("\n--- CLIMB: ACTUAL vs EXPECTED ---")
        print(climb.to_string(index=False))

    cruise = report["cruise"]
    if cruise:
        print("\n--- CRUISE: ACTUAL vs EXPECTED ---")
        for k, v in cruise.items():
            print(f"  {k:<24} {v}")

    desc = report["descent"]
    if desc:
        print("\n--- DESCENT ---")
        for k, v in desc.items():
            if k != "notes":
                print(f"  {k:<24} {v}")

    print("\n--- CAVEATS ---")
    for i, c in enumerate(report["caveats"], 1):
        print(f"  {i}. {c}")
    print("=" * 74)


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Available references:", list_available_references())
    print()

    # Reuse the synthetic A321-like flight from performance.py's test
    import subprocess, sys
    rng = np.random.default_rng(7)
    rows = []
    t = 0

    def push(alt, gs, vs, phase, lat, lon, oat=None, hw=20.0, cw=5.0):
        rows.append({
            "time_position": t, "baro_altitude": alt, "velocity": gs,
            "vertical_rate": vs, "phase": phase, "latitude": lat,
            "longitude": lon, "on_ground": phase == "ground",
            "oat_c": oat, "headwind_kts": hw, "crosswind_kts": cw,
            "true_track": 45.0,
        })

    for i in range(4):
        push(0, 12, 0, "ground", 33.64, -84.43, 28.0); t += 15

    alt = 0.0
    lat, lon = 33.64, -84.43
    while alt < 34000:
        vs = 2400 if alt < 10000 else (1800 if alt < 24000 else 1100)
        alt += vs * (10 / 60.0)
        gs = 180 + (alt / 34000) * 280
        phase = "takeoff" if alt < 1000 else "climb"
        push(min(alt, 34000), gs, vs, phase, lat, lon, 15 - 0.0019812 * alt + 3.0)
        lat += 0.004; lon += 0.004
        t += 10

    for i in range(60):
        push(34000, 465, 0, "cruise", lat, lon, -49.0); t += 30
        lat += 0.02; lon += 0.02

    while alt > 1500:
        vs = -1800 if alt > 10000 else -1000
        alt += vs * (10 / 60.0)
        gs = 420 - (34000 - alt) / 34000 * 240
        phase = "descent" if alt > 2500 else "landing"
        push(max(alt, 0), gs, vs, phase, lat, lon, 15 - 0.0019812 * alt + 3.0)
        lat += 0.003; lon += 0.003
        t += 10

    for i in range(4):
        push(0, 10, 0, "ground", lat, lon, 24.0); t += 15

    test_df = pd.DataFrame(rows)

    report = generate_report(test_df, "a321")
    print_report(report)