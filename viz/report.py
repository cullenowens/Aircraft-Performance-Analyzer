"""
report.py
Prepares summary metrics and comparison tables for display in the
dashboard. Like charts.py, this is pure view-prep code: it takes the
dicts/DataFrames produced by the analysis layer and reshapes them into
display-ready structures (formatted strings, labeled rows, delta signs).
It does NOT run analysis or touch the database.

Kept separate from app.py so the formatting logic can be tested on its
own and so app.py stays a thin layout shell.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Summary metric cards
# ---------------------------------------------------------------------------

def summary_cards(summary: dict) -> list[dict]:
    """
    Turn performance.flight_summary() output into a list of card specs
    for the dashboard's top row. Each card is {label, value, help}.

    Formatting is done here (not in the app) so units and rounding stay
    consistent and testable.
    """
    if not summary:
        return []

    def fmt(v, unit="", nd=0):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if nd == 0:
            return f"{v:,.0f}{unit}"
        return f"{v:,.{nd}f}{unit}"

    cards = [
        {"label": "Airborne time",
         "value": fmt(summary.get("airborne_duration_min"), " min", 1),
         "help": "Total time not on the ground."},
        {"label": "Track distance",
         "value": fmt(summary.get("track_distance_nm"), " nm", 1),
         "help": "Distance along the actual flown ground track."},
        {"label": "Max altitude",
         "value": fmt(summary.get("max_altitude_ft"), " ft"),
         "help": "Highest barometric altitude reached."},
        {"label": "Max Mach",
         "value": fmt(summary.get("max_mach"), "", 3),
         "help": "Highest Mach number observed."},
        {"label": "Max climb rate",
         "value": fmt(summary.get("max_roc_fpm"), " fpm"),
         "help": "Peak rate of climb."},
        {"label": "Weather coverage",
         "value": fmt((summary.get("weather_data_coverage") or 0) * 100, "%", 0),
         "help": "Share of samples with real NOAA weather data (rest fell back to ISA estimate)."},
    ]
    return cards


# ---------------------------------------------------------------------------
# Climb comparison table
# ---------------------------------------------------------------------------

def climb_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape poh_compare.compare_climb() into a clean display table with
    readable band labels, a signed delta, and a percent-of-expected
    column. Returns a DataFrame ready to hand to st.dataframe().
    """
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame()

    d = comparison_df.sort_values("band_low_ft").copy()

    out = pd.DataFrame({
        "Altitude band": [f"{int(lo/1000)}–{int(hi/1000)}k ft"
                          for lo, hi in zip(d["band_low_ft"], d["band_high_ft"])],
        "Actual (fpm)": d["actual_roc_fpm"].round(0).astype("Int64"),
        "Expected (fpm)": d["expected_roc_fpm"].round(0).astype("Int64"),
        "Δ (fpm)": d["delta_fpm"].round(0).astype("Int64"),
        "% of expected": d["pct_of_expected"].round(0).astype("Int64"),
        "ISA dev": d["isa_dev_c"].round(1),
        "In range": d["in_table_range"].map({True: "yes", False: "extrapolated"}),
    })
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cruise comparison rows
# ---------------------------------------------------------------------------

def cruise_rows(cruise: dict) -> list[dict]:
    """
    Turn poh_compare.compare_cruise() into a list of {label, value}
    rows for a small key-value table. Handles both Mach and KTAS
    reference types, and surfaces the limit-check flags.
    """
    if not cruise:
        return []

    speed_type = cruise.get("speed_type", "mach")
    unit = "" if speed_type == "mach" else " kt"
    nd = 3 if speed_type == "mach" else 1

    def fmt(v, u="", n=1):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:,.{n}f}{u}"

    rows = [
        {"label": "Primary level", "value": fmt(cruise.get("primary_level_ft"), " ft", 0)},
        {"label": f"Actual {speed_type}", "value": fmt(cruise.get("actual_speed"), unit, nd)},
        {"label": f"Expected {speed_type}", "value": fmt(cruise.get("expected_speed"), unit, nd)},
        {"label": "% of expected", "value": fmt(cruise.get("pct_of_expected"), "%", 0)},
        {"label": "ISA deviation", "value": fmt(cruise.get("avg_isa_dev_c"), "°C", 1)},
        {"label": "Cruise duration", "value": fmt(cruise.get("duration_min"), " min", 1)},
    ]

    # Surface limit checks only when present and meaningful
    if "exceeded_mmo" in cruise:
        rows.append({
            "label": f"Max Mach vs Mmo ({cruise.get('mmo')})",
            "value": f"{fmt(cruise.get('max_mach_observed'), '', 3)}  "
                     + ("⚠ EXCEEDED" if cruise.get("exceeded_mmo") else "within limit"),
        })
    if "above_service_ceiling" in cruise:
        rows.append({
            "label": f"Max alt vs ceiling ({fmt(cruise.get('service_ceiling_ft'), ' ft', 0)})",
            "value": "⚠ ABOVE" if cruise.get("above_service_ceiling") else "within limit",
        })

    return rows


# ---------------------------------------------------------------------------
# Descent summary rows
# ---------------------------------------------------------------------------

def descent_rows(descent: dict) -> list[dict]:
    """Turn poh_compare.compare_descent() into {label, value} rows."""
    if not descent:
        return []

    def fmt(v, u=""):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{v:,.0f}{u}"

    return [
        {"label": "Mean descent rate", "value": fmt(descent.get("mean_rod_fpm"), " fpm")},
        {"label": "Typical (reference)", "value": fmt(descent.get("typical_rod_fpm"), " fpm")},
        {"label": "Δ from typical", "value": fmt(descent.get("delta_fpm"), " fpm")},
        {"label": "Steepest band", "value": fmt(descent.get("steepest_band_fpm"), " fpm")},
    ]


# ---------------------------------------------------------------------------
# Phase summary table
# ---------------------------------------------------------------------------

def phase_table(phase_summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape phase_detector.phase_summary() into a display table, ordered
    chronologically (ground → takeoff → climb → cruise → descent →
    landing) rather than alphabetically.
    """
    if phase_summary_df is None or phase_summary_df.empty:
        return pd.DataFrame()

    order = ["ground", "takeoff", "climb", "cruise", "descent", "landing"]
    d = phase_summary_df.copy()
    d["_order"] = d["phase"].map({p: i for i, p in enumerate(order)}).fillna(99)
    d = d.sort_values("_order").drop(columns="_order")

    return pd.DataFrame({
        "Phase": d["phase"],
        "Duration (min)": d["duration_min"],
        "Avg altitude (ft)": d["avg_altitude_ft"].round(0).astype("Int64"),
        "Avg speed (kt)": d["avg_speed_kts"].round(0).astype("Int64"),
    }).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.db import get_connection, get_flight_dataframe, get_flight_meta
    from data.phase_detector import phase_summary
    from analysis.performance import flight_summary
    from analysis.poh_compare import generate_report

    conn = get_connection()
    df = get_flight_dataframe(conn, 16)
    meta = get_flight_meta(conn, 16)

    print("=== SUMMARY CARDS ===")
    for c in summary_cards(flight_summary(df)):
        print(f"  {c['label']:<18} {c['value']}")

    report = generate_report(df, aircraft_type=meta["aircraft_type"])

    print("\n=== CLIMB TABLE ===")
    print(climb_table(report["climb"]).to_string(index=False))

    print("\n=== CRUISE ROWS ===")
    for r in cruise_rows(report["cruise"]):
        print(f"  {r['label']:<28} {r['value']}")

    print("\n=== DESCENT ROWS ===")
    for r in descent_rows(report["descent"]):
        print(f"  {r['label']:<24} {r['value']}")

    print("\n=== PHASE TABLE ===")
    print(phase_table(phase_summary(df)).to_string(index=False))