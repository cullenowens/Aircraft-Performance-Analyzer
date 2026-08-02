"""
rating.py
Produces a "how closely did this flight match a typical operating
profile" assessment. Deliberately NOT a performance grade.

Why not a single blended score
-------------------------------
Aircraft weight is unknown from ADS-B and is one of the largest
drivers of climb performance. A number like "82/100" would claim a
precision this pipeline cannot support — a heavy aircraft flying
flawlessly and a light aircraft flying poorly can produce a nearly
identical climb deviation. Blending that into one score would hide
exactly the ambiguity a careful reader needs to see.

What this module answers instead
---------------------------------
1. Did the flight cross a hard limit (Mmo, service ceiling)? This does
   NOT depend on weight — a legitimate pass/fail.
2. What fraction of the flight fell within a normal operating envelope
   for this aircraft type — using the reference file's own stated
   purpose (see poh_data/*.json "source_confidence" fields): "a
   meaningful benchmark for did this flight fly a standard profile."
   This is reported as a plain count/fraction, never a percentage
   dressed up as a score out of 100.
3. Where climb deviates, is the deviation broad (spread across most of
   the climb — more consistent with a weight or thrust-setting effect
   that acts on the whole climb) or localized (concentrated in only
   the first band or two — more consistent with a technique like a
   noise-abatement reduced-thrust departure, which specifically
   targets the first few thousand feet)? This is a PATTERN
   OBSERVATION, not a diagnosis — ADS-B data alone cannot distinguish
   these causes with certainty, and the output says so explicitly.

Every function here returns plain data (dicts/lists of dicts) with a
"note" or "caveat" field attached — nothing is meant to be displayed
without its accompanying explanation.
"""

import numpy as np
import pandas as pd

# Tolerance bands. Wide on purpose: they're set to reflect genuine
# real-world variation (weight, wind, ATC, cost index), not to make
# every flight look "good". A band this wide passing is a weak signal;
# a band this wide FAILING is a much stronger one.
CLIMB_TOLERANCE_LOW_PCT  = 70    # below this = "notably below typical"
CLIMB_TOLERANCE_HIGH_PCT = 140   # above this = "notably above typical"

CRUISE_TOLERANCE_LOW_PCT  = 92   # cruise Mach is much less weight-sensitive
CRUISE_TOLERANCE_HIGH_PCT = 105  # than climb, so this band is tighter


# ---------------------------------------------------------------------------
# 1. Limits compliance — the one genuinely confident, weight-independent check
# ---------------------------------------------------------------------------

def limits_compliance(cruise: dict) -> dict:
    """
    Check observed values against hard published limits (Mmo, service
    ceiling). Unlike climb/cruise rate comparisons, this does not
    depend on aircraft weight — a real pass/fail.

    Returns
    -------
    dict: {"checks": [...], "overall": "PASS" | "FAIL" | "NOT AVAILABLE"}
    """
    checks = []

    if "exceeded_mmo" in cruise:
        checks.append({
            "check":  "Maximum operating Mach (Mmo)",
            "result": "FAIL" if cruise["exceeded_mmo"] else "PASS",
            "detail": f"Max Mach observed {cruise.get('max_mach_observed')} "
                     f"vs Mmo {cruise.get('mmo')}",
        })

    if "above_service_ceiling" in cruise:
        checks.append({
            "check":  "Service ceiling",
            "result": "FAIL" if cruise["above_service_ceiling"] else "PASS",
            "detail": f"Max altitude vs published ceiling "
                     f"{cruise.get('service_ceiling_ft'):,} ft" if cruise.get("service_ceiling_ft") else "",
        })

    if not checks:
        return {"checks": [], "overall": "NOT AVAILABLE"}

    overall = "FAIL" if any(c["result"] == "FAIL" for c in checks) else "PASS"
    return {"checks": checks, "overall": overall}


# ---------------------------------------------------------------------------
# 2. Climb adherence — per-band tolerance check + broad-vs-localized pattern
# ---------------------------------------------------------------------------

def climb_adherence(climb_df: pd.DataFrame) -> dict:
    """
    Classify each climb band as within/below/above the tolerance band,
    and characterize whether any below-tolerance bands are broadly
    spread (whole-climb effect) or localized to the lowest bands
    (technique/procedure effect) — a pattern observation, not a cause.

    Returns
    -------
    dict with per-band classifications, counts, and a plain-English
    pattern note that explicitly states what it can't determine.
    """
    if climb_df is None or climb_df.empty:
        return {"bands": [], "n_within": 0, "n_below": 0, "n_above": 0,
                "n_total": 0, "pattern": "no_data", "note": "No climb comparison data available."}

    d = climb_df.sort_values("band_low_ft").copy()

    def classify(pct):
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            return "unknown"
        if pct < CLIMB_TOLERANCE_LOW_PCT:
            return "below"
        if pct > CLIMB_TOLERANCE_HIGH_PCT:
            return "above"
        return "within"

    d["climb_class"] = d["pct_of_expected"].apply(classify)

    bands = [
        {"band_low_ft": int(r.band_low_ft), "band_high_ft": int(r.band_high_ft),
         "pct_of_expected": r.pct_of_expected, "classification": r.climb_class}
        for r in d.itertuples()
    ]

    n_within = int((d["climb_class"] == "within").sum())
    n_below  = int((d["climb_class"] == "below").sum())
    n_above  = int((d["climb_class"] == "above").sum())
    n_total  = len(d)

    # Pattern: is "below" concentrated in only the lowest 1-2 bands?
    below_bands = d[d["climb_class"] == "below"]
    if below_bands.empty:
        pattern = "no_deviation"
        note = "Climb rate stayed within typical range across all assessed altitude bands."
    else:
        lowest_two = set(d.sort_values("band_low_ft")["band_low_ft"].head(2))
        below_set = set(below_bands["band_low_ft"])
        if below_set.issubset(lowest_two):
            pattern = "localized_low_altitude"
            note = (
                "Below-typical climb rate is concentrated in the lowest altitude band(s) only. "
                "This shape is consistent with a reduced-thrust or noise-abatement departure "
                "technique (common near busy airports), which specifically targets the first "
                "few thousand feet — but it's also consistent with a heavy takeoff weight, "
                "which ADS-B cannot distinguish. Treat this as a pattern observation, not a cause."
            )
        else:
            pattern = "broad"
            note = (
                "Below-typical climb rate is spread across most of the climb, not just the "
                "lowest bands. This shape is more consistent with a whole-climb effect such as "
                "heavier-than-reference takeoff weight or a reduced climb thrust setting — but "
                "again, ADS-B data alone cannot confirm the cause."
            )

    return {
        "bands": bands, "n_within": n_within, "n_below": n_below,
        "n_above": n_above, "n_total": n_total, "pattern": pattern, "note": note,
    }


# ---------------------------------------------------------------------------
# 3. Cruise adherence — single value, tighter tolerance (less weight-sensitive)
# ---------------------------------------------------------------------------

def cruise_adherence(cruise: dict) -> dict:
    """
    Classify cruise speed against a tighter tolerance band than climb,
    since cruise Mach for a given aircraft type is set operationally
    (cost index, schedule) and is much less sensitive to weight than
    climb rate is.
    """
    if not cruise or cruise.get("pct_of_expected") is None:
        return {"classification": "no_data", "note": "No cruise comparison data available."}

    pct = cruise["pct_of_expected"]
    if pct < CRUISE_TOLERANCE_LOW_PCT:
        classification = "below"
        note = (
            f"Cruise speed was {pct:.0f}% of the typical reference figure. Slower-than-typical "
            f"cruise is usually a deliberate choice — cost index, ATC speed restriction, or "
            f"turbulence avoidance — rather than a performance shortfall."
        )
    elif pct > CRUISE_TOLERANCE_HIGH_PCT:
        classification = "above"
        note = (
            f"Cruise speed was {pct:.0f}% of the typical reference figure — faster than the "
            f"typical operating profile, possibly reflecting schedule recovery."
        )
    else:
        classification = "within"
        note = f"Cruise speed ({pct:.0f}% of typical) is consistent with normal operations."

    return {"classification": classification, "pct_of_expected": pct, "note": note}


# ---------------------------------------------------------------------------
# 4. Descent notes — deliberately descriptive, not graded
# ---------------------------------------------------------------------------

def descent_notes(descent: dict, df: pd.DataFrame) -> dict:
    """
    Descent rate is NOT scored against tolerance the way climb/cruise
    are. Step-down arrivals with real, multi-minute level-offs are
    routine at busy airports, and a duration-weighted average descent
    rate will legitimately read "shallow" against an idealized
    continuous-descent reference on almost any real arrival into
    controlled airspace. Scoring that would be misleading, so this
    function instead reports what happened descriptively.
    """
    if not descent:
        return {"note": "No descent comparison data available.", "n_level_offs": 0}

    # Count distinct cruise-labeled segments that occur AFTER the first
    # descent-phase row — i.e. level-offs during the arrival, not the
    # flight's real cruise segment.
    n_level_offs = 0
    if df is not None and not df.empty and "phase" in df.columns:
        d = df.sort_values("time_position").reset_index(drop=True)
        first_descent_idx = d.index[d["phase"] == "descent"]
        if len(first_descent_idx) > 0:
            after_tod = d.loc[first_descent_idx[0]:]
            seg_id = (after_tod["phase"] != after_tod["phase"].shift()).cumsum()
            level_segments = after_tod[after_tod["phase"] == "cruise"].groupby(seg_id).size()
            n_level_offs = int((level_segments >= 2).sum())  # require 2+ samples to count as real

    mean_rod = descent.get("mean_rod_fpm")
    typical  = descent.get("typical_rod_fpm")
    steepest = descent.get("steepest_band_fpm")

    note = (
        f"Mean descent rate was {mean_rod:,.0f} fpm against a typical reference of "
        f"{typical:,.0f} fpm, with a steepest band of {steepest:,.0f} fpm. "
    )
    if n_level_offs > 0:
        note += (
            f"{n_level_offs} distinct level-off segment(s) were detected during the descent, "
            f"consistent with a normal ATC-sequenced step-down arrival — this alone explains "
            f"most or all of the gap between the mean rate and the idealized continuous-descent "
            f"reference figure."
        )
    else:
        note += "No distinct level-off segments were detected during the descent."

    return {"note": note, "n_level_offs": n_level_offs, "mean_rod_fpm": mean_rod,
            "typical_rod_fpm": typical, "steepest_band_fpm": steepest}


# ---------------------------------------------------------------------------
# 5. Overall — headline word + explicit scope statement, never a number
# ---------------------------------------------------------------------------

def overall_rating(report: dict, df: pd.DataFrame) -> dict:
    """
    Combine the above into one dashboard-friendly headline. The
    headline is a WORD ("Nominal" / "Notable deviations" / "Significant
    deviations"), never a numeric score — deliberately, to avoid
    implying a precision this pipeline can't support. Always paired
    with an explicit scope statement about what it does and doesn't mean.

    Returns
    -------
    dict: {headline, scope_note, limits, climb, cruise, descent}
    """
    if report.get("reference_key") is None:
        return {
            "headline": "No reference available",
            "scope_note": "No reference performance data exists for this aircraft type, "
                          "so no profile-adherence assessment could be computed.",
            "limits": {}, "climb": {}, "cruise": {}, "descent": {},
        }

    limits = limits_compliance(report["cruise"])
    climb  = climb_adherence(report["climb"])
    cruise = cruise_adherence(report["cruise"])
    descent = descent_notes(report["descent"], df)

    # Fraction of assessed segments (climb bands + cruise) within tolerance
    total_assessed = climb["n_total"] + (1 if cruise.get("classification") not in (None, "no_data") else 0)
    total_within = climb["n_within"] + (1 if cruise.get("classification") == "within" else 0)
    fraction_within = (total_within / total_assessed) if total_assessed else None

    if limits["overall"] == "FAIL":
        headline = "Significant deviations"
    elif fraction_within is not None and fraction_within < 0.5:
        headline = "Significant deviations"
    elif fraction_within is not None and fraction_within < 0.8:
        headline = "Notable deviations"
    else:
        headline = "Nominal"

    scope_note = (
        "This is a PROFILE-ADHERENCE assessment — how closely this flight's climb and cruise "
        "numbers matched a typical operating profile for this aircraft type. It is NOT a "
        "performance grade. Aircraft weight is unknown from ADS-B and strongly affects climb "
        "rate, so a deviation here means 'this flight differed from a typical operation,' not "
        "'this flight underperformed.' Descent is reported descriptively rather than scored, "
        "since real step-down arrivals routinely and legitimately look 'shallow' against an "
        "idealized continuous-descent reference."
    )

    return {
        "headline": headline,
        "fraction_within": fraction_within,
        "total_assessed": total_assessed,
        "total_within": total_within,
        "scope_note": scope_note,
        "limits": limits,
        "climb": climb,
        "cruise": cruise,
        "descent": descent,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.db import get_connection, get_flight_dataframe, get_flight_meta
    from data.phase_detector import phase_summary
    from analysis.poh_compare import generate_report

    conn = get_connection()
    for fid in [16, 1]:
        meta = get_flight_meta(conn, fid)
        df = get_flight_dataframe(conn, fid)
        report = generate_report(df, aircraft_type=meta["aircraft_type"])

        rating = overall_rating(report, df)
        print(f"=== Flight #{fid} ({meta['callsign'].strip()}, {meta['aircraft_type']}) ===")
        print(f"Headline: {rating['headline']}")
        print(f"Assessed: {rating['total_within']}/{rating['total_assessed']} segments within tolerance")
        print(f"Limits: {rating['limits']['overall']}")
        print(f"Climb pattern: {rating['climb']['pattern']}")
        print(f"  {rating['climb']['note']}")
        print(f"Cruise: {rating['cruise']['classification']} — {rating['cruise']['note']}")
        print(f"Descent: {rating['descent']['note']}")
        print()