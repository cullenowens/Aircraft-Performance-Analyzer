"""
captions.py
Generates the explanatory text shown under each chart in the dashboard.

Every caption is computed from the SPECIFIC flight's own numbers, not
generic boilerplate — "this flight had an 18kt average tailwind" rather
than "tailwinds reduce fuel burn." The goal is to teach the underlying
aviation concept while grounding it in what actually happened on this
flight, so the explanation stays useful even to someone with no
aviation background.

Like charts.py and report.py, this module is pure view-prep: it reads
already-computed DataFrames/dicts and returns strings. No analysis, no
I/O.
"""

import numpy as np
import pandas as pd

from analysis.performance import _haversine_nm


def _fmt(v, unit="", nd=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "an unknown value"
    if nd == 0:
        return f"{v:,.0f}{unit}"
    return f"{v:,.{nd}f}{unit}"


# ---------------------------------------------------------------------------
# 1. Altitude profile
# ---------------------------------------------------------------------------

def altitude_caption(df: pd.DataFrame) -> str:
    """
    Explain the phase timeline and flag any level-offs during climb/descent.

    Takes the row-level flight DataFrame (one row per poll, with
    'time_position' and 'phase' columns) — NOT phase_detector's
    phase_summary() output, which has one row per phase and no
    per-sample timestamps.
    """
    if df is None or df.empty or "phase" not in df.columns:
        return "Altitude over time, colored by detected flight phase."

    d = df.sort_values("time_position").reset_index(drop=True)
    durations = d.groupby("phase")["time_position"].apply(lambda x: x.max() - x.min())

    climb_min   = durations.get("climb", 0) / 60
    cruise_min  = durations.get("cruise", 0) / 60
    descent_min = durations.get("descent", 0) / 60

    cruise_rows = d[d["phase"] == "cruise"]
    primary_alt = cruise_rows["baro_altitude"].mode().iloc[0] if not cruise_rows.empty else None

    # Count level-offs: cruise-labeled segments other than the main
    # (longest) one — these are ATC step-climbs/step-downs, not the
    # flight's real cruise altitude.
    seg_id = (d["phase"] != d["phase"].shift()).cumsum()
    cruise_segments = d[d["phase"] == "cruise"].groupby(seg_id).agg(
        dur=("time_position", lambda x: x.max() - x.min()))
    n_level_offs = max(0, len(cruise_segments) - 1)

    text = (
        f"This flight spent roughly {climb_min:.0f} min climbing, "
        f"{cruise_min:.0f} min at cruise"
        + (f" near {_fmt(primary_alt, ' ft')}" if primary_alt else "")
        + f", and {descent_min:.0f} min descending. "
    )
    if n_level_offs > 0:
        text += (
            f"{n_level_offs} additional level-off segment(s) appear outside the main cruise "
            f"block — these are almost always ATC-assigned altitude restrictions during climb "
            f"or a step-down arrival, not errors in the flight or the phase detection."
        )
    else:
        text += "No additional level-offs were detected outside the main cruise segment."
    return text


# ---------------------------------------------------------------------------
# 2. Speed / Mach
# ---------------------------------------------------------------------------

def speed_caption(df: pd.DataFrame) -> str:
    """
    Explain the groundspeed/TAS/wind relationship using this flight's
    actual average cruise headwind component.
    """
    if "phase" in df.columns:
        cruise = df[df["phase"] == "cruise"]
    else:
        cruise = df

    if cruise.empty or "headwind_kts" not in cruise.columns:
        return (
            "Groundspeed (blue) and true airspeed (orange) track together when there's no "
            "wind; the gap between them reflects the wind component along the flight's track."
        )

    mean_hw = cruise["headwind_kts"].mean()
    max_mach = df["mach"].max() if "mach" in df.columns else None

    if pd.isna(mean_hw):
        wind_text = "Wind data wasn't available to characterize the cruise wind component."
    elif mean_hw > 2:
        wind_text = (
            f"This flight cruised into an average headwind of {_fmt(mean_hw, ' kt')} — the "
            f"engines had to work harder to maintain groundspeed than they would in calm air, "
            f"which is why true airspeed (orange) sits above groundspeed (blue)."
        )
    elif mean_hw < -2:
        wind_text = (
            f"This flight had an average tailwind of {_fmt(abs(mean_hw), ' kt')} at cruise — "
            f"groundspeed (blue) ran ahead of true airspeed (orange) for free, which is why "
            f"airlines actively plan routes and altitudes to catch favorable winds like this "
            f"one; it improves fuel efficiency for the same engine setting."
        )
    else:
        wind_text = "Wind at cruise was close to calm on this flight, so groundspeed and true airspeed stayed close together."

    mach_text = f" Peak Mach reached was {_fmt(max_mach, '', 3)}." if max_mach else ""
    return wind_text + mach_text


# ---------------------------------------------------------------------------
# 3. Vertical rate
# ---------------------------------------------------------------------------

def vertical_rate_caption(df: pd.DataFrame) -> str:
    """
    Explain the threshold lines and flag any level-off flattening.
    Takes the row-level flight DataFrame (not phase_summary() output).
    """
    base = (
        "Vertical rate shows how fast the aircraft gained or lost altitude. The dashed lines "
        "mark the ±200 fpm thresholds used to classify climb vs descent vs level flight."
    )
    if df is None or df.empty or "phase" not in df.columns:
        return base

    d = df.sort_values("time_position")
    seg_id = (d["phase"] != d["phase"].shift()).cumsum()
    cruise_segments = d[d["phase"] == "cruise"].groupby(seg_id).size()
    n_flat_segments = (cruise_segments >= 2).sum()

    if n_flat_segments > 1:
        return (
            base + f" This flight shows {n_flat_segments} separate stretches near zero "
            f"vertical rate — sustained flat periods mid-climb or mid-descent like these "
            f"usually reflect an ATC-assigned altitude hold, not an error."
        )
    return base


# ---------------------------------------------------------------------------
# 4. Wind & temperature
# ---------------------------------------------------------------------------

def wind_temp_caption(df: pd.DataFrame) -> str:
    """Explain the temperature lapse and headwind-vs-altitude trend for this flight."""
    if df.empty:
        return "Headwind component and outside air temperature plotted against altitude."

    low = df[df["baro_altitude"] < 10000]
    high = df[df["baro_altitude"] > 25000]

    oat_low  = low["oat_c"].mean() if not low.empty and "oat_c" in df.columns else None
    oat_high = high["oat_c"].mean() if not high.empty and "oat_c" in df.columns else None
    hw_low   = low["headwind_kts"].mean() if not low.empty and "headwind_kts" in df.columns else None
    hw_high  = high["headwind_kts"].mean() if not high.empty and "headwind_kts" in df.columns else None

    text = (
        "Temperature drops and wind generally strengthens with altitude — this is the "
        "atmosphere the aircraft actually climbed and cruised through. "
    )
    if oat_low is not None and oat_high is not None and not (pd.isna(oat_low) or pd.isna(oat_high)):
        text += f"On this flight, OAT went from about {_fmt(oat_low, '°C')} below 10,000 ft to {_fmt(oat_high, '°C')} above 25,000 ft. "
    if hw_low is not None and hw_high is not None and not (pd.isna(hw_low) or pd.isna(hw_high)):
        direction_low = "headwind" if hw_low > 0 else "tailwind"
        direction_high = "headwind" if hw_high > 0 else "tailwind"
        text += (
            f"The wind component shifted from a {_fmt(abs(hw_low), ' kt')} {direction_low} "
            f"at lower altitude to a {_fmt(abs(hw_high), ' kt')} {direction_high} higher up."
        )

    if "oat_is_estimated" in df.columns and df["oat_is_estimated"].any():
        pct_est = df["oat_is_estimated"].mean() * 100
        text += f" Faint × markers ({pct_est:.0f}% of points) are ISA-estimated, not measured NOAA weather."

    return text


# ---------------------------------------------------------------------------
# 5. Climb comparison
# ---------------------------------------------------------------------------

def climb_comparison_caption(climb_adherence_result: dict) -> str:
    """
    Reuse the SAME classification the rating system computed, so the
    caption and the rating section never disagree with each other.
    """
    if not climb_adherence_result or climb_adherence_result.get("pattern") == "no_data":
        return "Actual vs typical rate of climb by altitude band."

    n_below = climb_adherence_result["n_below"]
    n_total = climb_adherence_result["n_total"]
    note = climb_adherence_result["note"]

    return f"{n_below} of {n_total} altitude bands fell notably below the typical reference rate. {note}"


# ---------------------------------------------------------------------------
# 6. Ground track
# ---------------------------------------------------------------------------

def ground_track_caption(df: pd.DataFrame, track_distance_nm: float | None) -> str:
    """
    Compare the actual flown distance to the direct great-circle
    distance between first and last tracked point, to show how much
    routing/vectoring added to the trip.
    """
    valid = df.dropna(subset=["latitude", "longitude"]).sort_values("time_position")
    if len(valid) < 2 or track_distance_nm is None:
        return "The aircraft's ground track, colored by flight phase."

    first, last = valid.iloc[0], valid.iloc[-1]
    direct_nm = float(_haversine_nm(first["latitude"], first["longitude"],
                                     last["latitude"], last["longitude"]))

    if direct_nm <= 0:
        return "The aircraft's ground track, colored by flight phase."

    extra_pct = (track_distance_nm - direct_nm) / direct_nm * 100

    text = (
        f"This flight covered {_fmt(track_distance_nm, ' nm', 1)} along its actual track, "
        f"versus a direct great-circle distance of {_fmt(direct_nm, ' nm', 1)} between its "
        f"first and last tracked points"
    )
    if extra_pct > 15:
        text += f" — about {extra_pct:.0f}% longer, suggesting noticeable vectoring, holding, or a routing that isn't a straight line (common near busy terminal airspace)."
    elif extra_pct > 3:
        text += f", about {extra_pct:.0f}% longer — a normal amount of routing overhead."
    else:
        text += ", a close-to-direct routing."
    return text


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.db import get_connection, get_flight_dataframe, get_flight_meta
    from analysis.atmosphere import add_atmospheric_columns
    from analysis.performance import flight_summary
    from analysis.rating import climb_adherence
    from analysis.poh_compare import generate_report

    conn = get_connection()
    df = add_atmospheric_columns(get_flight_dataframe(conn, 16))
    meta = get_flight_meta(conn, 16)
    summary = flight_summary(df)
    report = generate_report(df, aircraft_type=meta["aircraft_type"])

    print("ALTITUDE:", altitude_caption(df))
    print()
    print("SPEED:", speed_caption(df))
    print()
    print("VS:", vertical_rate_caption(df))
    print()
    print("WIND/TEMP:", wind_temp_caption(df))
    print()
    print("CLIMB:", climb_comparison_caption(climb_adherence(report["climb"])))
    print()
    print("TRACK:", ground_track_caption(df, summary.get("track_distance_nm")))