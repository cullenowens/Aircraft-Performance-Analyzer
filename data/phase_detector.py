"""
phase_detector.py
Labels each row of a flight DataFrame with a flight phase — ground,
takeoff, climb, cruise, descent, or landing — based on smoothed
vertical rate and altitude.

This is a batch operation, not a streaming one: the rolling-window
smoothing looks at rows both before and after each point (center=True),
so it needs the full trajectory in view. It cannot run mid-flight —
that's why poll_and_store() in poller.py uses a separate, simpler
live heuristic (_live_phase_hint) just for its status display and
landing-detection hook. label_phases() here is the source of truth,
run once after a flight lands, and its output is written back to the
database (see apply_and_store_phases()).
"""

import numpy as np
import pandas as pd

# Vertical rate thresholds (fpm) for classifying climb vs descent vs cruise
CLIMB_VS_THRESHOLD   = 200
DESCENT_VS_THRESHOLD = -200

# Altitude (ft) below which climb/descent is considered takeoff/landing
# rather than a generic climb or descent segment
LANDING_ALT_THRESHOLD = 2500

# Rolling window size (rows) for smoothing vertical_rate before thresholding.
# center=True in the rolling call means each smoothed value uses rows both
# before and after it, so phase transitions aren't lagged behind the real
# transition point.
SMOOTHING_WINDOW = 30


def label_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'phase' column to the DataFrame: one of
    'ground', 'takeoff', 'climb', 'cruise', 'descent', 'landing'.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: vertical_rate (fpm), baro_altitude (ft),
        on_ground (bool). Rows should already be sorted by time —
        this function sorts by time_position defensively if present.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'smoothed_vs' and 'phase' columns added.
    """
    df = df.copy()

    if "time_position" in df.columns:
        df = df.sort_values("time_position").reset_index(drop=True)

    # Smooth vertical_rate over a rolling window. min_periods=1 means
    # rows near the start/end of the flight still get a value (using
    # however many rows are available) instead of falling through to
    # cruise by default due to NaN.
    df["smoothed_vs"] = (
        df["vertical_rate"]
        .rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1)
        .mean()
    )

    # on_ground is checked first and unconditionally — a grounded
    # aircraft with near-zero smoothed VS would otherwise silently
    # fall through to "cruise" via the np.select default, which is
    # wrong (that's taxiing/parked, not cruising).
    conditions = [
        df["on_ground"] == True,  # noqa: E712 — explicit bool compare intentional for clarity
        (df["smoothed_vs"] < DESCENT_VS_THRESHOLD) & (df["baro_altitude"] < LANDING_ALT_THRESHOLD),
        (df["smoothed_vs"] > CLIMB_VS_THRESHOLD) & (df["baro_altitude"] < LANDING_ALT_THRESHOLD),
        df["smoothed_vs"] > CLIMB_VS_THRESHOLD,
        df["smoothed_vs"] < DESCENT_VS_THRESHOLD,
    ]
    choices = ["ground", "landing", "takeoff", "climb", "descent"]

    df["phase"] = np.select(conditions, choices, default="cruise")

    return df


def phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame showing the duration and average
    altitude/speed for each detected phase.

    Duration is calculated by identifying contiguous time segments of
    each phase and summing their durations — NOT a single global
    max-min span per phase. This matters because some phases occur in
    multiple disjoint segments within one flight (most notably
    'ground', which happens once before takeoff and again after
    landing). A naive global max-min for 'ground' would span from the
    first pre-departure timestamp to the last post-arrival timestamp —
    effectively the entire flight duration — rather than the actual
    time spent on the ground.

    Parameters
    ----------
    df : pd.DataFrame
        Output of label_phases() — must have 'phase', 'time_position',
        'baro_altitude', and 'velocity' columns.

    Returns
    -------
    pd.DataFrame
        One row per phase with duration_seconds, duration_min,
        avg_altitude_ft, avg_speed_kts.
    """
    if "phase" not in df.columns:
        raise ValueError("DataFrame must have a 'phase' column. Run label_phases() first.")
    if "time_position" not in df.columns:
        raise ValueError("DataFrame must have a 'time_position' column for duration calculation.")

    df = df.sort_values("time_position").reset_index(drop=True)

    # Identify contiguous segments: increments every time the phase
    # value changes from the previous row, so each run of consecutive
    # same-phase rows gets its own segment id
    segment_id = (df["phase"] != df["phase"].shift()).cumsum()

    segments = (
        df.groupby(segment_id)
        .agg(
            phase=("phase", "first"),
            start=("time_position", "min"),
            end=("time_position", "max"),
            avg_altitude_ft=("baro_altitude", "mean"),
            avg_speed_kts=("velocity", "mean"),
            row_count=("phase", "count"),
        )
        .reset_index(drop=True)
    )
    segments["duration_seconds"] = segments["end"] - segments["start"]

    # Sum durations across all segments of the same phase; average
    # altitude/speed weighted by each segment's row count so a long
    # segment isn't diluted by a short one when phases repeat
    def _weighted_mean(values, weights):
        weights = weights.astype(float)
        if weights.sum() == 0:
            return values.mean()
        return (values * weights).sum() / weights.sum()

    summary = (
        segments.groupby("phase")
        .apply(lambda g: pd.Series({
            "duration_seconds": int(g["duration_seconds"].sum()),
            "avg_altitude_ft": _weighted_mean(g["avg_altitude_ft"], g["row_count"]),
            "avg_speed_kts": _weighted_mean(g["avg_speed_kts"], g["row_count"]),
        }), include_groups=False)
        .reset_index()
    )

    summary[["avg_altitude_ft", "avg_speed_kts"]] = summary[["avg_altitude_ft", "avg_speed_kts"]].round(1)
    summary["duration_min"] = (summary["duration_seconds"] / 60).round(1)
    summary = summary[[
        "phase", "duration_seconds", "duration_min",
        "avg_altitude_ft", "avg_speed_kts"
    ]]

    return summary


def apply_and_store_phases(conn, flight_id: int) -> pd.DataFrame:
    """
    Pull all state vectors for a flight from the database, run batch
    phase detection over the full trajectory, and write the resulting
    phase labels back to those same rows.

    This is the batch step that runs once, right after a flight lands
    — called by the orchestrator (pipeline.py), not manually.

    Parameters
    ----------
    conn      : sqlite3.Connection
    flight_id : int   The flight to process

    Returns
    -------
    pd.DataFrame
        The labeled DataFrame (with 'phase' column), for immediate use
        by the caller (e.g. printing a summary) without a second DB read.
    """
    from data.db import get_flight_dataframe, update_phases, mark_flight_processed

    df = get_flight_dataframe(conn, flight_id)
    if df.empty:
        print(f"[phase_detector] No rows found for flight_id={flight_id}, nothing to process.")
        return df

    labeled = label_phases(df)

    # Write phase labels back — id column comes from the DB's own
    # primary key (state_vectors.id), fetched by get_flight_dataframe()
    phase_updates = list(zip(labeled["phase"], labeled["id"]))
    update_phases(conn, phase_updates)

    mark_flight_processed(conn, flight_id)

    return labeled


if __name__ == "__main__":
    # Simulate a flight with realistic dynamic polling intervals:
    # 10s during climb/descent, 30s during cruise — matching the poller
    CLIMB_ROWS   = 60
    CRUISE_ROWS  = 80
    DESCENT_ROWS = 40
    LAND_ROWS    = 10

    climb_times   = list(range(0, CLIMB_ROWS * 10, 10))
    cruise_times  = list(range(climb_times[-1] + 30, climb_times[-1] + 30 + CRUISE_ROWS * 30, 30))
    descent_times = list(range(cruise_times[-1] + 10, cruise_times[-1] + 10 + DESCENT_ROWS * 10, 10))
    land_times    = list(range(descent_times[-1] + 10, descent_times[-1] + 10 + LAND_ROWS * 10, 10))
    all_times     = climb_times + cruise_times + descent_times + land_times

    n = len(all_times)
    vs_base = (
        [800]  * CLIMB_ROWS
        + [0]  * CRUISE_ROWS
        + [-600] * DESCENT_ROWS
        + [-500] * LAND_ROWS
    )

    intervals = [10] * CLIMB_ROWS + [30] * CRUISE_ROWS + [10] * DESCENT_ROWS + [10] * LAND_ROWS
    alt, current_alt = [], 500.0
    for vs_val, dt in zip(vs_base, intervals):
        current_alt += (vs_val / 60) * dt
        alt.append(max(0, current_alt))

    rng = np.random.default_rng(42)
    vs_noisy = [v + rng.normal(0, 80) for v in vs_base]

    fake_df = pd.DataFrame({
        "time_position": all_times,
        "baro_altitude": alt,
        "velocity":      [300.0] * n,
        "vertical_rate": vs_noisy,
        "on_ground":     [False] * n,
    })

    result = label_phases(fake_df)

    print("Phase label counts (rows):")
    print(result["phase"].value_counts())
    print()
    print("Phase summary (timestamp-based durations):")
    print(phase_summary(result))
    print()
    print(f"Total flight time: {(all_times[-1] - all_times[0]) / 60:.1f} minutes")

    # Test the on_ground handling with a few grounded rows appended
    print("\n--- Testing on_ground handling ---")
    ground_df = pd.DataFrame({
        "time_position": [0, 10, 20],
        "baro_altitude": [0.0, 0.0, 0.0],
        "velocity": [0.0, 5.0, 8.0],
        "vertical_rate": [0.0, 0.0, 0.0],
        "on_ground": [True, True, True],
    })
    ground_result = label_phases(ground_df)
    print(ground_result[["time_position", "on_ground", "phase"]])
    assert (ground_result["phase"] == "ground").all(), "on_ground rows should be labeled 'ground', not 'cruise'"
    print("✓ on_ground rows correctly labeled 'ground'")