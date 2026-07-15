"""
phase_detector.py
Labels each row of the cleaned ADS-B DataFrame with a flight phase:
climb, cruise, descent, or landing.

Raw vertical_rate from ADS-B is noisy, so we smooth it with a rolling
average before thresholding. This prevents phase labels from flickering
between climb and cruise during normal turbulence or level-off.
"""

import pandas as pd

# Vertical speed thresholds in feet per minute.
# Any sustained VS above this is a climb; below the negative is a descent.
# 200 fpm gives plenty of margin over normal cruise noise (±50 fpm)
# while still catching real phase transitions.
CLIMB_VS_THRESHOLD = 200      # fpm
DESCENT_VS_THRESHOLD = -200   # fpm

# Altitude below which a descent is reclassified as landing approach.
# 2500 ft AGL is roughly the outer marker altitude for a typical ILS.
LANDING_ALT_THRESHOLD = 2500  # feet

# Rolling window size in rows for smoothing vertical_rate.
# At 1 Hz ADS-B data this is ~30 seconds — long enough to smooth out
# turbulence bumps, short enough to catch real climb/descent starts.
SMOOTHING_WINDOW = 30


def label_phases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'phase' column to the cleaned flight DataFrame.

    Phases (in order of detection priority):
    - 'landing'  : descending AND below LANDING_ALT_THRESHOLD ft
    - 'climb'    : smoothed vertical rate > CLIMB_VS_THRESHOLD fpm
    - 'descent'  : smoothed vertical rate < DESCENT_VS_THRESHOLD fpm
    - 'cruise'   : everything else

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from cleaner.clean(). Must contain
        'vertical_rate' and 'baro_altitude' columns.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with a new 'phase' string column and a
        'smoothed_vs' float column (useful for debugging/charting).
    """
    if df.empty:
        return df

    df = df.copy()

    # Smooth vertical_rate over a rolling window.
    # min_periods=1 means rows near the start of the flight still
    # get a value (using however many rows are available), rather
    # than being labeled NaN and falling through to cruise by default.
    df["smoothed_vs"] = (
        df["vertical_rate"]
        .rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1)
        .mean()
    )

    # Assign phases using vectorized np.select — cleaner than a loop,
    # and the order of conditions matters: landing is checked first
    # because a landing approach also satisfies the descent condition.
    import numpy as np

    conditions = [
        # Landing: descending AND low altitude
        (df["smoothed_vs"] < DESCENT_VS_THRESHOLD) & (df["baro_altitude"] < LANDING_ALT_THRESHOLD),
        # Climb: sustained positive vertical rate
        df["smoothed_vs"] > CLIMB_VS_THRESHOLD,
        # Descent: sustained negative vertical rate (above landing threshold)
        df["smoothed_vs"] < DESCENT_VS_THRESHOLD,
    ]
    choices = ["landing", "climb", "descent"]

    # Default (when none of the above match) is cruise.
    df["phase"] = np.select(conditions, choices, default="cruise")

    return df


def phase_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a summary DataFrame showing the duration and average
    altitude/speed for each detected phase. Useful for the dashboard.

    Parameters
    ----------
    df : pd.DataFrame
        Output of label_phases() — must have 'phase', 'baro_altitude',
        and 'velocity' columns.

    Returns
    -------
    pd.DataFrame
        One row per phase with duration (seconds), avg altitude (ft),
        and avg speed (kts).
    """
    if "phase" not in df.columns:
        raise ValueError("DataFrame must have a 'phase' column. Run label_phases() first.")

    summary = (
        df.groupby("phase")
        .agg(
            row_count=("phase", "count"),
            avg_altitude_ft=("baro_altitude", "mean"),
            avg_speed_kts=("velocity", "mean"),
        )
        .round(1)
        .reset_index()
    )

    # row_count ≈ seconds at 1 Hz, so rename it for clarity
    summary = summary.rename(columns={"row_count": "duration_seconds"})

    return summary


if __name__ == "__main__":
    import numpy as np

    # Simulate a simple climb -> cruise -> descent -> landing flight
    # with 400 rows (roughly 6-7 minutes of data at 1 Hz)
    n = 400
    time_pos = list(range(n))

    # Build a realistic vertical_rate profile with some noise
    vs = (
        [800] * 100    # climbing at 800 fpm
        + [0] * 100    # cruising
        + [-600] * 100 # descending
        + [-500] * 100 # landing approach (low altitude)
    )
    alt = []
    current_alt = 500
    for v in vs:
        current_alt += v / 60   # VS is fpm, each row is ~1 second
        alt.append(max(0, current_alt))

    # Add some realistic noise to vertical_rate
    rng = np.random.default_rng(42)
    vs_noisy = [v + rng.normal(0, 80) for v in vs]

    fake_df = pd.DataFrame({
        "time_position": time_pos,
        "baro_altitude": alt,
        "velocity": [120] * n,
        "vertical_rate": vs_noisy,
        "on_ground": [False] * n,
    })

    result = label_phases(fake_df)

    print("Phase label counts:")
    print(result["phase"].value_counts())
    print()
    print("Phase summary:")
    print(phase_summary(result))