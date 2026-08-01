"""
performance.py
Computes flight performance metrics from labeled, weather-enriched
telemetry.

Input is the DataFrame produced by the pipeline — a flight that has
already been polled, weather-enriched, and phase-labeled (see
db.get_flight_dataframe). This module adds the derived atmospheric
columns (via atmosphere.py) and then aggregates the trajectory into
the metrics that actually characterize how the aircraft performed:

  climb_performance()   — rate of climb by altitude band, time to altitude
  cruise_performance()  — cruise altitude, TAS, Mach, wind, ISA deviation
  descent_performance() — descent rate by band, top of descent
  flight_summary()      — distance, block/airborne time, altitude/speed peaks

All of these are descriptive — they say what the aircraft did. Compare
them against reference performance data using poh_compare.py.

Caveats worth carrying into any report generated from this:
  - TAS is derived from groundspeed + forecast wind, not measured.
    Wind forecast error propagates directly into TAS error.
  - Aircraft weight is unknown from ADS-B, and weight strongly affects
    climb performance. Comparisons against reference tables are
    therefore approximate unless a weight assumption is stated.
  - Sampling is 10-30 s, so brief transients are averaged away.
"""

import numpy as np
import pandas as pd

from analysis.atmosphere import add_atmospheric_columns

EARTH_RADIUS_NM = 3440.065


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles between arrays of points."""
    lat1, lon1, lat2, lon2 = map(lambda x: np.radians(np.asarray(x, dtype=float)),
                                 (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_NM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _segment_duration_s(sub: pd.DataFrame) -> float:
    """
    Total seconds covered by a subset of rows, summing contiguous runs
    rather than taking a single global span. Same reasoning as
    phase_detector.phase_summary — a phase can occur in several
    disjoint chunks, and a naive max-min would bridge the gaps.
    """
    if sub.empty:
        return 0.0
    t = sub["time_position"].astype(float).sort_values()
    # A gap larger than 5 minutes is treated as a break between segments
    breaks = t.diff() > 300
    seg = breaks.cumsum()
    return float(t.groupby(seg).agg(lambda x: x.max() - x.min()).sum())


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add atmospheric/airspeed columns and sort by time. Every function
    in this module calls this first, so it's safe (and idempotent) to
    call it yourself before doing custom analysis.
    """
    if df.empty:
        return df
    out = df.sort_values("time_position").reset_index(drop=True)
    return add_atmospheric_columns(out)


# ---------------------------------------------------------------------------
# Climb
# ---------------------------------------------------------------------------

def climb_performance(df: pd.DataFrame, band_ft: int = 5000) -> pd.DataFrame:
    """
    Break the climb into altitude bands and characterize each one.

    Rate of climb is computed from the actual altitude and time deltas
    across each band (not by averaging the reported vertical_rate
    field), because the reported value is instantaneous and noisy while
    the band-crossing figure is what a performance table would predict.

    A level-off (ATC-restricted climb, traffic separation, etc.) that
    happens to land inside a band is correctly excluded from the climb
    phase entirely by phase_detector — but naively taking
    (last_timestamp - first_timestamp) across the remaining rows in
    that band would still span the level-off's real-world duration,
    since the intervening excluded rows are gone from the group but
    the WALL-CLOCK TIME they occupied isn't. This silently dilutes the
    computed rate — a level-off of just a few minutes inside one band
    can drop the reported rate to a third of the aircraft's true
    climbing rate, on any flight, not just one with an unusual profile.

    The fix: track which maximal contiguous stretch of climb/takeoff
    phase each row belongs to (computed on the FULL flight sequence
    before filtering), then group by (band, segment) rather than just
    band. Two visits to the same band separated by an excluded
    level-off become two separate segments — their altitude changes
    and active durations are summed together, never spanned across
    the gap between them.

    Parameters
    ----------
    df      : pd.DataFrame  labeled flight data
    band_ft : int           altitude band height, default 5000 ft

    Returns
    -------
    pd.DataFrame with one row per band:
        band_low_ft, band_high_ft, roc_fpm, avg_tas_kts, avg_eas_kts,
        avg_mach, avg_isa_dev_c, avg_headwind_kts, duration_s, n_samples
    """
    d = prepare(df)
    if d.empty:
        return pd.DataFrame()
    d = d.sort_values("time_position").reset_index(drop=True)

    # Contiguous-phase segment id on the FULL sequence, before any
    # filtering — this is what lets us later tell "two separate visits
    # to this band" apart from "one continuous stretch through it".
    d["_segment_id"] = (d["phase"] != d["phase"].shift()).cumsum()

    climb = d[d["phase"].isin(["takeoff", "climb"])].copy()
    if climb.empty:
        return pd.DataFrame()
    climb["band"] = (climb["baro_altitude"] // band_ft).astype(int) * band_ft

    # Per (band, segment): altitude change and elapsed time. Because
    # _segment_id is constant only within one uninterrupted run of
    # climb/takeoff phase, first/last within a group here can never
    # silently span a level-off's duration.
    seg_rows = []
    for (band, seg_id), g in climb.groupby(["band", "_segment_id"]):
        g = g.sort_values("time_position")
        if len(g) < 2:
            continue
        d_alt = float(g["baro_altitude"].iloc[-1] - g["baro_altitude"].iloc[0])
        d_t   = float(g["time_position"].iloc[-1] - g["time_position"].iloc[0])
        if d_t <= 0:
            continue
        seg_rows.append({
            "band": int(band), "d_alt": d_alt, "d_t": d_t,
            "tas_sum":  float(g["tas_kts"].sum()),
            "eas_sum":  float(g["eas_kts"].sum()),
            "mach_sum": float(g["mach"].sum()),
            "isa_sum":  float(g["isa_dev_c"].sum()),
            "hw_sum":   float(pd.to_numeric(g.get("headwind_kts"), errors="coerce").fillna(0).sum()),
            "n": len(g),
        })

    if not seg_rows:
        return pd.DataFrame()

    seg_df = pd.DataFrame(seg_rows)

    # Aggregate segments sharing the same band: SUM altitude change and
    # active time across all visits to that band, then take the rate —
    # rather than averaging per-segment rates (which would weight a
    # 10-second segment equally to a 5-minute one).
    rows = []
    for band, g in seg_df.groupby("band"):
        total_alt = g["d_alt"].sum()
        total_t   = g["d_t"].sum()
        if total_t <= 0:
            continue
        n_total = int(g["n"].sum())
        rows.append({
            "band_low_ft":      int(band),
            "band_high_ft":     int(band + band_ft),
            "roc_fpm":          round(total_alt / (total_t / 60.0), 1),
            "avg_tas_kts":      round(g["tas_sum"].sum() / n_total, 1),
            "avg_eas_kts":      round(g["eas_sum"].sum() / n_total, 1),
            "avg_mach":         round(g["mach_sum"].sum() / n_total, 3),
            "avg_isa_dev_c":    round(g["isa_sum"].sum() / n_total, 1),
            "avg_headwind_kts": round(g["hw_sum"].sum() / n_total, 1),
            "duration_s":       int(total_t),
            "n_samples":        n_total,
        })

    return pd.DataFrame(rows).sort_values("band_low_ft").reset_index(drop=True)


def time_to_altitude(df: pd.DataFrame, targets_ft=(10000, 20000, 30000)) -> pd.DataFrame:
    """
    Time elapsed from first airborne sample until each target altitude
    is first reached during the climb.

    Returns NaN for targets the aircraft never reached (or that were
    already exceeded before tracking began — a real possibility if the
    poller was started mid-flight).
    """
    d = prepare(df)
    airborne = d[d["phase"] != "ground"]
    if airborne.empty:
        return pd.DataFrame()

    t0 = float(airborne["time_position"].iloc[0])
    start_alt = float(airborne["baro_altitude"].iloc[0])

    rows = []
    for target in targets_ft:
        reached = airborne[airborne["baro_altitude"] >= target]
        if reached.empty:
            elapsed = np.nan
            note = "not reached"
        elif start_alt >= target:
            elapsed = np.nan
            note = "already above at tracking start"
        else:
            elapsed = float(reached["time_position"].iloc[0]) - t0
            note = ""
        rows.append({
            "target_ft":   target,
            "time_s":      None if np.isnan(elapsed) else int(elapsed),
            "time_min":    None if np.isnan(elapsed) else round(elapsed / 60.0, 1),
            "note":        note,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cruise
# ---------------------------------------------------------------------------

def cruise_performance(df: pd.DataFrame) -> dict:
    """
    Characterize the cruise portion of the flight.

    Returns a dict (not a DataFrame) since cruise is best summarized as
    a single set of scalars. Altitude is reported as both mean and
    mode-like "primary level" — airliners often step-climb, so the mean
    can fall between two actual flight levels.
    """
    d = prepare(df)
    cruise = d[d["phase"] == "cruise"]
    if cruise.empty:
        return {}

    # Round to nearest 1000 ft to identify discrete flight levels flown
    levels = (cruise["baro_altitude"] / 1000).round().astype(int) * 1000
    level_counts = levels.value_counts().sort_values(ascending=False)

    return {
        "duration_s":          int(_segment_duration_s(cruise)),
        "duration_min":        round(_segment_duration_s(cruise) / 60.0, 1),
        "primary_level_ft":    int(level_counts.index[0]),
        "levels_flown_ft":     [int(x) for x in sorted(level_counts.index.tolist())],
        "mean_altitude_ft":    round(float(cruise["baro_altitude"].mean()), 0),
        "max_altitude_ft":     round(float(cruise["baro_altitude"].max()), 0),
        "avg_tas_kts":         round(float(cruise["tas_kts"].mean()), 1),
        "avg_groundspeed_kts": round(float(cruise["velocity"].mean()), 1),
        "avg_mach":            round(float(cruise["mach"].mean()), 3),
        "max_mach":            round(float(cruise["mach"].max()), 3),
        "avg_oat_c":           round(float(pd.to_numeric(cruise.get("oat_c"), errors="coerce").mean()), 1),
        "avg_isa_dev_c":       round(float(cruise["isa_dev_c"].mean()), 1),
        "avg_headwind_kts":    round(float(pd.to_numeric(cruise.get("headwind_kts"), errors="coerce").mean()), 1),
        "n_samples":           int(len(cruise)),
    }


# ---------------------------------------------------------------------------
# Descent
# ---------------------------------------------------------------------------

def descent_performance(df: pd.DataFrame, band_ft: int = 5000) -> pd.DataFrame:
    """
    Break the descent into altitude bands, same approach as
    climb_performance() — including the same fix for level-offs that
    land inside a band silently diluting the computed rate (see that
    function's docstring for the full explanation and worked example).
    Rate of descent is reported as a negative fpm value.
    """
    d = prepare(df)
    if d.empty:
        return pd.DataFrame()
    d = d.sort_values("time_position").reset_index(drop=True)
    d["_segment_id"] = (d["phase"] != d["phase"].shift()).cumsum()

    desc = d[d["phase"].isin(["descent", "landing"])].copy()
    if desc.empty:
        return pd.DataFrame()
    desc["band"] = (desc["baro_altitude"] // band_ft).astype(int) * band_ft

    seg_rows = []
    for (band, seg_id), g in desc.groupby(["band", "_segment_id"]):
        g = g.sort_values("time_position")
        if len(g) < 2:
            continue
        d_alt = float(g["baro_altitude"].iloc[-1] - g["baro_altitude"].iloc[0])
        d_t   = float(g["time_position"].iloc[-1] - g["time_position"].iloc[0])
        if d_t <= 0:
            continue
        seg_rows.append({
            "band": int(band), "d_alt": d_alt, "d_t": d_t,
            "tas_sum":  float(g["tas_kts"].sum()),
            "mach_sum": float(g["mach"].sum()),
            "n": len(g),
        })

    if not seg_rows:
        return pd.DataFrame()

    seg_df = pd.DataFrame(seg_rows)

    rows = []
    for band, g in seg_df.groupby("band"):
        total_alt = g["d_alt"].sum()
        total_t   = g["d_t"].sum()
        if total_t <= 0:
            continue
        n_total = int(g["n"].sum())
        rows.append({
            "band_low_ft":  int(band),
            "band_high_ft": int(band + band_ft),
            "rod_fpm":      round(total_alt / (total_t / 60.0), 1),
            "avg_tas_kts":  round(g["tas_sum"].sum() / n_total, 1),
            "avg_mach":     round(g["mach_sum"].sum() / n_total, 3),
            "duration_s":   int(total_t),
            "n_samples":    n_total,
        })

    return pd.DataFrame(rows).sort_values("band_low_ft", ascending=False).reset_index(drop=True)


def top_of_descent(df: pd.DataFrame) -> dict:
    """
    Locate the top of descent — the last sample at cruise before the
    sustained descent begins — and how far it was from the final
    tracked position.
    """
    d = prepare(df)
    desc = d[d["phase"] == "descent"]
    if desc.empty or len(d) < 2:
        return {}

    tod = desc.iloc[0]
    last = d.iloc[-1]
    dist = float(_haversine_nm(tod["latitude"], tod["longitude"],
                               last["latitude"], last["longitude"]))

    return {
        "tod_time_position": int(tod["time_position"]),
        "tod_altitude_ft":   round(float(tod["baro_altitude"]), 0),
        "tod_distance_to_end_nm": round(dist, 1),
        "descent_duration_min": round(
            float(last["time_position"] - tod["time_position"]) / 60.0, 1
        ),
    }


# ---------------------------------------------------------------------------
# Whole-flight summary
# ---------------------------------------------------------------------------

def flight_summary(df: pd.DataFrame) -> dict:
    """
    High-level totals for the whole tracked flight.

    Distance is integrated along the actual track (sum of great-circle
    hops between consecutive samples), so it reflects the routing
    actually flown rather than a straight origin-destination line.
    """
    d = prepare(df)
    if d.empty:
        return {}

    lat = d["latitude"].astype(float).to_numpy()
    lon = d["longitude"].astype(float).to_numpy()
    hops = _haversine_nm(lat[:-1], lon[:-1], lat[1:], lon[1:])
    track_distance = float(np.nansum(hops))

    airborne = d[d["phase"] != "ground"]
    airborne_s = _segment_duration_s(airborne) if not airborne.empty else 0.0
    total_s = float(d["time_position"].iloc[-1] - d["time_position"].iloc[0])

    # Fraction of rows whose weather came from real NOAA data rather
    # than an ISA fallback — reported so any downstream comparison can
    # state how much of the analysis rests on estimated temperature
    wx_coverage = float((~d["oat_is_estimated"]).mean()) if "oat_is_estimated" in d.columns else np.nan

    return {
        "n_samples":            int(len(d)),
        "tracked_duration_min": round(total_s / 60.0, 1),
        "airborne_duration_min": round(airborne_s / 60.0, 1),
        "track_distance_nm":    round(track_distance, 1),
        "max_altitude_ft":      round(float(d["baro_altitude"].max()), 0),
        "max_groundspeed_kts":  round(float(d["velocity"].max()), 1),
        "max_tas_kts":          round(float(d["tas_kts"].max()), 1),
        "max_mach":             round(float(d["mach"].max()), 3),
        "max_roc_fpm":          round(float(pd.to_numeric(d["vertical_rate"], errors="coerce").max()), 0),
        "max_rod_fpm":          round(float(pd.to_numeric(d["vertical_rate"], errors="coerce").min()), 0),
        "weather_data_coverage": round(wx_coverage, 3),
        "phases_observed":      sorted(d["phase"].unique().tolist()) if "phase" in d.columns else [],
    }


def compute_metrics(df: pd.DataFrame, band_ft: int = 5000) -> dict:
    """
    Run every analysis in this module and return the results together.
    This is the single entry point the pipeline / dashboard should use.
    """
    return {
        "summary":  flight_summary(df),
        "climb":    climb_performance(df, band_ft),
        "time_to_altitude": time_to_altitude(df),
        "cruise":   cruise_performance(df),
        "descent":  descent_performance(df, band_ft),
        "top_of_descent": top_of_descent(df),
    }


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Build a synthetic but physically plausible A321-like flight:
    # ground -> climb to FL340 -> cruise -> descent -> landing -> ground
    rng = np.random.default_rng(7)
    rows = []
    t = 0

    def push(alt, gs, vs, phase, lat, lon, oat=None, hw=20.0, cw=5.0):
        nonlocal_t = None
        rows.append({
            "time_position": t, "baro_altitude": alt, "velocity": gs,
            "vertical_rate": vs, "phase": phase, "latitude": lat,
            "longitude": lon, "on_ground": phase == "ground",
            "oat_c": oat, "headwind_kts": hw, "crosswind_kts": cw,
            "true_track": 45.0,
        })

    # Ground (pre-departure)
    for i in range(4):
        push(0, 12, 0, "ground", 33.64, -84.43, 28.0); t += 15

    # Takeoff + climb to FL340 at ~1800 fpm average
    alt = 0.0
    lat, lon = 33.64, -84.43
    while alt < 34000:
        vs = 2400 if alt < 10000 else (1800 if alt < 24000 else 1100)
        dt = 10
        alt += vs * (dt / 60.0)
        gs = 180 + (alt / 34000) * 280
        phase = "takeoff" if alt < 1000 else "climb"
        isa_t = 15 - 0.0019812 * alt
        push(min(alt, 34000), gs, vs, phase, lat, lon, isa_t + 3.0)
        lat += 0.004; lon += 0.004
        t += dt

    # Cruise at FL340
    for i in range(60):
        push(34000, 465, 0, "cruise", lat, lon, -49.0); t += 30
        lat += 0.02; lon += 0.02

    # Descent
    while alt > 1500:
        vs = -1800 if alt > 10000 else -1000
        dt = 10
        alt += vs * (dt / 60.0)
        gs = 420 - (34000 - alt) / 34000 * 240
        phase = "descent" if alt > 2500 else "landing"
        isa_t = 15 - 0.0019812 * alt
        push(max(alt, 0), gs, vs, phase, lat, lon, isa_t + 3.0)
        lat += 0.003; lon += 0.003
        t += dt

    # Ground (post-arrival)
    for i in range(4):
        push(0, 10, 0, "ground", lat, lon, 24.0); t += 15

    test_df = pd.DataFrame(rows)
    print(f"Synthetic flight: {len(test_df)} samples\n")

    metrics = compute_metrics(test_df)

    print("=== FLIGHT SUMMARY ===")
    for k, v in metrics["summary"].items():
        print(f"  {k:<24} {v}")

    print("\n=== CLIMB BY ALTITUDE BAND ===")
    print(metrics["climb"].to_string(index=False))

    print("\n=== TIME TO ALTITUDE ===")
    print(metrics["time_to_altitude"].to_string(index=False))

    print("\n=== CRUISE ===")
    for k, v in metrics["cruise"].items():
        print(f"  {k:<24} {v}")

    print("\n=== DESCENT BY ALTITUDE BAND ===")
    print(metrics["descent"].to_string(index=False))

    print("\n=== TOP OF DESCENT ===")
    for k, v in metrics["top_of_descent"].items():
        print(f"  {k:<24} {v}")