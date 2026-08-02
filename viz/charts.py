"""
charts.py
Plotly figure builders for the flight analysis dashboard.

Every function takes an already-prepared DataFrame (or a metrics/report
dict from the analysis layer) and returns a plotly.graph_objects.Figure.
None of these functions touch the database, run analysis, or read from
disk — they are pure view code, so they can be unit-tested against a
DataFrame and reused outside Streamlit if needed.

Design notes:
  - Flight phases share ONE consistent color map across every chart
    (PHASE_COLORS below), so a reader learns "orange = climb" once and
    it holds everywhere. The colors follow an intuitive energy gradient:
    ground neutral, takeoff/climb warm (energy going in), cruise calm
    blue (steady state), descent/landing cool (energy coming out).
  - Charts that compare actual vs expected always draw the reference as
    a muted dashed line and the actual as a solid saturated line, so
    "what happened" reads as foreground and "what was expected" as
    background — never the reverse.
  - Missing/estimated data is shown honestly (e.g. estimated-weather
    points are visually distinguishable) rather than silently blended
    in with measured data.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Shared visual vocabulary
# ---------------------------------------------------------------------------

PHASE_COLORS = {
    "ground":  "#9aa0a6",   # neutral grey — not flying
    "takeoff": "#e8710a",   # warm orange — max energy in
    "climb":   "#f9ab00",   # amber — energy in
    "cruise":  "#1a73e8",   # calm blue — steady state
    "descent": "#12b5cb",   # cool cyan — energy out
    "landing": "#9334e6",   # violet — final approach
}

# Phase display order (for legends / summary tables), chronological
PHASE_ORDER = ["ground", "takeoff", "climb", "cruise", "descent", "landing"]

ACTUAL_COLOR   = "#1a73e8"   # solid, saturated — foreground
EXPECTED_COLOR = "#80868b"   # muted grey — background reference
GRID_COLOR     = "#e8eaed"
AXIS_COLOR     = "#5f6368"

_BASE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, -apple-system, Segoe UI, sans-serif", size=13, color="#202124"),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="closest",
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor=GRID_COLOR, borderwidth=1),
)


def _style_axes(fig, xtitle=None, ytitle=None):
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
        linecolor=AXIS_COLOR, title_text=xtitle, title_font=dict(size=12, color=AXIS_COLOR),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID_COLOR, zeroline=False,
        linecolor=AXIS_COLOR, title_text=ytitle, title_font=dict(size=12, color=AXIS_COLOR),
    )
    return fig


def _minutes_from_start(df):
    """Elapsed minutes since the first sample — a readable x-axis."""
    t0 = df["time_position"].min()
    return (df["time_position"] - t0) / 60.0


# ---------------------------------------------------------------------------
# 1. Altitude profile colored by phase
# ---------------------------------------------------------------------------

def altitude_profile(df):
    """
    Altitude vs time, with the trace segmented and colored by flight
    phase. This is the signature chart — one glance shows the whole
    shape of the flight and where each phase happened.
    """
    df = df.sort_values("time_position").reset_index(drop=True)
    mins = _minutes_from_start(df)
    alt = df["baro_altitude"]

    fig = go.Figure()

    # Draw each contiguous phase run as its own colored segment. Using
    # contiguous runs (not one trace per phase label) keeps the line
    # visually connected and correctly ordered in time.
    if "phase" in df.columns:
        seg_id = (df["phase"] != df["phase"].shift()).cumsum()
        seen_phases = set()
        for _, g in df.groupby(seg_id):
            phase = g["phase"].iloc[0]
            idx = g.index
            # Include one point before the segment so segments visually
            # connect rather than showing gaps between phase changes
            start = max(idx[0] - 1, 0)
            seg_slice = df.loc[start:idx[-1]]
            fig.add_trace(go.Scatter(
                x=_minutes_from_start(df).loc[seg_slice.index],
                y=seg_slice["baro_altitude"],
                mode="lines",
                line=dict(color=PHASE_COLORS.get(phase, "#000"), width=2.5),
                name=phase,
                legendgroup=phase,
                showlegend=(phase not in seen_phases),
                hovertemplate=f"<b>{phase}</b><br>%{{y:,.0f}} ft<br>%{{x:.1f}} min<extra></extra>",
            ))
            seen_phases.add(phase)
    else:
        fig.add_trace(go.Scatter(x=mins, y=alt, mode="lines",
                                 line=dict(color=ACTUAL_COLOR, width=2.5)))

    fig.update_layout(**_BASE_LAYOUT, title="Altitude Profile")
    _style_axes(fig, "Time (minutes)", "Barometric altitude (ft)")
    return fig


# ---------------------------------------------------------------------------
# 2. Speed / Mach over time
# ---------------------------------------------------------------------------

def speed_profile(df):
    """
    Groundspeed and true airspeed over time on the left axis, Mach on
    the right. Shows the speed schedule and where the wind is helping
    or hurting (gap between GS and TAS = wind component).
    """
    df = df.sort_values("time_position").reset_index(drop=True)
    mins = _minutes_from_start(df)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=mins, y=df["velocity"], mode="lines", name="Groundspeed",
        line=dict(color="#1a73e8", width=2),
        hovertemplate="GS %{y:.0f} kt<br>%{x:.1f} min<extra></extra>",
    ), secondary_y=False)

    if "tas_kts" in df.columns:
        fig.add_trace(go.Scatter(
            x=mins, y=df["tas_kts"], mode="lines", name="True airspeed",
            line=dict(color="#e8710a", width=2, dash="dot"),
            hovertemplate="TAS %{y:.0f} kt<br>%{x:.1f} min<extra></extra>",
        ), secondary_y=False)

    if "mach" in df.columns:
        fig.add_trace(go.Scatter(
            x=mins, y=df["mach"], mode="lines", name="Mach",
            line=dict(color="#9334e6", width=1.5),
            hovertemplate="M %{y:.3f}<br>%{x:.1f} min<extra></extra>",
        ), secondary_y=True)

    fig.update_layout(**_BASE_LAYOUT, title="Speed & Mach")
    _style_axes(fig, "Time (minutes)")
    fig.update_yaxes(title_text="Speed (kt)", secondary_y=False,
                     showgrid=True, gridcolor=GRID_COLOR, linecolor=AXIS_COLOR)
    fig.update_yaxes(title_text="Mach", secondary_y=True,
                     showgrid=False, linecolor=AXIS_COLOR)
    return fig


# ---------------------------------------------------------------------------
# 3. Vertical rate over time
# ---------------------------------------------------------------------------

def vertical_rate_profile(df):
    """
    Vertical rate over time, colored above/below zero. Makes climb/
    descent segments and level-offs immediately visible, and is the
    clearest way to *see* the phase detector's decisions.
    """
    df = df.sort_values("time_position").reset_index(drop=True)
    mins = _minutes_from_start(df)
    vs = df["vertical_rate"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mins, y=vs, mode="lines", name="Vertical rate",
        line=dict(color="#5f6368", width=1),
        fill="tozeroy", fillcolor="rgba(26,115,232,0.08)",
        hovertemplate="%{y:,.0f} fpm<br>%{x:.1f} min<extra></extra>",
    ))
    # Reference lines at the climb/descent thresholds
    fig.add_hline(y=200, line=dict(color="#f9ab00", width=1, dash="dash"),
                  annotation_text="climb threshold", annotation_position="top left",
                  annotation_font_size=10)
    fig.add_hline(y=-200, line=dict(color="#12b5cb", width=1, dash="dash"),
                  annotation_text="descent threshold", annotation_position="bottom left",
                  annotation_font_size=10)

    fig.update_layout(**_BASE_LAYOUT, title="Vertical Rate")
    _style_axes(fig, "Time (minutes)", "Vertical rate (fpm)")
    return fig


# ---------------------------------------------------------------------------
# 4. Wind & temperature over altitude
# ---------------------------------------------------------------------------

def wind_temp_profile(df):
    """
    Headwind component and OAT vs altitude. Shows the atmosphere the
    aircraft climbed through — headwind building with altitude, temp
    dropping. Estimated-weather points (ISA fallback) are drawn faintly
    so measured vs estimated is honest and visible.
    """
    df = df.sort_values("baro_altitude").reset_index(drop=True)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if "headwind_kts" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["headwind_kts"], y=df["baro_altitude"], mode="markers",
            name="Headwind (+) / Tailwind (-)",
            marker=dict(color="#1a73e8", size=4, opacity=0.6),
            hovertemplate="%{x:.0f} kt @ %{y:,.0f} ft<extra></extra>",
        ), secondary_y=False)

    if "oat_c" in df.columns:
        # Split measured vs estimated if the flag is present
        if "oat_is_estimated" in df.columns:
            meas = df[~df["oat_is_estimated"]]
            est = df[df["oat_is_estimated"]]
            fig.add_trace(go.Scatter(
                x=meas["oat_c"], y=meas["baro_altitude"], mode="markers",
                name="OAT (measured)",
                marker=dict(color="#e8710a", size=4, opacity=0.7),
                hovertemplate="%{x:.0f}°C @ %{y:,.0f} ft<extra></extra>",
            ), secondary_y=True)
            if not est.empty:
                fig.add_trace(go.Scatter(
                    x=est["oat_c"], y=est["baro_altitude"], mode="markers",
                    name="OAT (ISA estimate)",
                    marker=dict(color="#e8710a", size=4, opacity=0.2, symbol="x"),
                    hovertemplate="%{x:.0f}°C (est) @ %{y:,.0f} ft<extra></extra>",
                ), secondary_y=True)
        else:
            fig.add_trace(go.Scatter(
                x=df["oat_c"], y=df["baro_altitude"], mode="markers",
                name="OAT", marker=dict(color="#e8710a", size=4, opacity=0.7),
            ), secondary_y=True)

    fig.update_layout(**_BASE_LAYOUT, title="Wind & Temperature vs Altitude")
    fig.update_yaxes(title_text="Altitude (ft)", showgrid=True, gridcolor=GRID_COLOR,
                     linecolor=AXIS_COLOR, secondary_y=False)
    fig.update_xaxes(title_text="Headwind (kt) / OAT (°C)", showgrid=True,
                     gridcolor=GRID_COLOR, linecolor=AXIS_COLOR)
    fig.update_yaxes(showticklabels=False, secondary_y=True)
    return fig


# ---------------------------------------------------------------------------
# 5. Actual vs expected climb rate by altitude band
# ---------------------------------------------------------------------------

def climb_comparison(comparison_df):
    """
    Grouped horizontal bars: actual vs expected rate of climb per
    altitude band. Takes the DataFrame from poh_compare.compare_climb().
    Bands outside the reference table's range are marked.
    """
    if comparison_df is None or comparison_df.empty:
        return _empty_fig("No climb comparison data available")

    d = comparison_df.sort_values("band_low_ft")
    band_labels = [f"{int(lo/1000)}–{int(hi/1000)}k ft"
                   for lo, hi in zip(d["band_low_ft"], d["band_high_ft"])]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=band_labels, x=d["expected_roc_fpm"], orientation="h",
        name="Expected", marker=dict(color=EXPECTED_COLOR),
        hovertemplate="Expected %{x:,.0f} fpm<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=band_labels, x=d["actual_roc_fpm"], orientation="h",
        name="Actual", marker=dict(color=ACTUAL_COLOR),
        hovertemplate="Actual %{x:,.0f} fpm<extra></extra>",
    ))
    fig.update_layout(**_BASE_LAYOUT, title="Climb Rate: Actual vs Expected",
                      barmode="group")
    _style_axes(fig, "Rate of climb (fpm)", "Altitude band")
    return fig


# ---------------------------------------------------------------------------
# 6. Ground track map
# ---------------------------------------------------------------------------

def ground_track(df):
    """
    Lat/lon ground track colored by phase, on a simple map. Uses
    scattergeo so it works offline without a map-tile token.
    """
    df = df.dropna(subset=["latitude", "longitude"]).sort_values("time_position")
    if df.empty:
        return _empty_fig("No position data available")

    fig = go.Figure()

    if "phase" in df.columns:
        for phase in PHASE_ORDER:
            g = df[df["phase"] == phase]
            if g.empty:
                continue
            fig.add_trace(go.Scattergeo(
                lat=g["latitude"], lon=g["longitude"], mode="markers",
                name=phase, marker=dict(color=PHASE_COLORS.get(phase), size=4),
                hovertemplate=f"<b>{phase}</b><br>%{{lat:.2f}}, %{{lon:.2f}}<extra></extra>",
            ))
    else:
        fig.add_trace(go.Scattergeo(
            lat=df["latitude"], lon=df["longitude"], mode="markers",
            marker=dict(color=ACTUAL_COLOR, size=4),
        ))

    fig.update_layout(
        **{k: v for k, v in _BASE_LAYOUT.items() if k != "hovermode"},
        title="Ground Track",
        geo=dict(
            scope="north america", projection_type="albers usa",
            showland=True, landcolor="#f8f9fa", showlakes=True, lakecolor="white",
            subunitcolor="#dadce0", countrycolor="#dadce0",
            fitbounds="locations",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _empty_fig(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=14, color=AXIS_COLOR), x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(**_BASE_LAYOUT)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


if __name__ == "__main__":
    # Smoke-test every chart against a real flight from the DB
    import sys
    sys.path.insert(0, "..")
    sys.path.insert(0, ".")
    from data.db import get_connection, get_flight_dataframe
    from analysis.atmosphere import add_atmospheric_columns
    from analysis.poh_compare import load_reference, compare_climb

    conn = get_connection()
    df = add_atmospheric_columns(get_flight_dataframe(conn, 16))

    charts = {
        "altitude_profile": altitude_profile(df),
        "speed_profile": speed_profile(df),
        "vertical_rate_profile": vertical_rate_profile(df),
        "wind_temp_profile": wind_temp_profile(df),
        "ground_track": ground_track(df),
    }

    ref = load_reference("a321")
    charts["climb_comparison"] = climb_comparison(compare_climb(df, ref))

    for name, fig in charts.items():
        n_traces = len(fig.data)
        print(f"  {name:<24} {n_traces} trace(s)  {'OK' if n_traces > 0 or 'empty' in name else 'CHECK'}")
    print("\nAll charts built without error.")