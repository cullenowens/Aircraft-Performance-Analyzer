"""
app.py
Streamlit dashboard for the Aircraft Performance Analyzer.

This is a thin layout shell: it picks a flight, pulls its already-
processed data from the database, runs the analysis layer, and hands
the results to viz/charts.py and viz/report.py for display. All the
real logic lives in those modules — app.py only arranges them.

Run with:
    .venv/bin/python3 -m streamlit run app.py

The dashboard is read-only over the database. It does NOT poll, write,
or modify anything — tracking new flights is done separately via
pipeline.py. This separation means the dashboard can be refreshed and
re-run freely without any risk to collected data or API credits.
"""

import streamlit as st

from data.db import get_connection, list_flights, get_flight_dataframe, get_flight_meta
from data.phase_detector import phase_summary
from analysis.atmosphere import add_atmospheric_columns
from analysis.performance import flight_summary
from analysis.poh_compare import generate_report
from analysis.rating import overall_rating

from viz import charts, report as report_view, captions


st.set_page_config(page_title="Aircraft Performance Analyzer", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_flight_list():
    conn = get_connection()
    return list_flights(conn)


@st.cache_data(show_spinner=False)
def load_flight(flight_id: int):
    """Load one flight's data + metadata + enriched frame + analysis report."""
    conn = get_connection()
    meta = get_flight_meta(conn, flight_id)
    df = get_flight_dataframe(conn, flight_id)
    enriched = add_atmospheric_columns(df)
    report = generate_report(df, aircraft_type=meta.get("aircraft_type") if meta else None)
    summary = flight_summary(df)
    phases = phase_summary(df)
    rating = overall_rating(report, df)
    return meta, enriched, report, summary, phases, rating


# ---------------------------------------------------------------------------
# Sidebar — flight picker
# ---------------------------------------------------------------------------

st.sidebar.title("✈ Flight Analyzer")

flights = load_flight_list()

if flights.empty:
    st.sidebar.warning("No flights in the database yet.")
    st.title("No flights to show")
    st.markdown(
        "Track a flight first with:\n\n"
        "```bash\n.venv/bin/python3 pipeline.py --icao24 <hex> --dep KATL --arr KJFK\n```"
    )
    st.stop()

# Build readable labels for the picker
def _flight_label(row):
    cs = (row["callsign"] or "?").strip()
    typ = row["aircraft_type"] or "?"
    return f"#{row['flight_id']} · {cs} · {row['dep_airport']}→{row['arr_airport']} · {typ}"

flights = flights.sort_values("flight_id", ascending=False)
options = {_flight_label(r): int(r["flight_id"]) for _, r in flights.iterrows()}

chosen_label = st.sidebar.selectbox("Select a flight", list(options.keys()))
flight_id = options[chosen_label]

band_ft = st.sidebar.select_slider(
    "Altitude band size", options=[2000, 2500, 5000, 10000], value=5000,
    help="Granularity of the climb/descent performance breakdown.",
)

meta, df, report, summary, phases, rating = load_flight(flight_id)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

cs = (meta.get("callsign") or "Unknown").strip()
st.title(f"{cs} — {meta.get('dep_airport','?')} → {meta.get('arr_airport','?')}")
st.caption(
    f"Aircraft type {meta.get('aircraft_type') or 'unknown'} · "
    f"ICAO24 {meta.get('icao24','?')} · flight #{flight_id} · status {meta.get('status','?')}"
)

# Summary cards row
cards = report_view.summary_cards(summary)
if cards:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.metric(card["label"], card["value"], help=card.get("help"))

# Profile-adherence headline — deliberately a word, not a score (see analysis/rating.py)
headline = rating.get("headline", "")
headline_color = {"Nominal": "🟢", "Notable deviations": "🟡",
                  "Significant deviations": "🔴", "No reference available": "⚪"}.get(headline, "")
st.markdown(f"**Profile adherence: {headline_color} {headline}**")
st.caption(rating.get("scope_note", ""))

st.divider()

# ---------------------------------------------------------------------------
# Charts — profile tab and comparison tab
# ---------------------------------------------------------------------------

tab_profile, tab_compare, tab_rating, tab_data = st.tabs(
    ["Flight profile", "Performance comparison", "Profile adherence", "Raw data"]
)

with tab_profile:
    st.plotly_chart(charts.altitude_profile(df), use_container_width=True)
    st.caption(captions.altitude_caption(df))

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.speed_profile(df), use_container_width=True)
        st.caption(captions.speed_caption(df))
    with c2:
        st.plotly_chart(charts.vertical_rate_profile(df), use_container_width=True)
        st.caption(captions.vertical_rate_caption(df))

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(charts.wind_temp_profile(df), use_container_width=True)
        st.caption(captions.wind_temp_caption(df))
    with c4:
        st.plotly_chart(charts.ground_track(df), use_container_width=True)
        st.caption(captions.ground_track_caption(df, summary.get("track_distance_nm")))

    st.subheader("Phase breakdown")
    phase_tbl = report_view.phase_table(phases)
    st.dataframe(phase_tbl, use_container_width=True, hide_index=True)


with tab_compare:
    # The reference caveat is the whole point — show it prominently, not buried
    ref_meta = report.get("reference_meta", {})
    if report.get("reference_key") is None:
        st.info(report.get("caveat_text", "No reference data available for this aircraft type."))
    else:
        with st.expander("⚠ About this reference data — read before interpreting", expanded=True):
            st.text(report.get("caveat_text", ""))

        st.subheader("Climb rate by altitude band")
        st.plotly_chart(charts.climb_comparison(report["climb"]), use_container_width=True)
        st.caption(captions.climb_comparison_caption(rating.get("climb", {})))
        st.dataframe(report_view.climb_table(report["climb"]),
                     use_container_width=True, hide_index=True)

        col_cruise, col_descent = st.columns(2)
        with col_cruise:
            st.subheader("Cruise")
            for r in report_view.cruise_rows(report["cruise"]):
                st.markdown(f"**{r['label']}**  ·  {r['value']}")
        with col_descent:
            st.subheader("Descent")
            for r in report_view.descent_rows(report["descent"]):
                st.markdown(f"**{r['label']}**  ·  {r['value']}")

    # Caveats always shown, whether or not a reference was available
    st.divider()
    st.subheader("Caveats")
    for c in report.get("caveats", []):
        st.markdown(f"- {c}")


with tab_rating:
    st.markdown(f"### {headline_color} {headline}")
    st.info(rating.get("scope_note", ""))

    if rating.get("headline") != "No reference available":
        st.subheader("Limits compliance")
        limits = rating.get("limits", {})
        if limits.get("checks"):
            for c in limits["checks"]:
                icon = "✅" if c["result"] == "PASS" else "❌"
                st.markdown(f"{icon} **{c['check']}** — {c['detail']}")
        else:
            st.caption("No limit checks available for this aircraft type.")

        st.subheader("Climb pattern")
        climb_r = rating.get("climb", {})
        st.markdown(
            f"**{climb_r.get('n_within', 0)} within tolerance · "
            f"{climb_r.get('n_below', 0)} below · {climb_r.get('n_above', 0)} above** "
            f"(of {climb_r.get('n_total', 0)} altitude bands)"
        )
        st.caption(climb_r.get("note", ""))

        st.subheader("Cruise")
        cruise_r = rating.get("cruise", {})
        st.caption(cruise_r.get("note", ""))

        st.subheader("Descent")
        descent_r = rating.get("descent", {})
        st.caption(descent_r.get("note", ""))
    else:
        st.caption("No profile-adherence assessment could be computed for this aircraft type.")


with tab_data:
    st.caption(
        "Raw per-sample data as stored, plus derived atmospheric columns. "
        "One row per poll."
    )
    display_cols = [
        "time_position", "phase", "baro_altitude", "velocity", "vertical_rate",
        "tas_kts", "mach", "oat_c", "isa_dev_c", "headwind_kts", "wx_station",
    ]
    present = [c for c in display_cols if c in df.columns]
    st.dataframe(df[present], use_container_width=True, hide_index=True)