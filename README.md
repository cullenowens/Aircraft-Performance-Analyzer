# Aircraft Performance Analyzer

A live ADS-B data pipeline that tracks real commercial flights from takeoff to
landing, enriches every position report with weather captured at the moment
it happened, automatically detects flight phases, and compares the resulting
climb/cruise/descent profile against published aircraft performance data —
all served through an interactive dashboard.

Built as a portfolio project to demonstrate end-to-end data engineering: live
ingestion, a real database, a physics-grounded analysis layer, and a
visualization frontend — using only free, publicly available data sources.

## Why this exists

I wanted to build something that touched every layer of a real data system —
not a notebook that reads a static CSV, but a pipeline that pulls live data
from an external API, has to handle that API's rate limits and quirks,
stores it durably, runs non-trivial physics and statistics on it, and
presents the result somewhere a non-technical person could actually use it.
Aviation was the domain because it's full of real, checkable physics (ISA
atmosphere, true airspeed, Mach number) and because free flight-tracking data
(OpenSky Network, NOAA aviation weather) is genuinely available to anyone.

**What this is:** a tool for understanding what a specific flight actually
did — its climb rate, cruise speed, the winds it flew through, and how that
compares to how that aircraft type typically operates.

**What this is *not*:** a certified performance-monitoring tool, a safety
analysis system, or a replacement for what an airline's own FOQA (Flight
Operations Quality Assurance) program does with real flight data recorder
output. See [Honest limitations](#honest-limitations) below — this is
discussed throughout because it shaped almost every design decision.

## Features

- **Live flight tracking** — polls OpenSky Network for a specific aircraft
  from takeoff through landing, with dynamic polling intervals (dense during
  climb/descent, sparse during cruise) to conserve API credits
- **Weather captured at the moment of flight** — NOAA winds-aloft data is
  looked up per position report as it's collected, not applied
  retroactively, using haversine-nearest-station selection
- **Automatic landing detection** — stops polling once a full flight profile
  (takeoff → climb → cruise → descent → landing → confirmed on ground) has
  been observed, with a safety-net fallback for imperfect coverage
- **Physics-based derived metrics** — ISA atmosphere model, density
  altitude, true airspeed (recovered from groundspeed + wind vector), Mach
  number
- **Automatic flight-phase detection** — time-based smoothed vertical rate
  classification, robust to the poller's variable sampling rate
- **Aircraft type auto-detection** — resolves ICAO24 hex → aircraft type via
  OpenSky's public aircraft database, no manual entry required
- **Reference performance comparison** — actual climb/cruise/descent
  measured against published performance data for 8 aircraft types, each
  file honestly graded for source confidence
- **Profile-adherence rating** — a qualitative assessment ("Nominal" /
  "Notable deviations" / "Significant deviations"), deliberately *not* a
  numeric score, with explicit reasoning for every classification
- **Interactive dashboard** — Streamlit + Plotly, with data-driven
  explanatory captions under every chart

## Architecture

```
                    ┌─────────────────┐
                    │  OpenSky Network │  (live ADS-B state vectors)
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐      ┌──────────────────┐
                    │   poller.py       │◄────►│  NOAA Aviation    │
                    │  (live polling +  │      │  Weather Center   │
                    │  weather-at-      │      └──────────────────┘
                    │  capture-time)    │
                    └────────┬─────────┘
                             │ writes rows with weather attached
                    ┌────────▼─────────┐
                    │   db.py (SQLite)  │  flights + state_vectors tables
                    └────────┬─────────┘
                             │ on landing confirmed
                    ┌────────▼─────────┐
                    │ phase_detector.py │  batch phase labeling,
                    │                   │  written back to DB
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼──────────┐ ┌─▼─────────────┐
     │  atmosphere.py  │ │performance.py│ │ poh_compare.py │
     │  (ISA physics)  │ │ (metrics)    │ │ (reference     │
     │                 │ │              │ │  comparison)   │
     └─────────────────┘ └──────────────┘ └───────┬────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │    rating.py       │
                                          │ (profile adherence)│
                                          └─────────┬─────────┘
                                                     │
                                          ┌──────────▼──────────┐
                                          │   app.py (Streamlit) │
                                          │  charts / captions /  │
                                          │  report view-prep     │
                                          └───────────────────────┘
```

`pipeline.py` is the single entry point that ties polling and phase
detection together — you run one command, it tracks a flight until it
lands, then automatically labels and stores the result. Everything from
`atmosphere.py` onward is a pure read over already-stored data, run fresh
each time the dashboard loads.

## Tech stack

| Layer | Tools |
|---|---|
| Data ingestion | OpenSky Network REST API (OAuth2), NOAA Aviation Weather Center API |
| Storage | SQLite |
| Analysis | pandas, numpy |
| Visualization | Streamlit, Plotly |
| Language | Python 3.13 |

## Setup

1. Create a free [OpenSky Network](https://opensky-network.org) account and
   register an API client to get `OPENSKY_CLIENT_ID` / `OPENSKY_CLIENT_SECRET`.
2. Copy `.env.example` to `.env` and fill in your credentials.
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install requests python-dotenv pandas numpy streamlit plotly
   ```

## Usage

**Track a flight from takeoff to landing (auto-detects aircraft type, stops
on confirmed landing):**
```bash
.venv/bin/python3 pipeline.py --icao24 <hex> --dep KATL --arr KJFK --callsign DAL123
```

**Find and auto-track upcoming departures by route/carrier:**
```bash
.venv/bin/python3 find_and_track.py --destination JFK,MIA --carrier DAL
```

**List everything currently tracked:**
```bash
.venv/bin/python3 pipeline.py --list
```

**Launch the dashboard:**
```bash
.venv/bin/python3 -m streamlit run app.py
```

## The analysis pipeline

1. **Ingestion** — `poller.py` polls `/states/all` for one aircraft,
   converting raw units (meters, m/s) to aviation units (feet, knots, fpm)
   once at write time.
2. **Weather enrichment** — for each position, the nearest NOAA winds-aloft
   reporting station is found (haversine distance), and wind/temperature are
   linearly interpolated between the two nearest published altitude bands.
3. **Phase detection** — vertical rate is smoothed with a *time-based*
   (not row-based) centered window, so classification stays consistent
   regardless of whether samples are 10 or 60 seconds apart, then classified
   into ground/takeoff/climb/cruise/descent/landing.
4. **Atmospheric physics** — ISA standard temperature and density ratio are
   computed from pressure altitude via the standard barometric formula; true
   airspeed is recovered from groundspeed by removing the forecast wind
   vector; Mach number follows from local speed of sound.
5. **Performance metrics** — climb/descent rate is computed per altitude
   band using *contiguous-segment-aware* duration tracking, so a real ATC
   level-off doesn't silently dilute the measured rate (see
   [Bugs found and fixed](#notable-bugs-found-and-fixed)).
6. **Reference comparison** — measured performance is compared against
   published data for the aircraft's type, each reference file explicitly
   graded for how solid its sourcing actually is.
7. **Rating** — a qualitative profile-adherence assessment, intentionally
   not a single number (see below).

## Honest limitations

This project tries to be upfront about what free, public data can and can't
support, rather than presenting derived numbers with false confidence:

- **Aircraft weight is unknown.** ADS-B doesn't transmit it, and weight is
  one of the largest drivers of climb performance. A climb that looks
  "underperforming" against the reference may simply be a heavier aircraft.
- **True airspeed is derived, not measured** — computed from groundspeed
  plus a *forecast* wind, not an onboard sensor. Wind forecast error
  propagates directly into TAS and Mach.
- **Reference data is a mix of confidence levels.** Some files (A350-900,
  737-800) are anchored to a published operational reference (SKYbrary);
  others are extrapolated from same-class aircraft because no public
  performance table exists for that specific type. Every file states its
  own sourcing and confidence, and the dashboard surfaces that text
  prominently rather than hiding it in a footnote.
- **No reference file is a certified POH/AFM.** Manufacturer performance
  tables are proprietary. What's here describes *typical* operation, which
  is a meaningful "did this flight look normal" benchmark — not a
  certified performance standard.
- **The rating is deliberately not a score.** See `analysis/rating.py` for
  the full reasoning — a blended numeric grade would imply a precision this
  pipeline cannot support.

## Notable bugs found and fixed

Documented here because finding and fixing them was as much the point of
this project as the final dashboard:

- **Duration-calculation bug (general):** an early version of the phase
  summary computed a phase's duration as `max(time) - min(time)` across all
  its occurrences. For a phase like "ground" that happens once before
  takeoff and again after landing, this silently spanned the *entire
  flight* as "ground duration." Fixed by summing duration over contiguous
  segments instead of taking one global span.
- **Same bug, subtler form, in performance metrics:** even after phase
  labeling correctly excluded a mid-climb/descent level-off from the
  climb/descent phase, the altitude-band rate calculation still measured
  elapsed wall-clock time across the gap the level-off left behind — a
  single 4-minute level-off could report a flight performing at 37% of its
  true climb/descent rate. Fixed the same way: contiguous-segment-aware
  grouping, verified against a clean synthetic case before trusting it on
  real data.
- **Row-based smoothing window:** flight-phase smoothing used a fixed
  30-*row* window, but the poller samples at different intervals depending
  on phase (10s during climb/descent, 60s during cruise) — so the same "30
  rows" represented wildly different real time depending on when it was
  centered. Fixed by converting to a genuinely time-based (seconds) centered
  window.
- **Weather station discretization artifact:** investigating why one flight
  showed an apparent Mmo (maximum operating Mach) exceedance revealed that
  the aircraft's groundspeed was completely flat at the time — what had
  actually happened was the nearest-weather-station lookup switched from one
  station to an adjacent one mid-flight, and the two stations' published
  wind differed sharply enough to produce a discontinuous jump in derived
  TAS. A real limitation of nearest-single-station lookup versus
  distance-weighted interpolation between multiple stations.

## Project structure

```
data/
  auth.py              OAuth2 token management for OpenSky
  db.py                SQLite schema + read/write helpers
  poller.py             Live polling, weather-at-capture, landing detection
  phase_detector.py     Flight phase classification (time-based smoothing)
  aircraft_lookup.py    ICAO24 -> aircraft typecode resolution
analysis/
  atmosphere.py         ISA atmosphere model, TAS, Mach, density altitude
  weather.py            NOAA winds-aloft fetch/parse/interpolate
  performance.py         Climb/cruise/descent/summary metrics
  poh_compare.py         Reference performance comparison engine
  rating.py              Profile-adherence assessment (qualitative, not scored)
viz/
  charts.py              Plotly figure builders
  report.py              View-prep tables/cards for the dashboard
  captions.py             Data-driven per-chart explanatory text
poh_data/                Reference performance data, 8 aircraft types
app.py                    Streamlit dashboard
pipeline.py                Orchestrator: poll -> phase-label -> ready to view
find_and_track.py          Auto-discovers and tracks upcoming departures
```

## Future work

- Weight-aware climb comparison, if a proxy for takeoff weight (e.g. typical
  payload for that route/carrier) can be reasonably estimated
- Distance-weighted multi-station wind interpolation instead of nearest-
  single-station, to remove the discretization artifact described above
- Deduplicate polled rows at ingestion (occasional identical-timestamp
  repeats from OpenSky currently pass through unfiltered — low impact today
  since duration/rate math is boundary-based, but worth closing)
- Multi-flight comparison view — the dashboard currently shows one flight at
  a time; the more interesting long-run question ("does this airport
  consistently fly reduced-thrust departures?") needs an aggregate view
  across many tracked flights