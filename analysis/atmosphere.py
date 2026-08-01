"""
atmosphere.py
International Standard Atmosphere (ISA) model and derived airspeed
calculations.

This module is pure physics — no network calls, no database access.
It converts the raw telemetry the poller collects (pressure altitude,
groundspeed, OAT, wind components) into the derived quantities that
performance analysis actually needs (density altitude, true airspeed,
Mach number, ISA temperature deviation).

A note on altitude:
  ADS-B `baro_altitude` is barometric altitude referenced to the
  standard 1013.25 hPa datum — i.e. it IS pressure altitude, not
  indicated/QNH altitude. That's convenient: it can be fed directly
  into the density altitude formulas below with no correction.

A note on airspeed:
  ADS-B gives groundspeed, not indicated or true airspeed. TAS is
  recovered by removing the wind vector (see true_airspeed_kts).
  Because the wind data comes from a forecast rather than onboard
  sensors, TAS here is an estimate — good enough for trend analysis
  and phase-level comparison, but not a substitute for FDR data.

ISA reference values:
  Sea level:   15.0 °C,  1013.25 hPa,  1.225 kg/m³
  Lapse rate:  1.98 °C per 1000 ft (troposphere)
  Tropopause:  36,089 ft,  -56.5 °C (isothermal above)
"""

import numpy as np

# ---------------------------------------------------------------------------
# ISA constants
# ---------------------------------------------------------------------------

ISA_SEA_LEVEL_TEMP_C   = 15.0
ISA_TROPOPAUSE_TEMP_C  = -56.5
LAPSE_RATE_C_PER_FT    = 0.0019812      # 1.98 °C per 1000 ft
TROPOPAUSE_FT          = 36089.0

KELVIN_OFFSET          = 273.15
ISA_SEA_LEVEL_TEMP_K   = ISA_SEA_LEVEL_TEMP_C + KELVIN_OFFSET   # 288.15 K

# Speed of sound coefficient: a_kts = 38.967 * sqrt(T_K)
# Derived from a = sqrt(gamma * R * T) with gamma=1.4, R=287 J/(kg·K),
# converted m/s -> knots. Gives 661.5 kts at ISA sea level.
SPEED_OF_SOUND_COEFF   = 38.967


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

def isa_temperature_c(pressure_alt_ft):
    """
    ISA standard temperature (°C) at a given pressure altitude.

    Linear lapse through the troposphere, isothermal above the
    tropopause at 36,089 ft.

    Accepts a scalar or array; returns the same shape.
    """
    pressure_alt_ft = np.asarray(pressure_alt_ft, dtype=float)
    tropo = ISA_SEA_LEVEL_TEMP_C - LAPSE_RATE_C_PER_FT * pressure_alt_ft
    return np.where(pressure_alt_ft < TROPOPAUSE_FT, tropo, ISA_TROPOPAUSE_TEMP_C)


def isa_deviation_c(pressure_alt_ft, oat_c):
    """
    ISA temperature deviation: how much warmer (+) or colder (-) the
    actual air is than standard for that altitude.

    This is the number that actually drives performance differences —
    "ISA+15" means noticeably degraded climb and higher TAS for a
    given indicated speed.
    """
    return np.asarray(oat_c, dtype=float) - isa_temperature_c(pressure_alt_ft)


# ---------------------------------------------------------------------------
# Pressure / density ratios
# ---------------------------------------------------------------------------

def pressure_ratio(pressure_alt_ft):
    """
    Delta — ratio of ambient static pressure to ISA sea level pressure.

    Uses the troposphere power law below 36,089 ft and the exponential
    stratosphere form above it (the two agree at the tropopause).
    """
    h = np.asarray(pressure_alt_ft, dtype=float)
    tropo  = np.power(np.clip(1.0 - 6.87535e-6 * h, 1e-9, None), 5.2559)
    strato = 0.223361 * np.exp(-4.80634e-5 * (h - TROPOPAUSE_FT))
    return np.where(h < TROPOPAUSE_FT, tropo, strato)


def density_ratio(pressure_alt_ft, oat_c):
    """
    Sigma — ratio of ambient air density to ISA sea level density,
    from the ideal gas relation sigma = delta / theta.
    """
    delta = pressure_ratio(pressure_alt_ft)
    theta = (np.asarray(oat_c, dtype=float) + KELVIN_OFFSET) / ISA_SEA_LEVEL_TEMP_K
    return delta / theta


def density_altitude_ft(pressure_alt_ft, oat_c):
    """
    Density altitude — the ISA altitude at which the ambient density
    would be standard. This is what the aircraft's aerodynamics and
    engines actually "feel".

    Note: density altitude is a performance concept aimed at takeoff,
    climb, and low-altitude operations. It's still mathematically
    well-defined at FL340, but comparing an airliner's cruise to a
    "density altitude" isn't especially meaningful — use ISA deviation
    and Mach up there instead.
    """
    sigma = density_ratio(pressure_alt_ft, oat_c)
    sigma = np.clip(sigma, 1e-9, None)
    return 145442.16 * (1.0 - np.power(sigma, 0.234969))


# ---------------------------------------------------------------------------
# Airspeed
# ---------------------------------------------------------------------------

def true_airspeed_kts(groundspeed_kts, headwind_kts=0.0, crosswind_kts=0.0):
    """
    Recover true airspeed from groundspeed by removing the wind vector.

    Working in a frame aligned with the aircraft's ground track:
        V_ground = (GS, 0)
        V_wind   = (-headwind, crosswind)     [headwind opposes motion]
        V_air    = V_ground - V_wind = (GS + headwind, -crosswind)
        TAS      = |V_air|

    So TAS = sqrt((GS + headwind)^2 + crosswind^2).

    The crosswind term is what accounts for crab — an aircraft holding
    a track through a crosswind is flying slightly "sideways" relative
    to its track, so its airspeed exceeds the along-track component.
    Usually a small correction, but it's free to include correctly.

    Sign convention matches weather._wind_components(): positive
    headwind means wind opposing the aircraft, so TAS > GS.
    """
    gs = np.asarray(groundspeed_kts, dtype=float)
    hw = np.asarray(headwind_kts, dtype=float)
    cw = np.asarray(crosswind_kts, dtype=float)
    return np.sqrt(np.square(gs + hw) + np.square(cw))


def speed_of_sound_kts(oat_c):
    """Local speed of sound (knots) for a given static air temperature."""
    t_k = np.asarray(oat_c, dtype=float) + KELVIN_OFFSET
    return SPEED_OF_SOUND_COEFF * np.sqrt(np.clip(t_k, 1e-9, None))


def mach_number(tas_kts, oat_c):
    """Mach number from true airspeed and static air temperature."""
    return np.asarray(tas_kts, dtype=float) / speed_of_sound_kts(oat_c)


def equivalent_airspeed_kts(tas_kts, pressure_alt_ft, oat_c):
    """
    Equivalent airspeed — TAS corrected back to sea level density.
    EAS = TAS * sqrt(sigma).

    Useful as a rough proxy for indicated airspeed (they differ only
    by compressibility correction, which is small below ~200 KIAS but
    grows at airliner speeds). Reported here as EAS rather than IAS
    precisely because that compressibility term is not applied.
    """
    sigma = density_ratio(pressure_alt_ft, oat_c)
    return np.asarray(tas_kts, dtype=float) * np.sqrt(np.clip(sigma, 0.0, None))


# ---------------------------------------------------------------------------
# DataFrame convenience wrapper
# ---------------------------------------------------------------------------

def add_atmospheric_columns(df):
    """
    Add derived atmospheric and airspeed columns to a flight DataFrame.

    Expects columns (as produced by the poller / stored in the DB):
        baro_altitude  — pressure altitude, feet
        velocity       — groundspeed, knots
        oat_c          — outside air temperature, °C  (may be NaN)
        headwind_kts   — headwind component, knots    (may be NaN)
        crosswind_kts  — crosswind component, knots   (may be NaN)

    Adds:
        isa_temp_c      — ISA standard temp at that altitude
        isa_dev_c       — actual minus standard
        density_alt_ft  — density altitude
        sigma           — density ratio
        tas_kts         — true airspeed
        mach            — Mach number
        eas_kts         — equivalent airspeed

    Rows with missing OAT fall back to ISA temperature so the
    calculation still produces a usable (if less accurate) result
    rather than propagating NaN through the whole analysis. An
    `oat_is_estimated` flag marks those rows so downstream reporting
    can be honest about which numbers came from real weather data.
    """
    import pandas as pd

    df = df.copy()

    alt = df["baro_altitude"].astype(float)

    df["isa_temp_c"] = isa_temperature_c(alt)

    if "oat_c" in df.columns:
        oat = pd.to_numeric(df["oat_c"], errors="coerce")
    else:
        oat = pd.Series(np.nan, index=df.index)

    df["oat_is_estimated"] = oat.isna()
    oat_filled = oat.fillna(pd.Series(df["isa_temp_c"], index=df.index))

    df["isa_dev_c"]      = oat_filled - df["isa_temp_c"]
    df["sigma"]          = density_ratio(alt, oat_filled)
    df["density_alt_ft"] = density_altitude_ft(alt, oat_filled)

    hw = pd.to_numeric(df.get("headwind_kts", 0.0), errors="coerce").fillna(0.0)
    cw = pd.to_numeric(df.get("crosswind_kts", 0.0), errors="coerce").fillna(0.0)
    gs = pd.to_numeric(df["velocity"], errors="coerce")

    df["tas_kts"] = true_airspeed_kts(gs, hw, cw)
    df["mach"]    = mach_number(df["tas_kts"], oat_filled)
    df["eas_kts"] = equivalent_airspeed_kts(df["tas_kts"], alt, oat_filled)

    return df


# ---------------------------------------------------------------------------
# Manual test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ISA temperature ===")
    for alt in [0, 5000, 10000, 20000, 36089, 40000]:
        print(f"  {alt:>6,} ft -> {float(isa_temperature_c(alt)):+6.1f} °C")

    print("\n=== Density altitude sanity checks ===")
    # At ISA conditions, density altitude should equal pressure altitude
    for alt in [0, 5000, 10000]:
        isa_t = float(isa_temperature_c(alt))
        da = float(density_altitude_ft(alt, isa_t))
        print(f"  PA {alt:>6,} ft @ ISA ({isa_t:+.1f}°C) -> DA {da:>8,.0f} ft   (should match PA)")

    # Hot day at sea level: DA should be well above field elevation
    da_hot = float(density_altitude_ft(0, 35.0))
    print(f"  PA      0 ft @ +35°C          -> DA {da_hot:>8,.0f} ft   (hot day, expect ~2,000+)")

    print("\n=== True airspeed ===")
    print(f"  GS 450, 50 kt headwind, 0 xwind -> TAS {float(true_airspeed_kts(450, 50, 0)):.1f} kts  (expect 500)")
    print(f"  GS 450, -50 kt (tailwind)       -> TAS {float(true_airspeed_kts(450, -50, 0)):.1f} kts  (expect 400)")
    print(f"  GS 450, 0 head, 40 kt xwind     -> TAS {float(true_airspeed_kts(450, 0, 40)):.1f} kts  (expect ~451.8)")

    print("\n=== Speed of sound / Mach ===")
    print(f"  a at ISA sea level: {float(speed_of_sound_kts(15.0)):.1f} kts  (expect ~661.5)")
    a340 = float(speed_of_sound_kts(-50.0))
    print(f"  a at -50°C:         {a340:.1f} kts")
    print(f"  TAS 480 @ -50°C  -> M{float(mach_number(480, -50.0)):.3f}")

    print("\n=== DataFrame wrapper ===")
    import pandas as pd
    test_df = pd.DataFrame({
        "baro_altitude": [1000.0, 15000.0, 34000.0, 34000.0],
        "velocity":      [180.0,  380.0,   460.0,   460.0],
        "oat_c":         [12.0,   -15.0,   -52.0,   np.nan],   # last row missing OAT
        "headwind_kts":  [5.0,    20.0,    35.0,    35.0],
        "crosswind_kts": [3.0,    -10.0,   15.0,    15.0],
    })
    out = add_atmospheric_columns(test_df)
    cols = ["baro_altitude", "oat_c", "isa_temp_c", "isa_dev_c",
            "density_alt_ft", "tas_kts", "mach", "oat_is_estimated"]
    print(out[cols].round(2).to_string(index=False))

    #TODO
    # figure out math behind stats