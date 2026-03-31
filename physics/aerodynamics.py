"""
Physics Engine: Air Density & Aerodynamic Modeling.

Implements the barometric formula to compute air density (ρ) based on:
  - Temperature (°F)
  - Altitude (feet above sea level)
  - Barometric pressure (millibars)

Also provides the fly-ball distance boost model:
  +10°F ≈ -2% air density ≈ +3.5 ft for a standard 400-ft fly ball.
"""

import math
from config import (
    RHO_SEA_LEVEL,
    TEMP_LAPSE_RATE,
    R_AIR,
    GRAVITY,
    DENSITY_TO_DISTANCE_COEFF,
)


def fahrenheit_to_kelvin(temp_f: float) -> float:
    """Convert Fahrenheit to Kelvin."""
    return (temp_f - 32.0) * 5.0 / 9.0 + 273.15


def calc_air_density(
    temp_f: float,
    altitude_ft: float,
    pressure_mb: float,
) -> float:
    """
    Calculate air density (kg/m³) using the ideal gas law with altitude correction.

    Uses the hypsometric/barometric formula:
        ρ = P / (R_air * T)
    where P is pressure at altitude (converted from mb to Pa) and T is in Kelvin.

    Parameters
    ----------
    temp_f : float
        Ambient temperature in Fahrenheit.
    altitude_ft : float
        Stadium altitude in feet above sea level.
    pressure_mb : float
        Surface barometric pressure in millibars (hPa).

    Returns
    -------
    float
        Air density in kg/m³. Sea level standard ≈ 1.225 kg/m³.

    Examples
    --------
    >>> round(calc_air_density(59, 0, 1013.25), 3)
    1.225
    >>> calc_air_density(70, 5200, 1013.25) < 1.0   # Denver, hot day
    True
    """
    temp_k = fahrenheit_to_kelvin(temp_f)
    altitude_m = altitude_ft * 0.3048

    # Altitude-corrected pressure using barometric formula
    # P(h) = P_surface * exp(-g*h / (R*T))
    pressure_pa = pressure_mb * 100.0   # mb → Pa
    pressure_at_alt = pressure_pa * math.exp(
        -(GRAVITY * altitude_m) / (R_AIR * temp_k)
    )

    # ρ = P / (R_air * T)
    rho = pressure_at_alt / (R_AIR * temp_k)
    return rho


def density_delta(
    temp_f: float,
    altitude_ft: float,
    pressure_mb: float,
) -> float:
    """
    Return Δρ relative to standard sea-level density (1.225 kg/m³).
    Negative means less dense air (ball carries further).
    """
    return calc_air_density(temp_f, altitude_ft, pressure_mb) - RHO_SEA_LEVEL


def temp_to_distance_boost(
    baseline_temp_f: float,
    current_temp_f: float,
    altitude_ft: float,
    pressure_mb: float,
    base_distance_ft: float = 400.0,
) -> float:
    """
    Estimate extra carry distance in feet due to temperature change.

    Calibrated to: +10°F ≈ +3.5 ft for a 400-ft standard fly ball.

    Parameters
    ----------
    baseline_temp_f : float
        Reference temperature (typically 72°F).
    current_temp_f : float
        Actual game-time temperature.
    altitude_ft : float
        Stadium altitude in feet.
    pressure_mb : float
        Barometric pressure in millibars.
    base_distance_ft : float
        Projected fly-ball distance under baseline conditions.

    Returns
    -------
    float
        Additional distance in feet (positive = further carry, negative = less).
    """
    rho_baseline = calc_air_density(baseline_temp_f, altitude_ft, pressure_mb)
    rho_current  = calc_air_density(current_temp_f,  altitude_ft, pressure_mb)
    delta_rho = rho_baseline - rho_current   # positive → current air is less dense

    # Linear calibration: Δρ of ~0.02 kg/m³ ≈ 3.5 ft for a 400-ft ball
    # Scale with base_distance for consistency
    distance_scale = base_distance_ft / 400.0
    boost = DENSITY_TO_DISTANCE_COEFF * delta_rho * distance_scale
    return boost


def air_density_score(
    temp_f: float,
    altitude_ft: float,
    pressure_mb: float,
) -> float:
    """
    Return a normalized 'air density advantage' score in [-1, 1].

    Positive = thinner air (hitter-friendly), Negative = denser air (pitcher-friendly).
    Anchored to Coors Field at 85°F summer day as max ≈ +1.0.
    """
    rho = calc_air_density(temp_f, altitude_ft, pressure_mb)
    delta = RHO_SEA_LEVEL - rho   # positive in thin air
    # Max delta observed (Coors, 95°F): ≈ 0.25 kg/m³
    score = max(-1.0, min(1.0, delta / 0.25))
    return round(score, 4)
