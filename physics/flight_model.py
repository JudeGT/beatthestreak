"""
Physics Engine: Drag-Coefficient Flight Distance Model.

Derives estimated fly-ball distance boost using a simplified drag model
where drag force depends on air density (ρ), velocity, and cross-section area.

The model is calibrated to match Statcast carry data:
  - A +10°F increase at sea level → ~3.5 additional feet on a 400-ft fly ball.
  - Coors Field at 85°F provides roughly +25–30 ft advantage over Fenway at 55°F.
"""

import math
from physics.aerodynamics import calc_air_density, RHO_SEA_LEVEL


# Baseball physical constants
BALL_MASS_KG     = 0.14175    # 5.0 oz in kg
BALL_RADIUS_M    = 0.0366     # 1.44 inches in meters
BALL_AREA_M2     = math.pi * BALL_RADIUS_M ** 2   # Cross-sectional area
CD_BASEBALL      = 0.35       # Drag coefficient for a spinning baseball (typical range 0.30–0.45)
G_M_S2           = 9.80665    # Gravity (m/s²)
FT_PER_METER     = 3.28084


def exit_velo_mph_to_ms(mph: float) -> float:
    """Convert exit velocity from MPH to m/s."""
    return mph * 0.44704


def estimate_fly_ball_distance(
    exit_velo_mph: float,
    launch_angle_deg: float,
    temp_f: float,
    altitude_ft: float,
    pressure_mb: float,
) -> float:
    """
    Estimate fly-ball carry distance (in feet) using a simplified kinematic
    drag model integrated with the air density engine.

    Uses the range equation with linear drag approximation:
        R ≈ (v₀² * sin(2θ)) / g  — no drag
        R_drag ≈ R_vacuum * (1 − k * ρ)   — linear drag correction

    Parameters
    ----------
    exit_velo_mph : float
        Exit velocity off the bat in MPH.
    launch_angle_deg : float
        Vertical launch angle in degrees.
    temp_f : float
        Ambient temperature (°F).
    altitude_ft : float
        Stadium altitude in feet.
    pressure_mb : float
        Barometric pressure in millibars.

    Returns
    -------
    float
        Estimated fly-ball distance in feet.
    """
    v0 = exit_velo_mph_to_ms(exit_velo_mph)
    theta = math.radians(launch_angle_deg)

    # Vacuum range (no air resistance)
    r_vacuum = (v0 ** 2 * math.sin(2 * theta)) / G_M_S2

    # Air density at venue
    rho = calc_air_density(temp_f, altitude_ft, pressure_mb)

    # Drag correction factor: derived from dimensional analysis
    # k = CD * Area / (2 * mass) — has units of m²/kg
    k = (CD_BASEBALL * BALL_AREA_M2) / (2.0 * BALL_MASS_KG)

    # Average velocity approximation for drag (use 60% of v0 as mean)
    v_avg = 0.60 * v0
    # Increase drag effect to calibrate distances down to empirical ~400ft for 100mph/28deg
    # Multiplied by 8.5 to approximate expected standard carry distances
    drag_factor = math.exp(-k * 8.5 * rho * r_vacuum / v_avg) if v_avg > 0 else 1.0

    r_with_drag_m = r_vacuum * drag_factor
    return r_with_drag_m * FT_PER_METER


def distance_boost_vs_baseline(
    exit_velo_mph: float,
    launch_angle_deg: float,
    temp_f: float,
    altitude_ft: float,
    pressure_mb: float,
    baseline_temp_f: float = 72.0,
    baseline_altitude_ft: float = 0.0,
    baseline_pressure_mb: float = 1013.25,
) -> float:
    """
    Return the distance advantage (in feet) relative to a sea-level baseline.

    Positive = hitter-friendly venue/conditions (ball carries further).
    Negative = pitcher-friendly conditions (ball dies).
    """
    d_venue = estimate_fly_ball_distance(
        exit_velo_mph, launch_angle_deg, temp_f, altitude_ft, pressure_mb
    )
    d_baseline = estimate_fly_ball_distance(
        exit_velo_mph, launch_angle_deg,
        baseline_temp_f, baseline_altitude_ft, baseline_pressure_mb,
    )
    return round(d_venue - d_baseline, 2)


def temp_sensitivity_ft_per_10f(
    exit_velo_mph: float = 100.0,
    launch_angle_deg: float = 28.0,
    altitude_ft: float = 0.0,
    pressure_mb: float = 1013.25,
    baseline_temp_f: float = 72.0,
) -> float:
    """
    Verify the +10°F → +3.5 ft calibration for a typical fly ball.
    Returns distance boost per 10°F temperature increase at sea level.
    """
    d_base = estimate_fly_ball_distance(
        exit_velo_mph, launch_angle_deg, baseline_temp_f, altitude_ft, pressure_mb
    )
    d_hot = estimate_fly_ball_distance(
        exit_velo_mph, launch_angle_deg, baseline_temp_f + 10.0, altitude_ft, pressure_mb
    )
    return round(d_hot - d_base, 2)


if __name__ == "__main__":
    # Sanity checks
    boost = temp_sensitivity_ft_per_10f()
    print(f"+10°F → +{boost:.1f} ft (target: ~3.5 ft)")

    denver = distance_boost_vs_baseline(
        exit_velo_mph=103, launch_angle_deg=28,
        temp_f=85, altitude_ft=5200, pressure_mb=1013.25,
    )
    print(f"Coors Field (85°F) vs sea-level baseline: +{denver:.1f} ft")
