"""
Physics Engine: Humidor / Coefficient of Restitution (COR) Adjustment.

Models how a baseball's bounciness changes based on relative humidity (RH)
in stadiums equipped with ball humidors (Coors, Chase, Globe Life).

A wetter ball has a lower COR → less explosive off the bat → fewer HRs.
A drier ball (altitude + low humidity) has a higher COR → more carry.

Reference: Studies show COR drops ~0.01–0.005 per 10% RH increase above 50%.
"""

from config import (
    HUMIDOR_STADIUMS,
    HUMIDOR_RH_TARGET,
    COR_DRY,
    COR_WET,
    COR_RH_SLOPE,
)


def cor_adjustment(
    stadium: str,
    ambient_rh_pct: float,
) -> float:
    """
    Calculate the effective Coefficient of Restitution (COR) for a baseball
    given the stadium and ambient relative humidity.

    Parameters
    ----------
    stadium : str
        Stadium name (must match STADIUM_ALTITUDE_FT keys in config.py).
    ambient_rh_pct : float
        Ambient relative humidity percentage (0–100).

    Returns
    -------
    float
        Effective COR of the baseball. Standard range: 0.510 – 0.535.
        Lower COR = less elastic = pitchers' park.

    Notes
    -----
    At humidor stadiums, the ball is stored at a target RH (typically 50%).
    The stored ball's moisture content drives COR, NOT the ambient humidity.
    We blend ambient RH and humidor target RH with a 30/70 weight to reflect
    partial atmospheric re-drying once a ball is removed from storage.
    """
    if stadium in HUMIDOR_STADIUMS:
        rh_target = HUMIDOR_RH_TARGET[stadium]
        # Blend: ball spends ~70% of life at stored humidity, 30% ambient
        effective_rh = 0.70 * rh_target + 0.30 * ambient_rh_pct
    else:
        # Non-humidor stadiums: ball absorbs purely from ambient air
        effective_rh = ambient_rh_pct

    # COR linear model: COR = COR_DRY + COR_RH_SLOPE * (RH - 0%)
    # COR_RH_SLOPE is negative (higher RH → lower COR)
    cor = COR_DRY + COR_RH_SLOPE * effective_rh

    # Clamp to physically reasonable range
    cor = max(0.490, min(0.545, cor))
    return round(cor, 4)


def cor_to_distance_effect(cor: float, base_distance_ft: float = 400.0) -> float:
    """
    Translate COR change into distance effect on a fly ball.

    A 0.01 increase in COR ≈ +4 feet on a 400-ft fly ball.
    (Based on Statcast humidor studies: ~4 ft/COR unit).

    Parameters
    ----------
    cor : float
        Effective COR of the ball.
    base_distance_ft : float
        Base projection distance (scales effect linearly).

    Returns
    -------
    float
        Distance adjustment in feet relative to COR_DRY baseline.
    """
    delta_cor = cor - COR_DRY
    # 0.01 COR delta ≈ 4 ft on a 400-ft ball → 400 ft/COR_unit
    distance_effect = delta_cor * 400.0 * (base_distance_ft / 400.0)
    return round(distance_effect, 2)


def humidor_babip_adjustment(stadium: str, ambient_rh_pct: float) -> float:
    """
    Return a BABIP multiplicative adjustment factor based on humidor effect.

    Typical range: 0.92 (Coors with humidor active, wet ball) to 1.05 (dry stadium).
    Coors pre-humidor (dry, high altitude) had HR rates ~40% above average.
    Post-humidor, the ball is effectively 'normalized'.

    Returns
    -------
    float
        Multiplier to apply to projected BABIP. 1.0 = neutral.
    """
    cor = cor_adjustment(stadium, ambient_rh_pct)
    # COR → BABIP adjustment: each 0.01 COR unit ≈ ±2% BABIP effect
    delta_cor = cor - COR_DRY
    adjustment = 1.0 + (delta_cor / 0.01) * 0.02
    return round(max(0.85, min(1.15, adjustment)), 4)


if __name__ == "__main__":
    for stadium in ["Coors Field", "Chase Field", "Fenway Park"]:
        for rh in [30, 50, 70]:
            cor = cor_adjustment(stadium, rh)
            dist = cor_to_distance_effect(cor)
            print(f"{stadium:30s} RH={rh:3d}%  COR={cor:.4f}  dist_adj={dist:+.1f} ft")
