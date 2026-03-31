"""
Physics Engine: Park Factors & Environment Composite Score.

Provides:
  1. Historical park factor lookup (HR, 1B, BABIP) by home team/stadium.
  2. Composite environment score combining park factor + COR + air density.
  3. DataFrame builder for the Gold layer physics join.
"""

import pandas as pd
from physics.aerodynamics import calc_air_density, air_density_score
from physics.humidor import cor_adjustment, humidor_babip_adjustment
from config import STADIUM_ALTITUDE_FT

# ── Park Factor Lookup ─────────────────────────────────────────────────────────
# Source: FanGraphs multi-year park factors (2021–2024 weighted average)
# Values > 1.0 = hitter-friendly, < 1.0 = pitcher-friendly
# Columns: team, stadium, hr_factor, singles_factor, babip_factor
_PARK_FACTOR_DATA = [
    ("COL", "Coors Field",              1.38, 1.17, 1.14),
    ("TEX", "Globe Life Field",         1.10, 1.05, 1.04),
    ("CIN", "Great American Ball Park", 1.09, 1.04, 1.03),
    ("PHI", "Citizens Bank Park",       1.07, 1.03, 1.02),
    ("CHC", "Wrigley Field",            1.06, 1.03, 1.02),
    ("ATL", "Truist Park",              1.05, 1.02, 1.01),
    ("HOU", "Minute Maid Park",         1.04, 1.01, 1.00),
    ("MIL", "American Family Field",    1.03, 1.01, 1.00),
    ("NYY", "Yankee Stadium",           1.08, 1.02, 1.01),
    ("BOS", "Fenway Park",              1.04, 1.07, 1.06),  # High singles/doubles park
    ("LAD", "Dodger Stadium",           0.94, 0.97, 0.97),
    ("SF",  "Oracle Park",              0.89, 0.96, 0.96),
    ("NYM", "Citi Field",               0.92, 0.97, 0.97),
    ("TB",  "Tropicana Field",          0.91, 0.98, 0.97),
    ("SD",  "Petco Park",               0.90, 0.97, 0.96),
    ("MIA", "loanDepot park",           0.93, 0.98, 0.98),
    ("SEA", "T-Mobile Park",            0.94, 0.99, 0.98),
    ("OAK", "Oakland Coliseum",         0.95, 0.99, 0.98),
    ("MIN", "Target Field",             0.98, 1.00, 1.00),
    ("DET", "Comerica Park",            0.97, 1.00, 0.99),
    ("PIT", "PNC Park",                 0.96, 0.99, 0.99),
    ("STL", "Busch Stadium",            0.97, 1.00, 0.99),
    ("WSH", "Nationals Park",           0.98, 1.00, 0.99),
    ("BAL", "Camden Yards",             1.02, 1.02, 1.01),
    ("CLE", "Progressive Field",        0.99, 1.00, 1.00),
    ("KC",  "Kauffman Stadium",         0.98, 1.00, 0.99),
    ("CWS", "Guaranteed Rate Field",    1.01, 1.01, 1.00),
    ("LAA", "Angel Stadium",            0.97, 0.99, 0.99),
    ("ARI", "Chase Field",              1.02, 1.01, 1.00),
    ("TOR", "Rogers Centre",            1.00, 1.00, 1.00),
]

PARK_FACTORS: dict[str, dict] = {
    row[0]: {
        "stadium":      row[1],
        "hr_factor":    row[2],
        "singles_factor": row[3],
        "babip_factor": row[4],
    }
    for row in _PARK_FACTOR_DATA
}

STADIUM_TO_TEAM: dict[str, str] = {v["stadium"]: k for k, v in PARK_FACTORS.items()}


def get_park_factor(team: str, factor_type: str = "babip_factor") -> float:
    """
    Return a specific park factor for a team.

    Parameters
    ----------
    team : str
        3-letter MLB team abbreviation (home team).
    factor_type : str
        One of 'hr_factor', 'singles_factor', 'babip_factor'.

    Returns
    -------
    float
        Park factor (1.0 = neutral).
    """
    pf = PARK_FACTORS.get(team.upper(), {})
    return pf.get(factor_type, 1.0)


def compute_env_composite(
    team: str,
    temp_f: float,
    humidity_pct: float,
    pressure_mb: float,
) -> dict:
    """
    Compute a composite environment score for a single game.

    Combines:
      - Air density advantage score [-1, 1]
      - COR adjustment (higher = hitter-friendly)
      - Humidor BABIP adjustment multiplier
      - Historical park BABIP factor

    Parameters
    ----------
    team : str
        Home team abbreviation.
    temp_f : float
        Game-time temperature (°F).
    humidity_pct : float
        Relative humidity (%).
    pressure_mb : float
        Barometric pressure (mb).

    Returns
    -------
    dict with keys:
        home_team, air_density, air_density_score, cor, humidor_babip_adj,
        park_babip_factor, env_composite
    """
    pf_info = PARK_FACTORS.get(team.upper(), {})
    stadium = pf_info.get("stadium", "Unknown")
    altitude_ft = STADIUM_ALTITUDE_FT.get(stadium, 0.0)

    rho       = calc_air_density(temp_f, altitude_ft, pressure_mb)
    rho_score = air_density_score(temp_f, altitude_ft, pressure_mb)
    cor       = cor_adjustment(stadium, humidity_pct)
    hum_adj   = humidor_babip_adjustment(stadium, humidity_pct)
    park_babip = pf_info.get("babip_factor", 1.0)

    # Composite: weighted blend of environment signals
    # Weights: air_density(40%) + park_factor(35%) + cor(25%)
    env_composite = (
        0.40 * rho_score +
        0.35 * (park_babip - 1.0) * 4 +     # scale to ~[-1,1] range
        0.25 * (cor - 0.520) * 50            # scale to ~[-1,1] range
    )
    env_composite = round(max(-1.0, min(1.0, env_composite)), 4)

    return {
        "home_team":         team.upper(),
        "stadium":           stadium,
        "altitude_ft":       altitude_ft,
        "air_density":       round(rho, 4),
        "air_density_score": rho_score,
        "cor_adjustment":    cor,
        "humidor_babip_adj": hum_adj,
        "park_babip_factor": park_babip,
        "env_composite":     env_composite,
    }


def build_physics_df(weather_data: dict[str, dict]) -> pd.DataFrame:
    """
    Build a physics DataFrame for all teams playing today.

    Parameters
    ----------
    weather_data : dict[str, WeatherData]
        Output of bronze.ingest_weather.fetch_all_weather().

    Returns
    -------
    pd.DataFrame
        One row per home team with all environment composite columns.
        Ready to join into the Gold table.
    """
    rows = []
    for team, w in weather_data.items():
        env = compute_env_composite(
            team=team,
            temp_f=w["temp_f"],
            humidity_pct=w["humidity_pct"],
            pressure_mb=w["pressure_mb"],
        )
        rows.append(env)
    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    # Example: print environment composite for all teams at typical conditions
    sample_weather = {
        team: {"temp_f": 72.0, "humidity_pct": 50.0, "pressure_mb": 1013.25}
        for team in PARK_FACTORS.keys()
    }
    df = build_physics_df(sample_weather)
    print(df[["home_team", "stadium", "air_density", "cor_adjustment", "env_composite"]]
          .sort_values("env_composite", ascending=False)
          .to_string(index=False))
