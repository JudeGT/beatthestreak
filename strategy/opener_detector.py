"""
Strategic Logic: Opener / Bullpen-Game Detector.

When a team uses an 'opener' (a reliever who starts 1–2 innings before
a 'bulk' pitcher takes over), batters face a different pitch mix than
when a traditional ace starts. This generally INCREASES hitter P(Hit)
because:
  - Openers are shorter and less optimal against specific batters
  - Bulk pitchers often have lower Stuff+ than traditional starters

Detection heuristic from Statcast:
  - True starter: pitched in the 1st inning, ≥ 3 innings total
  - Opener: pitched in the 1st inning, < 3 innings OR very small pitch count (<= 30)
"""

import logging
import duckdb
from config import DUCKDB_PATH, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

# P(Hit) boost multiplier when an opener is detected
OPENER_P_BOOST = 0.025   # +2.5 percentage points (additive)
BULLPEN_GAME_P_BOOST = 0.035  # +3.5 pp for full bullpen games (no starter)


def detect_openers_for_date(game_date: str) -> dict[str, bool]:
    """
    Detect which home teams are using an opener strategy on game_date.

    Parameters
    ----------
    game_date : str
        YYYY-MM-DD format.

    Returns
    -------
    dict[str, bool]
        Keyed by home_team abbreviation. True = opener detected.
    """
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        result = con.execute("""
            WITH starter_info AS (
                SELECT
                    home_team,
                    pitcher,
                    COUNT(*) AS total_pitches,
                    COUNT(DISTINCT inning) AS innings_pitched,
                    MIN(inning) AS first_inning,
                FROM statcast_raw
                WHERE game_date = ?
                  AND inning <= 2              -- Only consider pitchers who appeared early
                GROUP BY home_team, pitcher
            )
            SELECT
                home_team,
                -- Opener flag: pitched in inning 1 but < 3 innings OR <= 25 pitches
                CASE WHEN first_inning = 1
                      AND (innings_pitched < 3 OR total_pitches <= 25)
                    THEN TRUE ELSE FALSE END AS is_opener
            FROM starter_info
            WHERE first_inning = 1
        """, [game_date]).df()

        opener_map: dict[str, bool] = {}
        for _, row in result.iterrows():
            opener_map[row["home_team"]] = bool(row["is_opener"])

        log.info(f"Opener detection for {game_date}: {opener_map}")
        return opener_map
    finally:
        con.close()


def detect_bullpen_games(game_date: str) -> dict[str, bool]:
    """
    Detect full bullpen games (no pitcher threw in inning 1 with ≥ 50 pitches).
    Returns dict[home_team -> True if bullpen game].
    """
    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        result = con.execute("""
            WITH inn1_starters AS (
                SELECT home_team, MAX(COUNT(*)) AS max_pitches
                FROM statcast_raw
                WHERE game_date = ?
                  AND inning = 1
                GROUP BY home_team, pitcher
            )
            SELECT home_team,
                   CASE WHEN max_pitches < 50 THEN TRUE ELSE FALSE END AS is_bullpen_game
            FROM inn1_starters
        """, [game_date]).df()

        return {row["home_team"]: bool(row["is_bullpen_game"]) for _, row in result.iterrows()}
    finally:
        con.close()


def apply_opener_adjustment(
    candidates: list[dict],
    game_date: str,
) -> list[dict]:
    """
    Adjust P(Hit) upward for batters facing openers or bullpen-game teams.

    Parameters
    ----------
    candidates : list[dict]
        List of candidate dicts with keys: batter_id, home_team, away_team, p_hit.
    game_date : str
        Game date to check for opener usage.

    Returns
    -------
    list[dict]
        Same list with p_hit values adjusted and 'opener_adjusted' flag added.
    """
    try:
        opener_map   = detect_openers_for_date(game_date)
        bullpen_map  = detect_bullpen_games(game_date)
    except Exception as exc:
        log.warning(f"Opener detection failed ({exc}). Skipping adjustment.")
        return candidates

    for c in candidates:
        # The pitcher faces the batter on the VISITING team side of the game
        # Check if the home team (who fields the pitcher) uses an opener
        opp_team = c.get("home_team") or c.get("away_team")
        adjusted = False

        if opener_map.get(opp_team, False):
            c["p_hit"] = min(1.0, c["p_hit"] + OPENER_P_BOOST)
            c["opener_adjusted"] = True
            c["explanation"] = c.get("explanation", "") + " | Opener detected (+2.5%)"
            adjusted = True
        elif bullpen_map.get(opp_team, False):
            c["p_hit"] = min(1.0, c["p_hit"] + BULLPEN_GAME_P_BOOST)
            c["opener_adjusted"] = True
            c["explanation"] = c.get("explanation", "") + " | Bullpen game detected (+3.5%)"
            adjusted = True

        if not adjusted:
            c["opener_adjusted"] = False

    return candidates
