"""
Strategic Logic: Milestone Threshold & Pick Selection.

Implements the three-phase strategy:
  Phase 1 (0–10 games):   Aggressive   — P > 0.80, aggressive Double Down
  Phase 2 (11–40 games):  Opportunistic — P > 0.85
  Phase 3 (41–57 games):  Ultra-conservative — P > 0.92, 1 pick/day only
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import get_threshold, DOUBLE_DOWN_PHASE_0_10, MAX_DAILY_PICKS, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)


@dataclass
class Pick:
    batter_id:   int
    batter_name: str
    game_date:   str
    p_hit:       float
    home_team:   str
    away_team:   str
    stand:       str
    double_down: bool = False
    explanation: str  = ""

    def to_dict(self) -> dict:
        return {
            "batter_id":   self.batter_id,
            "batter_name": self.batter_name,
            "game_date":   self.game_date,
            "p_hit":       round(self.p_hit, 4),
            "home_team":   self.home_team,
            "away_team":   self.away_team,
            "stand":       self.stand,
            "double_down": self.double_down,
            "explanation": self.explanation,
        }


def get_phase(streak_len: int) -> str:
    """Return the name of the current strategy phase."""
    if streak_len <= 10:
        return "aggressive"
    elif streak_len <= 40:
        return "opportunistic"
    else:
        return "ultra_conservative"


def select_picks(
    candidates: list[dict],
    streak_len: int,
    double_down_budget: int = 1,
    force_single_pick: bool = False,
) -> list[Pick]:
    """
    Select today's picks from a ranked list of batter candidates.

    Parameters
    ----------
    candidates : list[dict]
        Ranked list of dicts with keys: batter_id, game_date, p_hit, home_team,
        away_team, stand. Must be sorted by p_hit descending.
    streak_len : int
        Current streak length (0 = start of streak attempt).
    double_down_budget : int
        Number of Double Down uses remaining.
    force_single_pick : bool
        If True, return at most 1 pick regardless of phase (ultra-conservative override).

    Returns
    -------
    list[Pick]
        Selected picks with Double Down flag set as appropriate.
    """
    threshold = get_threshold(streak_len)
    phase     = get_phase(streak_len)
    max_picks = 1 if (phase == "ultra_conservative" or force_single_pick) else MAX_DAILY_PICKS

    log.info(
        f"Streak={streak_len} | Phase={phase} | "
        f"Threshold={threshold:.2f} | Max picks={max_picks}"
    )

    # Filter by threshold
    qualified = [c for c in candidates if c["p_hit"] >= threshold]
    if not qualified:
        log.warning(f"No candidates meet P(Hit) ≥ {threshold:.2f} threshold today.")
        return []

    # Take top N picks
    selected_candidates = qualified[:max_picks]

    picks = []
    for i, c in enumerate(selected_candidates):
        # Double Down: apply to the top pick if budget available and phase is aggressive
        use_dd = (
            i == 0
            and double_down_budget > 0
            and phase == "aggressive"
            and DOUBLE_DOWN_PHASE_0_10
            and c["p_hit"] >= 0.85   # only DD on very high-confidence picks
        )
        pick = Pick(
            batter_id=c["batter_id"],
            batter_name=c.get("batter_name", "Unknown"),
            game_date=c["game_date"],
            p_hit=c["p_hit"],
            home_team=c["home_team"],
            away_team=c["away_team"],
            stand=c["stand"],
            double_down=use_dd,
            explanation=c.get("explanation", ""),
        )
        picks.append(pick)
        log.info(
            f"  Pick #{i+1}: batter={pick.batter_name} ({pick.batter_id}) "
            f"P={pick.p_hit:.3f} DD={pick.double_down}"
        )

    return picks


def should_use_streak_saver(
    current_p_hit: float,
    streak_len: int,
    savers_remaining: int,
) -> bool:
    """
    Decide whether to use a Streak Saver on the current pick.

    Rules:
    - Only use if P(Hit) < STREAK_SAVER_P_THRESHOLD (configured in config.py)
    - More aggressive saver usage in later phases (high streak value)
    - Never waste a saver in Phase 1 (streak < 11, easy to restart)
    """
    from config import STREAK_SAVER_P_THRESHOLD
    if savers_remaining <= 0:
        return False
    phase = get_phase(streak_len)
    if phase == "aggressive":   # Don't waste savers early
        return False
    threshold = STREAK_SAVER_P_THRESHOLD
    # In ultra-conservative phase, use saver more liberally
    if phase == "ultra_conservative":
        threshold = 0.65
    return current_p_hit < threshold
