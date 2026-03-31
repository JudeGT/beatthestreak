"""
Strategic Logic: Defensive Shift BABIP Recalibration for Left-Handed Hitters.

When an extreme defensive shift is applied against a left-handed hitter (LHH),
the probability of reaching on a ground ball to the right side drops significantly.
This module detects shift usage from the `if_fielding_alignment` Statcast field
and applies a BABIP penalty to the predicted P(Hit).

MLB banned the traditional 4-man shift in 2023 (ABS rule 5.02(c)).
We still track "shade" alignments which provide a smaller BABIP penalty.
"""

import logging
import duckdb
from config import DUCKDB_PATH, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

# BABIP penalty factors (multiplicative) by alignment type for LHH
# Source: Statcast shift analysis; 4-man abolished in 2023 so we track shade
SHIFT_BABIP_PENALTIES: dict[str, float] = {
    "4-man shift":          -0.045,   # Legacy (pre-2023): -4.5 pp
    "Shifted":              -0.030,   # Catch-all shifted alignment: -3.0 pp
    "Strategic":            -0.015,   # Shade / strategic positioning: -1.5 pp
    "Standard":              0.000,   # No adjustment
    "Unknown":               0.000,
}


def get_shift_usage(
    batter_id: int,
    game_date: str,
    con: duckdb.DuckDBPyConnection,
) -> str:
    """
    Determine the most common defensive alignment faced by a batter on game_date.
    Uses historical recent games (last 14 days) to establish tendency.

    Returns
    -------
    str
        Alignment category string matching SHIFT_BABIP_PENALTIES keys.
    """
    result = con.execute("""
        SELECT if_fielding_alignment, COUNT(*) AS cnt
        FROM statcast_raw
        WHERE batter = ?
          AND game_date BETWEEN DATE_TRUNC('day', CAST(? AS DATE) - INTERVAL '14 days')
                             AND CAST(? AS DATE)
          AND if_fielding_alignment IS NOT NULL
        GROUP BY if_fielding_alignment
        ORDER BY cnt DESC
        LIMIT 1
    """, [batter_id, game_date, game_date]).fetchone()

    if result:
        alignment = result[0]
        # Normalise to our penalty categories
        alignment_lower = alignment.lower()
        if "4-man" in alignment_lower or "extreme" in alignment_lower:
            return "4-man shift"
        elif "shift" in alignment_lower:
            return "Shifted"
        elif "strategic" in alignment_lower or "shade" in alignment_lower:
            return "Strategic"
        else:
            return "Standard"
    return "Unknown"


def apply_shift_recalibration(
    candidates: list[dict],
    game_date: str,
) -> list[dict]:
    """
    Apply defensive shift BABIP recalibration to left-handed hitter candidates.

    Only LHH (stand == 'L') are affected. Right-handed hitters and switch hitters
    are not recalibrated by this module.

    Parameters
    ----------
    candidates : list[dict]
        Candidate list with keys: batter_id, stand, p_hit, explanation.
    game_date : str
        Game date YYYY-MM-DD.

    Returns
    -------
    list[dict]
        Candidates with adjusted p_hit and explanation for LHH facing shifts.
    """
    lhh_candidates = [c for c in candidates if c.get("stand") == "L"]
    if not lhh_candidates:
        return candidates

    con = duckdb.connect(str(DUCKDB_PATH))
    try:
        for c in lhh_candidates:
            try:
                alignment = get_shift_usage(c["batter_id"], game_date, con)
                penalty   = SHIFT_BABIP_PENALTIES.get(alignment, 0.0)

                if penalty != 0.0:
                    old_p = c["p_hit"]
                    c["p_hit"] = max(0.0, c["p_hit"] + penalty)
                    c["shift_alignment"] = alignment
                    c["explanation"] = (
                        c.get("explanation", "")
                        + f" | LHH shift penalty ({alignment}: {penalty:+.1%})"
                    )
                    log.debug(
                        f"batter={c['batter_id']} LHH {alignment}: "
                        f"P={old_p:.3f} → {c['p_hit']:.3f}"
                    )
                else:
                    c["shift_alignment"] = alignment
            except Exception as exc:
                log.debug(f"Shift lookup failed for batter {c['batter_id']}: {exc}")
    finally:
        con.close()

    return candidates
