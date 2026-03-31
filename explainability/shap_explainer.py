"""
Explainability: SHAP-based per-pick explanations.

Uses SHAP KernelExplainer (model-agnostic) or a background-distribution
approximation to compute feature importances for each pick. Converts the
top SHAP contributors into natural-language explanation sentences.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import duckdb

from gold.gold_table import GOLD_FEATURE_COLS
from config import DUCKDB_PATH, MODEL_CHECKPOINT_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    log.warning("shap not installed. Run `pip install shap`.")

# Human-readable feature descriptions for SHAP explanation sentences
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "ghp_roll_7d":          "hot streak over the last 7 days (GHP)",
    "ghp_roll_14d":         "elevated GHP over the last 2 weeks",
    "ghp_roll_30d":         "strong 30-day hitting average",
    "ghp_roll_60d":         "consistent form over 2 months",
    "ghp_roll_120d":        "elite season-long hitting baseline",
    "h_pa_roll_7d":         "high H/PA rate last 7 days",
    "h_pa_roll_30d":        "strong H/PA rate last month",
    "xwoba_roll_7d":        "excellent expected wOBA last week",
    "xwoba_roll_30d":       "high expected wOBA last month",
    "barrel_roll_7d":       "high barrel rate last 7 days",
    "barrel_roll_30d":      "strong barrel contact rate last month",
    "exit_velo_roll_7d":    "elite exit velocity this week",
    "exit_velo_roll_30d":   "high exit velocity last month",
    "stuff_plus":           "pitcher Stuff+ (high = tougher pitcher)",
    "opp_pitcher_velo":     "opponent pitcher velocity",
    "opp_pitcher_spin":     "opponent pitcher spin rate",
    "opp_pitcher_tunnel":   "pitcher tunnel consistency (low = more deceptive)",
    "babip_park_adjusted":  "park-adjusted BABIP projection",
    "squared_up_rate":      "squared-up bat contact rate",
    "env_composite":        "hitter-friendly environment composite score",
    "air_density":          "air density (low = ball carries further)",
    "cor_adjustment":       "ball bounciness (COR) adjustment",
    "humidity_pct":         "ambient humidity",
    "temp_f":               "game-time temperature",
    "pressure_mb":          "barometric pressure",
    "same_hand_matchup":    "same-handedness pitcher-batter matchup",
    "stand_enc":            "batter batting side (L/R)",
}


def _load_background_data(n_samples: int = 100) -> np.ndarray:
    """Load a background sample from the gold table for SHAP."""
    con = duckdb.connect(str(DUCKDB_PATH))
    available = [c for c in GOLD_FEATURE_COLS]
    col_str   = ", ".join(available)
    df = con.execute(
        f"SELECT {col_str} FROM gold_features ORDER BY RANDOM() LIMIT {n_samples}"
    ).df()
    con.close()
    return df[available].fillna(0).values.astype(np.float32)


def shap_values_for_pick(
    batter_id: int,
    game_date: str,
    checkpoint_path: Optional[Path] = None,
    n_background: int = 50,
    top_n: int = 5,
) -> dict:
    """
    Compute SHAP feature importances for a specific (batter, game_date) prediction.

    Uses SHAP KernelExplainer with a background dataset from the gold table.
    Note: KernelExplainer can be slow; for production, consider TreeExplainer
    if a gradient-boosted surrogate is available.

    Parameters
    ----------
    batter_id : int
    game_date : str
    checkpoint_path : Path, optional
    n_background : int
        Number of background samples for SHAP baseline.
    top_n : int
        Number of top features to return.

    Returns
    -------
    dict with keys:
        batter_id, game_date, p_hit, top_features, explanation_text
    """
    if not SHAP_AVAILABLE:
        return {
            "batter_id": batter_id,
            "game_date": game_date,
            "p_hit": None,
            "top_features": [],
            "explanation_text": "SHAP not available. Install with: pip install shap",
        }

    from models.predict import load_model, build_env_vector
    import torch

    model, device, _ = load_model(checkpoint_path)

    con = duckdb.connect(str(DUCKDB_PATH))
    env_vec = build_env_vector(batter_id, game_date, con)

    # Background data
    available = [c for c in GOLD_FEATURE_COLS]
    bg_df = con.execute(
        f"SELECT {', '.join(available)} FROM gold_features "
        f"ORDER BY RANDOM() LIMIT {n_background}"
    ).df()
    con.close()

    if env_vec is None:
        return {"batter_id": batter_id, "game_date": game_date, "error": "Not in gold table"}

    background = bg_df[available].fillna(0).values.astype(np.float32)

    # Define prediction function that takes env features only (SHAP interface)
    # PA sequence is zero-padded for SHAP analysis (env features drive most variation)
    def predict_fn(env_arr: np.ndarray) -> np.ndarray:
        n = len(env_arr)
        pa_seq = torch.zeros(n, 100, 8, dtype=torch.float32).to(device)   # zero-pad
        env_t  = torch.tensor(env_arr, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = model.predict_prob(pa_seq, env_t).cpu().numpy()
        return probs

    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_vals   = explainer.shap_values(env_vec.reshape(1, -1), nsamples=100)
    shap_arr    = np.array(shap_vals).flatten()

    # Match to feature names
    feature_names = [c for c in GOLD_FEATURE_COLS if c in available]
    shap_df = pd.DataFrame({
        "feature":    feature_names[:len(shap_arr)],
        "shap_value": shap_arr[:len(feature_names)],
    })
    shap_df["abs_shap"] = shap_df["shap_value"].abs()
    top = shap_df.nlargest(top_n, "abs_shap")

    # Build natural-language explanation
    explanation_parts = []
    for _, row in top.iterrows():
        feat   = row["feature"]
        val    = row["shap_value"]
        desc   = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " "))
        direction = "boosted" if val > 0 else "reduced"
        explanation_parts.append(
            f"{desc.capitalize()} {direction} P(Hit) by {abs(val):.1%}"
        )

    p_hit = float(predict_fn(env_vec.reshape(1, -1))[0])
    explanation_text = "; ".join(explanation_parts)

    return {
        "batter_id":        batter_id,
        "game_date":        game_date,
        "p_hit":            round(p_hit, 4),
        "top_features": [
            {
                "feature":     r["feature"],
                "shap_value":  round(r["shap_value"], 4),
                "description": FEATURE_DESCRIPTIONS.get(r["feature"], r["feature"]),
            }
            for _, r in top.iterrows()
        ],
        "explanation_text": explanation_text,
    }
