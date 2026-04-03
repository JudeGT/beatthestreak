"""
HTL Inference Pipeline.

Loads the best model checkpoint and produces P(Hit) for a given
(batter_id, game_date) using:
  - Gold table features for the environment/pitcher context
  - PA sequence built from the batter's recent plate appearance history
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import duckdb

from models.htl_model import HTLModel
from models.train import PA_SEQ_FEATURES, N_PA_FEATURES, load_training_data
from gold.gold_table import GOLD_FEATURE_COLS
from config import DUCKDB_PATH, MODEL_CHECKPOINT_DIR, LSTM_SEQUENCE_LEN, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

_model_cache: dict = {}   # cache loaded model to avoid re-loading


def load_model(
    checkpoint_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[HTLModel, torch.device, dict]:
    """
    Load the best checkpoint. Caches in-process to avoid repeated disk reads.

    Returns
    -------
    (model, device, checkpoint_meta)
    """
    global _model_cache

    if checkpoint_path is None:
        checkpoint_path = MODEL_CHECKPOINT_DIR / "htl_best.pt"

    ckpt_key = str(checkpoint_path)
    if ckpt_key in _model_cache:
        return _model_cache[ckpt_key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. "
            "Run `python models/train.py` to train the model first."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_pa    = ckpt.get("n_pa_features", N_PA_FEATURES)
    n_env   = ckpt.get("n_env_features", len(GOLD_FEATURE_COLS))

    model = HTLModel(n_pa_features=n_pa, n_env_features=n_env).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    log.info(
        f"Loaded checkpoint from epoch {ckpt.get('epoch')} | "
        f"Val AUC={ckpt.get('val_auc', '?'):.4f}"
    )
    _model_cache[ckpt_key] = (model, device, ckpt)
    return model, device, ckpt


def build_env_vector(
    batter: int,
    game_date: str,
    con: duckdb.DuckDBPyConnection,
) -> Optional[np.ndarray]:
    """
    Fetch the Gold table row for (batter, game_date) as a feature vector.
    """
    available = [c for c in GOLD_FEATURE_COLS]
    col_str = ", ".join(c for c in available)
    row = con.execute(
        f"SELECT {col_str} FROM gold_features WHERE batter=? AND game_date=?",
        [batter, game_date]
    ).fetchone()

    if row is None:
        log.warning(f"No gold_features row for batter={batter}, date={game_date}")
        return None
    return np.array(row, dtype=np.float32)


def build_pa_sequence(
    batter: int,
    game_date: str,
    con: duckdb.DuckDBPyConnection,
    seq_len: int = LSTM_SEQUENCE_LEN,
) -> np.ndarray:
    """
    Build the PA sequence for a batter: last `seq_len` PAs before game_date.
    Zero-padded if fewer than seq_len PAs available.
    """
    df = con.execute(f"""
        SELECT
            COALESCE(launch_speed, 90.0)     AS exit_velo,
            COALESCE(launch_angle, 15.0)     AS launch_angle,
            COALESCE(xba, 0.250)             AS xba,
            COALESCE(xwoba, 0.320)           AS xwoba,
            COALESCE(barrel_flag, 0)         AS barrel_flag,
            0.0                              AS pitch_type_enc,
            CASE WHEN p_throws='L' THEN 1.0 ELSE 0.0 END AS pitcher_hand_enc,
            is_hit::DOUBLE                   AS result_enc,
        FROM pa_grain
        WHERE batter = ?
          AND game_date < ?
        ORDER BY game_date DESC, at_bat_number DESC
        LIMIT {seq_len}
    """, [batter, game_date]).df()

    seq = np.zeros((seq_len, N_PA_FEATURES), dtype=np.float32)
    if not df.empty:
        vals = df[PA_SEQ_FEATURES].fillna(0).values.astype(np.float32)
        n = min(len(vals), seq_len)
        seq[-n:] = vals[:n][::-1]   # right-align, most recent last
    return seq


def predict_hit_prob(
    batter_id: int,
    game_date: str,
    checkpoint_path: Optional[Path] = None,
) -> float:
    """
    Predict P(Hit) for a specific batter on a specific game date.

    Parameters
    ----------
    batter_id : int
        MLB batter MLBAM ID.
    game_date : str
        Game date in 'YYYY-MM-DD' format.
    checkpoint_path : Path, optional
        Override for model checkpoint path.

    Returns
    -------
    float
        P(Hit) in [0, 1]. Higher = more likely to get a hit.

    Raises
    ------
    FileNotFoundError
        If no checkpoint exists (model not trained yet).
    ValueError
        If the batter/date combination is not in the gold table.
    """
    model, device, _ = load_model(checkpoint_path)
    con = duckdb.connect(str(DUCKDB_PATH))

    env_vec = build_env_vector(batter_id, game_date, con)
    if env_vec is None:
        con.close()
        raise ValueError(f"Batter {batter_id} not found in gold_features for {game_date}")

    pa_seq = build_pa_sequence(batter_id, game_date, con)
    con.close()

    # To tensors
    env_t = torch.tensor(env_vec, dtype=torch.float32).unsqueeze(0).to(device)
    pa_t  = torch.tensor(pa_seq,  dtype=torch.float32).unsqueeze(0).to(device)

    prob = model.predict_prob(pa_t, env_t).item()
    log.debug(f"batter={batter_id} date={game_date} P(Hit)={prob:.4f}")
    return round(prob, 4)


def rank_batters_for_date(
    game_date: str,
    min_prob: float = 0.70,
    checkpoint_path: Optional[Path] = None,
) -> list[dict]:
    """
    Rank all batters playing on game_date by P(Hit), filtered by min_prob.

    Returns
    -------
    list[dict] sorted by P(Hit) descending. Each dict has:
        batter_id, game_date, p_hit, home_team, away_team, stand, batter_name
    """
    con = duckdb.connect(str(DUCKDB_PATH))
    candidates = con.execute("""
        SELECT DISTINCT batter, batter_name, home_team, away_team, stand
        FROM gold_features
        WHERE game_date = ?
    """, [game_date]).df()
    con.close()

    if candidates.empty:
        log.warning(f"No gold_features data for {game_date}")
        return []

    results = []
    for _, row in candidates.iterrows():
        try:
            p = predict_hit_prob(int(row["batter"]), game_date, checkpoint_path)
            if p >= min_prob:
                results.append({
                    "batter_id":   int(row["batter"]),
                    "batter_name": str(row["batter_name"]),
                    "game_date":   game_date,
                    "p_hit":       p,
                    "home_team":   row["home_team"],
                    "away_team":   row["away_team"],
                    "stand":       row["stand"],
                })
        except Exception as exc:
            log.debug(f"Skipping batter {row['batter']}: {exc}")

    results.sort(key=lambda r: r["p_hit"], reverse=True)
    return results
