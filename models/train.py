"""
HTL Training Loop.

Trains the HTLModel with:
  - BCEWithLogitsLoss (Bernoulli loss for binary hit prediction)
  - AdamW optimizer with cosine LR schedule
  - Early stopping with patience
  - Stratified 80/10/10 train/val/test split
  - Checkpoint saving of best validation AUC
"""

from typing import Optional
import logging
import json
from pathlib import Path

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import duckdb
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from models.htl_model import HTLModel
from gold.gold_table import GOLD_FEATURE_COLS
from config import (
    DUCKDB_PATH, MODEL_CHECKPOINT_DIR,
    TRAIN_EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, TRAIN_VAL_TEST_SPLIT, SUBSAMPLE_RATIO,
    LSTM_SEQUENCE_LEN, LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

# ── PA Sequence feature columns (ordered, from pa_grain) ─────────────────────
PA_SEQ_FEATURES = [
    "exit_velo", "launch_angle", "xba", "xwoba",
    "barrel_flag", "pitch_type_enc", "pitcher_hand_enc", "result_enc",
]
N_PA_FEATURES = len(PA_SEQ_FEATURES)


class BatterDataset(Dataset):
    """
    PyTorch Dataset for HTL training.

    Each sample contains:
      - pa_seq   : (seq_len, n_pa_features)  — last 100 PAs
      - env_feat : (n_env_features,)         — Gold table row features
      - label    : float                     — 1.0 = hit, 0.0 = no hit
    """

    def __init__(
        self,
        env_features: np.ndarray,   # shape (N, n_env_features)
        pa_sequences: np.ndarray,   # shape (N, seq_len, n_pa_features)
        labels: np.ndarray,         # shape (N,)
    ):
        self.env_features = torch.tensor(env_features, dtype=torch.float32)
        self.pa_sequences  = torch.tensor(pa_sequences,  dtype=torch.float32)
        self.labels        = torch.tensor(labels,        dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pa_sequences[idx], self.env_features[idx], self.labels[idx]


def load_training_data(duckdb_path: Path = DUCKDB_PATH):
    """
    Load gold features and PA sequences from DuckDB.

    Returns
    -------
    env_features : np.ndarray (N, n_env_features)
    pa_sequences : np.ndarray (N, seq_len, n_pa_features)
    labels       : np.ndarray (N,)
    """
    con = duckdb.connect(str(duckdb_path))

    # Load gold table (build if missing)
    log.info("Loading gold_features from DuckDB...")
    try:
        gold_df = con.execute("SELECT * FROM gold_features ORDER BY batter, game_date").df()
    except duckdb.CatalogException as exc:
        log.warning("gold_features table missing in DuckDB — attempting to build gold table now...")
        from gold.gold_table import run_gold_table
        try:
            # Attempt to build gold table (this requires Silver tables to exist)
            run_gold_table()
            gold_df = con.execute("SELECT * FROM gold_features ORDER BY batter, game_date").df()
        except duckdb.CatalogException as exc2:
            raise RuntimeError(
                "Failed to build gold_features: upstream tables (e.g. silver_features) are missing. "
                "Run the data pipeline first: python cli.py pipeline --season <YEAR>"
            ) from exc2
    available_features = [c for c in GOLD_FEATURE_COLS if c in gold_df.columns]
    env_features = gold_df[available_features].fillna(0).values.astype(np.float32)
    labels       = gold_df["hit_label"].values.astype(np.float32)
    batters      = gold_df["batter"].values
    dates        = gold_df["game_date"].values

    log.info(f"Gold table: {len(env_features):,} samples, {env_features.shape[1]} env features")

    # Build PA sequences from pa_grain table
    log.info("Building PA sequences from pa_grain...")
    pa_grain = con.execute("""
        SELECT batter, game_date,
               COALESCE(launch_speed, 90.0)     AS exit_velo,
               COALESCE(launch_angle, 15.0)     AS launch_angle,
               COALESCE(xba, 0.250)             AS xba,
               COALESCE(xwoba, 0.320)           AS xwoba,
               COALESCE(barrel_flag, 0)         AS barrel_flag,
               0.0                              AS pitch_type_enc,
               CASE WHEN p_throws='L' THEN 1.0 ELSE 0.0 END AS pitcher_hand_enc,
               is_hit::DOUBLE                   AS result_enc
        FROM pa_grain
        ORDER BY batter, game_date, at_bat_number
    """).df()
    con.close()

    # Build per-sample PA sequences (last LSTM_SEQUENCE_LEN PAs before game_date)
    pa_sequences = np.zeros(
        (len(env_features), LSTM_SEQUENCE_LEN, N_PA_FEATURES), dtype=np.float32
    )

    batter_pa = pa_grain.groupby("batter")
    for i, (batter, game_date) in enumerate(zip(batters, dates)):
        try:
            group = batter_pa.get_group(int(batter))
            past_pas = group[group["game_date"] < game_date].tail(LSTM_SEQUENCE_LEN)
            n = len(past_pas)
            if n > 0:
                seq = past_pas[PA_SEQ_FEATURES].fillna(0).values.astype(np.float32)
                pa_sequences[i, -n:, :] = seq   # right-align (most recent = last)
        except KeyError:
            pass   # No PA history: sequence stays zero-padded

    log.info(f"PA sequences built: shape {pa_sequences.shape}")
    return env_features, pa_sequences, labels


def train_model(
    checkpoint_dir: Path = MODEL_CHECKPOINT_DIR,
    n_env_features: Optional[int] = None,
) -> Path:
    """
    Full training pipeline: load data → split → train → save best checkpoint.

    Returns
    -------
    Path to the saved best model checkpoint.
    """
    env_features, pa_sequences, labels = load_training_data()
    N = len(labels)

    # Subsample for faster training (if SUBSAMPLE_RATIO < 1.0)
    if SUBSAMPLE_RATIO < 1.0:
        subsample_size = int(N * SUBSAMPLE_RATIO)
        indices = np.random.choice(N, size=subsample_size, replace=False, random_state=42)
        env_features = env_features[indices]
        pa_sequences = pa_sequences[indices]
        labels = labels[indices]
        N = subsample_size
        log.info(f"Subsampled to {N:,} examples ({SUBSAMPLE_RATIO:.0%} of original)")

    if n_env_features is None:
        n_env_features = env_features.shape[1]

    # ── Stratified split ────────────────────────────────────────────────────
    train_ratio, val_ratio, test_ratio = TRAIN_VAL_TEST_SPLIT
    idx = np.arange(N)
    train_idx, tmp_idx = train_test_split(
        idx, test_size=(1 - train_ratio), stratify=labels, random_state=42
    )
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=(1 - val_ratio_adjusted),
        stratify=labels[tmp_idx], random_state=42
    )

    # ── Scaler ─────────────────────────────────────────────────────────────
    # Fit scalers only on training data
    env_scaler = StandardScaler()
    seq_scaler = StandardScaler() # Flatten for sequence scaling

    # Env Features
    env_features[train_idx] = env_scaler.fit_transform(env_features[train_idx])
    env_features[val_idx]   = env_scaler.transform(env_features[val_idx])
    env_features[test_idx]  = env_scaler.transform(env_features[test_idx])

    # Sequence Features (reshape to (N*T, D) to scale across all time steps)
    T, D = pa_sequences.shape[1], pa_sequences.shape[2]
    train_seq_flat = pa_sequences[train_idx].reshape(-1, D)
    val_seq_flat   = pa_sequences[val_idx].reshape(-1, D)
    test_seq_flat  = pa_sequences[test_idx].reshape(-1, D)

    seq_scaler.fit(train_seq_flat)
    pa_sequences[train_idx] = seq_scaler.transform(train_seq_flat).reshape(-1, T, D)
    pa_sequences[val_idx]   = seq_scaler.transform(val_seq_flat).reshape(-1, T, D)
    pa_sequences[test_idx]  = seq_scaler.transform(test_seq_flat).reshape(-1, T, D)

    log.info(f"Features normalized: env mean={env_scaler.mean_[0]:.2f} seq std={seq_scaler.scale_[0]:.2f}")

    log.info(f"Split: train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")

    def make_dataset(idx_arr):
        return BatterDataset(
            env_features[idx_arr],
            pa_sequences[idx_arr],
            labels[idx_arr],
        )

    # Weighted sampler to handle class imbalance (~28% hit rate)
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels.astype(int))
    weights = 1.0 / class_counts[train_labels.astype(int)]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    # Detect device early to configure data loading appropriately
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_pin_memory = device.type == "cuda"  # Only pin memory if using GPU

    # Multi-core data loading (reduce num_workers if memory issues occur)
    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(make_dataset(train_idx), batch_size=BATCH_SIZE, sampler=sampler, num_workers=num_workers, pin_memory=use_pin_memory)
    val_loader   = DataLoader(make_dataset(val_idx),   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)
    test_loader  = DataLoader(make_dataset(test_idx),  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=use_pin_memory)

    # ── Model & Optimizer ───────────────────────────────────────────────────
    # Configure multi-threading for CPU training
    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)
    
    log.info(f"Training on: {device}")
    log.info(f"CPU threads: {num_threads}, Data workers: {num_workers}")

    model = HTLModel(
        n_pa_features=N_PA_FEATURES,
        n_env_features=n_env_features,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    # ── Training Loop ────────────────────────────────────────────────────────
    best_val_auc = 0.0
    patience_counter = 0
    checkpoint_path = checkpoint_dir / "htl_best.pt"
    metrics_path    = checkpoint_dir / "training_metrics.json"
    history = []

    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        train_losses = []
        for pa_seq, env_feat, y in tqdm(train_loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS} Train"):
            pa_seq, env_feat, y = pa_seq.to(device), env_feat.to(device), y.to(device)
            optimizer.zero_grad()
            logit = model(pa_seq, env_feat)
            loss  = criterion(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_losses, val_probs, val_targets = [], [], []
        with torch.no_grad():
            for pa_seq, env_feat, y in tqdm(val_loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS} Val"):
                pa_seq, env_feat, y = pa_seq.to(device), env_feat.to(device), y.to(device)
                logit = model(pa_seq, env_feat)
                val_losses.append(criterion(logit, y).item())
                val_probs.extend(torch.sigmoid(logit).cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_auc = roc_auc_score(val_targets, val_probs) if SKLEARN_AVAILABLE else 0.0
        val_ap  = average_precision_score(val_targets, val_probs) if SKLEARN_AVAILABLE else 0.0

        log.info(
            f"Epoch {epoch:3d}/{TRAIN_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Val AP: {val_ap:.4f}"
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 6),
            "val_loss":   round(avg_val_loss, 6),
            "val_auc":    round(val_auc, 6),
            "val_ap":     round(val_ap, 6),
        }
        history.append(epoch_metrics)

        # Checkpoint on best AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "n_pa_features": N_PA_FEATURES,
                "n_env_features": n_env_features,
                "env_scaler_mean": env_scaler.mean_.tolist(),
                "env_scaler_scale": env_scaler.scale_.tolist(),
                "seq_scaler_mean": seq_scaler.mean_.tolist(),
                "seq_scaler_scale": seq_scaler.scale_.tolist(),
            }, checkpoint_path)
            log.info(f"  ✓ New best checkpoint saved (AUC={val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                log.info(f"Early stopping at epoch {epoch} (patience={EARLY_STOPPING_PATIENCE})")
                break

    # Save training history
    with open(metrics_path, "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Training complete. Best val AUC: {best_val_auc:.4f}")
    log.info(f"Checkpoint: {checkpoint_path}")
    return checkpoint_path


if __name__ == "__main__":
    train_model()
