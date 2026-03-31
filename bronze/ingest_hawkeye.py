"""
Bronze Layer: Hawk-Eye 3D skeletal coordinate interface.

Hawk-Eye data (joint positions, bat-path kinematics) is an MLB commercial
product. This module defines the schema and ingestion interface expected
by the Silver feature engineering layer. When data is unavailable, all
functions return None and the pipeline zero-pads the biomechanical
features during training.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config import BRONZE_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)


# Expected columns when Hawk-Eye data IS available
HAWKEYE_SCHEMA = [
    "game_pk",
    "at_bat_number",
    "batter",
    "game_date",
    # 3D wrist positions (meters, right-handed coordinate system)
    "wrist_lead_x", "wrist_lead_y", "wrist_lead_z",
    "wrist_rear_x",  "wrist_rear_y",  "wrist_rear_z",
    # Hip–shoulder separation angle (degrees)
    "hip_shoulder_sep_deg",
    # Head stability metric (standard deviation of head position mid-swing)
    "head_stability_sd",
    # Bat-path metrics
    "squared_up_flag",       # 1 if bat/ball contact ≥ 80% efficiency
    "attack_angle_deg",
    "bat_speed_mph",
]


def load_hawkeye_csv(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a Hawk-Eye CSV export and validate its schema.

    Parameters
    ----------
    path : Path
        Path to the Hawk-Eye CSV file.

    Returns
    -------
    pd.DataFrame or None
        Parsed DataFrame conforming to HAWKEYE_SCHEMA, or None if unavailable.
    """
    if not path.exists():
        log.warning(f"Hawk-Eye file not found: {path}. Returning None.")
        return None

    try:
        df = pd.read_csv(path)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        missing = [c for c in HAWKEYE_SCHEMA if c not in df.columns]
        if missing:
            log.warning(f"Hawk-Eye file missing columns: {missing}. Returning None.")
            return None
        df["game_date"] = pd.to_datetime(df["game_date"])
        log.info(f"Loaded {len(df):,} Hawk-Eye rows from {path}")
        return df[HAWKEYE_SCHEMA]
    except Exception as exc:
        log.error(f"Failed to parse Hawk-Eye CSV {path}: {exc}")
        return None


def get_hawkeye_features(
    batter_id: int,
    game_date: str,
    bronze_dir: Path = BRONZE_DIR,
) -> Optional[dict]:
    """
    Retrieve Hawk-Eye features for a specific batter on a specific game date.

    Returns None when data is unavailable. The model will zero-pad.
    """
    pattern = f"hawkeye_{game_date[:4]}*.csv"
    files = sorted(bronze_dir.glob(pattern))
    if not files:
        log.debug("No Hawk-Eye files found. Returning None.")
        return None

    frames = []
    for f in files:
        df = load_hawkeye_csv(f)
        if df is not None:
            frames.append(df)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    mask = (combined["batter"] == batter_id) & (
        combined["game_date"].dt.strftime("%Y-%m-%d") == game_date
    )
    sub = combined[mask]
    if sub.empty:
        return None

    # Aggregate across plate appearances for the game
    numeric_cols = [
        c for c in HAWKEYE_SCHEMA
        if c not in ("game_pk", "at_bat_number", "batter", "game_date", "squared_up_flag")
    ]
    row = sub[numeric_cols].mean().to_dict()
    row["squared_up_rate"] = float(sub["squared_up_flag"].mean())
    return row
