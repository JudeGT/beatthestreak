"""
Project DiMaggio — Central Configuration
All runtime constants, thresholds, and environment loading in one place.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
DUCKDB_PATH = Path(os.getenv("DUCKDB_PATH", str(DATA_DIR / "dimaggio.duckdb")))
MODEL_CHECKPOINT_DIR = ROOT_DIR / os.getenv("MODEL_CHECKPOINT_DIR", "models/checkpoints")

for _d in [BRONZE_DIR, SILVER_DIR, GOLD_DIR, MODEL_CHECKPOINT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API Keys ──────────────────────────────────────────────────────────────────
OPENWEATHERMAP_API_KEY: str = os.getenv("OPENWEATHERMAP_API_KEY", "")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# ── Milestone Thresholds ──────────────────────────────────────────────────────
# Minimum P(Hit) required to make a pick / use Double Down
MILESTONE_PHASES = {
    "aggressive":     (0,  10,  0.80),   # (streak_min, streak_max, p_threshold)
    "opportunistic":  (11, 40,  0.85),
    "ultra_conservative": (41, 57, 0.92),
}

def get_threshold(streak_len: int) -> float:
    """Return P(Hit) threshold for the current streak phase."""
    for phase, (lo, hi, thresh) in MILESTONE_PHASES.items():
        if lo <= streak_len <= hi:
            return thresh
    return 0.92  # beyond 57 — still apply max caution


# ── Double Down / Streak Saver Config ─────────────────────────────────────────
DOUBLE_DOWN_PHASE_0_10 = True         # Aggressive doubling in early streak
MAX_DAILY_PICKS = 2                   # Max concurrent picks (Beat the Streak rules)
STREAK_SAVER_P_THRESHOLD = 0.50      # Only use saver if P(Hit) < this

# ── Stadium Data ──────────────────────────────────────────────────────────────
# altitude_ft: feet above sea level for air density calculation
STADIUM_ALTITUDE_FT: dict[str, float] = {
    "Coors Field":              5200.0,
    "Chase Field":               1100.0,
    "Globe Life Field":           551.0,
    "Dodger Stadium":             395.0,
    "Oracle Park":                 10.0,
    "Fenway Park":                 20.0,
    "Yankee Stadium":              55.0,
    "Wrigley Field":              595.0,
    "Truist Park":                1050.0,
    "American Family Field":      635.0,
    "T-Mobile Park":               17.0,
    "PNC Park":                   730.0,
    "Great American Ball Park":   487.0,
    "Kauffman Stadium":           750.0,
    "Target Field":               815.0,
    "Guaranteed Rate Field":      595.0,
    "Progressive Field":          650.0,
    "Comerica Park":              600.0,
    "Busch Stadium":              465.0,
    "Minute Maid Park":            22.0,
    "Angel Stadium":               160.0,
    "Petco Park":                  16.0,
    "loanDepot park":               6.0,
    "Nationals Park":              25.0,
    "Camden Yards":                30.0,
    "Citizens Bank Park":          20.0,
    "Citi Field":                  20.0,
    "Rogers Centre":              300.0,
    "Oakland Coliseum":            12.0,
    "Tropicana Field":             43.0,
}

# ── Humidor Stadiums ──────────────────────────────────────────────────────────
# Stadiums with active humidity/humidor systems (affects COR of baseball)
HUMIDOR_STADIUMS: set[str] = {
    "Coors Field",
    "Chase Field",
    "Globe Life Field",
}

# Baseline relative humidity targets (%) when humidor is active
HUMIDOR_RH_TARGET: dict[str, float] = {
    "Coors Field":    50.0,
    "Chase Field":    50.0,
    "Globe Life Field": 50.0,
}

# ── Neural Network Hyperparameters ────────────────────────────────────────────
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_SEQUENCE_LEN = 50          # last N plate appearances
TRANSFORMER_D_MODEL = 128
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2
DROPOUT = 0.3

# ── Training Hyperparameters ──────────────────────────────────────────────────
TRAIN_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
TRAIN_VAL_TEST_SPLIT = (0.80, 0.10, 0.10)
SUBSAMPLE_RATIO = 1  # Use 10% of data for faster training (set to 1.0 for full dataset)

# ── Pitcher Archetypes ────────────────────────────────────────────────────────
NUM_PITCHER_ARCHETYPES = 8

# ── Rolling Window Sizes (days) ───────────────────────────────────────────────
ROLLING_WINDOWS = [7, 14, 30, 60, 120]

# ── Aerodynamics Constants ────────────────────────────────────────────────────
# Standard sea-level air density (kg/m³) at 59°F / 15°C, 1013.25 mb
RHO_SEA_LEVEL = 1.225
# Lapse rate (K/m) for barometric formula approximation
TEMP_LAPSE_RATE = 0.0065
R_AIR = 287.058          # specific gas constant for dry air (J/(kg·K))
GRAVITY = 9.80665        # m/s²
# Distance bonus per unit decrease in air density (calibrated empirically)
# +10°F ≈ -2% ρ ≈ +3.5 ft for a 400-ft fly ball
DENSITY_TO_DISTANCE_COEFF = 1600.0   # ft·m³/kg

# ── Coefficient of Restitution ─────────────────────────────────────────────────
# COR of a dry MLB baseball at standard humidity
COR_DRY = 0.530
# COR of a humidified ball (Coors pre-humidor era reference)
COR_WET = 0.510
# Linear approximation: COR drops by this much per 1% RH above baseline
COR_RH_SLOPE = -0.0002
