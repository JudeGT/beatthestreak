# Project DiMaggio — ⚾ MLB Beat the Streak Prediction Engine

> *"Hit. Every. Day."* — Targeting a 57-game streak with >80% per-pick accuracy.

---

## Architecture Overview

```
Bronze Layer  →  Silver Layer  →  Gold Layer  →  HTL Model  →  Strategy  →  API / CLI / Dashboard
(Statcast raw)   (DuckDB feats)   (Inference)    (LSTM+Attn)   (RL + Rules)  (FastAPI + Streamlit)
```

## Project Structure

```
BeatTheStreakV3/
├── bronze/                # Raw data ingestion
│   ├── ingest_statcast.py   # Statcast pitch data via pybaseball
│   ├── ingest_weather.py    # OpenWeatherMap API
│   └── ingest_hawkeye.py    # Hawk-Eye 3D skeletal interface (stub)
├── silver/                # DuckDB feature engineering
│   ├── rolling_windows.py   # 7/14/30/60/120-day GHP + H/PA rolling avgs
│   ├── pitcher_archetypes.py# K-Means (k=8) pitcher clustering
│   └── feature_engineering.py
├── gold/                  # Inference table
│   └── gold_table.py        # Stuff+, Squared-Up rate, park-adjusted BABIP
├── physics/               # Environment engine
│   ├── aerodynamics.py      # Air density ρ (barometric formula)
│   ├── flight_model.py      # Drag coefficient, +10°F → +3.5 ft model
│   ├── humidor.py           # COR adjustment (Coors/Chase/Globe Life)
│   └── park_factors.py      # All 30 MLB park factors + env composite
├── models/                # HTL Neural Network
│   ├── lstm_temporal.py     # Bidirectional LSTM over last 100 PAs
│   ├── transformer_attention.py  # Transformer over env/pitcher features
│   ├── htl_model.py         # Gated fusion model (BCEWithLogitsLoss)
│   ├── train.py             # Training loop (AdamW, cosine LR, AUC ckpt)
│   └── predict.py           # Inference: P(Hit) + batter ranking
├── strategy/              # Strategic logic
│   ├── milestone_logic.py   # 3-phase thresholds + pick selection
│   ├── rl_agent.py          # DQN agent (Double Down / Streak Saver)
│   ├── opener_detector.py   # Opener/bullpen game detector (+2.5%/+3.5%)
│   └── shift_recalibration.py  # LHH BABIP penalty for defensive shifts
├── explainability/
│   └── shap_explainer.py    # SHAP KernelExplainer + natural-language output
├── api/
│   ├── main.py              # FastAPI: /health /predict /picks /explain
│   └── schemas.py           # Pydantic request/response models
├── dashboard/
│   └── app.py               # Streamlit dashboard (gauges, SHAP waterfall)
├── tests/                 # pytest test suite
├── cli.py                 # Rich CLI entry point
├── config.py              # All constants, thresholds, hyperparameters
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OPENWEATHERMAP_API_KEY
```

### 3. Run the Data Pipeline (one season)
```bash
python cli.py pipeline --season 2024
```
This runs: Statcast ingestion → rolling windows → pitcher archetypes → feature engineering → gold table.

### 4. Train the HTL Model
```bash
python cli.py train
```
Trains on the gold table data. Checkpoint saved to `models/checkpoints/htl_best.pt`.

### 5. Get Daily Picks
```bash
# Console picks for a given date and current streak
python cli.py picks --date 2025-06-15 --streak 23

# JSON output
python cli.py picks --date 2025-06-15 --streak 23 --json
```

### 6. Get a Pick Explanation
```bash
python cli.py explain --batter 660271 --date 2025-06-15
```

### 7. Start the API Server
```bash
uvicorn api.main:app --reload
# → http://127.0.0.1:8000/docs
```

### 8. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
# (Requires API server to be running)
```

---

## Strategy Phases

| Streak | Phase | Min P(Hit) | Double Down | Max Picks/Day |
|--------|-------|-----------|-------------|---------------|
| 0–10   | Aggressive | 0.80 | Yes (≥0.85) | 2 |
| 11–40  | Opportunistic | 0.85 | No | 2 |
| 41–57  | Ultra-Conservative | 0.92 | No | 1 |

---

## Running Tests
```bash
pytest tests/ -v --tb=short
```

---

## API Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Model + DB health check |
| GET | `/predict?batter_id=…&date=…` | Raw P(Hit) for one batter |
| GET | `/picks?date=…&streak_len=…` | Ranked picks with strategy |
| GET | `/explain?batter_id=…&date=…` | SHAP explanation for a pick |

---

## Key Technologies

- **Data**: `pybaseball` (Statcast), `DuckDB` (SQL feature engineering), `pyarrow` (Parquet)
- **Physics**: Custom barometric formula + drag coefficient model + COR humidor model
- **ML**: `PyTorch` (HTL: Bidirectional LSTM + Transformer), `scikit-learn` (K-Means archetypes)
- **RL**: `DQN` (gymnasium) for Double Down / Streak Saver optimization
- **Explainability**: `shap` (KernelExplainer)
- **API**: `FastAPI` + `Pydantic`
- **Dashboard**: `Streamlit` + `Plotly`
- **CLI**: `Click` + `Rich`
