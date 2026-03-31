"""
FastAPI Application: Project DiMaggio prediction API.

Endpoints:
  GET /health            — system health check
  GET /picks             — ranked daily picks with strategy applied
  GET /predict           — raw P(Hit) for a specific batter/date
  GET /explain           — SHAP-based natural-language explanation for a pick
"""

import logging
from pathlib import Path
from typing import Optional

import duckdb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    HealthResponse, PredictionResponse,
    PicksResponse, PickResult, ExplanationResponse, ShapFeature,
)
from models.predict import predict_hit_prob, rank_batters_for_date
from strategy.milestone_logic import select_picks, get_threshold, get_phase
from strategy.opener_detector import apply_opener_adjustment
from strategy.shift_recalibration import apply_shift_recalibration
from strategy.rl_agent import StreakDQNAgent, StreakState
from explainability.shap_explainer import shap_values_for_pick
from config import DUCKDB_PATH, MODEL_CHECKPOINT_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Project DiMaggio API",
    description="ML-powered MLB Beat the Streak pick engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── RL Agent singleton (loaded lazily) ────────────────────────────────────────
_rl_agent: Optional[StreakDQNAgent] = None

def get_rl_agent() -> StreakDQNAgent:
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = StreakDQNAgent(epsilon=0.0)  # Greedy mode for inference
        rl_path = MODEL_CHECKPOINT_DIR / "rl_agent.pt"
        if rl_path.exists():
            _rl_agent.load(rl_path)
        else:
            log.warning("No RL agent checkpoint found. Using untrained agent (recommend training).")
    return _rl_agent


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the model checkpoint and database are accessible."""
    model_loaded = (MODEL_CHECKPOINT_DIR / "htl_best.pt").exists()
    try:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        duckdb_ready = "gold_features" in tables
        con.close()
    except Exception:
        duckdb_ready = False

    return HealthResponse(status="ok", model_loaded=model_loaded, duckdb_ready=duckdb_ready)


# ── Predict ────────────────────────────────────────────────────────────────────

@app.get("/predict", response_model=PredictionResponse)
def predict(
    batter_id: int = Query(..., description="MLB batter MLBAM ID"),
    date:       str = Query(..., description="Game date YYYY-MM-DD"),
):
    """Return raw P(Hit) for a specific batter on a specific game date."""
    try:
        p = predict_hit_prob(batter_id, date)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # Fetch metadata from gold table
    try:
        con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        row = con.execute(
            "SELECT home_team, away_team, stand, env_composite "
            "FROM gold_features WHERE batter=? AND game_date=?",
            [batter_id, date]
        ).fetchone()
        con.close()
        home_team, away_team, stand, env_composite = (row or (None, None, None, None))
    except Exception:
        home_team = away_team = stand = env_composite = None

    return PredictionResponse(
        batter_id=batter_id,
        game_date=date,
        p_hit=p,
        home_team=home_team,
        away_team=away_team,
        stand=stand,
        env_composite=env_composite,
    )


# ── Picks ──────────────────────────────────────────────────────────────────────

@app.get("/picks", response_model=PicksResponse)
def picks(
    date:               str = Query(..., description="Game date YYYY-MM-DD"),
    streak_len:         int = Query(0,  description="Current streak length"),
    double_down_budget: int = Query(1,  description="Double Down uses remaining"),
    savers_remaining:   int = Query(1,  description="Streak Saver uses remaining"),
    min_prob:           float = Query(0.70, description="Minimum P(Hit) to consider"),
):
    """
    Return today's ranked picks with strategy (milestone thresholds, Double Down, 
    opener/shift adjustments) applied.
    """
    try:
        candidates = rank_batters_for_date(date, min_prob=min_prob)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if not candidates:
        return PicksResponse(
            game_date=date,
            streak_len=streak_len,
            threshold_used=get_threshold(streak_len),
            phase=get_phase(streak_len),
            picks=[],
        )

    # Apply strategic adjustments
    candidates = apply_opener_adjustment(candidates, date)
    candidates = apply_shift_recalibration(candidates, date)

    # Re-sort after adjustments
    candidates.sort(key=lambda c: c["p_hit"], reverse=True)

    # Milestone pick selection
    selected = select_picks(candidates, streak_len, double_down_budget)

    # RL recommendation
    rl_agent = get_rl_agent()
    rl_state  = StreakState(
        streak_length=streak_len,
        p_hit=candidates[0]["p_hit"] if candidates else 0.75,
        double_downs_remaining=double_down_budget,
        streak_savers_remaining=savers_remaining,
    )
    rl_rec = rl_agent.recommend_action(rl_state)

    return PicksResponse(
        game_date=date,
        streak_len=streak_len,
        threshold_used=get_threshold(streak_len),
        phase=get_phase(streak_len),
        picks=[PickResult(**p.to_dict()) for p in selected],
        rl_recommendation=rl_rec,
    )


# ── Explain ────────────────────────────────────────────────────────────────────

@app.get("/explain", response_model=ExplanationResponse)
def explain(
    batter_id: int = Query(..., description="MLB batter MLBAM ID"),
    date:       str = Query(..., description="Game date YYYY-MM-DD"),
    top_n:      int = Query(5,  description="Number of top SHAP features to return"),
):
    """Return a SHAP-based natural-language explanation for a pick."""
    try:
        result = shap_values_for_pick(batter_id, date, top_n=top_n)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SHAP error: {exc}")

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return ExplanationResponse(
        batter_id=result["batter_id"],
        game_date=result["game_date"],
        p_hit=result.get("p_hit"),
        top_features=[ShapFeature(**f) for f in result.get("top_features", [])],
        explanation_text=result.get("explanation_text", ""),
    )
