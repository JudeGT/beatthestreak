"""
API: Pydantic request/response schemas for the FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    batter_id: int = Field(..., description="MLB batter MLBAM ID")
    game_date: str = Field(..., description="Game date YYYY-MM-DD", example="2025-06-15")


class PredictionResponse(BaseModel):
    batter_id: int
    game_date: str
    p_hit: float = Field(..., description="P(Hit) in [0,1]")
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    stand: Optional[str] = None
    env_composite: Optional[float] = None


class PickResult(BaseModel):
    batter_id: int
    game_date: str
    p_hit: float
    home_team: Optional[str] = None
    away_team: Optional[str] = None
    stand: Optional[str] = None
    double_down: bool = False
    explanation: Optional[str] = None


class PicksResponse(BaseModel):
    game_date: str
    streak_len: int
    threshold_used: float
    phase: str
    picks: list[PickResult]
    rl_recommendation: Optional[dict] = None


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    description: str


class ExplanationResponse(BaseModel):
    batter_id: int
    game_date: str
    p_hit: Optional[float] = None
    top_features: list[ShapFeature] = []
    explanation_text: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    duckdb_ready: bool
