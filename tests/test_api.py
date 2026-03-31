"""
Tests: FastAPI Endpoints — /health, /picks, /predict, /explain.

These tests use httpx TestClient and mock the underlying model/DB calls
so they run without a trained checkpoint or DuckDB data.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Patch heavy imports before loading the API
with patch("models.predict.load_model", return_value=(MagicMock(), "cpu", {})), \
     patch("strategy.rl_agent.StreakDQNAgent"):
    from api.main import app

client = TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_has_status_ok(self):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_expected_keys(self):
        data = client.get("/health").json()
        assert "model_loaded" in data
        assert "duckdb_ready" in data


# ── /predict ───────────────────────────────────────────────────────────────────

class TestPredict:

    @patch("api.main.predict_hit_prob", return_value=0.8750)
    @patch("api.main.duckdb.connect")
    def test_predict_returns_200(self, mock_db, mock_predict):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = ("NYY", "BOS", "R", 0.12)
        mock_db.return_value.__enter__ = lambda s: mock_con
        mock_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_db.return_value = mock_con

        resp = client.get("/predict", params={"batter_id": 660271, "date": "2025-06-15"})
        assert resp.status_code == 200

    @patch("api.main.predict_hit_prob", return_value=0.8750)
    @patch("api.main.duckdb.connect")
    def test_predict_response_schema(self, mock_db, mock_predict):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = None
        mock_db.return_value = mock_con

        resp = client.get("/predict", params={"batter_id": 660271, "date": "2025-06-15"})
        data = resp.json()
        assert "p_hit" in data
        assert "batter_id" in data
        assert data["p_hit"] == pytest.approx(0.875, abs=0.001)

    @patch("api.main.predict_hit_prob", side_effect=ValueError("Not found"))
    def test_predict_404_on_missing_batter(self, mock_predict):
        resp = client.get("/predict", params={"batter_id": 9999999, "date": "2025-06-15"})
        assert resp.status_code == 404

    @patch("api.main.predict_hit_prob", side_effect=FileNotFoundError("No checkpoint"))
    def test_predict_503_on_missing_model(self, mock_predict):
        resp = client.get("/predict", params={"batter_id": 660271, "date": "2025-06-15"})
        assert resp.status_code == 503


# ── /picks ─────────────────────────────────────────────────────────────────────

class TestPicks:

    @patch("api.main.rank_batters_for_date", return_value=[])
    def test_picks_empty_when_no_candidates(self, mock_rank):
        resp = client.get("/picks", params={"date": "2025-06-15", "streak_len": 0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["picks"] == []

    @patch("api.main.rank_batters_for_date", return_value=[
        {"batter_id": 100, "game_date": "2025-06-15", "p_hit": 0.91,
         "home_team": "NYY", "away_team": "BOS", "stand": "R"},
        {"batter_id": 200, "game_date": "2025-06-15", "p_hit": 0.87,
         "home_team": "LAD", "away_team": "SF",  "stand": "L"},
    ])
    @patch("api.main.apply_opener_adjustment", side_effect=lambda c, d: c)
    @patch("api.main.apply_shift_recalibration", side_effect=lambda c, d: c)
    def test_picks_returns_qualified_picks(self, mock_shift, mock_opener, mock_rank):
        resp = client.get("/picks", params={"date": "2025-06-15", "streak_len": 5})
        assert resp.status_code == 200
        data = resp.json()
        # Both candidates have p_hit >= 0.80 (aggressive threshold)
        assert len(data["picks"]) >= 1

    @patch("api.main.rank_batters_for_date", return_value=[
        {"batter_id": 100, "game_date": "2025-06-15", "p_hit": 0.91,
         "home_team": "NYY", "away_team": "BOS", "stand": "R"},
    ])
    @patch("api.main.apply_opener_adjustment", side_effect=lambda c, d: c)
    @patch("api.main.apply_shift_recalibration", side_effect=lambda c, d: c)
    def test_picks_response_schema(self, mock_shift, mock_opener, mock_rank):
        resp = client.get("/picks", params={"date": "2025-06-15", "streak_len": 0})
        data = resp.json()
        assert "picks" in data
        assert "phase" in data
        assert "threshold_used" in data
        assert "streak_len" in data
        assert data["streak_len"] == 0

    @patch("api.main.rank_batters_for_date", side_effect=FileNotFoundError("no model"))
    def test_picks_503_no_model(self, mock_rank):
        resp = client.get("/picks", params={"date": "2025-06-15", "streak_len": 0})
        assert resp.status_code == 503

    def test_picks_missing_date_returns_422(self):
        resp = client.get("/picks", params={"streak_len": 0})
        assert resp.status_code == 422


# ── /explain ───────────────────────────────────────────────────────────────────

class TestExplain:

    @patch("api.main.shap_values_for_pick", return_value={
        "batter_id": 660271,
        "game_date": "2025-06-15",
        "p_hit": 0.88,
        "top_features": [
            {"feature": "ghp_roll_7d", "shap_value": 0.04,
             "description": "hot streak over the last 7 days (GHP)"},
        ],
        "explanation_text": "Hot streak over the last 7 days (GHP) boosted P(Hit) by 4.0%",
    })
    def test_explain_returns_200(self, mock_shap):
        resp = client.get("/explain", params={"batter_id": 660271, "date": "2025-06-15"})
        assert resp.status_code == 200

    @patch("api.main.shap_values_for_pick", return_value={
        "batter_id": 660271,
        "game_date": "2025-06-15",
        "p_hit": 0.88,
        "top_features": [
            {"feature": "ghp_roll_7d", "shap_value": 0.04,
             "description": "hot streak over the last 7 days (GHP)"},
        ],
        "explanation_text": "Hot streak test",
    })
    def test_explain_response_schema(self, mock_shap):
        resp = client.get("/explain", params={"batter_id": 660271, "date": "2025-06-15"})
        data = resp.json()
        assert "explanation_text" in data
        assert "top_features" in data
        assert "p_hit" in data

    @patch("api.main.shap_values_for_pick", return_value={"error": "Not in gold table"})
    def test_explain_404_not_in_table(self, mock_shap):
        resp = client.get("/explain", params={"batter_id": 9999, "date": "2025-06-15"})
        assert resp.status_code == 404
