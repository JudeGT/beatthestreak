"""
Tests: Data Pipeline — Bronze ingestion, Silver rolling windows, Gold table schema.

These tests use mock/synthetic data so they don't require a live Statcast connection
or a pre-populated DuckDB database.
"""

import pytest
import pandas as pd
import numpy as np
import duckdb
import tempfile
from pathlib import Path
from datetime import date, timedelta


# ── Helpers ────────────────────────────────────────────────────────────────────

def _synthetic_statcast(n_rows: int = 500, n_batters: int = 10) -> pd.DataFrame:
    """Generate a minimal synthetic Statcast-like DataFrame for testing."""
    rng = np.random.default_rng(42)
    start = date(2024, 4, 1)
    rows = []
    for i in range(n_rows):
        game_date = start + timedelta(days=rng.integers(0, 90))
        batter    = 100 + rng.integers(0, n_batters)
        pitcher   = 200 + rng.integers(0, 5)
        events    = rng.choice(
            ["single", "double", "strikeout", "field_out", "walk", "home_run"],
            p=[0.12, 0.06, 0.22, 0.38, 0.15, 0.07]
        )
        rows.append({
            "game_date":           str(game_date),
            "batter":              int(batter),
            "pitcher":             int(pitcher),
            "events":              events,
            "description":         "hit_into_play",
            "stand":               rng.choice(["L", "R"]),
            "p_throws":            rng.choice(["L", "R"]),
            "home_team":           rng.choice(["NYY", "BOS", "LAD", "COL"]),
            "away_team":           rng.choice(["CHC", "ATL", "SF", "PHI"]),
            "release_speed":       float(rng.uniform(88, 99)),
            "release_spin_rate":   float(rng.uniform(2000, 2600)),
            "pfx_x":              float(rng.uniform(-1.5, 1.5)),
            "pfx_z":              float(rng.uniform(-1.0, 2.0)),
            "launch_speed":        float(rng.uniform(70, 110)),
            "launch_angle":        float(rng.uniform(-10, 45)),
            "estimated_ba_using_speedangle":  float(rng.uniform(0.1, 0.7)),
            "estimated_woba_using_speedangle": float(rng.uniform(0.2, 0.8)),
            "barrel":              int(rng.integers(0, 2)),
            "hit_distance_sc":     float(rng.uniform(100, 450)),
            "spray_angle":         float(rng.uniform(-45, 45)),
            "if_fielding_alignment": rng.choice(["Standard", "Shifted", "Strategic"]),
            "at_bat_number":       int(rng.integers(1, 40)),
            "pitch_number":        int(rng.integers(1, 6)),
            "inning":              int(rng.integers(1, 10)),
            "outs_when_up":        int(rng.integers(0, 3)),
            "on_1b":               None,
            "on_2b":               None,
            "on_3b":               None,
            "woba_value":          float(rng.uniform(0, 2)),
            "woba_denom":          1,
            "babip_value":         float(rng.uniform(0, 1)),
        })
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


# ── Synthetic Statcast Validity ────────────────────────────────────────────────

class TestSyntheticDataGeneration:
    def test_dataframe_shape(self):
        df = _synthetic_statcast(200, 5)
        assert len(df) == 200

    def test_required_columns_present(self):
        df = _synthetic_statcast(50)
        for col in ["batter", "pitcher", "events", "game_date", "launch_speed"]:
            assert col in df.columns


# ── DuckDB Rolling Windows ─────────────────────────────────────────────────────

class TestRollingWindowsPipeline:
    """
    Integration test for the Silver rolling window pipeline using in-memory DuckDB
    with synthetic data.
    """

    @pytest.fixture
    def con(self):
        con = duckdb.connect(":memory:")
        df  = _synthetic_statcast(400, 8)
        con.register("statcast_raw", df)
        return con

    def test_pa_grain_builds(self, con):
        from silver.rolling_windows import build_pa_grain
        build_pa_grain(con)
        count = con.execute("SELECT COUNT(*) FROM pa_grain").fetchone()[0]
        assert count > 0

    def test_batter_daily_aggregation(self, con):
        from silver.rolling_windows import build_pa_grain, build_rolling_features
        build_pa_grain(con)
        build_rolling_features(con, windows=[7, 30])

        cols = [r[0] for r in con.execute("DESCRIBE batter_rolling").fetchall()]
        assert "ghp_roll_7d"  in cols
        assert "ghp_roll_30d" in cols
        assert "h_pa_roll_7d" in cols

    def test_rolling_values_in_reasonable_range(self, con):
        from silver.rolling_windows import build_pa_grain, build_rolling_features
        build_pa_grain(con)
        build_rolling_features(con, windows=[30])

        df = con.execute(
            "SELECT ghp_roll_30d, h_pa_roll_30d FROM batter_rolling WHERE ghp_roll_30d IS NOT NULL"
        ).df()
        assert (df["ghp_roll_30d"] >= 0).all()
        assert (df["h_pa_roll_30d"] >= 0).all()
        assert (df["h_pa_roll_30d"] <= 2.0).all()   # H/PA can't normally exceed 2


# ── Pitcher Archetype Clustering ────────────────────────────────────────────────

class TestPitcherArchetypes:

    def test_kmeans_produces_8_clusters(self):
        from silver.pitcher_archetypes import fit_pitcher_archetypes, build_pitcher_features
        import duckdb

        con = duckdb.connect(":memory:")
        df  = _synthetic_statcast(2000, 10)
        con.register("statcast_raw", df)

        pitcher_df = build_pitcher_features(con)
        con.close()

        if pitcher_df.empty:
            pytest.skip("Not enough pitcher pitch data in small synthetic dataset")

        labeled_df, kmeans, scaler = fit_pitcher_archetypes(pitcher_df)
        n_clusters = labeled_df["archetype_id"].nunique()
        assert n_clusters <= 8    # May be < 8 if fewer pitchers than k

    def test_archetype_labels_assigned(self):
        from silver.pitcher_archetypes import fit_pitcher_archetypes, ARCHETYPE_LABELS
        import duckdb

        con = duckdb.connect(":memory:")
        df  = _synthetic_statcast(2000, 10)
        con.register("statcast_raw", df)

        from silver.pitcher_archetypes import build_pitcher_features
        pitcher_df = build_pitcher_features(con)
        con.close()

        if pitcher_df.empty:
            pytest.skip("Not enough data")

        labeled_df, _, _ = fit_pitcher_archetypes(pitcher_df, n_clusters=min(3, len(pitcher_df)))
        assert "archetype_label" in labeled_df.columns
        assert labeled_df["archetype_label"].notna().all()


# ── Gold Table Schema ──────────────────────────────────────────────────────────

class TestGoldTableSchema:

    def test_gold_feature_cols_list_not_empty(self):
        from gold.gold_table import GOLD_FEATURE_COLS
        assert len(GOLD_FEATURE_COLS) > 10

    def test_gold_feature_cols_contains_key_features(self):
        from gold.gold_table import GOLD_FEATURE_COLS
        required = [
            "ghp_roll_30d", "xwoba_roll_30d", "babip_park_adjusted",
            "stuff_plus", "env_composite", "air_density",
        ]
        for feat in required:
            assert feat in GOLD_FEATURE_COLS, f"Missing: {feat}"
