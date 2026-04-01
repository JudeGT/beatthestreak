"""
Gold Layer: Produce a flattened inference table for the HTL model.

Incorporates:
  - All Silver features (rolling windows, pitcher archetypes)
  - Stuff+ proxy metric
  - 3D bat-path Squared-Up rate (from Hawk-Eye or barrel% proxy)
  - Park-adjusted BABIP
  - Physics environment composite score
  - Pitcher archetype one-hot encoding
"""

import logging
import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

from config import DUCKDB_PATH, NUM_PITCHER_ARCHETYPES, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

# Gold feature columns (schema contract for the model)
GOLD_FEATURE_COLS = [
    # Rolling GHP
    "ghp_roll_7d", "ghp_roll_14d", "ghp_roll_30d", "ghp_roll_60d", "ghp_roll_120d",
    # Rolling H/PA
    "h_pa_roll_7d", "h_pa_roll_14d", "h_pa_roll_30d",
    # Quality metrics
    "xwoba_roll_7d", "xwoba_roll_14d", "xwoba_roll_30d",
    "barrel_roll_7d", "barrel_roll_30d",
    "exit_velo_roll_7d", "exit_velo_roll_30d",
    # Pitcher context
    "stuff_plus", "opp_pitcher_velo", "opp_pitcher_spin", "opp_pitcher_tunnel",
    # One-hot pitcher archetypes (0..7)
    *[f"archetype_{i}" for i in range(NUM_PITCHER_ARCHETYPES)],
    # Park & environment
    "babip_park_adjusted", "squared_up_rate", "env_composite",
    "air_density", "cor_adjustment", "humidity_pct",
    "temp_f", "pressure_mb",
    # Matchup context
    "same_hand_matchup",
    # Categorical (encoded)
    "stand_enc",    # 0=R, 1=L
]


def build_gold_table(
    con: duckdb.DuckDBPyConnection,
    park_factors: dict[str, float] | None = None,
    physics_df: pd.DataFrame | None = None,
    hawkeye_df: pd.DataFrame | None = None,
) -> None:
    """
    Produce the gold inference table from silver_features.

    Parameters
    ----------
    con : DuckDBPyConnection
    park_factors : dict [team -> park_factor float], optional
        Park factor per home team. Defaults to 1.0 if absent.
    physics_df : pd.DataFrame, optional
        Columns: [home_team, air_density, cor_adjustment, env_composite].
        Output from the physics engine for today's games.
    hawkeye_df : pd.DataFrame, optional
        Columns: [batter, game_date, squared_up_rate].
        From bronze/ingest_hawkeye.py. Zero-padded if None.
    """
    log.info("Building gold_features table...")

    # Register optional DataFrames
    if physics_df is not None:
        con.register("physics_df", physics_df)
        physics_join = """
            LEFT JOIN physics_df phys ON sf.home_team = phys.home_team
        """
        physics_cols = """
            COALESCE(phys.air_density, 1.225)      AS air_density,
            COALESCE(phys.cor_adjustment, 0.530)   AS cor_adjustment,
            COALESCE(phys.env_composite, 0.0)      AS env_composite,
        """
    else:
        physics_join = ""
        physics_cols = """
            1.225  AS air_density,
            0.530  AS cor_adjustment,
            0.0    AS env_composite,
        """

    if hawkeye_df is not None:
        con.register("hawkeye_df", hawkeye_df)
        hawkeye_join = """
            LEFT JOIN hawkeye_df he
              ON sf.batter = he.batter AND sf.game_date = he.game_date
        """
        hawkeye_col = "COALESCE(he.squared_up_rate, sf.barrel_roll_30d)"
    else:
        hawkeye_join = ""
        hawkeye_col = "sf.barrel_roll_30d"   # proxy when Hawk-Eye unavailable

    # Build park factor CASE expression
    if park_factors:
        pf_cases = "\n".join(
            f"WHEN sf.home_team = '{team}' THEN {pf:.4f}"
            for team, pf in park_factors.items()
        )
        park_factor_expr = f"CASE {pf_cases} ELSE 1.0 END"
    else:
        park_factor_expr = "1.0"

    # Archetype one-hot encoding
    archetype_onehot = "\n".join(
        f"CASE WHEN sf.opp_pitcher_archetype = {i} THEN 1 ELSE 0 END AS archetype_{i},"
        for i in range(NUM_PITCHER_ARCHETYPES)
    )

    # Check if silver_features_daily exists
    has_daily = con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name = 'silver_features_daily'").fetchone()[0] > 0
    
    if not has_daily:
        con.execute("CREATE TEMP TABLE silver_features_daily AS SELECT * FROM silver_features WHERE 1=0")

    con.execute(f"""
        CREATE OR REPLACE TABLE gold_features AS
        SELECT
            -- Identity (not model features, used for lookup)
            sf.batter,
            sf.game_date,
            sf.home_team,
            sf.away_team,
            sf.stand,
            sf.hit_label,

            -- Rolling features
            COALESCE(sf.ghp_roll_7d,    0.0) AS ghp_roll_7d,
            COALESCE(sf.ghp_roll_14d,   0.0) AS ghp_roll_14d,
            COALESCE(sf.ghp_roll_30d,   0.0) AS ghp_roll_30d,
            COALESCE(sf.ghp_roll_60d,   0.0) AS ghp_roll_60d,
            COALESCE(sf.ghp_roll_120d,  0.0) AS ghp_roll_120d,
            COALESCE(sf.h_pa_roll_7d,   0.0) AS h_pa_roll_7d,
            COALESCE(sf.h_pa_roll_14d,  0.0) AS h_pa_roll_14d,
            COALESCE(sf.h_pa_roll_30d,  0.0) AS h_pa_roll_30d,
            COALESCE(sf.xwoba_roll_7d,  0.0) AS xwoba_roll_7d,
            COALESCE(sf.xwoba_roll_14d, 0.0) AS xwoba_roll_14d,
            COALESCE(sf.xwoba_roll_30d, 0.0) AS xwoba_roll_30d,
            COALESCE(sf.barrel_roll_7d, 0.0) AS barrel_roll_7d,
            COALESCE(sf.barrel_roll_30d,0.0) AS barrel_roll_30d,
            COALESCE(sf.exit_velo_roll_7d,  0.0) AS exit_velo_roll_7d,
            COALESCE(sf.exit_velo_roll_30d, 0.0) AS exit_velo_roll_30d,

            -- Stuff+ proxy: normalized pitcher quality vs. league average
            -- (release_speed / 94.0 + spin_rate / 2400.0) * 100 / 2 → centered ~100
            ROUND(
                ((COALESCE(sf.opp_pitcher_velo, 94.0) / 94.0)
               + (COALESCE(sf.opp_pitcher_spin, 2400.0) / 2400.0)) * 50.0
            , 1) AS stuff_plus,

            COALESCE(sf.opp_pitcher_velo,    94.0) AS opp_pitcher_velo,
            COALESCE(sf.opp_pitcher_spin,  2400.0) AS opp_pitcher_spin,
            COALESCE(sf.opp_pitcher_tunnel,   3.0) AS opp_pitcher_tunnel,
            COALESCE(sf.opp_pitcher_archetype, -1) AS opp_pitcher_archetype,

            -- One-hot archetype
            {archetype_onehot}

            -- Park-adjusted BABIP: raw xBA * park_factor
            ROUND(
                COALESCE(sf.xwoba_roll_30d, 0.300) * {park_factor_expr}
            , 4) AS babip_park_adjusted,

            -- Squared-up rate (Hawk-Eye if available, else barrel proxy)
            COALESCE({hawkeye_col}, 0.0) AS squared_up_rate,

            -- Physics / environment
            {physics_cols}

            -- Weather context
            COALESCE(sf.temp_f,        72.0) AS temp_f,
            COALESCE(sf.pressure_mb, 1013.0) AS pressure_mb,
            COALESCE(sf.humidity_pct,  50.0) AS humidity_pct,

            -- Matchup
            COALESCE(sf.same_hand_matchup, 0) AS same_hand_matchup,
            CASE WHEN sf.stand = 'L' THEN 1 ELSE 0 END AS stand_enc,

        FROM (
            SELECT * FROM silver_features
            UNION ALL
            SELECT * FROM silver_features_daily WHERE 1=1 -- Only if exists
        ) sf
        {hawkeye_join}
        {physics_join}
        ORDER BY sf.batter, sf.game_date
    """)

    count = con.execute("SELECT COUNT(*) FROM gold_features").fetchone()[0]
    log.info(f"gold_features built: {count:,} rows.")


def load_gold_table(
    con: duckdb.DuckDBPyConnection,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load gold_features into X, y arrays ready for model training/inference.

    Returns
    -------
    X : pd.DataFrame — feature matrix
    y : pd.Series   — binary hit label
    """
    if feature_cols is None:
        feature_cols = GOLD_FEATURE_COLS

    df = con.execute("SELECT * FROM gold_features").df()

    available_features = [c for c in feature_cols if c in df.columns]
    X = df[available_features].astype(np.float32)
    y = df["hit_label"].astype(np.float32)
    return X, y


def run_gold_table(
    park_factors: dict[str, float] | None = None,
    physics_df: pd.DataFrame | None = None,
    hawkeye_df: pd.DataFrame | None = None,
) -> None:
    con = duckdb.connect(str(DUCKDB_PATH))
    build_gold_table(con, park_factors=park_factors, physics_df=physics_df, hawkeye_df=hawkeye_df)
    con.close()
    log.info("Gold layer complete.")


if __name__ == "__main__":
    run_gold_table()
