"""
Silver Layer: Feature engineering — join batter rolling windows,
pitcher archetypes, and park/weather factors into a unified silver feature table.
"""

import logging
import duckdb
import pandas as pd
from pathlib import Path

from config import DUCKDB_PATH, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)


def build_silver_features(
    con: duckdb.DuckDBPyConnection,
    include_weather: bool = False,
    weather_df: pd.DataFrame | None = None,
) -> None:
    """
    Join batter_rolling + pitcher_archetypes into `silver_features`.

    Parameters
    ----------
    con : DuckDBPyConnection
        Active DuckDB connection (must already have batter_rolling,
        pa_grain, and pitcher_archetypes tables).
    include_weather : bool
        If True, expects `weather_df` to be provided and will join it.
    weather_df : pd.DataFrame, optional
        DataFrame with columns [home_team, temp_f, humidity_pct, pressure_mb].
    """
    log.info("Building silver_features table...")

    if include_weather and weather_df is not None:
        con.register("weather_df", weather_df)
        weather_join = """
            LEFT JOIN weather_df w
              ON br.home_team = w.home_team
        """
        weather_cols = """
            w.temp_f,
            w.humidity_pct,
            w.pressure_mb,
        """
    else:
        weather_join = ""
        weather_cols = """
            NULL::DOUBLE AS temp_f,
            NULL::DOUBLE AS humidity_pct,
            NULL::DOUBLE AS pressure_mb,
        """

    con.execute(f"""
        CREATE OR REPLACE TABLE silver_features AS
        SELECT
            -- Identity
            br.batter,
            br.game_date,
            br.home_team,
            br.away_team,
            br.stand,
            CAST(br.batter AS VARCHAR) AS batter_name,

            -- Daily outcome
            br.hits,
            br.pas,
            COALESCE(br.hits::DOUBLE / NULLIF(br.pas, 0), 0.0) AS h_pa_today,

            -- Rolling windows (GHP + H/PA)
            br.ghp_roll_7d,
            br.ghp_roll_14d,
            br.ghp_roll_30d,
            br.ghp_roll_60d,
            br.ghp_roll_120d,
            br.h_pa_roll_7d,
            br.h_pa_roll_14d,
            br.h_pa_roll_30d,
            br.h_pa_roll_60d,
            br.h_pa_roll_120d,
            br.xwoba_roll_7d,
            br.xwoba_roll_14d,
            br.xwoba_roll_30d,
            br.barrel_roll_7d,
            br.barrel_roll_30d,
            br.exit_velo_roll_7d,
            br.exit_velo_roll_30d,

            -- Opponent pitcher archetype (majority archetype for the game)
            pa.archetype_id      AS opp_pitcher_archetype,
            pa.archetype_label   AS opp_pitcher_archetype_label,
            pa.avg_release_speed AS opp_pitcher_velo,
            pa.avg_spin_rate     AS opp_pitcher_spin,
            pa.tunnel_consistency AS opp_pitcher_tunnel,

            -- Split: batter hand vs pitcher hand
            CASE WHEN br.stand = 'L' AND pg.p_throws = 'L' THEN 1 ELSE 0 END AS same_hand_matchup,

            -- Weather context
            {weather_cols}

            -- Hit label for training
            CASE WHEN br.hits > 0 THEN 1 ELSE 0 END AS hit_label,

        FROM batter_rolling br

        -- Join pitcher archetype for the game
        LEFT JOIN (
            SELECT
                pg_inner.game_date,
                pg_inner.home_team,
                pg_arch.archetype_id,
                pg_arch.archetype_label,
                pg_arch.avg_release_speed,
                pg_arch.avg_spin_rate,
                pg_arch.tunnel_consistency,
                pg_inner.p_throws,
            FROM (
                SELECT DISTINCT game_date, home_team, away_team, pitcher, p_throws
                FROM pa_grain
            ) pg_inner
            LEFT JOIN pitcher_archetypes pg_arch
              ON pg_inner.pitcher = pg_arch.pitcher
        ) pa
          ON br.game_date = pa.game_date
         AND (br.home_team = pa.home_team OR br.away_team = pa.home_team)

        -- Used to get p_throws for same-hand split
        LEFT JOIN (
            SELECT DISTINCT game_date, home_team, away_team, p_throws
            FROM pa_grain
        ) pg
          ON br.game_date = pg.game_date
         AND (br.home_team = pg.home_team OR br.away_team = pg.home_team)

        {weather_join}

        WHERE br.pas > 0
        ORDER BY br.batter, br.game_date
    """)

    count = con.execute("SELECT COUNT(*) FROM silver_features").fetchone()[0]
    log.info(f"silver_features built: {count:,} rows.")


def build_silver_daily(
    con: duckdb.DuckDBPyConnection,
    date_str: str,
    weather_df: pd.DataFrame | None = None,
) -> None:
    """
    Build 'silver_features_daily' for a future/today date using previous stats.
    
    This handles the case where Statcast (pa_grain) doesn't have today's data yet.
    """
    log.info(f"Building silver_features_daily for {date_str}...")

    # Ensure daily_lineups view exists
    lineup_path = Path("data/bronze") / f"daily_lineups_{date_str}.parquet"
    if not lineup_path.exists():
        log.warning(f"No daily lineups file found at {lineup_path}. Skipping silver_daily.")
        return

    con.execute(f"CREATE OR REPLACE VIEW daily_lineups_raw AS SELECT * FROM read_parquet('{lineup_path}')")

    if weather_df is not None:
        con.register("weather_df", weather_df)
        weather_join = "LEFT JOIN weather_df w ON dl.home_team = w.home_team"
        weather_cols = "w.temp_f, w.humidity_pct, w.pressure_mb"
    else:
        weather_join = ""
        weather_cols = "72.0 AS temp_f, 50.0 AS humidity_pct, 1013.0 AS pressure_mb"

    # Assemble the feature vector for today's lineups
    con.execute(f"""
        CREATE OR REPLACE TABLE silver_features_daily AS
        SELECT
            dl.batter,
            dl.game_date,
            dl.home_team,
            dl.away_team,
            dl.stand,
            COALESCE(dl.batter_name, CAST(br.batter AS VARCHAR)) AS batter_name,
            
            -- Dummy outcomes for today (not used for inference, but matches schema)
            0::BIGINT AS hits,
            0::BIGINT AS pas,
            0.0::DOUBLE AS h_pa_today,

            -- Most recent rolling windows (from batter_rolling)
            br.ghp_roll_7d, br.ghp_roll_14d, br.ghp_roll_30d, br.ghp_roll_60d, br.ghp_roll_120d,
            br.h_pa_roll_7d, br.h_pa_roll_14d, br.h_pa_roll_30d, br.h_pa_roll_60d, br.h_pa_roll_120d,
            br.xwoba_roll_7d, br.xwoba_roll_14d, br.xwoba_roll_30d,
            br.barrel_roll_7d, br.barrel_roll_30d,
            br.exit_velo_roll_7d, br.exit_velo_roll_30d,

            -- Opponent pitcher statistics (from pitcher_archetypes)
            pa.archetype_id      AS opp_pitcher_archetype,
            pa.archetype_label   AS opp_pitcher_archetype_label,
            pa.avg_release_speed AS opp_pitcher_velo,
            pa.avg_spin_rate     AS opp_pitcher_spin,
            pa.tunnel_consistency AS opp_pitcher_tunnel,

            -- Matchup: Left/Right split
            CASE WHEN dl.stand = 'L' AND pa_throws.p_throws = 'L' THEN 1 ELSE 0 END AS same_hand_matchup,

            {weather_cols},

            -- Hit label (None for inference)
            0::BIGINT AS hit_label

        FROM daily_lineups_raw dl
        
        -- Get most recent rolling row for this batter
        LEFT JOIN (
            SELECT batter, MAX(game_date) as last_date
            FROM batter_rolling
            WHERE game_date < '{date_str}'
            GROUP BY batter
        ) last_br ON dl.batter = last_br.batter
        LEFT JOIN batter_rolling br 
          ON dl.batter = br.batter AND last_br.last_date = br.game_date

        -- Get pitcher archetype (fixed metrics for the pitcher)
        LEFT JOIN pitcher_archetypes pa ON dl.pitcher = pa.pitcher

        -- Get pitcher throws for the same_hand context
        -- (Ideally statsapi would provide p_throws, but we can look it up or add it to ingest_daily)
        LEFT JOIN (
             SELECT DISTINCT pitcher, p_throws FROM pa_grain
        ) pa_throws ON dl.pitcher = pa_throws.pitcher

        {weather_join}
    """)
    
    count = con.execute("SELECT COUNT(*) FROM silver_features_daily").fetchone()[0]
    log.info(f"silver_features_daily built: {count:,} rows.")


def run_silver_features(
    weather_df: pd.DataFrame | None = None,
) -> None:
    con = duckdb.connect(str(DUCKDB_PATH))
    build_silver_features(
        con,
        include_weather=(weather_df is not None),
        weather_df=weather_df,
    )
    con.close()
    log.info("Silver feature engineering complete.")


if __name__ == "__main__":
    run_silver_features()
