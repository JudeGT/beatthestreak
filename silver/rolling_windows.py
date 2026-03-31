"""
Silver Layer: Rolling window feature engineering using DuckDB.

Computes 7/14/30/60/120-day rolling averages for:
  - GHP (Grand Hit Probability proxy): estimated_ba + (H / PA)
  - H/PA (hits per plate appearance)
  - xBA, xwOBA, barrel rate, launch speed, launch angle

All computations run directly in DuckDB for high throughput.
"""

import logging
import duckdb
import pandas as pd
from pathlib import Path

from config import DUCKDB_PATH, ROLLING_WINDOWS, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DUCKDB_PATH))


def build_pa_grain(con: duckdb.DuckDBPyConnection) -> None:
    """
    Create a plate-appearance grain table from raw Statcast pitch data.
    One row per (batter, game_date, at_bat_number).
    """
    log.info("Building plate-appearance grain table...")
    con.execute("""
        CREATE OR REPLACE TABLE pa_grain AS
        SELECT
            batter,
            CAST(game_date AS DATE)       AS game_date,
            at_bat_number,
            pitcher,
            home_team,
            away_team,
            stand,
            p_throws,
            if_fielding_alignment,
            -- Terminal event of the PA (last pitch row)
            LAST(events)                  AS pa_event,
            LAST(description)             AS pa_desc,
            -- Hit indicator (1 = single/double/triple/HR)
            CASE WHEN LAST(events) IN
                ('single','double','triple','home_run') THEN 1 ELSE 0
            END                           AS is_hit,
            -- PA indicator (AB + walks + HBP + sac flies)
            CASE WHEN LAST(events) NOT IN
                ('caught_stealing_2b','caught_stealing_3b',
                 'caught_stealing_home','pickoff_1b','pickoff_2b',
                 'pickoff_3b','runner_double_play') THEN 1 ELSE 0
            END                           AS is_pa,
            -- Plate discipline / quality metrics (last pitch values)
            MAX(launch_speed)             AS launch_speed,
            MAX(launch_angle)             AS launch_angle,
            MAX(estimated_ba_using_speedangle)   AS xba,
            MAX(estimated_woba_using_speedangle) AS xwoba,
            -- Statcast 'barrel' may be missing in some parquet exports; use a proxy
            MAX(CASE WHEN estimated_woba_using_speedangle >= 0.65 THEN 1 ELSE 0 END) AS barrel_flag,
            MAX(hit_distance_sc)          AS hit_distance
        FROM statcast_raw
        GROUP BY batter, CAST(game_date AS DATE), at_bat_number,
                 pitcher, home_team, away_team, stand, p_throws,
                 if_fielding_alignment
    """)
    log.info("pa_grain table created.")


def register_statcast_parquet(
    con: duckdb.DuckDBPyConnection,
    parquet_dir: Path,
) -> None:
    """Register all Statcast Parquet files as the `statcast_raw` view."""
    pattern = str(parquet_dir / "statcast_*.parquet")
    con.execute(f"""
        CREATE OR REPLACE VIEW statcast_raw AS
        SELECT * FROM read_parquet('{pattern}', union_by_name=true)
    """)
    log.info(f"Registered Statcast view from {pattern}")


def build_rolling_features(
    con: duckdb.DuckDBPyConnection,
    windows: list[int] = None,
) -> None:
    """
    Build rolling window averages for each batter over specified day windows.
    Creates table `batter_rolling` with one row per (batter, game_date).
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    log.info(f"Computing rolling windows: {windows} days...")

    # Aggregate to daily batter stats first
    con.execute("""
        CREATE OR REPLACE TABLE batter_daily AS
        SELECT
            batter,
            game_date,
            SUM(is_hit)           AS hits,
            SUM(is_pa)            AS pas,
            AVG(xba)              AS avg_xba,
            AVG(xwoba)            AS avg_xwoba,
            AVG(barrel_flag)      AS barrel_rate,
            AVG(launch_speed)     AS avg_exit_velo,
            AVG(launch_angle)     AS avg_launch_angle,
            home_team,
            away_team,
            stand,
        FROM pa_grain
        GROUP BY batter, game_date, home_team, away_team, stand
    """)

    # Build rolling SELECT clauses for each window
    window_selects = []
    for w in windows:
        window_selects.extend([
            f"""
            AVG(hits / NULLIF(pas, 0)) OVER (
                PARTITION BY batter
                ORDER BY game_date
                RANGE BETWEEN INTERVAL '{w} days' PRECEDING AND CURRENT ROW
            ) AS h_pa_roll_{w}d,
            AVG(avg_xba + COALESCE(hits / NULLIF(pas, 0), 0)) OVER (
                PARTITION BY batter
                ORDER BY game_date
                RANGE BETWEEN INTERVAL '{w} days' PRECEDING AND CURRENT ROW
            ) AS ghp_roll_{w}d,
            AVG(avg_xwoba) OVER (
                PARTITION BY batter
                ORDER BY game_date
                RANGE BETWEEN INTERVAL '{w} days' PRECEDING AND CURRENT ROW
            ) AS xwoba_roll_{w}d,
            AVG(barrel_rate) OVER (
                PARTITION BY batter
                ORDER BY game_date
                RANGE BETWEEN INTERVAL '{w} days' PRECEDING AND CURRENT ROW
            ) AS barrel_roll_{w}d,
            AVG(avg_exit_velo) OVER (
                PARTITION BY batter
                ORDER BY game_date
                RANGE BETWEEN INTERVAL '{w} days' PRECEDING AND CURRENT ROW
            ) AS exit_velo_roll_{w}d,
            """
        ])

    window_sql = "\n".join(window_selects)
    con.execute(f"""
        CREATE OR REPLACE TABLE batter_rolling AS
        SELECT
            batter,
            game_date,
            hits,
            pas,
            avg_xba,
            avg_xwoba,
            barrel_rate,
            avg_exit_velo,
            avg_launch_angle,
            home_team,
            away_team,
            stand,
            {window_sql}
        FROM batter_daily
        ORDER BY batter, game_date
    """)
    log.info("batter_rolling table created.")


def run_silver_rolling(parquet_dir: Path) -> None:
    """End-to-end: register Parquet → build PA grain → build rolling features."""
    con = _connect()
    register_statcast_parquet(con, parquet_dir)
    build_pa_grain(con)
    build_rolling_features(con)
    con.close()
    log.info("Silver rolling-window layer complete.")


if __name__ == "__main__":
    from config import BRONZE_DIR
    run_silver_rolling(BRONZE_DIR)
