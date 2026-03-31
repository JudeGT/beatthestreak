"""
Bronze Layer: Statcast pitch-level data ingestion via pybaseball.
Saves raw data as Parquet files to the bronze data directory.
"""

import logging
from pathlib import Path
from datetime import date, timedelta

import pandas as pd
import pybaseball

from config import BRONZE_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

# Suppress pybaseball's verbose output
pybaseball.cache.enable()


def ingest_statcast(
    start_date: str,
    end_date: str,
    team: str | None = None,
    output_dir: Path = BRONZE_DIR,
) -> Path:
    """
    Pull pitch-level Statcast data from Baseball Savant via pybaseball.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    team : str, optional
        3-letter team abbreviation to filter (e.g. 'NYY'). None = all teams.
    output_dir : Path
        Directory to save the resulting Parquet file.

    Returns
    -------
    Path
        Path to the saved Parquet file.
    """
    log.info(f"Ingesting Statcast data: {start_date} → {end_date}, team={team}")

    try:
        if team:
            df = pybaseball.statcast(start_dt=start_date, end_dt=end_date, team=team)
        else:
            df = pybaseball.statcast(start_dt=start_date, end_dt=end_date)
    except Exception as exc:
        log.error(f"Statcast ingestion failed: {exc}")
        raise

    if df is None or df.empty:
        log.warning("No Statcast data returned for the specified date range.")
        raise ValueError(f"Empty Statcast result for {start_date}–{end_date}")

    log.info(f"Fetched {len(df):,} pitch rows.")

    # Standardise column names to snake_case
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Keep only the columns we need for feature engineering
    KEEP_COLS = [
        "game_date", "batter", "pitcher", "events", "description",
        "stand", "p_throws", "home_team", "away_team",
        "release_speed", "release_spin_rate",
        "pfx_x", "pfx_z",
        "plate_x", "plate_z",
        "launch_speed", "launch_angle",
        "estimated_ba_using_speedangle",   # xBA
        "estimated_woba_using_speedangle", # xwOBA
        "woba_value", "woba_denom",
        "babip_value",
        "barrel",
        "hit_distance_sc",
        "spray_angle",
        "if_fielding_alignment",           # shift detection
        "at_bat_number", "pitch_number",
        "outs_when_up", "inning",
        "on_1b", "on_2b", "on_3b",        # baserunner context
    ]
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()

    # Save
    out_path = output_dir / f"statcast_{start_date}_{end_date}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info(f"Saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def ingest_statcast_season(
    season: int,
    output_dir: Path = BRONZE_DIR,
    chunk_days: int = 14,
) -> list[Path]:
    """
    Ingest a full season of Statcast data in bi-weekly chunks (avoids timeouts).

    Returns
    -------
    list[Path]
        List of Parquet file paths written.
    """
    paths = []
    start = date(season, 4, 1)     # Season typically starts early April
    end = date(season, 10, 1)      # Ends start of October (pre-playoffs)
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end)
        try:
            path = ingest_statcast(
                str(current), str(chunk_end), output_dir=output_dir
            )
            paths.append(path)
        except Exception as exc:
            log.warning(f"Chunk {current}→{chunk_end} failed: {exc}. Skipping.")
        current = chunk_end + timedelta(days=1)

    log.info(f"Season {season} ingestion complete. {len(paths)} chunks written.")
    return paths


def load_statcast_bronze(output_dir: Path = BRONZE_DIR) -> pd.DataFrame:
    """
    Load all Parquet files in the bronze directory into a single DataFrame.
    """
    parquet_files = sorted(output_dir.glob("statcast_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Statcast Parquet files found in {output_dir}")

    frames = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(frames, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df.sort_values("game_date", inplace=True)
    log.info(f"Loaded {len(df):,} rows from {len(parquet_files)} Parquet files.")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Statcast data to Bronze layer")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--team", default=None, help="Team abbreviation (optional)")
    parser.add_argument("--season", type=int, default=None, help="Full season year (overrides --start/--end)")
    args = parser.parse_args()

    if args.season:
        ingest_statcast_season(args.season)
    else:
        ingest_statcast(args.start, args.end, team=args.team)
