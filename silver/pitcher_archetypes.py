"""
Silver Layer: Pitcher Archetype Clustering (K-Means, k=8).

Groups pitchers into 8 archetypes based on:
  - Average release speed (velocity)
  - Average spin rate
  - Horizontal movement (pfx_x)
  - Vertical movement (pfx_z)
  - Tunnel consistency (standard deviation of pfx_x + pfx_z across pitches)

Archetype labels are written back to DuckDB as `pitcher_archetypes`.
"""

import logging
import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pathlib import Path

from config import DUCKDB_PATH, NUM_PITCHER_ARCHETYPES, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

ARCHETYPE_FEATURES = [
    "avg_release_speed",
    "avg_spin_rate",
    "avg_pfx_x",
    "avg_pfx_z",
    "tunnel_consistency",   # lower = more consistent = more deceptive
]

ARCHETYPE_LABELS: dict[int, str] = {
    0: "Power Flamethrower",
    1: "Sweeping Slider Specialist",
    2: "Soft-Tossing Groundballer",
    3: "High-Spin Curveballer",
    4: "Command Finesse Artist",
    5: "Two-Seam Sinker Heavy",
    6: "Mixed-Arsenal Deceptor",
    7: "Four-Seam/Change Tunnel Artist",
}


def build_pitcher_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Aggregate Statcast pitch-level data to one row per pitcher with
    archetype-defining features.
    """
    log.info("Aggregating pitcher features from statcast_raw...")
    df = con.execute("""
        SELECT
            pitcher,
            COUNT(*)                      AS total_pitches,
            AVG(release_speed)            AS avg_release_speed,
            AVG(release_spin_rate)        AS avg_spin_rate,
            AVG(pfx_x)                   AS avg_pfx_x,
            AVG(pfx_z)                   AS avg_pfx_z,
            -- Tunnel consistency: lower stddev = more consistent release tunnel
            STDDEV(pfx_x) + STDDEV(pfx_z) AS tunnel_consistency,
            -- Extra context (not used for clustering, stored for reference)
            AVG(plate_x)                 AS avg_plate_x,
            AVG(plate_z)                 AS avg_plate_z,
        FROM statcast_raw
        WHERE release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
        GROUP BY pitcher
        HAVING COUNT(*) >= 50     -- Minimum pitch sample threshold
    """).df()
    log.info(f"Pitcher feature matrix: {df.shape[0]} pitchers x {df.shape[1]} cols")
    return df


def fit_pitcher_archetypes(
    pitcher_df: pd.DataFrame,
    n_clusters: int = NUM_PITCHER_ARCHETYPES,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Run K-Means clustering on pitcher features.

    Returns
    -------
    pitcher_df : pd.DataFrame
        Original DataFrame with added `archetype_id` and `archetype_label` columns.
    kmeans : KMeans
        Fitted K-Means model (for inference on new pitchers).
    scaler : StandardScaler
        Fitted scaler (saved alongside the model for inference).
    """
    X = pitcher_df[ARCHETYPE_FEATURES].dropna()
    pitcher_df = pitcher_df.loc[X.index].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log.info(f"Fitting K-Means with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X_scaled)

    pitcher_df["archetype_id"]    = labels
    pitcher_df["archetype_label"] = [ARCHETYPE_LABELS.get(l, f"Archetype-{l}") for l in labels]

    # Log cluster summary
    for aid, aname in ARCHETYPE_LABELS.items():
        n = (labels == aid).sum()
        log.info(f"  [{aid}] {aname}: {n} pitchers")

    return pitcher_df, kmeans, scaler


def predict_archetype(
    pitcher_id: int,
    con: duckdb.DuckDBPyConnection,
    kmeans: KMeans,
    scaler: StandardScaler,
) -> dict:
    """
    Look up a pitcher's archetype from the stored table, or predict on the fly.

    Returns
    -------
    dict with keys: archetype_id, archetype_label
    """
    row = con.execute(
        "SELECT archetype_id, archetype_label FROM pitcher_archetypes WHERE pitcher = ?",
        [pitcher_id]
    ).fetchone()

    if row:
        return {"archetype_id": row[0], "archetype_label": row[1]}

    # Pitcher not in lookup — predict from current pitch data
    df = con.execute("""
        SELECT
            AVG(release_speed)            AS avg_release_speed,
            AVG(release_spin_rate)        AS avg_spin_rate,
            AVG(pfx_x)                   AS avg_pfx_x,
            AVG(pfx_z)                   AS avg_pfx_z,
            STDDEV(pfx_x) + STDDEV(pfx_z) AS tunnel_consistency
        FROM statcast_raw
        WHERE pitcher = ?
    """, [pitcher_id]).df()

    if df.empty or df[ARCHETYPE_FEATURES].isna().all(axis=None):
        return {"archetype_id": -1, "archetype_label": "Unknown"}

    X_scaled = scaler.transform(df[ARCHETYPE_FEATURES].fillna(0))
    aid = int(kmeans.predict(X_scaled)[0])
    return {"archetype_id": aid, "archetype_label": ARCHETYPE_LABELS.get(aid, f"Archetype-{aid}")}


def run_pitcher_archetypes() -> tuple[KMeans, StandardScaler]:
    """
    End-to-end: build pitcher features → fit K-Means → persist to DuckDB.

    Returns
    -------
    (KMeans, StandardScaler)  — for saving and inference use.
    """
    con = duckdb.connect(str(DUCKDB_PATH))
    pitcher_df = build_pitcher_features(con)
    labeled_df, kmeans, scaler = fit_pitcher_archetypes(pitcher_df)

    # Write labeled table to DuckDB
    con.execute("DROP TABLE IF EXISTS pitcher_archetypes")
    con.execute("CREATE TABLE pitcher_archetypes AS SELECT * FROM labeled_df")
    log.info("pitcher_archetypes table written to DuckDB.")
    con.close()
    return kmeans, scaler


if __name__ == "__main__":
    kmeans, scaler = run_pitcher_archetypes()
    print("Pitcher archetype clustering complete.")
