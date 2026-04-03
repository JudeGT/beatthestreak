"""
Backtest Script: Evaluate HTL Model accuracy on historical data.
Takes past games, builds inference features, and compares predictions to actual outcomes.
"""

import logging
import duckdb
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

from config import DUCKDB_PATH, LOG_LEVEL
from models.predict import predict_hit_prob

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

def run_backtest(n_samples: int = 1000):
    con = duckdb.connect(str(DUCKDB_PATH))
    
    # Check if batter_name exists (to handle tables made before the update)
    cols = con.execute("DESCRIBE gold_features").df()['column_name'].tolist()
    name_col = "batter_name" if "batter_name" in cols else "CAST(batter AS VARCHAR) AS batter_name"

    # Fetch random historical samples from the gold table
    log.info(f"Fetching {n_samples} historical samples for backtesting...")
    test_samples = con.execute(f"""
        SELECT batter, {name_col}, game_date, hit_label
        FROM gold_features
        WHERE hit_label IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, [n_samples]).df()
    con.close()
    
    if test_samples.empty:
        print("Error: No historical data found in gold_features. Run the pipeline first.")
        return

    results = []
    
    print(f"\n🚀 Running backtest on {len(test_samples)} samples...\n")
    
    for _, row in tqdm(test_samples.iterrows(), total=len(test_samples)):
        batter_id = int(row['batter'])
        game_date = row['game_date']
        actual = int(row['hit_label'])
        
        try:
            # Predict
            p_hit = predict_hit_prob(batter_id, game_date)
            
            results.append({
                'batter': row['batter_name'],
                'date': game_date,
                'p_hit': p_hit,
                'actual': actual,
                'correct': 1 if (p_hit >= 0.80 and actual == 1) or (p_hit < 0.80 and actual == 0) else 0
            })
        except Exception as exc:
            if n_samples <= 10: # Only print if small batch to avoid spam
                print(f"Prediction failed for batter {batter_id} on {game_date}: {exc}")
            log.debug(f"Failed to predict for {batter_id} on {game_date}: {exc}")
            continue

    if not results:
        print("No predictions were successful.")
        return

    df = pd.DataFrame(results)
    
    # Metrics
    y_true = df['actual']
    y_prob = df['p_hit']
    y_pred = (y_prob >= 0.75).astype(int) # Threshold of 0.75 for "Hit" prediction
    
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    
    print("\n" + "="*40)
    print("       BACKTEST RESULTS       ")
    print("="*40)
    print(f"Total Samples:    {len(df)}")
    print(f"Accuracy:         {acc:.2%}")
    print(f"ROC AUC:          {auc:.4f}")
    print(f"Precision:        {prec:.2%}")
    print(f"Recall:           {rec:.2%}")
    print("-" * 40)
    
    # Hit rate at different thresholds
    for thresh in [0.70, 0.80, 0.85, 0.90]:
        high_prob = df[df['p_hit'] >= thresh]
        if not high_prob.empty:
            actual_hit_rate = high_prob['actual'].mean()
            print(f"Actual Hit Rate (at P >= {thresh:.2f}): {actual_hit_rate:.2%}")
    
    print("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run model backtest")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    args = parser.parse_args()
    
    run_backtest(n_samples=args.n)
