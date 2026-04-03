"""
Bronze Layer: Real-time daily lineup and probable pitcher ingestion via MLB Stats API.
"""

import logging
from datetime import datetime
from pathlib import Path
import statsapi
import pandas as pd

from config import BRONZE_DIR, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

def fetch_daily_lineups(date_str: str = None) -> pd.DataFrame:
    """
    Fetch today's schedule, probable pitchers, and starting lineups.
    
    Parameters
    ----------
    date_str : str, optional
        Date in 'YYYY-MM-DD' format. Defaults to today.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [game_date, game_id, home_team, away_team, 
                                 batter, batter_name, pitcher, stand, team]
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
        
    log.info(f"Fetching daily lineups for {date_str}...")
    
    # Fetch schedule with hydrations
    try:
        data = statsapi.get('schedule', {
            'sportId': 1,
            'date': date_str,
            'hydrate': 'probablePitcher,lineups,team'
        })
    except Exception as exc:
        log.error(f"Failed to fetch schedule from MLB API: {exc}")
        return pd.DataFrame()

    rows = []
    for date_info in data.get('dates', []):
        for game in date_info.get('games', []):
            game_id = game['gamePk']
            home_team = game['teams']['home']['team']['abbreviation']
            away_team = game['teams']['away']['team']['abbreviation']
            
            # Probable Pitchers
            home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('id')
            away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('id')
            
            # Get lineups if available
            lineups = game.get('lineups', {})
            
            # If lineups aren't posted yet, we might want to "guess" or wait.
            # For now, we only include confirmed starting players.
            
            # Process Home Batters vs Away Pitcher
            for p in lineups.get('homePlayers', []):
                rows.append({
                    'game_date': date_str,
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'batter': p['id'],
                    'batter_name': p.get('fullName', 'Unknown'),
                    'pitcher': away_pitcher,
                    'stand': p.get('batSide', {}).get('code', 'R'), # Default R if missing
                    'team': home_team,
                    'is_home': True
                })
                
            # Process Away Batters vs Home Pitcher
            for p in lineups.get('awayPlayers', []):
                rows.append({
                    'game_date': date_str,
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'batter': p['id'],
                    'batter_name': p.get('fullName', 'Unknown'),
                    'pitcher': home_pitcher,
                    'stand': p.get('batSide', {}).get('code', 'R'),
                    'team': away_team,
                    'is_home': False
                })
                
    if not rows:
        log.warning(f"No lineups found for {date_str}. They might not be posted yet.")
        return pd.DataFrame()
        
    df = pd.DataFrame(rows)
    log.info(f"Fetched {len(df)} batters across {len(data.get('dates', [{}])[0].get('games', []))} games.")
    return df

def save_daily_lineups(df: pd.DataFrame, output_dir: Path = BRONZE_DIR) -> Path:
    """Save daily lineups to a Parquet file."""
    if df.empty:
        return None
    date_str = df['game_date'].iloc[0]
    out_path = output_dir / f"daily_lineups_{date_str}.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved daily lineups to {out_path}")
    return out_path

if __name__ == "__main__":
    df = fetch_daily_lineups()
    if not df.empty:
        save_daily_lineups(df)
