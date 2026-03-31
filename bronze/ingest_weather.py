"""
Bronze Layer: Real-time weather ingestion via OpenWeatherMap API.
Returns temperature, humidity, and barometric pressure per MLB venue city.
"""

import logging
import requests
from typing import TypedDict

from config import OPENWEATHERMAP_API_KEY, LOG_LEVEL

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger(__name__)

# Mapping of MLB home team abbreviation → venue city name (for OWM query)
TEAM_CITY: dict[str, str] = {
    "ARI": "Phoenix,US",
    "ATL": "Atlanta,US",
    "BAL": "Baltimore,US",
    "BOS": "Boston,US",
    "CHC": "Chicago,US",
    "CWS": "Chicago,US",
    "CIN": "Cincinnati,US",
    "CLE": "Cleveland,US",
    "COL": "Denver,US",
    "DET": "Detroit,US",
    "HOU": "Houston,US",
    "KC":  "Kansas City,US",
    "LAA": "Anaheim,US",
    "LAD": "Los Angeles,US",
    "MIA": "Miami,US",
    "MIL": "Milwaukee,US",
    "MIN": "Minneapolis,US",
    "NYM": "New York,US",
    "NYY": "New York,US",
    "OAK": "Oakland,US",
    "PHI": "Philadelphia,US",
    "PIT": "Pittsburgh,US",
    "SD":  "San Diego,US",
    "SF":  "San Francisco,US",
    "SEA": "Seattle,US",
    "STL": "St. Louis,US",
    "TB":  "St. Petersburg,US",
    "TEX": "Arlington,US",
    "TOR": "Toronto,CA",
    "WSH": "Washington,US",
}


class WeatherData(TypedDict):
    team: str
    city: str
    temp_f: float
    humidity_pct: float
    pressure_mb: float
    description: str


def fetch_weather(team: str) -> WeatherData:
    """
    Fetch current weather for an MLB team's home city.

    Parameters
    ----------
    team : str
        3-letter MLB team abbreviation (e.g. 'COL').

    Returns
    -------
    WeatherData
        Dict with temp_f, humidity_pct, pressure_mb, description.

    Raises
    ------
    ValueError
        If the team abbreviation is unknown or the API key is missing.
    RuntimeError
        If the OpenWeatherMap API request fails.
    """
    if not OPENWEATHERMAP_API_KEY:
        raise ValueError(
            "OPENWEATHERMAP_API_KEY is not set. Add it to your .env file."
        )

    city = TEAM_CITY.get(team.upper())
    if city is None:
        raise ValueError(f"Unknown team abbreviation: '{team}'")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "imperial",   # Fahrenheit
    }

    log.debug(f"Fetching weather for {team} ({city})")
    resp = requests.get(url, params=params, timeout=10)

    if resp.status_code != 200:
        raise RuntimeError(
            f"OpenWeatherMap API error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    weather: WeatherData = {
        "team": team.upper(),
        "city": city,
        "temp_f":       float(data["main"]["temp"]),
        "humidity_pct": float(data["main"]["humidity"]),
        "pressure_mb":  float(data["main"]["pressure"]),
        "description":  data["weather"][0]["description"],
    }
    log.info(
        f"[{team}] {city}: {weather['temp_f']:.1f}°F, "
        f"{weather['humidity_pct']:.0f}% RH, "
        f"{weather['pressure_mb']:.1f} mb"
    )
    return weather


def fetch_all_weather(teams: list[str] | None = None) -> dict[str, WeatherData]:
    """
    Fetch weather for all teams (or a specified subset).

    Returns
    -------
    dict[str, WeatherData]
        Keyed by team abbreviation. Teams that fail are silently omitted
        with a warning logged.
    """
    if teams is None:
        teams = list(TEAM_CITY.keys())

    results: dict[str, WeatherData] = {}
    for team in teams:
        try:
            results[team] = fetch_weather(team)
        except Exception as exc:
            log.warning(f"Weather fetch failed for {team}: {exc}")
    return results


if __name__ == "__main__":
    import json, sys
    team_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if team_arg:
        print(json.dumps(fetch_weather(team_arg), indent=2))
    else:
        data = fetch_all_weather()
        print(json.dumps(data, indent=2))
