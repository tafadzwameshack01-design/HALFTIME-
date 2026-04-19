# CONTRACT: data_sources/odds_api.py
# Classes: OddsApi
# Methods: __init__, __repr__, get_ht_odds

from config import *

import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class OddsApi:
    """The Odds API client (free tier: 500 req/month)."""

    SPORT_KEYS: dict = {
        "bundesliga":   "soccer_germany_bundesliga",
        "eredivisie":   "soccer_netherlands_eredivisie",
        "belgian_pro":  "soccer_belgium_first_div",
        "super_lig":    "soccer_turkey_super_league",
        "championship": "soccer_england_league1",
    }

    def __init__(self, registry) -> None:
        """Args: registry — SourceRegistry instance."""
        self._registry = registry

    def get_ht_odds(self, sport_key: str, regions: str = "eu") -> pd.DataFrame:
        """
        Fetch bookmaker odds for a sport and convert to implied probabilities.

        Args:
            sport_key: The Odds API sport key string.
            regions: Bookmaker regions to query (default 'eu').

        Returns:
            pd.DataFrame: Columns — home_team, away_team, bookmaker,
                market, implied_prob. Empty DataFrame on failure or no key.
        """
        if not ODDS_API_KEY:
            logger.debug("OddsApi: no API key configured.")
            return pd.DataFrame()
        if not self._registry.before_call("odds_api"):
            return pd.DataFrame()

        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds"
        params = {
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "apiKey": ODDS_API_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._registry.after_call("odds_api", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("OddsApi.get_ht_odds(%r) failed: %s", sport_key, exc)
            self._registry.after_call("odds_api", False, str(exc))
            return pd.DataFrame()

        records = []
        for event in data:
            try:
                home_team = event.get("home_team", "")
                away_team = event.get("away_team", "")
                for bookmaker in event.get("bookmakers", []):
                    bk_name = bookmaker.get("title", "")
                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key", "")
                        for outcome in market.get("outcomes", []):
                            decimal_odds = float(outcome.get("price", 2.0) or 2.0)
                            # Guard against division by zero
                            implied_prob = round(1.0 / decimal_odds, 6) if decimal_odds > 0 else 0.5
                            records.append({
                                "home_team": home_team,
                                "away_team": away_team,
                                "bookmaker": bk_name,
                                "market": market_key,
                                "outcome_name": outcome.get("name", ""),
                                "decimal_odds": decimal_odds,
                                "implied_prob": implied_prob,
                            })
            except Exception as exc:
                logger.debug("OddsApi event parse error: %s", exc)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def __repr__(self) -> str:
        return f"OddsApi(base={ODDS_API_BASE!r}, key_set={bool(ODDS_API_KEY)})"
