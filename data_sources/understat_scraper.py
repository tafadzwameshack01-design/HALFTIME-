# CONTRACT: data_sources/understat_scraper.py
# Classes: UnderstatScraper
# Methods: __init__, __repr__, get_league_xg, get_match_xg

from config import *

import json
import logging
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class UnderstatScraper:
    """Scrapes xG data from understat.com. Fully defensive — never raises."""

    def __init__(self, registry) -> None:
        """Args: registry — SourceRegistry instance."""
        self._registry = registry

    def get_league_xg(self, league_name: str, season: int) -> pd.DataFrame:
        """
        Scrape team xG statistics for a league season from Understat.

        Args:
            league_name: Understat league name (e.g. 'Bundesliga').
            season: Season year (e.g. 2024).

        Returns:
            pd.DataFrame: Columns — team_name, xg_per_game, xga_per_game,
                npxg_per_game. Empty DataFrame on any failure.
        """
        if not league_name:
            return pd.DataFrame()
        if not self._registry.before_call("understat"):
            return pd.DataFrame()

        url = f"https://understat.com/league/{league_name}/{season}"
        try:
            time.sleep(SCRAPER_DELAY_SECONDS)
            resp = requests.get(url, headers=SCRAPER_HEADERS, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            self._registry.after_call("understat", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("UnderstatScraper.get_league_xg(%r, %r) failed: %s", league_name, season, exc)
            self._registry.after_call("understat", False, str(exc))
            return pd.DataFrame()

        try:
            soup = BeautifulSoup(resp.text, "lxml")
            scripts = soup.find_all("script")
            teams_data_raw = None
            for script in scripts:
                if script.string and "teamsData" in script.string:
                    # Extract JSON from: var teamsData = JSON.parse('...');
                    match = re.search(r"teamsData\s*=\s*JSON\.parse\('(.+?)'\)", script.string)
                    if match:
                        teams_data_raw = match.group(1)
                        break

            if not teams_data_raw:
                logger.debug("UnderstatScraper: teamsData not found for %r", league_name)
                return pd.DataFrame()

            # Unescape the JSON string
            teams_data_raw = teams_data_raw.encode().decode("unicode_escape")
            teams_data = json.loads(teams_data_raw)

            records = []
            for team_id, team_info in teams_data.items():
                try:
                    title = team_info.get("title", "")
                    history = team_info.get("history", [])
                    if not history:
                        continue
                    xg_vals = [float(h.get("xG", 0) or 0) for h in history]
                    xga_vals = [float(h.get("xGA", 0) or 0) for h in history]
                    npxg_vals = [float(h.get("npxG", 0) or 0) for h in history]
                    n = len(history) or 1
                    records.append({
                        "team_name": title,
                        "xg_per_game": round(sum(xg_vals) / n, 4),
                        "xga_per_game": round(sum(xga_vals) / n, 4),
                        "npxg_per_game": round(sum(npxg_vals) / n, 4),
                    })
                except Exception as exc:
                    logger.debug("UnderstatScraper team parse error: %s", exc)

            return pd.DataFrame(records) if records else pd.DataFrame()

        except Exception as exc:
            logger.warning("UnderstatScraper.get_league_xg parse error: %s", exc)
            return pd.DataFrame()

    def get_match_xg(self, match_id: str) -> "dict | None":
        """
        Scrape per-match xG totals from Understat.

        Args:
            match_id: Understat match ID string.

        Returns:
            dict | None: {"home_xg": float, "away_xg": float} or None on failure.
        """
        if not self._registry.before_call("understat"):
            return None
        url = f"https://understat.com/match/{match_id}"
        try:
            time.sleep(SCRAPER_DELAY_SECONDS)
            resp = requests.get(url, headers=SCRAPER_HEADERS, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            self._registry.after_call("understat", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("UnderstatScraper.get_match_xg(%r) failed: %s", match_id, exc)
            self._registry.after_call("understat", False, str(exc))
            return None

        try:
            soup = BeautifulSoup(resp.text, "lxml")
            scripts = soup.find_all("script")
            shots_raw = None
            for script in scripts:
                if script.string and "shotsData" in script.string:
                    match = re.search(r"shotsData\s*=\s*JSON\.parse\('(.+?)'\)", script.string)
                    if match:
                        shots_raw = match.group(1)
                        break

            if not shots_raw:
                return None

            shots_raw = shots_raw.encode().decode("unicode_escape")
            shots_data = json.loads(shots_raw)

            home_xg = sum(float(s.get("xG", 0) or 0) for s in shots_data.get("h", []))
            away_xg = sum(float(s.get("xG", 0) or 0) for s in shots_data.get("a", []))
            return {"home_xg": round(home_xg, 4), "away_xg": round(away_xg, 4)}

        except Exception as exc:
            logger.warning("UnderstatScraper.get_match_xg parse error: %s", exc)
            return None

    def __repr__(self) -> str:
        return "UnderstatScraper(source=understat.com)"
