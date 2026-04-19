# CONTRACT: data_sources/football_data_org.py
# Classes: FootballDataOrg
# Methods: __init__, __repr__, get_matches, get_todays_fixtures, get_standings

from config import *

import logging
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class FootballDataOrg:
    """football-data.org API client (free tier)."""

    def __init__(self, registry) -> None:
        """Args: registry — SourceRegistry instance."""
        self._registry = registry
        self._headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}

    def get_matches(self, competition_code: str, date_from: str, date_to: str) -> pd.DataFrame:
        """
        Fetch matches for a competition within a date range.

        Skips leagues where competition_code is None (e.g. super_lig).

        Args:
            competition_code: football-data.org code (e.g. 'BL1'). None = skip.
            date_from: Start date string YYYY-MM-DD.
            date_to: End date string YYYY-MM-DD.

        Returns:
            pd.DataFrame: Standardised match rows. Empty DataFrame on failure or None code.
        """
        if not competition_code:
            logger.debug("FootballDataOrg.get_matches: competition_code is None, skipping.")
            return pd.DataFrame()
        if not FOOTBALL_DATA_KEY:
            logger.debug("FootballDataOrg: no API key configured.")
            return pd.DataFrame()
        if not self._registry.before_call("football_data_org"):
            return pd.DataFrame()

        url = f"{FOOTBALL_DATA_BASE}/competitions/{competition_code}/matches"
        params = {"dateFrom": date_from, "dateTo": date_to}
        try:
            resp = requests.get(url, headers=self._headers, params=params, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._registry.after_call("football_data_org", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("FootballDataOrg.get_matches(%r) failed: %s", competition_code, exc)
            self._registry.after_call("football_data_org", False, str(exc))
            return pd.DataFrame()

        records = []
        for match in data.get("matches", []):
            try:
                score = match.get("score", {})
                ht = score.get("halfTime", {}) or {}
                ft = score.get("fullTime", {}) or {}
                records.append({
                    "match_id": str(match.get("id", "")),
                    "home_team_id": str(match.get("homeTeam", {}).get("id", "")),
                    "away_team_id": str(match.get("awayTeam", {}).get("id", "")),
                    "home_team_name": match.get("homeTeam", {}).get("name", ""),
                    "away_team_name": match.get("awayTeam", {}).get("name", ""),
                    "kickoff_utc": match.get("utcDate", ""),
                    "home_ht_goals": ht.get("home") if ht.get("home") is not None else None,
                    "away_ht_goals": ht.get("away") if ht.get("away") is not None else None,
                    "home_ft_goals": ft.get("home") if ft.get("home") is not None else None,
                    "away_ft_goals": ft.get("away") if ft.get("away") is not None else None,
                    "status": match.get("status", ""),
                })
            except Exception as exc:
                logger.debug("FootballDataOrg match parse error: %s", exc)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_todays_fixtures(self, competition_code: str) -> list:
        """
        Return today's scheduled fixtures for a competition.

        Args:
            competition_code: football-data.org code or None.

        Returns:
            list[dict]: Fixture dicts with status == 'SCHEDULED'.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df = self.get_matches(competition_code, today, today)
        if df.empty:
            return []
        scheduled = df[df["status"] == "SCHEDULED"]
        return scheduled.to_dict("records")

    def get_standings(self, competition_code: str) -> pd.DataFrame:
        """
        Fetch current league standings.

        Args:
            competition_code: football-data.org code or None.

        Returns:
            pd.DataFrame: Standings data. Empty on failure.
        """
        if not competition_code or not FOOTBALL_DATA_KEY:
            return pd.DataFrame()
        if not self._registry.before_call("football_data_org"):
            return pd.DataFrame()
        url = f"{FOOTBALL_DATA_BASE}/competitions/{competition_code}/standings"
        try:
            resp = requests.get(url, headers=self._headers, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._registry.after_call("football_data_org", True)
            standings = data.get("standings", [])
            if standings:
                table = standings[0].get("table", [])
                records = [{"position": t.get("position"), "team": t.get("team", {}).get("name", ""),
                             "played": t.get("playedGames"), "won": t.get("won"),
                             "draw": t.get("draw"), "lost": t.get("lost"),
                             "goals_for": t.get("goalsFor"), "goals_against": t.get("goalsAgainst"),
                             "points": t.get("points")} for t in table]
                return pd.DataFrame(records)
            return pd.DataFrame()
        except requests.exceptions.RequestException as exc:
            logger.warning("FootballDataOrg.get_standings(%r) failed: %s", competition_code, exc)
            self._registry.after_call("football_data_org", False, str(exc))
            return pd.DataFrame()

    def __repr__(self) -> str:
        return f"FootballDataOrg(base={FOOTBALL_DATA_BASE!r}, key_set={bool(FOOTBALL_DATA_KEY)})"
