# CONTRACT: data_sources/openligadb.py
# Classes: OpenLigaDB
# Methods: __init__, __repr__, get_matches, get_team_list

from config import *

import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class OpenLigaDB:
    """OpenLigaDB client — completely free, no API key required."""

    def __init__(self, registry) -> None:
        """Args: registry — SourceRegistry instance."""
        self._registry = registry

    def get_matches(self, season: int, league_shortcut: str = "bl1") -> pd.DataFrame:
        """
        Fetch all matches for a season from OpenLigaDB.

        Args:
            season: Season year (e.g. 2024).
            league_shortcut: OpenLigaDB league shortcut (e.g. 'bl1').

        Returns:
            pd.DataFrame: Columns — match_id, home_team_id, away_team_id,
                home_team_name, away_team_name, match_date_utc,
                home_ht_goals, away_ht_goals, home_ft_goals, away_ft_goals,
                is_finished. Returns empty DataFrame on failure.
        """
        if not self._registry.before_call("openligadb"):
            return pd.DataFrame()
        url = f"{OPENLIGADB_BASE}/getmatchdata/{league_shortcut}/{season}"
        try:
            resp = requests.get(url, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._registry.after_call("openligadb", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("OpenLigaDB.get_matches(%r, %r) failed: %s", league_shortcut, season, exc)
            self._registry.after_call("openligadb", False, str(exc))
            return pd.DataFrame()

        records = []
        for match in data:
            try:
                match_id = str(match.get("MatchID", ""))
                home_team = match.get("Team1", {})
                away_team = match.get("Team2", {})
                match_results = match.get("MatchResults", [])

                home_ht, away_ht, home_ft, away_ft = 0, 0, 0, 0
                for result in match_results:
                    result_type = result.get("ResultTypeID", 0)
                    if result_type == 1:  # HT
                        home_ht = int(result.get("PointsTeam1", 0) or 0)
                        away_ht = int(result.get("PointsTeam2", 0) or 0)
                    elif result_type == 2:  # FT
                        home_ft = int(result.get("PointsTeam1", 0) or 0)
                        away_ft = int(result.get("PointsTeam2", 0) or 0)

                match_date_str = match.get("MatchDateTimeUTC", "") or match.get("MatchDateTime", "")

                records.append({
                    "match_id": match_id,
                    "home_team_id": str(home_team.get("TeamId", "")),
                    "away_team_id": str(away_team.get("TeamId", "")),
                    "home_team_name": home_team.get("TeamName", ""),
                    "away_team_name": away_team.get("TeamName", ""),
                    "match_date_utc": match_date_str,
                    "home_ht_goals": home_ht,
                    "away_ht_goals": away_ht,
                    "home_ft_goals": home_ft,
                    "away_ft_goals": away_ft,
                    "is_finished": bool(match.get("MatchIsFinished", False)),
                })
            except Exception as exc:
                logger.debug("OpenLigaDB match parse error: %s", exc)
                continue

        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_team_list(self, league_shortcut: str = "bl1", season: int = 2024) -> pd.DataFrame:
        """
        Fetch available teams for a league season.

        Args:
            league_shortcut: OpenLigaDB league shortcut.
            season: Season year.

        Returns:
            pd.DataFrame: Columns — team_id, team_name, short_name.
        """
        if not self._registry.before_call("openligadb"):
            return pd.DataFrame()
        url = f"{OPENLIGADB_BASE}/getavailableteams/{league_shortcut}/{season}"
        try:
            resp = requests.get(url, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._registry.after_call("openligadb", True)
        except requests.exceptions.RequestException as exc:
            logger.warning("OpenLigaDB.get_team_list failed: %s", exc)
            self._registry.after_call("openligadb", False, str(exc))
            return pd.DataFrame()

        records = [{"team_id": str(t.get("TeamId", "")),
                    "team_name": t.get("TeamName", ""),
                    "short_name": t.get("ShortName", "")} for t in data]
        return pd.DataFrame(records) if records else pd.DataFrame()

    def __repr__(self) -> str:
        return f"OpenLigaDB(base={OPENLIGADB_BASE!r})"
