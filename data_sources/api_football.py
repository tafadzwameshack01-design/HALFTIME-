# CONTRACT: data_sources/api_football.py
# Classes: ApiFootball
# Methods: __init__, __repr__, get_fixtures, get_h2h, get_team_stats, get_injuries

from config import *

import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class ApiFootball:
    """API-Football v3 client (free tier: 100 req/day)."""

    def __init__(self, registry) -> None:
        """Args: registry — SourceRegistry instance."""
        self._registry = registry
        self._headers = {"x-apisports-key": API_FOOTBALL_KEY}

    def _get(self, endpoint: str, params: dict) -> "dict | None":
        """
        Make a GET request to API-Football. Guards rate limit before calling.

        Args:
            endpoint: API path (e.g. 'fixtures').
            params: Query parameters dict.

        Returns:
            dict | None: Parsed JSON response, or None on failure.
        """
        if not API_FOOTBALL_KEY:
            return None
        if not self._registry.before_call("api_football"):
            return None
        url = f"{API_FOOTBALL_BASE}/{endpoint}"
        try:
            resp = requests.get(url, headers=self._headers, params=params, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            self._registry.after_call("api_football", True)
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("ApiFootball.%s(%r) failed: %s", endpoint, params, exc)
            self._registry.after_call("api_football", False, str(exc))
            return None

    def get_fixtures(self, league_id: int, season: int, date: str = None) -> pd.DataFrame:
        """
        Fetch fixtures for a league/season, optionally filtered by date.

        Args:
            league_id: API-Football league ID.
            season: Season year (e.g. 2024).
            date: Optional date string YYYY-MM-DD to filter to a single day.

        Returns:
            pd.DataFrame: Standardised fixture rows. Empty on failure.
        """
        params = {"league": league_id, "season": season}
        if date:
            params["date"] = date
        data = self._get("fixtures", params)
        if not data:
            return pd.DataFrame()
        records = []
        for fix in data.get("response", []):
            try:
                fixture = fix.get("fixture", {})
                teams = fix.get("teams", {})
                goals = fix.get("goals", {})
                score = fix.get("score", {})
                ht = score.get("halftime", {}) or {}
                records.append({
                    "fixture_id": str(fixture.get("id", "")),
                    "home_team_id": str(teams.get("home", {}).get("id", "")),
                    "away_team_id": str(teams.get("away", {}).get("id", "")),
                    "home_team_name": teams.get("home", {}).get("name", ""),
                    "away_team_name": teams.get("away", {}).get("name", ""),
                    "kickoff_utc": fixture.get("date", ""),
                    "home_ht_goals": ht.get("home"),
                    "away_ht_goals": ht.get("away"),
                    "home_ft_goals": goals.get("home"),
                    "away_ft_goals": goals.get("away"),
                    "status_short": fixture.get("status", {}).get("short", ""),
                })
            except Exception as exc:
                logger.debug("ApiFootball fixture parse error: %s", exc)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_h2h(self, team1_id: int, team2_id: int, last: int = 6) -> pd.DataFrame:
        """
        Fetch head-to-head fixture history between two teams.

        Args:
            team1_id: API-Football team ID for team 1.
            team2_id: API-Football team ID for team 2.
            last: Number of most recent H2H matches to return.

        Returns:
            pd.DataFrame: H2H fixtures with HT scores. Empty on failure.
        """
        data = self._get("fixtures/headtohead", {"h2h": f"{team1_id}-{team2_id}", "last": last})
        if not data:
            return pd.DataFrame()
        records = []
        for fix in data.get("response", []):
            try:
                teams = fix.get("teams", {})
                score = fix.get("score", {})
                ht = score.get("halftime", {}) or {}
                records.append({
                    "home_team_id": str(teams.get("home", {}).get("id", "")),
                    "away_team_id": str(teams.get("away", {}).get("id", "")),
                    "home_team_name": teams.get("home", {}).get("name", ""),
                    "away_team_name": teams.get("away", {}).get("name", ""),
                    "home_ht_goals": ht.get("home"),
                    "away_ht_goals": ht.get("away"),
                    "match_date": fix.get("fixture", {}).get("date", ""),
                })
            except Exception as exc:
                logger.debug("ApiFootball H2H parse error: %s", exc)
        return pd.DataFrame(records) if records else pd.DataFrame()

    def get_team_stats(self, league_id: int, season: int, team_id: int) -> dict:
        """
        Fetch team statistics for a season.

        Args:
            league_id: API-Football league ID.
            season: Season year.
            team_id: API-Football team ID.

        Returns:
            dict: shots_on_target_avg, corners_avg, goals_scored_avg,
                goals_conceded_avg. All 0.0 on failure.
        """
        default = {"shots_on_target_avg": 0.0, "corners_avg": 0.0,
                   "goals_scored_avg": 0.0, "goals_conceded_avg": 0.0}
        data = self._get("teams/statistics", {"league": league_id, "season": season, "team": team_id})
        if not data or not data.get("response"):
            return default
        try:
            resp = data["response"]
            goals = resp.get("goals", {})
            shots = resp.get("shots", {})
            return {
                "shots_on_target_avg": float(shots.get("on", {}).get("average", {}).get("total", 0) or 0),
                "corners_avg": 0.0,  # API-Football does not expose corners directly in stats
                "goals_scored_avg": float(goals.get("for", {}).get("average", {}).get("total", 0) or 0),
                "goals_conceded_avg": float(goals.get("against", {}).get("average", {}).get("total", 0) or 0),
            }
        except Exception as exc:
            logger.warning("ApiFootball.get_team_stats parse error: %s", exc)
            return default

    def get_injuries(self, league_id: int, season: int, team_id: int) -> list:
        """
        Fetch active injuries for a team.

        Args:
            league_id: API-Football league ID.
            season: Season year.
            team_id: API-Football team ID.

        Returns:
            list[dict]: Active injury records. Empty list on failure.
        """
        data = self._get("injuries", {"league": league_id, "season": season, "team": team_id})
        if not data:
            return []
        records = []
        for injury in data.get("response", []):
            try:
                player = injury.get("player", {})
                records.append({
                    "player_name": player.get("name", ""),
                    "player_id": str(player.get("id", "")),
                    "type": injury.get("fixture", {}).get("type", ""),
                    "reason": injury.get("fixture", {}).get("reason", ""),
                })
            except Exception as exc:
                logger.debug("ApiFootball injury parse error: %s", exc)
        return records

    def __repr__(self) -> str:
        return f"ApiFootball(base={API_FOOTBALL_BASE!r}, key_set={bool(API_FOOTBALL_KEY)})"
