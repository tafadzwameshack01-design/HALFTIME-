# CONTRACT: data_sources/espn_api.py
# Classes: ESPNApi
# Methods: __init__, __repr__, get_scoreboard, get_live_matches,
#          get_todays_fixtures, get_match_summary, get_ht_score

from config import *

import logging
from datetime import datetime, timezone

import requests

from utils.helpers import parse_espn_datetime, build_match_id

logger = logging.getLogger(__name__)


class ESPNApi:
    """ESPN hidden API client. No API key required."""

    def __init__(self, registry) -> None:
        """Args: registry — SourceRegistry instance."""
        self._registry = registry

    def get_scoreboard(self, league_key: str, date_str: str = None) -> list:
        """
        Fetch scoreboard events for a league on a given date.

        Args:
            league_key: Key into LEAGUES config.
            date_str: Date string in YYYYMMDD format. Defaults to today.

        Returns:
            list: Raw event dicts from ESPN response["events"], or [] on failure.
        """
        if not self._registry.before_call("espn"):
            return []
        slug = LEAGUES.get(league_key, {}).get("espn_slug", "")
        if not slug:
            return []
        url = f"{ESPN_BASE}/{slug}/scoreboard"
        params = {}
        if date_str:
            params["dates"] = date_str
        try:
            resp = requests.get(url, params=params, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._registry.after_call("espn", True)
            return data.get("events", [])
        except requests.exceptions.RequestException as exc:
            logger.warning("ESPNApi.get_scoreboard(%r) failed: %s", league_key, exc)
            self._registry.after_call("espn", False, str(exc))
            return []

    def get_live_matches(self, league_key: str) -> list:
        """
        Return currently live first-half matches for a league.

        Args:
            league_key: Key into LEAGUES config.

        Returns:
            list: Parsed match dicts for in-progress matches.
        """
        events = self.get_scoreboard(league_key)
        matches = []
        for event in events:
            try:
                status = event.get("status", {})
                state = status.get("type", {}).get("state", "")
                if state != "in":
                    continue
                parsed = self._parse_event(event, league_key)
                if parsed:
                    matches.append(parsed)
            except Exception as exc:
                logger.debug("ESPNApi.get_live_matches parse error: %s", exc)
        return matches

    def get_todays_fixtures(self, league_key: str) -> list:
        """
        Return today's pre-match fixtures for a league.

        Args:
            league_key: Key into LEAGUES config.

        Returns:
            list: Parsed fixture dicts for scheduled matches.
        """
        events = self.get_scoreboard(league_key)
        fixtures = []
        for event in events:
            try:
                status = event.get("status", {})
                state = status.get("type", {}).get("state", "")
                if state != "pre":
                    continue
                parsed = self._parse_event(event, league_key)
                if parsed:
                    fixtures.append(parsed)
            except Exception as exc:
                logger.debug("ESPNApi.get_todays_fixtures parse error: %s", exc)
        return fixtures

    def get_match_summary(self, league_key: str, event_id: str) -> "dict | None":
        """
        Fetch full match summary from ESPN.

        Args:
            league_key: Key into LEAGUES config.
            event_id: ESPN event identifier string.

        Returns:
            dict | None: Full summary dict, or None on failure.
        """
        if not self._registry.before_call("espn"):
            return None
        slug = LEAGUES.get(league_key, {}).get("espn_slug", "")
        url = f"{ESPN_BASE}/{slug}/summary"
        try:
            resp = requests.get(url, params={"event": event_id}, timeout=ESPN_TIMEOUT)
            resp.raise_for_status()
            self._registry.after_call("espn", True)
            return resp.json()
        except requests.exceptions.RequestException as exc:
            logger.warning("ESPNApi.get_match_summary(%r) failed: %s", event_id, exc)
            self._registry.after_call("espn", False, str(exc))
            return None

    def get_ht_score(self, league_key: str, event_id: str) -> "tuple | None":
        """
        Extract halftime score from ESPN match summary.

        Parses competitions[0].competitors[i].linescores[0].value.

        Args:
            league_key: Key into LEAGUES config.
            event_id: ESPN event identifier.

        Returns:
            tuple[int, int] | None: (home_ht_goals, away_ht_goals) or None.
        """
        summary = self.get_match_summary(league_key, event_id)
        if not summary:
            return None
        try:
            competitions = summary.get("header", {}).get("competitions", [])
            if not competitions:
                # Try top-level competitions key
                competitions = summary.get("competitions", [])
            if not competitions:
                return None
            competitors = competitions[0].get("competitors", [])
            if len(competitors) < 2:
                return None

            def _extract_ht(competitor: dict) -> int:
                linescores = competitor.get("linescores", [])
                if linescores:
                    val = linescores[0].get("value", 0)
                    return int(float(val))
                return 0

            # ESPN orders home team first (homeAway == "home")
            home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            return (_extract_ht(home_comp), _extract_ht(away_comp))
        except Exception as exc:
            logger.warning("ESPNApi.get_ht_score(%r) parse error: %s", event_id, exc)
            return None

    def _parse_event(self, event: dict, league_key: str) -> "dict | None":
        """
        Parse an ESPN event dict into a standardised match dict.

        Args:
            event: Raw ESPN event object.
            league_key: League identifier.

        Returns:
            dict | None: Standardised match dict, or None if parsing fails.
        """
        try:
            competitions = event.get("competitions", [])
            if not competitions:
                return None
            comp = competitions[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                return None

            home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

            home_team_id = str(home_comp.get("team", {}).get("id", ""))
            away_team_id = str(away_comp.get("team", {}).get("id", ""))
            home_team_name = home_comp.get("team", {}).get("displayName", "Unknown")
            away_team_name = away_comp.get("team", {}).get("displayName", "Unknown")

            kickoff_str = event.get("date", "")
            kickoff_utc = parse_espn_datetime(kickoff_str)

            # Current score
            home_score = int(float(home_comp.get("score", 0) or 0))
            away_score = int(float(away_comp.get("score", 0) or 0))

            # HT score from linescores
            home_ht = 0
            away_ht = 0
            home_ls = home_comp.get("linescores", [])
            away_ls = away_comp.get("linescores", [])
            if home_ls:
                home_ht = int(float(home_ls[0].get("value", 0)))
            if away_ls:
                away_ht = int(float(away_ls[0].get("value", 0)))

            # Current match minute
            current_minute = 0
            status = event.get("status", {})
            clock = status.get("displayClock", "0:00")
            try:
                current_minute = int(clock.split(":")[0])
            except Exception:
                current_minute = 0

            event_id = str(event.get("id", ""))
            match_id = build_match_id(home_team_id, away_team_id, kickoff_str[:10])

            venue = comp.get("venue", {}).get("fullName", "")

            return {
                "match_id": match_id,
                "espn_event_id": event_id,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_name": home_team_name,
                "away_team_name": away_team_name,
                "kickoff_utc": kickoff_utc.isoformat(),
                "league_key": league_key,
                "venue": venue,
                "home_score": home_score,
                "away_score": away_score,
                "home_ht_score": home_ht,
                "away_ht_score": away_ht,
                "current_minute": current_minute,
                "status_state": status.get("type", {}).get("state", ""),
            }
        except Exception as exc:
            logger.debug("ESPNApi._parse_event error: %s", exc)
            return None

    def __repr__(self) -> str:
        return f"ESPNApi(base={ESPN_BASE!r})"
