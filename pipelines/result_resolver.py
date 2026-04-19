# CONTRACT: pipelines/result_resolver.py
# Classes: ResultResolver
# Methods: __init__, __repr__, maybe_run_resolver,
#          resolve_pending_predictions, _fetch_ht_score_from_espn, resolve_single

from config import *

import logging
import os
from datetime import datetime, timedelta, timezone

import requests

logger = logging.getLogger(__name__)


class ResultResolver:
    """
    Resolves pending predictions by fetching actual HT scores from ESPN.

    Uses a Streamlit-Cloud-safe timestamp pattern (DB cache) instead of
    background threads to check if enough time has passed since the last run.
    Called on every page load from app.py via maybe_run_resolver().
    """

    def __init__(self, db, espn, online_learner_map: dict) -> None:
        """
        Args:
            db: DatabaseManager instance.
            espn: ESPNApi instance.
            online_learner_map: {league_key: OnlineLearner} dict.
        """
        self._db = db
        self._espn = espn
        self._ol_map = online_learner_map

    def maybe_run_resolver(self) -> None:
        """
        Run resolve_pending_predictions() if enough time has elapsed since the
        last run (RESOLVER_INTERVAL_MINUTES).

        Checks the last-run timestamp stored in the match_cache table under
        match_id=RESOLVER_CACHE_KEY, source='resolver_ts'. Updates the timestamp
        after each successful run.
        """
        try:
            cached = self._db.get_cached_match(RESOLVER_CACHE_KEY, "resolver_ts")
            if cached:
                last_run_str = cached.get("last_run_at", "")
                if last_run_str:
                    last_run = datetime.fromisoformat(last_run_str)
                    if last_run.tzinfo is None:
                        last_run = last_run.replace(tzinfo=timezone.utc)
                    elapsed_minutes = (datetime.now(timezone.utc) - last_run).total_seconds() / 60.0
                    if elapsed_minutes < RESOLVER_INTERVAL_MINUTES:
                        return  # Too soon to run again

            # Enough time has passed — run resolver
            count = self.resolve_pending_predictions()
            logger.info("ResultResolver.maybe_run_resolver: resolved %d predictions.", count)

            # Update last-run timestamp
            self._db.cache_match(
                match_id=RESOLVER_CACHE_KEY,
                league_key="",
                source="resolver_ts",
                data={"last_run_at": datetime.now(timezone.utc).isoformat()},
            )
        except Exception as exc:
            logger.error("ResultResolver.maybe_run_resolver() error: %s", exc)

    def resolve_pending_predictions(self) -> int:
        """
        Fetch HT scores for all pending predictions and resolve them.

        Returns:
            int: Number of predictions successfully resolved.
        """
        resolved_count = 0
        try:
            pending = self._db.get_pending_predictions(
                older_than_hours=RESOLVER_MATCH_GRACE_HOURS
            )
            if not pending:
                return 0

            # Group by (match_id, league_key, espn_event_id) to avoid duplicate ESPN calls
            seen_matches: dict = {}  # match_id -> (home_ht, away_ht) or None

            for pred in pending:
                match_id   = pred.get("match_id", "")
                league_key = pred.get("league_key", "")
                market     = pred.get("market", "")

                if not match_id or not market:
                    continue

                # Fetch HT score once per match
                if match_id not in seen_matches:
                    ht_score = self._fetch_ht_score_from_espn(match_id, league_key)
                    seen_matches[match_id] = ht_score

                ht_score = seen_matches[match_id]
                if ht_score is None:
                    continue

                home_ht, away_ht = ht_score

                # Resolve in DB
                self._db.resolve_prediction(match_id, market, home_ht, away_ht)

                # Trigger online learner update
                try:
                    ol = self._ol_map.get(league_key)
                    if ol is not None:
                        features_json = pred.get("features_json", {})
                        if isinstance(features_json, str):
                            import json
                            features_json = json.loads(features_json) if features_json else {}
                        # Determine actual outcome for this market
                        total = home_ht + away_ht
                        market_outcomes = {
                            "HT_over_0.5":  1 if total > 0 else 0,
                            "HT_under_0.5": 1 if total <= 0 else 0,
                            "HT_over_1.5":  1 if total > 1 else 0,
                            "HT_under_1.5": 1 if total <= 1 else 0,
                            "HT_over_2.5":  1 if total > 2 else 0,
                            "HT_under_2.5": 1 if total <= 2 else 0,
                        }
                        actual_outcome = market_outcomes.get(market, 0)
                        ol.process_new_result(match_id, market, features_json, actual_outcome)
                except Exception as exc:
                    logger.warning("ResultResolver online_learner update error: %s", exc)

                resolved_count += 1

                # Log to resolver log file
                self._log_resolution(match_id, market, home_ht, away_ht, league_key)

        except Exception as exc:
            logger.error("ResultResolver.resolve_pending_predictions() error: %s", exc)

        return resolved_count

    def _fetch_ht_score_from_espn(
        self, match_id: str, league_key: str
    ) -> "tuple | None":
        """
        Attempt to fetch the halftime score for a match from ESPN.

        Args:
            match_id: Our internal match ID (used to look up the ESPN event ID
                from the match cache if available).
            league_key: League identifier.

        Returns:
            tuple[int, int] | None: (home_ht_goals, away_ht_goals) or None.
        """
        try:
            # Try to find ESPN event ID from cache
            cached = self._db.get_cached_match(match_id, "espn")
            espn_event_id = None
            if cached:
                espn_event_id = cached.get("espn_event_id")

            if not espn_event_id:
                # Fall back: search today's scoreboard for this league
                events = self._espn.get_scoreboard(league_key)
                for event in events:
                    parsed = self._espn._parse_event(event, league_key)
                    if parsed and parsed.get("match_id") == match_id:
                        espn_event_id = parsed.get("espn_event_id")
                        break

            if not espn_event_id:
                return None

            ht_score = self._espn.get_ht_score(league_key, str(espn_event_id))
            return ht_score

        except Exception as exc:
            logger.warning(
                "ResultResolver._fetch_ht_score_from_espn(%r) error: %s",
                match_id, exc
            )
            return None

    def resolve_single(self, match_id: str, league_key: str) -> bool:
        """
        Manually resolve a single match (called from the UI).

        Args:
            match_id: Internal match identifier.
            league_key: League identifier.

        Returns:
            bool: True if the match was successfully resolved.
        """
        try:
            ht_score = self._fetch_ht_score_from_espn(match_id, league_key)
            if ht_score is None:
                return False
            home_ht, away_ht = ht_score
            for market in MARKETS:
                self._db.resolve_prediction(match_id, market, home_ht, away_ht)
            self._log_resolution(match_id, "all_markets", home_ht, away_ht, league_key)
            return True
        except Exception as exc:
            logger.error("ResultResolver.resolve_single(%r) error: %s", match_id, exc)
            return False

    def _log_resolution(
        self, match_id: str, market: str,
        home_ht: int, away_ht: int, league_key: str
    ) -> None:
        """Write a resolution event line to the resolver log file."""
        try:
            os.makedirs(os.path.dirname(RESOLVER_LOG), exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            line = f"{now} | RESOLVED | {league_key} | {match_id} | {market} | HT: {home_ht}-{away_ht}\n"
            with open(RESOLVER_LOG, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception as exc:
            logger.debug("_log_resolution write error: %s", exc)

    def __repr__(self) -> str:
        return f"ResultResolver(leagues={list(self._ol_map.keys())})"
