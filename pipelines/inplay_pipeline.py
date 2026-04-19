# CONTRACT: pipelines/inplay_pipeline.py
# Classes: InPlayPipeline
# Methods: __init__, __repr__, get_live_matches, compute_live_prediction,
#          get_probability_history, store_probability_checkpoint

from config import *

import json
import logging

logger = logging.getLogger(__name__)


class InPlayPipeline:
    """
    Live in-play prediction pipeline.

    Polls ESPN for live matches, merges live stats with cached pre-match
    features, and runs the ensemble to update probabilities every minute.
    Stores probability checkpoints for trend visualisation.
    At minute >= 40: marks prediction as is_final_call=True.
    """

    def __init__(self, db, espn, ensemble_map: dict, fe) -> None:
        """
        Args:
            db: DatabaseManager instance.
            espn: ESPNApi instance.
            ensemble_map: {league_key: EnsemblePredictor} dict.
            fe: FeatureEngineer instance.
        """
        self._db = db
        self._espn = espn
        self._ensemble_map = ensemble_map
        self._fe = fe

    def get_live_matches(self, league_keys: list) -> list:
        """
        Fetch all currently live first-half matches across specified leagues.

        Args:
            league_keys: List of league identifiers to poll.

        Returns:
            list[dict]: Live match dicts from ESPN, one per live match.
        """
        all_live = []
        for league_key in league_keys:
            try:
                live = self._espn.get_live_matches(league_key)
                all_live.extend(live)
            except Exception as exc:
                logger.warning("InPlayPipeline.get_live_matches(%r) error: %s", league_key, exc)
        return all_live

    def compute_live_prediction(
        self,
        match_id: str,
        league_key: str,
        current_minute: int,
        live_data: dict,
    ) -> dict:
        """
        Compute an updated ensemble prediction using live match statistics.

        Retrieves cached pre-match features, merges with live_data, runs
        the ensemble, and stores a probability checkpoint for trend charting.
        At current_minute >= 40, sets is_final_call=True.

        Args:
            match_id: Internal match identifier.
            league_key: League identifier.
            current_minute: Current match minute (0-45).
            live_data: Dict of live statistics from ESPN (home_score,
                away_score, home_shots, away_shots, home_possession, etc.).

        Returns:
            dict: Full prediction dict with 'markets', 'is_final_call',
                'current_minute', 'match_id'. Empty dict on failure.
        """
        try:
            # Load cached pre-match features
            cached = self._db.get_cached_match(match_id, "prematch_features")
            if not cached:
                logger.debug("InPlayPipeline: no prematch cache for %r.", match_id)
                return {}

            prematch_features = cached.get("features", {})
            fixture = cached.get("fixture", {})

            home_id = fixture.get("home_team_id", "")
            away_id = fixture.get("away_team_id", "")

            # Build inplay features
            inplay_features = self._fe.build_inplay_features(
                prematch_features, live_data, current_minute
            )

            # Convert to array and normalise
            feature_array = self._fe.features_to_array(inplay_features, INPLAY_FEATURE_COUNT)

            # Pad or trim to PREMATCH_FEATURE_COUNT for model compatibility
            # (models trained on prematch features; use first PREMATCH_FEATURE_COUNT cols)
            feature_array_trimmed = feature_array[:, :PREMATCH_FEATURE_COUNT]
            feature_array_norm = self._fe.transform_features(feature_array_trimmed, league_key)

            ensemble = self._ensemble_map.get(league_key)
            if ensemble is None:
                return {}

            market_results = ensemble.predict(
                home_id, away_id, feature_array_norm, pipeline_type="inplay"
            )

            is_final_call = current_minute >= 40

            # Store probability checkpoints for trend chart
            for market, result in market_results.items():
                if result.get("should_predict", False):
                    self.store_probability_checkpoint(
                        match_id, market, current_minute,
                        result.get("calibrated_prob", 0.5)
                    )

            return {
                "match_id":       match_id,
                "league_key":     league_key,
                "current_minute": current_minute,
                "is_final_call":  is_final_call,
                "markets":        market_results,
                "live_data":      live_data,
                "fixture":        fixture,
            }

        except Exception as exc:
            logger.error("InPlayPipeline.compute_live_prediction(%r) error: %s", match_id, exc)
            return {}

    def get_probability_history(self, match_id: str, market: str) -> list:
        """
        Retrieve the stored probability trend history for a match/market.

        Args:
            match_id: Internal match identifier.
            market: Market string (e.g. 'HT_over_1.5').

        Returns:
            list[dict]: Each item has 'minute' (int) and 'prob' (float),
                ordered by minute ascending. Empty list if no history.
        """
        try:
            safe_market = market.replace(".", "_").replace("/", "_")
            cache_key = f"inplay_hist_{match_id}_{safe_market}"
            cached = self._db.get_cached_match(cache_key, "inplay_history")
            if not cached:
                return []
            history = cached.get("history", [])
            return sorted(history, key=lambda x: x.get("minute", 0))
        except Exception as exc:
            logger.warning("InPlayPipeline.get_probability_history(%r, %r) error: %s",
                           match_id, market, exc)
            return []

    def store_probability_checkpoint(
        self, match_id: str, market: str, minute: int, prob: float
    ) -> None:
        """
        Append a probability checkpoint to the inplay history cache.

        Args:
            match_id: Internal match identifier.
            market: Market string.
            minute: Current match minute.
            prob: Calibrated probability at this minute.
        """
        try:
            safe_market = market.replace(".", "_").replace("/", "_")
            cache_key = f"inplay_hist_{match_id}_{safe_market}"

            # Load existing history
            existing = self._db.get_cached_match(cache_key, "inplay_history")
            history = existing.get("history", []) if existing else []

            # Append new checkpoint
            history.append({"minute": int(minute), "prob": round(float(prob), 6)})

            # Keep only the last 50 checkpoints to avoid bloat
            history = history[-50:]

            self._db.cache_match(
                match_id=cache_key,
                league_key="",
                source="inplay_history",
                data={"history": history},
            )
        except Exception as exc:
            logger.debug("InPlayPipeline.store_probability_checkpoint error: %s", exc)

    def __repr__(self) -> str:
        return f"InPlayPipeline(leagues={list(self._ensemble_map.keys())})"
