# CONTRACT: pipelines/prematch_pipeline.py
# Classes: PreMatchPipeline
# Methods: __init__, __repr__, run, get_todays_fixtures,
#          is_model_ready, get_training_progress

from config import *

import logging
import uuid
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class PreMatchPipeline:
    """
    Daily pre-match prediction pipeline.

    Fetches today's fixtures, builds features, runs ensemble, and logs predictions.
    Degrades gracefully to DC-only statistical mode when ML model is not ready.
    """

    def __init__(
        self,
        db,
        registry,
        espn,
        fdorg,
        apif,
        ensemble_map: dict,
        fe,
    ) -> None:
        """
        Args:
            db: DatabaseManager instance.
            registry: SourceRegistry instance.
            espn: ESPNApi instance.
            fdorg: FootballDataOrg instance.
            apif: ApiFootball instance.
            ensemble_map: {league_key: EnsemblePredictor} dict.
            fe: FeatureEngineer instance.
        """
        self._db = db
        self._registry = registry
        self._espn = espn
        self._fdorg = fdorg
        self._apif = apif
        self._ensemble_map = ensemble_map
        self._fe = fe

    def run(self, league_keys: list = None) -> list:
        """
        Execute the pre-match prediction pipeline for all active leagues.

        Args:
            league_keys: Optional list of leagues to run. Defaults to ACTIVE_LEAGUE_KEYS.

        Returns:
            list[dict]: All prediction dicts (including NO_BET). Each dict contains
                match metadata, all 6 market prediction sub-dicts, and a
                'model_ready' boolean.
        """
        if league_keys is None:
            league_keys = ACTIVE_LEAGUE_KEYS

        all_predictions = []

        for league_key in league_keys:
            try:
                fixtures = self.get_todays_fixtures(league_key)
                if not fixtures:
                    logger.info("PreMatchPipeline: no fixtures for %r.", league_key)
                    continue

                model_ready, reason = self.is_model_ready(league_key)
                ensemble = self._ensemble_map.get(league_key)

                if ensemble is None:
                    logger.warning("PreMatchPipeline: no ensemble for %r.", league_key)
                    continue

                for fixture in fixtures:
                    try:
                        home_id   = fixture.get("home_team_id", "")
                        away_id   = fixture.get("away_team_id", "")
                        home_name = fixture.get("home_team_name", "")
                        away_name = fixture.get("away_team_name", "")
                        kickoff   = fixture.get("kickoff_utc", "")
                        match_id  = fixture.get("match_id", uuid.uuid4().hex[:16])

                        match_date = datetime.now(timezone.utc)
                        if kickoff:
                            try:
                                match_date = datetime.fromisoformat(
                                    kickoff.replace("Z", "+00:00")
                                )
                            except Exception:
                                pass

                        # Build pre-match features
                        dc_model = ensemble._dc
                        xg_model = ensemble._ol._fe  # access via ensemble internals
                        # Get xg model directly from ensemble's context
                        try:
                            xg_model_obj = None
                            # Try to find synthetic xg in the system dict stored on ensemble
                            if hasattr(ensemble, '_xg_model'):
                                xg_model_obj = ensemble._xg_model
                        except Exception:
                            xg_model_obj = None

                        features = self._fe.build_prematch_features(
                            home_id, away_id, league_key, match_date, dc_model, xg_model_obj
                        )

                        # Convert to array and normalise
                        feature_array = self._fe.features_to_array(features, PREMATCH_FEATURE_COUNT)
                        feature_array_norm = self._fe.transform_features(feature_array, league_key)

                        # Run ensemble
                        market_results = ensemble.predict(
                            home_id, away_id, feature_array_norm, pipeline_type="prematch"
                        )

                        prediction_record = {
                            "match_id":        match_id,
                            "home_team":       home_name,
                            "away_team":       away_name,
                            "league_key":      league_key,
                            "match_date":      match_date.strftime("%Y-%m-%d"),
                            "kickoff_utc":     kickoff,
                            "model_ready":     model_ready,
                            "model_reason":    reason,
                            "fixture":         fixture,
                            "features":        features,
                            "markets":         market_results,
                        }

                        # Log each market prediction to DB
                        for market, result in market_results.items():
                            self._db.log_prediction({
                                "id":               uuid.uuid4().hex[:16],
                                "match_id":         match_id,
                                "home_team":        home_name,
                                "away_team":        away_name,
                                "league_key":       league_key,
                                "match_date":       match_date.strftime("%Y-%m-%d"),
                                "kickoff_utc":      kickoff,
                                "market":           market,
                                "predicted_prob":   result.get("calibrated_prob"),
                                "predicted_outcome": 1 if "over" in market else 0,
                                "confidence_label": result.get("confidence_label"),
                                "dixon_coles_prob": result.get("dixon_coles_prob"),
                                "xgb_prob":         result.get("xgb_prob"),
                                "sgd_adjustment":   result.get("sgd_adjustment"),
                                "ensemble_prob":    result.get("raw_ensemble_prob"),
                                "features":         features,
                                "pipeline_type":    "prematch",
                            })

                            # Cache fixture data for in-play pipeline
                            self._db.cache_match(
                                match_id=match_id,
                                league_key=league_key,
                                source="prematch_features",
                                data={"features": features, "fixture": fixture},
                            )

                        all_predictions.append(prediction_record)

                    except Exception as exc:
                        logger.warning(
                            "PreMatchPipeline fixture error (%r): %s",
                            fixture.get("home_team_name", "?"), exc
                        )

            except Exception as exc:
                logger.error("PreMatchPipeline league error (%r): %s", league_key, exc)

        logger.info("PreMatchPipeline.run(): %d predictions generated.", len(all_predictions))
        return all_predictions

    def get_todays_fixtures(self, league_key: str) -> list:
        """
        Fetch today's fixtures using source priority: ESPN → FootballDataOrg → ApiFootball.

        Args:
            league_key: League identifier.

        Returns:
            list[dict]: Fixture dicts. Empty list if all sources fail.
        """
        # Priority 1: ESPN (no key required)
        fixtures = self._espn.get_todays_fixtures(league_key)
        if fixtures:
            return fixtures

        # Priority 2: football-data.org (skip if no code or no key)
        football_data_code = LEAGUES.get(league_key, {}).get("football_data_code")
        if football_data_code:
            fd_fixtures = self._fdorg.get_todays_fixtures(football_data_code)
            if fd_fixtures:
                # Normalise to standard format
                return [self._normalise_fdorg_fixture(f, league_key) for f in fd_fixtures]

        # Priority 3: API-Football
        api_football_id = LEAGUES.get(league_key, {}).get("api_football_id")
        api_football_season = LEAGUES.get(league_key, {}).get("api_football_season", 2024)
        if api_football_id:
            from datetime import datetime, timezone
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            df = self._apif.get_fixtures(api_football_id, api_football_season, date=today)
            if not df.empty:
                return df.to_dict("records")

        return []

    def _normalise_fdorg_fixture(self, fixture: dict, league_key: str) -> dict:
        """Normalise a football-data.org fixture dict to the standard format."""
        from utils.helpers import build_match_id
        kickoff = fixture.get("kickoff_utc", "")
        home_id = str(fixture.get("home_team_id", ""))
        away_id = str(fixture.get("away_team_id", ""))
        return {
            "match_id":       build_match_id(home_id, away_id, kickoff[:10]),
            "home_team_id":   home_id,
            "away_team_id":   away_id,
            "home_team_name": fixture.get("home_team_name", ""),
            "away_team_name": fixture.get("away_team_name", ""),
            "kickoff_utc":    kickoff,
            "league_key":     league_key,
            "venue":          "",
        }

    def is_model_ready(self, league_key: str) -> tuple:
        """
        Check if the ML model has sufficient resolved predictions.

        Args:
            league_key: League identifier.

        Returns:
            tuple[bool, str]: (ready, reason_string).
        """
        count = self._db.get_resolved_prediction_count(league_key)
        if count < MIN_TRAINING_MATCHES:
            return (
                False,
                f"Warming up: {count}/{MIN_TRAINING_MATCHES} resolved matches. "
                f"Using Dixon-Coles statistical model only."
            )
        return (True, "ML model active.")

    def get_training_progress(self, league_key: str) -> dict:
        """
        Return training data accumulation progress for cold-start UI display.

        Args:
            league_key: League identifier.

        Returns:
            dict: count, required, pct (0.0-1.0).
        """
        count = self._db.get_resolved_prediction_count(league_key)
        pct = min(count / MIN_TRAINING_MATCHES, 1.0) if MIN_TRAINING_MATCHES > 0 else 1.0
        return {"count": count, "required": MIN_TRAINING_MATCHES, "pct": round(pct, 4)}

    def __repr__(self) -> str:
        return f"PreMatchPipeline(leagues={ACTIVE_LEAGUE_KEYS})"
