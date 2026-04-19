# CONTRACT: models/online_learner.py
# Classes: SampleWeightManager, OnlineLearner
# SampleWeightManager methods: __init__, __repr__, add_sample, get_weights,
#   get_sample_count, samples_since_last_retrain, mark_retrained, save, load
# OnlineLearner methods: __init__, __repr__, process_new_result,
#   get_sgd_adjustment, get_rolling_brier_score, force_retrain,
#   save_sgd_state, load_sgd_state

from config import *

import logging
import os

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

from utils.calibration import BrierScorer

logger = logging.getLogger(__name__)


class SampleWeightManager:
    """
    Manages exponential time-decay weights for training samples.

    Weight formula: w_i = DECAY_LAMBDA ^ (N - i)
    where N = total samples, i = sample index (0 = oldest).
    Weights are normalised to sum to 1.
    """

    def __init__(self, league_key: str) -> None:
        self._league_key = league_key
        self._samples: list = []          # list of (features_dict, outcome, market)
        self._retrain_counter: int = 0    # samples added since last retrain

    def add_sample(self, features: dict, outcome: int, market: str) -> None:
        """
        Append a new resolved training sample.

        Args:
            features: Feature dict from FeatureEngineer.
            outcome: Actual binary outcome (0 or 1).
            market: Market string.
        """
        self._samples.append((features, int(outcome), market))
        self._retrain_counter += 1

    def get_weights(self) -> np.ndarray:
        """
        Compute and return normalised decay weights for all samples.

        Returns:
            np.ndarray: Shape (n_samples,). Sums to 1.0.
        """
        n = len(self._samples)
        if n == 0:
            return np.array([])
        indices = np.arange(n)
        weights = DECAY_LAMBDA ** (n - 1 - indices)
        total = weights.sum()
        return weights / total if total > 0 else np.ones(n) / n

    def get_sample_count(self) -> int:
        """Return total number of stored samples."""
        return len(self._samples)

    def samples_since_last_retrain(self) -> int:
        """Return number of samples added since the last retrain event."""
        return self._retrain_counter

    def mark_retrained(self) -> None:
        """Reset the retrain counter after a full XGBoost retrain."""
        self._retrain_counter = 0

    def save(self) -> None:
        """Persist sample store to disk."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, f"swm_{self._league_key}.joblib")
            joblib.dump({"samples": self._samples, "counter": self._retrain_counter}, path)
        except Exception as exc:
            logger.error("SampleWeightManager.save() error: %s", exc)

    def load(self) -> bool:
        """
        Load sample store from disk.

        Returns:
            bool: True if loaded successfully.
        """
        path = os.path.join(MODEL_DIR, f"swm_{self._league_key}.joblib")
        if not os.path.exists(path):
            return False
        try:
            state = joblib.load(path)
            self._samples = state["samples"]
            self._retrain_counter = state["counter"]
            return True
        except Exception as exc:
            logger.error("SampleWeightManager.load() error: %s", exc)
            return False

    def __repr__(self) -> str:
        return (
            f"SampleWeightManager(league={self._league_key!r}, "
            f"samples={len(self._samples)}, since_retrain={self._retrain_counter})"
        )


class OnlineLearner:
    """
    Online learning engine that updates model weights after every resolved match.

    Architecture:
    - SampleWeightManager: tracks all samples with exponential decay weights
    - SGDClassifier (one per market): applies partial_fit() after every result
    - Triggers full XGBoost retrain every RETRAIN_EVERY_N new resolved samples
    - Uses st.cache_data timestamp pattern (no background thread) for Streamlit Cloud safety
    """

    def __init__(
        self,
        league_key: str,
        db,
        xgb_model,
        fe,
    ) -> None:
        """
        Args:
            league_key: League identifier.
            db: DatabaseManager instance.
            xgb_model: XGBHalfTimeClassifier instance.
            fe: FeatureEngineer instance.
        """
        self._league_key = league_key
        self._db = db
        self._xgb = xgb_model
        self._fe = fe

        self._swm = SampleWeightManager(league_key)
        self._swm.load()

        self._brier_scorers: dict = {market: BrierScorer() for market in MARKETS}

        # SGD live-adjustment layer — one per market (SIM-BUG-07 fix: initialise classes)
        self._sgd: dict = {}
        for market in MARKETS:
            sgd = SGDClassifier(
                loss="log_loss",
                learning_rate="constant",
                eta0=SGD_LEARNING_RATE,
                random_state=42,
                max_iter=1,
                warm_start=True,
            )
            # Initialise with a zero-vector so classes=[0,1] are registered
            zero_X = np.zeros((1, PREMATCH_FEATURE_COUNT))
            sgd.partial_fit(zero_X, [0], classes=np.array([0, 1]))
            self._sgd[market] = sgd

        self.load_sgd_state()

    def process_new_result(
        self,
        match_id: str,
        market: str,
        features: dict,
        actual_outcome: int,
    ) -> None:
        """
        Process a newly resolved match result to update online learning state.

        Steps:
        1. Add sample to SampleWeightManager.
        2. SGDClassifier.partial_fit() on normalised features.
        3. Update BrierScorer for this market.
        4. If Brier degraded: log warning and DB event.
        5. If samples_since_last_retrain >= RETRAIN_EVERY_N: trigger force_retrain().
        6. Log update event to DB.

        Args:
            match_id: ESPN match identifier.
            market: Market string (e.g. 'HT_over_1.5').
            features: Pre-match feature dict.
            actual_outcome: Binary outcome (0 or 1).
        """
        try:
            # 1. Store sample with weight
            self._swm.add_sample(features, actual_outcome, market)

            # 2. SGD partial fit on normalised features
            try:
                arr = self._fe.features_to_array(features, PREMATCH_FEATURE_COUNT)
                arr_norm = self._fe.transform_features(arr, self._league_key)
                sgd = self._sgd.get(market)
                if sgd is not None:
                    sgd.partial_fit(arr_norm, [actual_outcome])
            except Exception as exc:
                logger.warning("OnlineLearner SGD partial_fit error: %s", exc)

            # 3. Update Brier scorer
            old_brier = self._brier_scorers[market].rolling_score() if market in self._brier_scorers else 0.25
            if market in self._brier_scorers:
                # Estimate predicted prob from SGD (best available signal pre-retrain)
                try:
                    arr = self._fe.features_to_array(features, PREMATCH_FEATURE_COUNT)
                    arr_norm = self._fe.transform_features(arr, self._league_key)
                    pred_prob = self.get_sgd_adjustment(arr_norm, market)
                except Exception:
                    pred_prob = 0.5
                self._brier_scorers[market].update(pred_prob, actual_outcome)
            new_brier = self._brier_scorers[market].rolling_score() if market in self._brier_scorers else 0.25

            # 4. Check for degradation
            retrained = False
            if self._brier_scorers[market].is_degraded(old_brier):
                logger.warning(
                    "OnlineLearner: Brier score degraded for %r/%r (%.4f -> %.4f).",
                    self._league_key, market, old_brier, new_brier
                )

            # 5. Trigger full retrain if threshold reached
            if self._swm.samples_since_last_retrain() >= RETRAIN_EVERY_N:
                self.force_retrain(self._league_key)
                retrained = True

            # 6. Log to DB
            self._db.log_model_update({
                "league_key": self._league_key,
                "market": market,
                "trigger": "new_result",
                "samples_added": 1,
                "new_brier_score": new_brier,
                "old_brier_score": old_brier,
                "retrained": retrained,
            })

            self._swm.save()
            self.save_sgd_state()

        except Exception as exc:
            logger.error("OnlineLearner.process_new_result() error: %s", exc)

    def get_sgd_adjustment(self, feature_array: np.ndarray, market: str) -> float:
        """
        Return the SGD-adjusted probability for a market.

        Args:
            feature_array: Pre-normalised feature array shape (1, n_features).
            market: Market string.

        Returns:
            float: Probability from SGD model. Returns 0.5 on any failure or
                if the SGD has not yet been updated past its zero-vector init.
        """
        sgd = self._sgd.get(market)
        if sgd is None:
            return 0.5
        try:
            proba = sgd.predict_proba(feature_array)
            return float(proba[0][1])
        except Exception:
            return 0.5

    def get_rolling_brier_score(self, market: str) -> float:
        """
        Return rolling Brier score for a market.

        Args:
            market: Market string.

        Returns:
            float: Current rolling Brier score, or 0.25 baseline if unavailable.
        """
        scorer = self._brier_scorers.get(market)
        if scorer is None:
            return 0.25
        return scorer.rolling_score()

    def force_retrain(self, league_key: str) -> None:
        """
        Trigger a full XGBoost retrain using all stored samples with decay weights.

        Pulls training data from DB, applies current SampleWeightManager weights,
        refits the scaler and XGBHalfTimeClassifier, then marks the retrain.
        Uses DB cache to store retrain timestamp (Streamlit-Cloud-safe — no threads).

        Args:
            league_key: League to retrain.
        """
        try:
            logger.info("OnlineLearner.force_retrain() triggered for %r.", league_key)

            # Pull training samples for each market
            all_X: list = []
            y_dict: dict = {m: [] for m in MARKETS}

            for market in MARKETS:
                df = self._db.get_training_samples(league_key, market, limit=5000)
                if df.empty:
                    continue
                # Build feature arrays from stored feature JSON columns
                feat_cols = [c for c in PREMATCH_FEATURE_NAMES if c in df.columns]
                if not feat_cols:
                    continue
                X_market = df[feat_cols].fillna(0.0).values
                y_market = df["actual_outcome"].fillna(0).astype(int).values
                for i in range(len(X_market)):
                    if len(all_X) <= i:
                        all_X.append(X_market[i])
                    y_dict[market].append(y_market[i])

            if not all_X:
                logger.warning("OnlineLearner.force_retrain: no training data available.")
                return

            X = np.array(all_X)
            weights = self._swm.get_weights()
            # Pad or trim weights to match X
            if len(weights) != len(X):
                weights = np.ones(len(X)) / len(X)

            # Refit scaler
            self._fe.fit_scaler(X, league_key)
            X_norm = self._fe.transform_features(X, league_key)

            # Convert y_dict lists to arrays
            y_arrays = {m: np.array(v) for m, v in y_dict.items() if len(v) == len(X)}

            if y_arrays:
                self._xgb.fit(X_norm, y_arrays, weights)

            self._swm.mark_retrained()

            # Store retrain timestamp in DB cache (Streamlit-Cloud-safe pattern)
            from datetime import datetime, timezone
            self._db.cache_match(
                match_id=f"retrain_{league_key}",
                league_key=league_key,
                source="retrain_ts",
                data={"retrained_at": datetime.now(timezone.utc).isoformat()},
            )

            logger.info("OnlineLearner.force_retrain() complete for %r.", league_key)

        except Exception as exc:
            logger.error("OnlineLearner.force_retrain() error: %s", exc)

    def save_sgd_state(self) -> None:
        """Persist all SGD classifiers to disk."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, f"sgd_{self._league_key}.joblib")
            joblib.dump(self._sgd, path)
        except Exception as exc:
            logger.error("OnlineLearner.save_sgd_state() error: %s", exc)

    def load_sgd_state(self) -> bool:
        """
        Load persisted SGD classifiers from disk.

        Returns:
            bool: True if loaded successfully.
        """
        path = os.path.join(MODEL_DIR, f"sgd_{self._league_key}.joblib")
        if not os.path.exists(path):
            return False
        try:
            loaded = joblib.load(path)
            # Only replace if all expected markets are present
            if all(m in loaded for m in MARKETS):
                self._sgd = loaded
                logger.info("OnlineLearner SGD state loaded for %r.", self._league_key)
                return True
            return False
        except Exception as exc:
            logger.error("OnlineLearner.load_sgd_state() error: %s", exc)
            return False

    def __repr__(self) -> str:
        return (
            f"OnlineLearner(league={self._league_key!r}, "
            f"samples={self._swm.get_sample_count()}, "
            f"since_retrain={self._swm.samples_since_last_retrain()})"
        )
