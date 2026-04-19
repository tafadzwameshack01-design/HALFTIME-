# CONTRACT: models/xgb_classifier.py
# Classes: XGBHalfTimeClassifier
# Methods: __init__, __repr__, fit, predict_proba,
#          get_feature_importance, save, load, is_fitted

from config import *

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class XGBHalfTimeClassifier:
    """
    XGBoost-based halftime Over/Under classifier.

    Maintains one XGBClassifier per market (6 total).
    Uses XGB_EVAL_SPLIT (10%) internal validation for early stopping.
    Feature arrays must be pre-normalized by the caller (via FeatureEngineer.transform_features).
    """

    def __init__(self, league_key: str) -> None:
        self._league_key = league_key
        self._models: dict = {}      # market -> XGBClassifier
        self._fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y_dict: dict,
        sample_weights: np.ndarray,
    ) -> None:
        """
        Fit one XGBClassifier per market.

        Args:
            X: Feature matrix, shape (n_samples, n_features). Pre-normalized.
            y_dict: {market_name: np.ndarray of 0/1 labels}.
            sample_weights: Decay-weighted array of shape (n_samples,).

        Raises:
            ValueError: If X has fewer than 10 samples (too small for eval split).
        """
        if len(X) < 10:
            raise ValueError(f"XGBHalfTimeClassifier.fit: need >= 10 samples, got {len(X)}.")

        for market, y in y_dict.items():
            try:
                # Internal 90/10 train/val split for early stopping (SIM-BUG-06 fix)
                X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
                    X, y, sample_weights,
                    test_size=XGB_EVAL_SPLIT,
                    random_state=42,
                    stratify=y if len(np.unique(y)) > 1 else None,
                )

                params = dict(XGB_PARAMS)
                model = XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    sample_weight=w_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                self._models[market] = model
                logger.info(
                    "XGBHalfTimeClassifier fitted market=%r for %r on %d samples.",
                    market, self._league_key, len(X_train)
                )
            except Exception as exc:
                logger.error("XGBHalfTimeClassifier.fit market=%r error: %s", market, exc)

        if self._models:
            self._fitted = True
            self.save()

    def predict_proba(self, feature_array: np.ndarray) -> dict:
        """
        Predict probability for each market.

        Args:
            feature_array: Shape (1, n_features). Must be pre-normalized.

        Returns:
            dict: {market: probability_float} for all markets in MARKETS.
                Returns 0.5 for any market whose model is not fitted.
        """
        results = {}
        for market in MARKETS:
            model = self._models.get(market)
            if model is None:
                results[market] = 0.5
                continue
            try:
                proba = model.predict_proba(feature_array)
                # Class 1 probability
                results[market] = float(proba[0][1])
            except Exception as exc:
                logger.warning("XGBHalfTimeClassifier.predict_proba market=%r error: %s", market, exc)
                results[market] = 0.5
        return results

    def get_feature_importance(self, market: str) -> dict:
        """
        Return feature importance scores for a given market.

        Args:
            market: Market string (e.g. 'HT_over_1.5').

        Returns:
            dict: {feature_name: importance_score} sorted descending.
                Empty dict if model not available for this market.
        """
        model = self._models.get(market)
        if model is None:
            return {}
        try:
            scores = model.get_booster().get_score(importance_type="gain")
            # Map f0, f1, ... to actual feature names
            named = {}
            for fkey, score in scores.items():
                try:
                    idx = int(fkey.replace("f", ""))
                    fname = PREMATCH_FEATURE_NAMES[idx] if idx < len(PREMATCH_FEATURE_NAMES) else fkey
                except (ValueError, IndexError):
                    fname = fkey
                named[fname] = round(float(score), 4)
            return dict(sorted(named.items(), key=lambda x: x[1], reverse=True))
        except Exception as exc:
            logger.warning("XGBHalfTimeClassifier.get_feature_importance error: %s", exc)
            return {}

    def save(self) -> None:
        """Save each market model to MODEL_DIR/xgb_{league_key}_{market}.joblib."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            for market, model in self._models.items():
                safe_market = market.replace(".", "_").replace("/", "_")
                path = os.path.join(MODEL_DIR, f"xgb_{self._league_key}_{safe_market}.joblib")
                joblib.dump(model, path)
            logger.info("XGBHalfTimeClassifier saved %d models for %r.", len(self._models), self._league_key)
        except Exception as exc:
            logger.error("XGBHalfTimeClassifier.save() error: %s", exc)

    def load(self) -> bool:
        """
        Load all market models from disk.

        Returns:
            bool: True if at least one model was loaded.
        """
        loaded = 0
        for market in MARKETS:
            safe_market = market.replace(".", "_").replace("/", "_")
            path = os.path.join(MODEL_DIR, f"xgb_{self._league_key}_{safe_market}.joblib")
            if os.path.exists(path):
                try:
                    self._models[market] = joblib.load(path)
                    loaded += 1
                except Exception as exc:
                    logger.warning("XGBHalfTimeClassifier.load market=%r error: %s", market, exc)
        self._fitted = loaded > 0
        if loaded:
            logger.info("XGBHalfTimeClassifier loaded %d models for %r.", loaded, self._league_key)
        return self._fitted

    def is_fitted(self) -> bool:
        """Return True if at least one market model is fitted."""
        return self._fitted

    def __repr__(self) -> str:
        return (
            f"XGBHalfTimeClassifier(league={self._league_key!r}, "
            f"fitted={self._fitted}, markets={list(self._models.keys())})"
        )
