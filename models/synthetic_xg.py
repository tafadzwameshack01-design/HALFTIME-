# CONTRACT: models/synthetic_xg.py
# Classes: SyntheticXGEstimator
# Methods: __init__, __repr__, fit, predict, save, load, is_fitted

from config import *

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class SyntheticXGEstimator:
    """
    Estimates expected goals (xG) from freely available match statistics.

    Features: shots_on_target, dangerous_attacks, corners,
              possession_pct, aerial_duels_won.
    Target: actual_goals (used as proxy for xG during training).
    Output clipped to [0.0, 5.0].
    Falls back to league average if model not fitted.
    """

    def __init__(self, league_key: str) -> None:
        self._league_key = league_key
        self._model: "LinearRegression | None" = None
        self._fitted: bool = False
        self._feature_names: list = [
            "shots_on_target", "dangerous_attacks",
            "corners", "possession_pct", "aerial_duels_won",
        ]

    def fit(self, training_data: pd.DataFrame) -> None:
        """
        Fit a LinearRegression model on historical match statistics.

        Args:
            training_data: DataFrame with columns matching self._feature_names
                plus an 'actual_goals' target column.
        """
        required = self._feature_names + ["actual_goals"]
        missing = [c for c in required if c not in training_data.columns]
        if missing:
            logger.warning("SyntheticXGEstimator.fit: missing columns %s — using defaults.", missing)
            # Fill missing columns with 0
            for col in missing:
                training_data[col] = 0.0

        try:
            X = training_data[self._feature_names].fillna(0.0).values
            y = training_data["actual_goals"].fillna(0.0).values
            self._model = LinearRegression()
            self._model.fit(X, y)
            self._fitted = True
            logger.info("SyntheticXGEstimator fitted for %r on %d samples.", self._league_key, len(y))
            self.save()
        except Exception as exc:
            logger.error("SyntheticXGEstimator.fit() error: %s", exc)

    def predict(
        self,
        shots_on_target: float,
        dangerous_attacks: float,
        corners: float,
        possession_pct: float,
        aerial_duels_won: float = 0.0,
    ) -> float:
        """
        Predict expected goals given match statistics.

        Args:
            shots_on_target: Average shots on target per game.
            dangerous_attacks: Average dangerous attacks per game.
            corners: Average corners per game.
            possession_pct: Average possession percentage.
            aerial_duels_won: Average aerial duels won per game (optional).

        Returns:
            float: Estimated xG clipped to [0.0, 5.0]. Returns league average
                if model is not fitted.
        """
        if not self._fitted or self._model is None:
            # Fall back to league average as a proxy
            lc = LEAGUES.get(self._league_key, {})
            return float(lc.get("avg_ht_goals_over_05", 0.85)) * 0.5

        try:
            X = np.array([[shots_on_target, dangerous_attacks,
                           corners, possession_pct, aerial_duels_won]])
            raw = float(self._model.predict(X)[0])
            return float(np.clip(raw, 0.0, 5.0))
        except Exception as exc:
            logger.warning("SyntheticXGEstimator.predict() error: %s", exc)
            return 0.5

    def save(self) -> None:
        """Save model to MODEL_DIR/synthetic_xg_{league_key}.joblib."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, f"synthetic_xg_{self._league_key}.joblib")
            joblib.dump({"model": self._model, "fitted": self._fitted,
                         "feature_names": self._feature_names}, path)
            logger.info("SyntheticXGEstimator saved: %s", path)
        except Exception as exc:
            logger.error("SyntheticXGEstimator.save() error: %s", exc)

    def load(self) -> bool:
        """
        Load a saved model from disk.

        Returns:
            bool: True if loaded successfully.
        """
        path = os.path.join(MODEL_DIR, f"synthetic_xg_{self._league_key}.joblib")
        if not os.path.exists(path):
            return False
        try:
            state = joblib.load(path)
            self._model = state["model"]
            self._fitted = state["fitted"]
            self._feature_names = state.get("feature_names", self._feature_names)
            logger.info("SyntheticXGEstimator loaded: %s", path)
            return True
        except Exception as exc:
            logger.error("SyntheticXGEstimator.load() error: %s", exc)
            return False

    def is_fitted(self) -> bool:
        """Return True if the model has been fitted."""
        return self._fitted

    def __repr__(self) -> str:
        return (
            f"SyntheticXGEstimator(league={self._league_key!r}, "
            f"fitted={self._fitted})"
        )
