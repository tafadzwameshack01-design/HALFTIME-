# CONTRACT: models/ensemble.py
# Classes: EnsemblePredictor
# Methods: __init__, __repr__, predict, _platt_scale,
#          _assign_confidence, fit_platt_scaler

from config import *

import logging
import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Final prediction layer combining Dixon-Coles, XGBoost, and SGD outputs.

    Steps:
    1. Dixon-Coles HT over/under probabilities (base layer)
    2. XGBoost per-market probabilities (ML layer)
    3. SGD live-adjustment probability
    4. Weighted ensemble (weights from DB — respects UI slider override)
    5. Platt scaling calibration
    6. Confidence assignment and threshold gating
    """

    def __init__(
        self,
        league_key: str,
        dc_model,
        xgb_model,
        online_learner,
        fe,
        db,
    ) -> None:
        """
        Args:
            league_key: League identifier.
            dc_model: DixonColesModel instance.
            xgb_model: XGBHalfTimeClassifier instance.
            online_learner: OnlineLearner instance.
            fe: FeatureEngineer instance.
            db: DatabaseManager instance.
        """
        self._league_key = league_key
        self._dc = dc_model
        self._xgb = xgb_model
        self._ol = online_learner
        self._fe = fe
        self._db = db

    def predict(
        self,
        home_team_id: str,
        away_team_id: str,
        feature_array: np.ndarray,
        pipeline_type: str = "prematch",
    ) -> dict:
        """
        Produce a full ensemble prediction for all 6 markets.

        Args:
            home_team_id: Home team identifier.
            away_team_id: Away team identifier.
            feature_array: Pre-normalised shape (1, n_features) numpy array.
            pipeline_type: 'prematch' or 'inplay'.

        Returns:
            dict: Keyed by market. Each value is a dict:
                {dixon_coles_prob, xgb_prob, sgd_adjustment, raw_ensemble_prob,
                 calibrated_prob, confidence_label, should_predict, threshold}
        """
        # 1. Dixon-Coles probabilities
        try:
            dc_probs = self._dc.predict_ht_over_under(home_team_id, away_team_id)
        except Exception as exc:
            logger.warning("EnsemblePredictor DC predict error: %s", exc)
            dc_probs = {m: 0.5 for m in MARKETS}

        # 2. XGBoost probabilities
        try:
            xgb_probs = self._xgb.predict_proba(feature_array) if self._xgb.is_fitted() else {m: 0.5 for m in MARKETS}
        except Exception as exc:
            logger.warning("EnsemblePredictor XGB predict error: %s", exc)
            xgb_probs = {m: 0.5 for m in MARKETS}

        # 3. SGD adjustments
        sgd_adjustments = {}
        for market in MARKETS:
            try:
                sgd_adjustments[market] = self._ol.get_sgd_adjustment(feature_array, market)
            except Exception:
                sgd_adjustments[market] = 0.5

        # 4. Get active ensemble weights (respects UI override)
        try:
            weight_dc, weight_xgb = self._db.get_active_ensemble_weights()
        except Exception:
            weight_dc, weight_xgb = ENSEMBLE_WEIGHT_DC, ENSEMBLE_WEIGHT_XGB

        results = {}
        for market in MARKETS:
            dc_p   = float(dc_probs.get(market, 0.5))
            xgb_p  = float(xgb_probs.get(market, 0.5))
            sgd_p  = float(sgd_adjustments.get(market, 0.5))

            # 5. Raw weighted ensemble
            if self._xgb.is_fitted():
                raw = weight_dc * dc_p + weight_xgb * xgb_p
            else:
                # Cold start: DC only
                raw = dc_p

            # Clip to valid probability range
            raw = float(np.clip(raw, 0.0, 1.0))

            # 6. Platt calibration
            calibrated = self._platt_scale(raw, market)

            # 7. Confidence assignment
            label, should_predict = self._assign_confidence(calibrated, market)

            results[market] = {
                "dixon_coles_prob":  round(dc_p, 6),
                "xgb_prob":          round(xgb_p, 6),
                "sgd_adjustment":    round(sgd_p, 6),
                "raw_ensemble_prob": round(raw, 6),
                "calibrated_prob":   round(calibrated, 6),
                "confidence_label":  label,
                "should_predict":    should_predict,
                "threshold":         CONFIDENCE_THRESHOLDS.get(market, 0.75),
                "pipeline_type":     pipeline_type,
            }

        return results

    def _platt_scale(self, raw_prob: float, market: str) -> float:
        """
        Apply Platt scaling (logistic regression calibration) to a raw probability.

        Loads the fitted scaler from MODEL_DIR. Returns raw_prob unchanged if:
        - Scaler file not found (cold start)
        - Fewer than PLATT_MIN_SAMPLES were used to fit (unreliable)

        Args:
            raw_prob: Uncalibrated ensemble probability.
            market: Market string.

        Returns:
            float: Calibrated probability in [0, 1].
        """
        safe_market = market.replace(".", "_").replace("/", "_")
        path = os.path.join(MODEL_DIR, f"platt_{self._league_key}_{safe_market}.joblib")
        if not os.path.exists(path):
            return raw_prob
        try:
            state = joblib.load(path)
            if state.get("n_samples", 0) < PLATT_MIN_SAMPLES:
                return raw_prob
            scaler: LogisticRegression = state["model"]
            calibrated = scaler.predict_proba([[raw_prob]])[0][1]
            return float(np.clip(calibrated, 0.0, 1.0))
        except Exception as exc:
            logger.debug("_platt_scale(%r, %r) error: %s", market, self._league_key, exc)
            return raw_prob

    def _assign_confidence(self, prob: float, market: str) -> tuple:
        """
        Assign a confidence label and prediction gate based on threshold.

        Thresholds (market-specific from CONFIDENCE_THRESHOLDS):
        - prob >= threshold + 0.05 → HIGH, should_predict=True
        - prob >= threshold        → MEDIUM, should_predict=True
        - prob < threshold         → NO_BET, should_predict=False

        Args:
            prob: Calibrated probability.
            market: Market string.

        Returns:
            tuple[str, bool]: (confidence_label, should_predict).
        """
        threshold = CONFIDENCE_THRESHOLDS.get(market, 0.75)
        if prob >= threshold + 0.05:
            return ("HIGH", True)
        elif prob >= threshold:
            return ("MEDIUM", True)
        else:
            return ("NO_BET", False)

    def fit_platt_scaler(self, market: str) -> None:
        """
        Fit a Platt scaler using resolved predictions from the database.

        Pulls raw_ensemble_prob and actual_outcome for the market, fits
        a LogisticRegression, and saves to MODEL_DIR.

        Args:
            market: Market string.
        """
        try:
            df = self._db.get_training_samples(self._league_key, market, limit=5000)
            if df.empty or "ensemble_prob" not in df.columns or "actual_outcome" not in df.columns:
                logger.debug("fit_platt_scaler: insufficient data for %r/%r.", self._league_key, market)
                return

            df = df.dropna(subset=["ensemble_prob", "actual_outcome"])
            n = len(df)
            if n < PLATT_MIN_SAMPLES:
                logger.debug("fit_platt_scaler: need %d samples, got %d.", PLATT_MIN_SAMPLES, n)
                return

            X = df["ensemble_prob"].values.reshape(-1, 1)
            y = df["actual_outcome"].astype(int).values

            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)

            safe_market = market.replace(".", "_").replace("/", "_")
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, f"platt_{self._league_key}_{safe_market}.joblib")
            joblib.dump({"model": model, "n_samples": n}, path)
            logger.info("Platt scaler fitted for %r/%r on %d samples.", self._league_key, market, n)

        except Exception as exc:
            logger.error("fit_platt_scaler(%r, %r) error: %s", market, self._league_key, exc)

    def __repr__(self) -> str:
        return (
            f"EnsemblePredictor(league={self._league_key!r}, "
            f"dc_fitted={self._dc.is_fitted()}, "
            f"xgb_fitted={self._xgb.is_fitted()})"
        )
