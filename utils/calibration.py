# CONTRACT: utils/calibration.py
# Classes: BrierScorer, CalibrationCurve

from config import *

import logging
from collections import deque

import numpy as np
import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


class BrierScorer:
    """Rolling Brier score tracker for a single market."""

    def __init__(self) -> None:
        """Initialise with a deque bounded by BRIER_WINDOW."""
        self._history: deque = deque(maxlen=BRIER_WINDOW)

    def update(self, predicted_prob: float, actual_outcome: int) -> None:
        """
        Record a new prediction-outcome pair.

        Args:
            predicted_prob: Calibrated probability in [0, 1].
            actual_outcome: 1 if event occurred, 0 otherwise.
        """
        try:
            self._history.append((float(predicted_prob), int(actual_outcome)))
        except Exception as exc:
            logger.warning("BrierScorer.update error: %s", exc)

    def rolling_score(self, window: int = BRIER_WINDOW) -> float:
        """
        Compute rolling Brier score over the last ``window`` predictions.

        Returns:
            float: Mean squared error between prob and outcome.
                Returns 0.25 (random-classifier baseline) if fewer than
                5 observations are available.
        """
        items = list(self._history)[-window:]
        if len(items) < 5:
            return 0.25  # random baseline
        total = sum((p - o) ** 2 for p, o in items)
        return round(total / len(items), 6)

    def is_degraded(self, baseline: float) -> bool:
        """
        Return True if current rolling score exceeds baseline + threshold.

        Args:
            baseline: Reference Brier score to compare against.

        Returns:
            bool: True if model performance has significantly worsened.
        """
        return self.rolling_score() > baseline + BRIER_DEGRADATION_THRESHOLD

    def to_dict(self) -> dict:
        """Return current scorer state as a serialisable dict."""
        return {
            "history_len": len(self._history),
            "rolling_score": self.rolling_score(),
            "window": BRIER_WINDOW,
        }

    def __repr__(self) -> str:
        return f"BrierScorer(n={len(self._history)}, score={self.rolling_score():.4f})"


class CalibrationCurve:
    """Static methods for computing and plotting the Silver calibration test."""

    @staticmethod
    def compute(predicted_probs: list, actual_outcomes: list, n_bins: int = 10) -> pd.DataFrame:
        """
        Bin predictions and compute actual outcome frequency per bin.

        Args:
            predicted_probs: List of predicted probabilities in [0, 1].
            actual_outcomes: List of 0/1 outcomes.
            n_bins: Number of equal-width bins. Defaults to 10.

        Returns:
            pd.DataFrame: Columns — bin_center (float), actual_freq (float),
                count (int). Rows with count=0 are excluded.
        """
        if not predicted_probs or not actual_outcomes:
            return pd.DataFrame(columns=["bin_center", "actual_freq", "count"])
        try:
            probs = np.array(predicted_probs, dtype=float)
            outcomes = np.array(actual_outcomes, dtype=float)
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            records = []
            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                # Include upper bound only for the last bin
                if i < n_bins - 1:
                    mask = (probs >= lo) & (probs < hi)
                else:
                    mask = (probs >= lo) & (probs <= hi)
                count = int(mask.sum())
                if count == 0:
                    continue
                bin_center = round((lo + hi) / 2, 4)
                actual_freq = round(float(outcomes[mask].mean()), 4)
                records.append({"bin_center": bin_center, "actual_freq": actual_freq, "count": count})
            return pd.DataFrame(records)
        except Exception as exc:
            logger.error("CalibrationCurve.compute error: %s", exc)
            return pd.DataFrame(columns=["bin_center", "actual_freq", "count"])

    @staticmethod
    def plot(calibration_df: pd.DataFrame) -> go.Figure:
        """
        Produce a Plotly calibration curve with a perfect-calibration diagonal.

        Args:
            calibration_df: DataFrame as returned by :meth:`compute`.

        Returns:
            go.Figure: Plotly figure ready for ``st.plotly_chart()``.
        """
        fig = go.Figure()

        # Perfect calibration diagonal
        fig.add_trace(go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            line=dict(color=COLOR_MUTED, dash="dash", width=1),
            name="Perfect calibration",
        ))

        if not calibration_df.empty:
            # Bubble size proportional to prediction count
            max_count = calibration_df["count"].max() if calibration_df["count"].max() > 0 else 1
            sizes = (calibration_df["count"] / max_count * 20 + 6).tolist()

            fig.add_trace(go.Scatter(
                x=calibration_df["bin_center"].tolist(),
                y=calibration_df["actual_freq"].tolist(),
                mode="markers+lines",
                marker=dict(
                    size=sizes,
                    color=COLOR_HIGH,
                    line=dict(color=COLOR_BORDER, width=1),
                ),
                line=dict(color=COLOR_HIGH, width=2),
                name="Model calibration",
                text=[f"n={c}" for c in calibration_df["count"].tolist()],
                hovertemplate="Predicted: %{x:.2f}<br>Actual: %{y:.2f}<br>%{text}<extra></extra>",
            ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=COLOR_SURFACE,
            plot_bgcolor=COLOR_SURFACE,
            title=dict(text="Calibration Curve", font=dict(color=COLOR_TEXT, size=14)),
            xaxis=dict(title="Predicted Probability", range=[0, 1],
                       gridcolor=COLOR_BORDER, color=COLOR_TEXT),
            yaxis=dict(title="Actual Frequency", range=[0, 1],
                       gridcolor=COLOR_BORDER, color=COLOR_TEXT),
            legend=dict(font=dict(color=COLOR_TEXT)),
            margin=dict(l=40, r=20, t=40, b=40),
        )
        return fig
