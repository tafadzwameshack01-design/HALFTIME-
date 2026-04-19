# CONTRACT: models/dixon_coles.py
# Classes: DixonColesModel
# Methods: __init__, __repr__, fit, _negative_log_likelihood, _tau,
#          predict_scoreline_grid, predict_ht_over_under, get_team_params,
#          save, load, is_fitted

from config import *

import logging
import os

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from utils.helpers import InsufficientDataError

logger = logging.getLogger(__name__)


class DixonColesModel:
    """
    Dixon-Coles Poisson goal model re-parameterised for halftime prediction.

    HT lambda = attack_home * defense_away * home_advantage * HT_POISSON_RATE_FACTOR
    Tau correction applied to (0,0),(1,0),(0,1),(1,1) scorelines using DC_RHO.
    MLE optimised with L-BFGS-B and exponential time-decay sample weights.
    """

    def __init__(self, league_key: str) -> None:
        self._league_key = league_key
        self._attack_params: dict = {}
        self._defense_params: dict = {}
        self._home_advantage: float = 1.0
        self._teams: list = []
        self._fitted: bool = False

    def fit(self, match_history: pd.DataFrame) -> None:
        """
        Fit Dixon-Coles parameters via MLE on historical HT scorelines.

        Args:
            match_history: DataFrame with columns — home_team_id, away_team_id,
                home_ht_goals, away_ht_goals, match_date.

        Raises:
            InsufficientDataError: If len(match_history) < DC_MIN_MATCHES.
        """
        if len(match_history) < DC_MIN_MATCHES:
            raise InsufficientDataError(
                f"DixonColesModel requires >= {DC_MIN_MATCHES} matches, "
                f"got {len(match_history)} for {self._league_key}."
            )

        # Drop rows with missing scores
        df = match_history.dropna(subset=["home_ht_goals", "away_ht_goals"]).copy()
        df["home_ht_goals"] = df["home_ht_goals"].astype(int)
        df["away_ht_goals"] = df["away_ht_goals"].astype(int)

        # All unique teams
        self._teams = sorted(
            list(set(df["home_team_id"].tolist() + df["away_team_id"].tolist()))
        )
        n_teams = len(self._teams)
        team_idx = {t: i for i, t in enumerate(self._teams)}

        # Compute time-decay weights
        if "match_date" in df.columns:
            try:
                df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
                today = pd.Timestamp.now()
                df["days_ago"] = (today - df["match_date"]).dt.days.fillna(365).clip(lower=0)
                weights = np.exp(-0.01 * df["days_ago"].values)
            except Exception:
                weights = np.ones(len(df))
        else:
            weights = np.ones(len(df))

        # Initial parameter vector: [attack * n_teams, defense * n_teams, home_advantage]
        # Constrain first defense param = 1.0 (identification)
        init_attack = np.zeros(n_teams)
        init_defense = np.zeros(n_teams)
        init_home = np.log(1.1)
        x0 = np.concatenate([init_attack, init_defense, [init_home]])

        home_idx = df["home_team_id"].map(team_idx).values
        away_idx = df["away_team_id"].map(team_idx).values
        home_goals = df["home_ht_goals"].values
        away_goals = df["away_ht_goals"].values

        def neg_ll(params):
            return self._negative_log_likelihood(
                params, n_teams, home_idx, away_idx, home_goals, away_goals, weights
            )

        result = minimize(
            neg_ll,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )

        # Unpack optimised parameters
        attack = result.x[:n_teams]
        defense = result.x[n_teams: 2 * n_teams]
        home_adv = float(result.x[2 * n_teams])

        # Normalise: subtract mean attack so parameters are identifiable
        attack -= attack.mean()
        defense -= defense.mean()

        self._attack_params = {t: float(np.exp(attack[i])) for i, t in enumerate(self._teams)}
        self._defense_params = {t: float(np.exp(defense[i])) for i, t in enumerate(self._teams)}
        self._home_advantage = float(np.exp(home_adv))
        self._fitted = True
        logger.info("DixonColesModel fitted for %r: %d teams, %d matches.", self._league_key, n_teams, len(df))
        self.save()

    def _negative_log_likelihood(
        self, params: np.ndarray, n_teams: int,
        home_idx: np.ndarray, away_idx: np.ndarray,
        home_goals: np.ndarray, away_goals: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute weighted negative log-likelihood of the Dixon-Coles model.

        Args:
            params: Flattened parameter vector [attack * n, defense * n, home_adv].
            n_teams: Number of teams.
            home_idx: Array of home team indices.
            away_idx: Array of away team indices.
            home_goals: Array of observed home HT goals.
            away_goals: Array of observed away HT goals.
            weights: Per-match time-decay weights.

        Returns:
            float: Negative log-likelihood (to be minimised).
        """
        attack  = params[:n_teams]
        defense = params[n_teams: 2 * n_teams]
        home_adv = params[2 * n_teams]

        total_ll = 0.0
        for i in range(len(home_goals)):
            lam = np.exp(attack[home_idx[i]] + defense[away_idx[i]] + home_adv) * HT_POISSON_RATE_FACTOR
            mu  = np.exp(attack[away_idx[i]] + defense[home_idx[i]]) * HT_POISSON_RATE_FACTOR

            # Clamp to avoid log(0)
            lam = max(lam, 1e-6)
            mu  = max(mu, 1e-6)

            hg = int(home_goals[i])
            ag = int(away_goals[i])

            ll = (
                poisson.logpmf(hg, lam)
                + poisson.logpmf(ag, mu)
                + np.log(max(self._tau(hg, ag, lam, mu), 1e-10))
            )
            total_ll += weights[i] * ll

        return -total_ll

    def _tau(self, x: int, y: int, lam: float, mu: float) -> float:
        """
        Dixon-Coles low-score correction factor (tau).

        Args:
            x: Home goals.
            y: Away goals.
            lam: Home expected goals (HT-adjusted).
            mu: Away expected goals (HT-adjusted).

        Returns:
            float: Correction multiplier. 1.0 for scores outside DC_TAU_SCORELINES.
        """
        rho = DC_RHO
        if (x, y) == (0, 0):
            return 1.0 - lam * mu * rho
        elif (x, y) == (1, 0):
            return 1.0 + mu * rho
        elif (x, y) == (0, 1):
            return 1.0 + lam * rho
        elif (x, y) == (1, 1):
            return 1.0 - rho
        return 1.0

    def predict_scoreline_grid(self, home_team_id: str, away_team_id: str) -> np.ndarray:
        """
        Compute the HT scoreline probability matrix.

        Args:
            home_team_id: Home team identifier.
            away_team_id: Away team identifier.

        Returns:
            np.ndarray: Shape (DC_MAX_GOALS+1, DC_MAX_GOALS+1). grid[i][j] =
                P(home_ht_goals=i, away_ht_goals=j). Falls back to league-average
                Poisson if either team is unknown.
        """
        if not self._fitted:
            return self._fallback_grid()

        # Use league averages for unknown teams
        home_attack  = self._attack_params.get(home_team_id, float(np.mean(list(self._attack_params.values()))) if self._attack_params else 1.0)
        home_defense = self._defense_params.get(home_team_id, float(np.mean(list(self._defense_params.values()))) if self._defense_params else 1.0)
        away_attack  = self._attack_params.get(away_team_id, float(np.mean(list(self._attack_params.values()))) if self._attack_params else 1.0)
        away_defense = self._defense_params.get(away_team_id, float(np.mean(list(self._defense_params.values()))) if self._defense_params else 1.0)

        lam = home_attack * away_defense * self._home_advantage * HT_POISSON_RATE_FACTOR
        mu  = away_attack * home_defense * HT_POISSON_RATE_FACTOR

        lam = max(lam, 1e-6)
        mu  = max(mu, 1e-6)

        grid = np.zeros((DC_MAX_GOALS + 1, DC_MAX_GOALS + 1))
        for i in range(DC_MAX_GOALS + 1):
            for j in range(DC_MAX_GOALS + 1):
                grid[i][j] = (
                    poisson.pmf(i, lam)
                    * poisson.pmf(j, mu)
                    * self._tau(i, j, lam, mu)
                )

        # Normalise so probabilities sum to 1
        total = grid.sum()
        if total > 0:
            grid /= total

        return grid

    def _fallback_grid(self) -> np.ndarray:
        """Return a uniform Poisson grid using league-average rates."""
        lc = LEAGUES.get(self._league_key, {})
        avg = lc.get("avg_ht_goals_over_05", 0.85)
        lam = mu = avg / 2.0 * HT_POISSON_RATE_FACTOR
        grid = np.zeros((DC_MAX_GOALS + 1, DC_MAX_GOALS + 1))
        for i in range(DC_MAX_GOALS + 1):
            for j in range(DC_MAX_GOALS + 1):
                grid[i][j] = poisson.pmf(i, lam) * poisson.pmf(j, mu)
        total = grid.sum()
        if total > 0:
            grid /= total
        return grid

    def predict_ht_over_under(self, home_team_id: str, away_team_id: str) -> dict:
        """
        Derive HT Over/Under probabilities from the scoreline grid.

        Args:
            home_team_id: Home team identifier.
            away_team_id: Away team identifier.

        Returns:
            dict: Keys for all 6 MARKETS with probability floats.
        """
        grid = self.predict_scoreline_grid(home_team_id, away_team_id)
        probs = {}
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                total = i + j
                p = float(grid[i][j])
                if total > 0:
                    probs["HT_over_0.5"]  = probs.get("HT_over_0.5", 0.0) + p
                if total > 1:
                    probs["HT_over_1.5"]  = probs.get("HT_over_1.5", 0.0) + p
                if total > 2:
                    probs["HT_over_2.5"]  = probs.get("HT_over_2.5", 0.0) + p

        probs["HT_under_0.5"] = round(1.0 - probs.get("HT_over_0.5", 0.0), 6)
        probs["HT_under_1.5"] = round(1.0 - probs.get("HT_over_1.5", 0.0), 6)
        probs["HT_under_2.5"] = round(1.0 - probs.get("HT_over_2.5", 0.0), 6)
        probs["HT_over_0.5"]  = round(probs.get("HT_over_0.5", 0.0), 6)
        probs["HT_over_1.5"]  = round(probs.get("HT_over_1.5", 0.0), 6)
        probs["HT_over_2.5"]  = round(probs.get("HT_over_2.5", 0.0), 6)
        return probs

    def get_team_params(self) -> pd.DataFrame:
        """
        Return a DataFrame of all team attack/defense/home_advantage parameters.

        Returns:
            pd.DataFrame: Columns — team_id, attack_param, defense_param, home_advantage.
        """
        if not self._fitted:
            return pd.DataFrame()
        records = [
            {
                "team_id": t,
                "attack_param": self._attack_params.get(t, 1.0),
                "defense_param": self._defense_params.get(t, 1.0),
                "home_advantage": self._home_advantage,
            }
            for t in self._teams
        ]
        return pd.DataFrame(records)

    def save(self) -> None:
        """Save fitted model to MODEL_DIR/dc_{league_key}.joblib."""
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            path = os.path.join(MODEL_DIR, f"dc_{self._league_key}.joblib")
            joblib.dump({
                "attack_params": self._attack_params,
                "defense_params": self._defense_params,
                "home_advantage": self._home_advantage,
                "teams": self._teams,
                "fitted": self._fitted,
            }, path)
            logger.info("DixonColesModel saved: %s", path)
        except Exception as exc:
            logger.error("DixonColesModel.save() error: %s", exc)

    def load(self) -> bool:
        """
        Load a previously saved model from disk.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        path = os.path.join(MODEL_DIR, f"dc_{self._league_key}.joblib")
        if not os.path.exists(path):
            return False
        try:
            state = joblib.load(path)
            self._attack_params  = state["attack_params"]
            self._defense_params = state["defense_params"]
            self._home_advantage = state["home_advantage"]
            self._teams          = state["teams"]
            self._fitted         = state["fitted"]
            logger.info("DixonColesModel loaded: %s", path)
            return True
        except Exception as exc:
            logger.error("DixonColesModel.load() error: %s", exc)
            return False

    def is_fitted(self) -> bool:
        """Return True if the model has been fitted."""
        return self._fitted

    def __repr__(self) -> str:
        return (
            f"DixonColesModel(league={self._league_key!r}, "
            f"fitted={self._fitted}, teams={len(self._teams)})"
        )
