# CONTRACT: utils/feature_engineering.py
# Classes: FeatureEngineer
# Methods: __init__, __repr__, build_prematch_features, _get_ht_form,
#          _get_h2h_stats, _get_days_rest, _get_form_pts,
#          _get_shot_corner_avgs, build_inplay_features,
#          features_to_array, fit_scaler, transform_features

from config import *

import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Constructs pre-match (31) and in-play (42) feature vectors for ML models."""

    def __init__(self, db) -> None:
        """Args: db — DatabaseManager instance."""
        self._db = db
        os.makedirs(MODEL_DIR, exist_ok=True)

    def build_prematch_features(self, home_team_id: str, away_team_id: str,
                                 league_key: str, match_date, dc_model, xg_model) -> dict:
        """Build 31-feature pre-match vector. Returns 0.0 for unavailable features. Never raises."""
        f: dict = {k: 0.0 for k in PREMATCH_FEATURE_NAMES}

        # Features 0-3: HT goals form last 10
        try:
            home_ht = self._get_ht_form(home_team_id, league_key, 10, "home")
            f["home_ht_goals_scored_avg_l10"] = home_ht.get("scored", 0.0)
            f["home_ht_goals_conceded_avg_l10"] = home_ht.get("conceded", 0.0)
        except Exception as exc:
            logger.debug("home HT form error: %s", exc)
        try:
            away_ht = self._get_ht_form(away_team_id, league_key, 10, "away")
            f["away_ht_goals_scored_avg_l10"] = away_ht.get("scored", 0.0)
            f["away_ht_goals_conceded_avg_l10"] = away_ht.get("conceded", 0.0)
        except Exception as exc:
            logger.debug("away HT form error: %s", exc)

        # Features 4-7: H2H
        try:
            h2h = self._get_h2h_stats(home_team_id, away_team_id, league_key)
            f["h2h_ht_goals_avg"] = h2h.get("avg_total", 0.0)
            f["h2h_ht_over05_rate"] = h2h.get("over05_rate", 0.0)
            f["h2h_ht_over15_rate"] = h2h.get("over15_rate", 0.0)
            f["h2h_ht_over25_rate"] = h2h.get("over25_rate", 0.0)
        except Exception as exc:
            logger.debug("H2H stats error: %s", exc)

        # Features 8-10: League averages
        lc = LEAGUES.get(league_key, {})
        f["league_avg_ht_goals_over_05"] = float(lc.get("avg_ht_goals_over_05", 0.0))
        f["league_avg_ht_goals_over_15"] = float(lc.get("avg_ht_goals_over_15", 0.0))
        f["league_avg_ht_goals_over_25"] = float(lc.get("avg_ht_goals_over_25", 0.0))

        # Features 11-13: Synthetic xG
        try:
            if xg_model is not None and xg_model.is_fitted():
                home_sc = self._get_shot_corner_avgs(home_team_id, league_key, 10)
                away_sc = self._get_shot_corner_avgs(away_team_id, league_key, 10)
                home_xg = xg_model.predict(home_sc.get("shots_on_target", 0.0),
                                           home_sc.get("dangerous_attacks", 0.0),
                                           home_sc.get("corners", 0.0),
                                           home_sc.get("possession_pct", 50.0))
                away_xg = xg_model.predict(away_sc.get("shots_on_target", 0.0),
                                           away_sc.get("dangerous_attacks", 0.0),
                                           away_sc.get("corners", 0.0),
                                           away_sc.get("possession_pct", 50.0))
                f["home_xg_synthetic"] = float(home_xg)
                f["away_xg_synthetic"] = float(away_xg)
                f["xg_differential"] = float(home_xg - away_xg)
        except Exception as exc:
            logger.debug("Synthetic xG error: %s", exc)

        # Features 14-18: DC team params
        try:
            if dc_model is not None and dc_model.is_fitted():
                params_df = dc_model.get_team_params()
                if not params_df.empty:
                    hr = params_df[params_df["team_id"] == home_team_id]
                    ar = params_df[params_df["team_id"] == away_team_id]
                    if not hr.empty:
                        f["home_attack_param"] = float(hr.iloc[0].get("attack_param", 0.0))
                        f["home_defense_param"] = float(hr.iloc[0].get("defense_param", 0.0))
                        f["home_advantage_param"] = float(hr.iloc[0].get("home_advantage", 0.0))
                    if not ar.empty:
                        f["away_attack_param"] = float(ar.iloc[0].get("attack_param", 0.0))
                        f["away_defense_param"] = float(ar.iloc[0].get("defense_param", 0.0))
        except Exception as exc:
            logger.debug("DC team params error: %s", exc)

        # Features 19-21: DC HT probs
        try:
            if dc_model is not None and dc_model.is_fitted():
                dc_probs = dc_model.predict_ht_over_under(home_team_id, away_team_id)
                f["dc_ht_over05_prob"] = float(dc_probs.get("HT_over_0.5", 0.0))
                f["dc_ht_over15_prob"] = float(dc_probs.get("HT_over_1.5", 0.0))
                f["dc_ht_over25_prob"] = float(dc_probs.get("HT_over_2.5", 0.0))
        except Exception as exc:
            logger.debug("DC HT probs error: %s", exc)

        # Features 22-23: Days rest
        try:
            f["home_days_rest"] = self._get_days_rest(home_team_id, league_key, match_date)
        except Exception as exc:
            logger.debug("home days rest error: %s", exc)
        try:
            f["away_days_rest"] = self._get_days_rest(away_team_id, league_key, match_date)
        except Exception as exc:
            logger.debug("away days rest error: %s", exc)

        # Features 24-25: Form pts/game last 5
        try:
            f["home_form_pts_per_game_l5"] = self._get_form_pts(home_team_id, league_key, 5, match_date)
        except Exception as exc:
            logger.debug("home form pts error: %s", exc)
        try:
            f["away_form_pts_per_game_l5"] = self._get_form_pts(away_team_id, league_key, 5, match_date)
        except Exception as exc:
            logger.debug("away form pts error: %s", exc)

        # Features 26-29: Shots and corners
        try:
            hsc = self._get_shot_corner_avgs(home_team_id, league_key, 10)
            f["home_shots_on_target_avg_l10"] = hsc.get("shots_on_target", 0.0)
            f["home_corners_avg_l10"] = hsc.get("corners", 0.0)
        except Exception as exc:
            logger.debug("home shots/corners error: %s", exc)
        try:
            asc = self._get_shot_corner_avgs(away_team_id, league_key, 10)
            f["away_shots_on_target_avg_l10"] = asc.get("shots_on_target", 0.0)
            f["away_corners_avg_l10"] = asc.get("corners", 0.0)
        except Exception as exc:
            logger.debug("away shots/corners error: %s", exc)

        # Feature 30: Day of week
        try:
            f["day_of_week"] = float(match_date.weekday())
        except Exception:
            f["day_of_week"] = 0.0

        return f

    def _get_ht_form(self, team_id: str, league_key: str, last_n: int, home_away: str) -> dict:
        """Compute average HT goals scored/conceded for a team."""
        default = {"scored": 0.0, "conceded": 0.0}
        try:
            df = self._db.get_training_samples(league_key, "HT_over_1.5", limit=500)
            if df.empty:
                return default
            if home_away == "home":
                if "home_team" in df.columns:
                    mask = df["home_team"].astype(str).str.lower().str.contains(team_id[:6].lower(), na=False)
                else:
                    return default
                scored_col, conceded_col = "actual_ht_home", "actual_ht_away"
            else:
                if "away_team" in df.columns:
                    mask = df["away_team"].astype(str).str.lower().str.contains(team_id[:6].lower(), na=False)
                else:
                    return default
                scored_col, conceded_col = "actual_ht_away", "actual_ht_home"
            filtered = df[mask].tail(last_n)
            if filtered.empty or scored_col not in filtered.columns:
                return default
            scored = filtered[scored_col].dropna().astype(float).mean()
            conceded = filtered[conceded_col].dropna().astype(float).mean()
            return {"scored": round(float(scored), 4) if not np.isnan(scored) else 0.0,
                    "conceded": round(float(conceded), 4) if not np.isnan(conceded) else 0.0}
        except Exception as exc:
            logger.debug("_get_ht_form error: %s", exc)
            return default

    def _get_h2h_stats(self, home_team_id: str, away_team_id: str, league_key: str) -> dict:
        """Compute H2H halftime goal statistics."""
        default = {"avg_total": 0.0, "over05_rate": 0.0, "over15_rate": 0.0, "over25_rate": 0.0}
        try:
            df = self._db.get_training_samples(league_key, "HT_over_1.5", limit=1000)
            if df.empty or "home_team" not in df.columns:
                return default
            h_id = home_team_id[:6].lower()
            a_id = away_team_id[:6].lower()
            home_col = df["home_team"].astype(str).str.lower()
            away_col = df["away_team"].astype(str).str.lower()
            mask = ((home_col.str.contains(h_id, na=False) & away_col.str.contains(a_id, na=False)) |
                    (home_col.str.contains(a_id, na=False) & away_col.str.contains(h_id, na=False)))
            h2h_df = df[mask].tail(6)
            if h2h_df.empty or "actual_ht_home" not in h2h_df.columns:
                return default
            totals = h2h_df["actual_ht_home"].fillna(0).astype(float) + h2h_df["actual_ht_away"].fillna(0).astype(float)
            n = len(totals)
            return {"avg_total": round(float(totals.mean()), 4),
                    "over05_rate": round(float((totals > 0).sum() / n), 4),
                    "over15_rate": round(float((totals > 1).sum() / n), 4),
                    "over25_rate": round(float((totals > 2).sum() / n), 4)}
        except Exception as exc:
            logger.debug("_get_h2h_stats error: %s", exc)
            return default

    def _get_days_rest(self, team_id: str, league_key: str, before_date) -> float:
        """Estimate days since team's last match, capped at 30."""
        try:
            df = self._db.get_training_samples(league_key, "HT_over_1.5", limit=500)
            if df.empty or "match_date" not in df.columns:
                return 7.0
            before_str = before_date.strftime("%Y-%m-%d") if hasattr(before_date, "strftime") else str(before_date)[:10]
            team_short = team_id[:6].lower()
            if "home_team" in df.columns and "away_team" in df.columns:
                mask = (df["home_team"].astype(str).str.lower().str.contains(team_short, na=False) |
                        df["away_team"].astype(str).str.lower().str.contains(team_short, na=False))
                team_matches = df[mask]
            else:
                team_matches = df
            date_mask = team_matches["match_date"].astype(str) < before_str
            recent = team_matches[date_mask]
            if recent.empty:
                return 7.0
            last_date_str = recent["match_date"].astype(str).max()
            last_date = datetime.strptime(last_date_str[:10], "%Y-%m-%d")
            fixture_date = before_date if hasattr(before_date, "strftime") else datetime.strptime(str(before_date)[:10], "%Y-%m-%d")
            return float(min(max((fixture_date - last_date).days, 0), 30))
        except Exception as exc:
            logger.debug("_get_days_rest error: %s", exc)
            return 7.0

    def _get_form_pts(self, team_id: str, league_key: str, last_n: int, before_date) -> float:
        """Average points per game over last n matches (3=W, 1=D, 0=L)."""
        try:
            df = self._db.get_training_samples(league_key, "HT_over_1.5", limit=500)
            if df.empty:
                return 1.0
            team_short = team_id[:6].lower()
            before_str = before_date.strftime("%Y-%m-%d") if hasattr(before_date, "strftime") else str(before_date)[:10]
            pts_list = []
            for _, row in df.iterrows():
                if str(row.get("match_date", ""))[:10] >= before_str:
                    continue
                home_name = str(row.get("home_team", "")).lower()
                away_name = str(row.get("away_team", "")).lower()
                ht_home = float(row.get("actual_ht_home") or 0)
                ht_away = float(row.get("actual_ht_away") or 0)
                if team_short in home_name:
                    pts_list.append(3.0 if ht_home > ht_away else 1.0 if ht_home == ht_away else 0.0)
                elif team_short in away_name:
                    pts_list.append(3.0 if ht_away > ht_home else 1.0 if ht_away == ht_home else 0.0)
            if not pts_list:
                return 1.0
            recent = pts_list[-last_n:]
            return round(float(sum(recent) / len(recent)), 4)
        except Exception as exc:
            logger.debug("_get_form_pts error: %s", exc)
            return 1.0

    def _get_shot_corner_avgs(self, team_id: str, league_key: str, last_n: int) -> dict:
        """Return average shots, corners, dangerous attacks, possession for a team."""
        default = {"shots_on_target": 4.5, "corners": 4.5, "dangerous_attacks": 50.0, "possession_pct": 50.0}
        try:
            df = self._db.get_training_samples(league_key, "HT_over_1.5", limit=500)
            if df.empty:
                return default
            team_short = team_id[:6].lower()
            avgs = {}
            for col_key, col_name in [("shots_on_target", "home_shots_on_target_avg_l10"),
                                       ("corners", "home_corners_avg_l10"),
                                       ("dangerous_attacks", "home_dangerous_attacks_avg"),
                                       ("possession_pct", "home_possession_pct")]:
                if col_name in df.columns and "home_team" in df.columns:
                    mask = df["home_team"].astype(str).str.lower().str.contains(team_short, na=False)
                    vals = df[mask][col_name].dropna().tail(last_n)
                    avgs[col_key] = round(float(vals.mean()), 4) if not vals.empty else default[col_key]
                else:
                    avgs[col_key] = default[col_key]
            return avgs
        except Exception as exc:
            logger.debug("_get_shot_corner_avgs error: %s", exc)
            return default

    def build_inplay_features(self, prematch_features: dict, live_data: dict, current_minute: int) -> dict:
        """Build 42-feature in-play vector by merging prematch + live data."""
        f: dict = {k: float(v) for k, v in prematch_features.items()}
        current_ht_home = float(live_data.get("home_score", 0) or 0)
        current_ht_away = float(live_data.get("away_score", 0) or 0)
        live_home_shots = float(live_data.get("home_shots", 0) or 0)
        live_away_shots = float(live_data.get("away_shots", 0) or 0)
        live_home_poss  = float(live_data.get("home_possession", 50) or 50)
        live_home_da    = float(live_data.get("home_dangerous_attacks", 0) or 0)
        live_away_da    = float(live_data.get("away_dangerous_attacks", 0) or 0)

        f["current_minute"]             = float(max(0, min(45, current_minute)))
        f["current_ht_goals_total"]     = current_ht_home + current_ht_away
        f["live_home_shots"]            = live_home_shots
        f["live_away_shots"]            = live_away_shots
        f["live_home_possession_pct"]   = live_home_poss
        f["live_home_dangerous_attacks"] = live_home_da
        f["live_away_dangerous_attacks"] = live_away_da
        f["live_xg_home"]               = round(live_home_shots * 0.10, 4)
        f["live_xg_away"]               = round(live_away_shots * 0.10, 4)
        safe_min = max(1, current_minute)
        f["goals_per_minute_pace"]      = round(f["current_ht_goals_total"] / safe_min, 6)
        f["pressure_index"]             = round((live_home_da + live_away_da + live_home_shots * 2 + live_away_shots * 2) / safe_min, 6)
        return f

    def features_to_array(self, features: dict, feature_count: int) -> np.ndarray:
        """Convert feature dict to ordered (1, feature_count) numpy array."""
        if feature_count == PREMATCH_FEATURE_COUNT:
            name_list = PREMATCH_FEATURE_NAMES
        elif feature_count == INPLAY_FEATURE_COUNT:
            name_list = INPLAY_FEATURE_NAMES
        else:
            raise ValueError(f"feature_count must be {PREMATCH_FEATURE_COUNT} or {INPLAY_FEATURE_COUNT}, got {feature_count}.")
        arr = np.array([float(features.get(n, 0.0) or 0.0) for n in name_list], dtype=np.float64)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.shape[0] != feature_count:
            raise ValueError(f"features_to_array produced {arr.shape[0]}, expected {feature_count}.")
        return arr.reshape(1, -1)

    def fit_scaler(self, feature_matrix: np.ndarray, league_key: str) -> None:
        """Fit StandardScaler on training data and save to MODEL_DIR."""
        try:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(scaler, os.path.join(MODEL_DIR, f"scaler_{league_key}.joblib"))
            logger.info("Scaler fitted for %s (shape=%s)", league_key, feature_matrix.shape)
        except Exception as exc:
            logger.error("fit_scaler(%r) error: %s", league_key, exc)

    def transform_features(self, feature_array: np.ndarray, league_key: str) -> np.ndarray:
        """Apply fitted StandardScaler. Returns raw array if scaler not found."""
        scaler_path = os.path.join(MODEL_DIR, f"scaler_{league_key}.joblib")
        if not os.path.exists(scaler_path):
            logger.debug("Scaler not found for %r — returning raw.", league_key)
            return feature_array
        try:
            return joblib.load(scaler_path).transform(feature_array)
        except Exception as exc:
            logger.error("transform_features(%r) error: %s", league_key, exc)
            return feature_array

    def __repr__(self) -> str:
        return f"FeatureEngineer(prematch={PREMATCH_FEATURE_COUNT}, inplay={INPLAY_FEATURE_COUNT})"
