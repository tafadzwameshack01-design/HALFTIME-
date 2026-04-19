# CONTRACT: database/db_manager.py
# Classes: _StorageBackend(ABC), _SQLiteBackend, _SupabaseBackend, DatabaseManager

from config import *

import json
import logging
import os
import sqlite3
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


class _StorageBackend(ABC):
    @abstractmethod
    def log_prediction(self, prediction: dict) -> str: ...
    @abstractmethod
    def resolve_prediction(self, match_id: str, market: str, actual_ht_home: int, actual_ht_away: int) -> None: ...
    @abstractmethod
    def get_pending_predictions(self, older_than_hours: float = 2.0) -> list: ...
    @abstractmethod
    def get_resolved_prediction_count(self, league_key: str) -> int: ...
    @abstractmethod
    def get_training_samples(self, league_key: str, market: str, limit: int = 5000) -> pd.DataFrame: ...
    @abstractmethod
    def get_accuracy_stats(self, league_key: str = None, market: str = None, days: int = 30) -> dict: ...
    @abstractmethod
    def get_brier_score(self, league_key: str, market: str, window: int = 50) -> float: ...
    @abstractmethod
    def get_feature_drift(self, league_key: str, market: str) -> dict: ...
    @abstractmethod
    def cache_match(self, match_id: str, league_key: str, source: str, data: dict) -> None: ...
    @abstractmethod
    def get_cached_match(self, match_id: str, source: str) -> "dict | None": ...
    @abstractmethod
    def is_cache_fresh(self, match_id: str, source: str, ttl_seconds: int) -> bool: ...
    @abstractmethod
    def track_api_call(self, source_name: str, success: bool, error: str = "") -> None: ...
    @abstractmethod
    def get_api_usage(self, source_name: str) -> dict: ...
    @abstractmethod
    def log_model_update(self, event: dict) -> None: ...
    @abstractmethod
    def get_model_update_log(self, limit: int = 20) -> list: ...
    @abstractmethod
    def save_team_ratings(self, ratings: list) -> None: ...
    @abstractmethod
    def get_team_ratings(self, league_key: str) -> pd.DataFrame: ...
    @abstractmethod
    def log_performance_snapshot(self, snapshot: dict) -> None: ...
    @abstractmethod
    def get_performance_history(self, league_key: str, market: str, days: int = 90) -> pd.DataFrame: ...
    @abstractmethod
    def save_ensemble_weights(self, weight_dc: float, weight_xgb: float) -> None: ...
    @abstractmethod
    def get_active_ensemble_weights(self) -> tuple: ...
    @abstractmethod
    def log_feature_importance(self, league_key: str, market: str, importance: dict) -> None: ...
    @abstractmethod
    def get_feature_importance_history(self, league_key: str, market: str, weeks: int = 8) -> list: ...


class _SQLiteBackend(_StorageBackend):
    """Thread-safe SQLite backend using a single persistent connection."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)
        self._conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
        self._conn.row_factory = sqlite3.Row
        self._apply_schema()

    def _apply_schema(self) -> None:
        schema_path = os.path.join(BASE_DIR, "database", "schema.sql")
        try:
            with open(schema_path, "r", encoding="utf-8") as fh:
                sql = fh.read()
            with self._lock:
                self._conn.executescript(sql)
                self._conn.commit()
        except Exception as exc:
            logger.error("Failed to apply schema: %s", exc)

    def _execute_write(self, sql: str, params: tuple = ()) -> "sqlite3.Cursor | None":
        try:
            with self._lock:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
        except Exception as exc:
            logger.error("SQLite write error: %s | SQL: %s", exc, sql[:120])
            return None

    def _execute_read(self, sql: str, params: tuple = ()) -> list:
        try:
            cur = self._conn.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            logger.error("SQLite read error: %s | SQL: %s", exc, sql[:120])
            return []

    def log_prediction(self, prediction: dict) -> str:
        pred_id = prediction.get("id", uuid.uuid4().hex[:16])
        self._execute_write(
            """INSERT OR IGNORE INTO predictions (id,match_id,home_team,away_team,league_key,
               match_date,kickoff_utc,market,predicted_prob,predicted_outcome,confidence_label,
               dixon_coles_prob,xgb_prob,sgd_adjustment,ensemble_prob,features_json,pipeline_type,created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pred_id, prediction.get("match_id"), prediction.get("home_team"),
             prediction.get("away_team"), prediction.get("league_key"),
             prediction.get("match_date"), prediction.get("kickoff_utc"),
             prediction.get("market"), prediction.get("predicted_prob"),
             prediction.get("predicted_outcome"), prediction.get("confidence_label"),
             prediction.get("dixon_coles_prob"), prediction.get("xgb_prob"),
             prediction.get("sgd_adjustment"), prediction.get("ensemble_prob"),
             json.dumps(prediction.get("features", {})),
             prediction.get("pipeline_type", "prematch"),
             datetime.now(timezone.utc).isoformat()),
        )
        return pred_id

    def resolve_prediction(self, match_id: str, market: str, actual_ht_home: int, actual_ht_away: int) -> None:
        total = actual_ht_home + actual_ht_away
        market_outcomes = {
            "HT_over_0.5": 1 if total > 0 else 0, "HT_under_0.5": 1 if total <= 0 else 0,
            "HT_over_1.5": 1 if total > 1 else 0, "HT_under_1.5": 1 if total <= 1 else 0,
            "HT_over_2.5": 1 if total > 2 else 0, "HT_under_2.5": 1 if total <= 2 else 0,
        }
        actual_outcome = market_outcomes.get(market, 0)
        rows = self._execute_read(
            "SELECT predicted_outcome FROM predictions WHERE match_id=? AND market=? AND resolved_at IS NULL",
            (match_id, market))
        if not rows:
            return
        is_correct = 1 if rows[0].get("predicted_outcome", -1) == actual_outcome else 0
        self._execute_write(
            """UPDATE predictions SET actual_ht_home=?,actual_ht_away=?,actual_outcome=?,
               is_correct=?,resolved_at=? WHERE match_id=? AND market=? AND resolved_at IS NULL""",
            (actual_ht_home, actual_ht_away, actual_outcome, is_correct,
             datetime.now(timezone.utc).isoformat(), match_id, market))

    def get_pending_predictions(self, older_than_hours: float = 2.0) -> list:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).isoformat()
        return self._execute_read(
            "SELECT * FROM predictions WHERE resolved_at IS NULL AND kickoff_utc < ? ORDER BY kickoff_utc ASC",
            (cutoff,))

    def get_resolved_prediction_count(self, league_key: str) -> int:
        rows = self._execute_read(
            "SELECT COUNT(*) AS cnt FROM predictions WHERE league_key=? AND resolved_at IS NOT NULL",
            (league_key,))
        return rows[0]["cnt"] if rows else 0

    def get_training_samples(self, league_key: str, market: str, limit: int = 5000) -> pd.DataFrame:
        rows = self._execute_read(
            "SELECT * FROM predictions WHERE league_key=? AND market=? AND resolved_at IS NOT NULL ORDER BY created_at DESC LIMIT ?",
            (league_key, market, limit))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        try:
            features_expanded = df["features_json"].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
            feature_df = pd.json_normalize(features_expanded)
            df = pd.concat([df.drop(columns=["features_json"]), feature_df], axis=1)
        except Exception as exc:
            logger.warning("Could not expand features_json: %s", exc)
        return df

    def get_accuracy_stats(self, league_key: str = None, market: str = None, days: int = 30) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        conditions = ["resolved_at IS NOT NULL", "created_at >= ?"]
        params: list = [cutoff]
        if league_key:
            conditions.append("league_key = ?"); params.append(league_key)
        if market:
            conditions.append("market = ?"); params.append(market)
        rows = self._execute_read(f"SELECT is_correct, confidence_label FROM predictions WHERE {' AND '.join(conditions)}", tuple(params))
        if not rows:
            return {"total": 0, "correct": 0, "accuracy": 0.0, "no_bet_count": 0, "suppression_rate": 0.0}
        total = len(rows)
        correct = sum(1 for r in rows if r.get("is_correct") == 1)
        no_bet = sum(1 for r in rows if r.get("confidence_label") == "NO_BET")
        return {"total": total, "correct": correct,
                "accuracy": round(correct / total, 4) if total else 0.0,
                "no_bet_count": no_bet,
                "suppression_rate": round(no_bet / total, 4) if total else 0.0}

    def get_brier_score(self, league_key: str, market: str, window: int = BRIER_WINDOW) -> float:
        rows = self._execute_read(
            """SELECT predicted_prob, actual_outcome FROM predictions
               WHERE league_key=? AND market=? AND resolved_at IS NOT NULL
               AND predicted_prob IS NOT NULL AND actual_outcome IS NOT NULL
               ORDER BY resolved_at DESC LIMIT ?""",
            (league_key, market, window))
        if len(rows) < 5:
            return 0.25
        return round(sum((r["predicted_prob"] - r["actual_outcome"]) ** 2 for r in rows) / len(rows), 6)

    def get_feature_drift(self, league_key: str, market: str) -> dict:
        rows = self._execute_read(
            "SELECT importance_json FROM feature_importance_log WHERE league_key=? AND market=? ORDER BY logged_at DESC LIMIT 2",
            (league_key, market))
        if len(rows) < 2:
            return {}
        try:
            latest = json.loads(rows[0]["importance_json"])
            previous = json.loads(rows[1]["importance_json"])
            drift = {k: round(latest.get(k, 0.0) - previous.get(k, 0.0), 6)
                     for k in set(list(latest.keys()) + list(previous.keys()))}
            return dict(sorted(drift.items(), key=lambda x: abs(x[1]), reverse=True))
        except Exception as exc:
            logger.warning("get_feature_drift error: %s", exc)
            return {}

    def cache_match(self, match_id: str, league_key: str, source: str, data: dict) -> None:
        self._execute_write(
            """INSERT INTO match_cache (match_id,league_key,source,data_json,cached_at) VALUES (?,?,?,?,?)
               ON CONFLICT(match_id,source) DO UPDATE SET data_json=excluded.data_json,cached_at=excluded.cached_at,league_key=excluded.league_key""",
            (match_id, league_key, source, json.dumps(data), datetime.now(timezone.utc).isoformat()))

    def get_cached_match(self, match_id: str, source: str) -> "dict | None":
        rows = self._execute_read("SELECT data_json FROM match_cache WHERE match_id=? AND source=?", (match_id, source))
        if not rows:
            return None
        try:
            return json.loads(rows[0]["data_json"])
        except Exception:
            return None

    def is_cache_fresh(self, match_id: str, source: str, ttl_seconds: int) -> bool:
        rows = self._execute_read("SELECT cached_at FROM match_cache WHERE match_id=? AND source=?", (match_id, source))
        if not rows:
            return False
        try:
            cached_at = datetime.fromisoformat(rows[0]["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - cached_at).total_seconds() < ttl_seconds
        except Exception:
            return False

    def track_api_call(self, source_name: str, success: bool, error: str = "") -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()
        self._execute_write(
            """INSERT INTO api_usage (source_name,usage_date,call_count,error_count,last_call_at,last_error)
               VALUES (?,?,1,?,?,?)
               ON CONFLICT(source_name,usage_date) DO UPDATE SET
               call_count=call_count+1,error_count=error_count+?,last_call_at=excluded.last_call_at,
               last_error=CASE WHEN excluded.last_error!='' THEN excluded.last_error ELSE last_error END""",
            (source_name, today, 1 if not success else 0, now_iso, error, 1 if not success else 0))

    def get_api_usage(self, source_name: str) -> dict:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._execute_read("SELECT * FROM api_usage WHERE source_name=? AND usage_date=?", (source_name, today))
        if not rows:
            return {"call_count": 0, "error_count": 0, "last_call_at": None, "last_error": ""}
        return rows[0]

    def log_model_update(self, event: dict) -> None:
        self._execute_write(
            """INSERT INTO online_learning_log (updated_at,league_key,market,trigger,samples_added,
               new_brier_score,old_brier_score,retrained) VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.now(timezone.utc).isoformat(), event.get("league_key", ""),
             event.get("market", ""), event.get("trigger", ""),
             event.get("samples_added", 0), event.get("new_brier_score"),
             event.get("old_brier_score"), 1 if event.get("retrained") else 0))

    def get_model_update_log(self, limit: int = 20) -> list:
        return self._execute_read("SELECT * FROM online_learning_log ORDER BY updated_at DESC LIMIT ?", (limit,))

    def save_team_ratings(self, ratings: list) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        for r in ratings:
            self._execute_write(
                """INSERT INTO team_ratings (team_id,team_name,league_key,attack_param,defense_param,home_advantage,last_updated)
                   VALUES (?,?,?,?,?,?,?) ON CONFLICT DO NOTHING""",
                (r.get("team_id",""), r.get("team_name",""), r.get("league_key",""),
                 r.get("attack_param"), r.get("defense_param"), r.get("home_advantage"), now_iso))
            self._execute_write(
                """UPDATE team_ratings SET attack_param=?,defense_param=?,home_advantage=?,last_updated=?
                   WHERE team_id=? AND league_key=?""",
                (r.get("attack_param"), r.get("defense_param"), r.get("home_advantage"),
                 now_iso, r.get("team_id",""), r.get("league_key","")))

    def get_team_ratings(self, league_key: str) -> pd.DataFrame:
        rows = self._execute_read("SELECT * FROM team_ratings WHERE league_key=? ORDER BY team_name", (league_key,))
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def log_performance_snapshot(self, snapshot: dict) -> None:
        self._execute_write(
            """INSERT INTO model_performance (snapshot_date,league_key,market,total_predictions,
               correct_predictions,accuracy,brier_score,avg_confidence,created_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (snapshot.get("snapshot_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
             snapshot.get("league_key",""), snapshot.get("market",""),
             snapshot.get("total_predictions", 0), snapshot.get("correct_predictions", 0),
             snapshot.get("accuracy"), snapshot.get("brier_score"),
             snapshot.get("avg_confidence"), datetime.now(timezone.utc).isoformat()))

    def get_performance_history(self, league_key: str, market: str, days: int = 90) -> pd.DataFrame:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = self._execute_read(
            "SELECT * FROM model_performance WHERE league_key=? AND market=? AND snapshot_date>=? ORDER BY snapshot_date ASC",
            (league_key, market, cutoff))
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def save_ensemble_weights(self, weight_dc: float, weight_xgb: float) -> None:
        with self._lock:
            self._conn.execute("UPDATE ensemble_weights_override SET is_active=0")
            self._conn.execute(
                "INSERT INTO ensemble_weights_override (set_at,weight_dc,weight_xgb,is_active) VALUES (?,?,?,1)",
                (datetime.now(timezone.utc).isoformat(), weight_dc, weight_xgb))
            self._conn.commit()

    def get_active_ensemble_weights(self) -> tuple:
        rows = self._execute_read(
            "SELECT weight_dc,weight_xgb FROM ensemble_weights_override WHERE is_active=1 ORDER BY set_at DESC LIMIT 1")
        if not rows:
            return (ENSEMBLE_WEIGHT_DC, ENSEMBLE_WEIGHT_XGB)
        return (rows[0]["weight_dc"], rows[0]["weight_xgb"])

    def log_feature_importance(self, league_key: str, market: str, importance: dict) -> None:
        self._execute_write(
            "INSERT INTO feature_importance_log (logged_at,league_key,market,importance_json) VALUES (?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), league_key, market, json.dumps(importance)))

    def get_feature_importance_history(self, league_key: str, market: str, weeks: int = 8) -> list:
        rows = self._execute_read(
            "SELECT logged_at,importance_json FROM feature_importance_log WHERE league_key=? AND market=? ORDER BY logged_at DESC LIMIT ?",
            (league_key, market, weeks))
        result = []
        for row in rows:
            try:
                result.append({"logged_at": row["logged_at"], "importance": json.loads(row["importance_json"])})
            except Exception as exc:
                logger.warning("importance parse error: %s", exc)
        return result

    def __repr__(self) -> str:
        return f"_SQLiteBackend(path={SQLITE_PATH!r})"


class _SupabaseBackend(_StorageBackend):
    """Supabase backend using supabase-py v2 API."""

    def __init__(self) -> None:
        try:
            from supabase import create_client
            self._client = create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as exc:
            logger.error("Supabase client init failed: %s", exc)
            raise

    def _insert(self, table: str, data: dict) -> None:
        try:
            self._client.table(table).insert(data).execute()
        except Exception as exc:
            logger.error("Supabase insert error [%s]: %s", table, exc)

    def _upsert(self, table: str, data: dict) -> None:
        try:
            self._client.table(table).upsert(data).execute()
        except Exception as exc:
            logger.error("Supabase upsert error [%s]: %s", table, exc)

    def _select(self, table: str, filters: dict = None, limit: int = None) -> list:
        try:
            query = self._client.table(table).select("*")
            if filters:
                for col, val in filters.items():
                    query = query.eq(col, val)
            if limit:
                query = query.limit(limit)
            resp = query.execute()
            return resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase select error [%s]: %s", table, exc)
            return []

    def log_prediction(self, prediction: dict) -> str:
        pred_id = prediction.get("id", uuid.uuid4().hex[:16])
        self._insert("predictions", {
            "id": pred_id, "match_id": prediction.get("match_id"),
            "home_team": prediction.get("home_team"), "away_team": prediction.get("away_team"),
            "league_key": prediction.get("league_key"), "match_date": prediction.get("match_date"),
            "kickoff_utc": prediction.get("kickoff_utc"), "market": prediction.get("market"),
            "predicted_prob": prediction.get("predicted_prob"),
            "predicted_outcome": prediction.get("predicted_outcome"),
            "confidence_label": prediction.get("confidence_label"),
            "dixon_coles_prob": prediction.get("dixon_coles_prob"),
            "xgb_prob": prediction.get("xgb_prob"), "sgd_adjustment": prediction.get("sgd_adjustment"),
            "ensemble_prob": prediction.get("ensemble_prob"),
            "features_json": json.dumps(prediction.get("features", {})),
            "pipeline_type": prediction.get("pipeline_type", "prematch"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        return pred_id

    def resolve_prediction(self, match_id: str, market: str, actual_ht_home: int, actual_ht_away: int) -> None:
        total = actual_ht_home + actual_ht_away
        market_outcomes = {
            "HT_over_0.5": 1 if total > 0 else 0, "HT_under_0.5": 1 if total <= 0 else 0,
            "HT_over_1.5": 1 if total > 1 else 0, "HT_under_1.5": 1 if total <= 1 else 0,
            "HT_over_2.5": 1 if total > 2 else 0, "HT_under_2.5": 1 if total <= 2 else 0,
        }
        actual_outcome = market_outcomes.get(market, 0)
        try:
            rows = self._select("predictions", {"match_id": match_id, "market": market})
            if not rows:
                return
            is_correct = 1 if rows[0].get("predicted_outcome", -1) == actual_outcome else 0
            (self._client.table("predictions").update({
                "actual_ht_home": actual_ht_home, "actual_ht_away": actual_ht_away,
                "actual_outcome": actual_outcome, "is_correct": is_correct,
                "resolved_at": datetime.now(timezone.utc).isoformat()
            }).eq("match_id", match_id).eq("market", market).is_("resolved_at", "null").execute())
        except Exception as exc:
            logger.error("Supabase resolve_prediction error: %s", exc)

    def get_pending_predictions(self, older_than_hours: float = 2.0) -> list:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=older_than_hours)).isoformat()
        try:
            resp = (self._client.table("predictions").select("*")
                    .is_("resolved_at", "null").lt("kickoff_utc", cutoff).execute())
            return resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_pending_predictions error: %s", exc)
            return []

    def get_resolved_prediction_count(self, league_key: str) -> int:
        try:
            resp = (self._client.table("predictions").select("id", count="exact")
                    .eq("league_key", league_key).not_.is_("resolved_at", "null").execute())
            return resp.count or 0
        except Exception as exc:
            logger.error("Supabase get_resolved_prediction_count error: %s", exc)
            return 0

    def get_training_samples(self, league_key: str, market: str, limit: int = 5000) -> pd.DataFrame:
        try:
            resp = (self._client.table("predictions").select("*")
                    .eq("league_key", league_key).eq("market", market)
                    .not_.is_("resolved_at", "null")
                    .order("created_at", desc=True).limit(limit).execute())
            rows = resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_training_samples error: %s", exc)
            return pd.DataFrame()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        try:
            features_expanded = df["features_json"].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
            feature_df = pd.json_normalize(features_expanded)
            df = pd.concat([df.drop(columns=["features_json"]), feature_df], axis=1)
        except Exception as exc:
            logger.warning("Supabase features_json expand error: %s", exc)
        return df

    def get_accuracy_stats(self, league_key: str = None, market: str = None, days: int = 30) -> dict:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        try:
            query = (self._client.table("predictions").select("is_correct,confidence_label")
                     .not_.is_("resolved_at", "null").gte("created_at", cutoff))
            if league_key:
                query = query.eq("league_key", league_key)
            if market:
                query = query.eq("market", market)
            rows = query.execute().data or []
        except Exception as exc:
            logger.error("Supabase get_accuracy_stats error: %s", exc)
            return {"total": 0, "correct": 0, "accuracy": 0.0, "no_bet_count": 0, "suppression_rate": 0.0}
        total = len(rows)
        correct = sum(1 for r in rows if r.get("is_correct") == 1)
        no_bet = sum(1 for r in rows if r.get("confidence_label") == "NO_BET")
        return {"total": total, "correct": correct,
                "accuracy": round(correct / total, 4) if total else 0.0,
                "no_bet_count": no_bet,
                "suppression_rate": round(no_bet / total, 4) if total else 0.0}

    def get_brier_score(self, league_key: str, market: str, window: int = BRIER_WINDOW) -> float:
        try:
            resp = (self._client.table("predictions")
                    .select("predicted_prob,actual_outcome")
                    .eq("league_key", league_key).eq("market", market)
                    .not_.is_("resolved_at", "null")
                    .order("resolved_at", desc=True).limit(window).execute())
            rows = resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_brier_score error: %s", exc)
            return 0.25
        if len(rows) < 5:
            return 0.25
        return round(sum((r["predicted_prob"] - r["actual_outcome"]) ** 2 for r in rows) / len(rows), 6)

    def get_feature_drift(self, league_key: str, market: str) -> dict:
        try:
            resp = (self._client.table("feature_importance_log").select("importance_json")
                    .eq("league_key", league_key).eq("market", market)
                    .order("logged_at", desc=True).limit(2).execute())
            rows = resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_feature_drift error: %s", exc)
            return {}
        if len(rows) < 2:
            return {}
        try:
            latest = json.loads(rows[0]["importance_json"])
            previous = json.loads(rows[1]["importance_json"])
            drift = {k: round(latest.get(k, 0.0) - previous.get(k, 0.0), 6)
                     for k in set(list(latest.keys()) + list(previous.keys()))}
            return dict(sorted(drift.items(), key=lambda x: abs(x[1]), reverse=True))
        except Exception as exc:
            logger.warning("Supabase get_feature_drift parse error: %s", exc)
            return {}

    def cache_match(self, match_id: str, league_key: str, source: str, data: dict) -> None:
        self._upsert("match_cache", {"match_id": match_id, "league_key": league_key, "source": source,
                                      "data_json": json.dumps(data), "cached_at": datetime.now(timezone.utc).isoformat()})

    def get_cached_match(self, match_id: str, source: str) -> "dict | None":
        rows = self._select("match_cache", {"match_id": match_id, "source": source})
        if not rows:
            return None
        try:
            return json.loads(rows[0]["data_json"])
        except Exception:
            return None

    def is_cache_fresh(self, match_id: str, source: str, ttl_seconds: int) -> bool:
        rows = self._select("match_cache", {"match_id": match_id, "source": source})
        if not rows:
            return False
        try:
            cached_at = datetime.fromisoformat(rows[0]["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - cached_at).total_seconds() < ttl_seconds
        except Exception:
            return False

    def track_api_call(self, source_name: str, success: bool, error: str = "") -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        existing = self._select("api_usage", {"source_name": source_name, "usage_date": today})
        now_iso = datetime.now(timezone.utc).isoformat()
        if existing:
            update = {"call_count": existing[0]["call_count"] + 1, "last_call_at": now_iso}
            if not success:
                update["error_count"] = existing[0]["error_count"] + 1
                update["last_error"] = error
            try:
                self._client.table("api_usage").update(update).eq("id", existing[0]["id"]).execute()
            except Exception as exc:
                logger.error("Supabase track_api_call update error: %s", exc)
        else:
            self._insert("api_usage", {"source_name": source_name, "usage_date": today,
                                        "call_count": 1, "error_count": 1 if not success else 0,
                                        "last_call_at": now_iso, "last_error": error if not success else ""})

    def get_api_usage(self, source_name: str) -> dict:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self._select("api_usage", {"source_name": source_name, "usage_date": today})
        if not rows:
            return {"call_count": 0, "error_count": 0, "last_call_at": None, "last_error": ""}
        return rows[0]

    def log_model_update(self, event: dict) -> None:
        self._insert("online_learning_log", {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "league_key": event.get("league_key", ""), "market": event.get("market", ""),
            "trigger": event.get("trigger", ""), "samples_added": event.get("samples_added", 0),
            "new_brier_score": event.get("new_brier_score"), "old_brier_score": event.get("old_brier_score"),
            "retrained": 1 if event.get("retrained") else 0})

    def get_model_update_log(self, limit: int = 20) -> list:
        try:
            resp = (self._client.table("online_learning_log").select("*")
                    .order("updated_at", desc=True).limit(limit).execute())
            return resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_model_update_log error: %s", exc)
            return []

    def save_team_ratings(self, ratings: list) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        for r in ratings:
            self._upsert("team_ratings", {"team_id": r.get("team_id",""), "team_name": r.get("team_name",""),
                                           "league_key": r.get("league_key",""),
                                           "attack_param": r.get("attack_param"),
                                           "defense_param": r.get("defense_param"),
                                           "home_advantage": r.get("home_advantage"), "last_updated": now_iso})

    def get_team_ratings(self, league_key: str) -> pd.DataFrame:
        rows = self._select("team_ratings", {"league_key": league_key})
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def log_performance_snapshot(self, snapshot: dict) -> None:
        self._insert("model_performance", {
            "snapshot_date": snapshot.get("snapshot_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
            "league_key": snapshot.get("league_key",""), "market": snapshot.get("market",""),
            "total_predictions": snapshot.get("total_predictions", 0),
            "correct_predictions": snapshot.get("correct_predictions", 0),
            "accuracy": snapshot.get("accuracy"), "brier_score": snapshot.get("brier_score"),
            "avg_confidence": snapshot.get("avg_confidence"),
            "created_at": datetime.now(timezone.utc).isoformat()})

    def get_performance_history(self, league_key: str, market: str, days: int = 90) -> pd.DataFrame:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            resp = (self._client.table("model_performance").select("*")
                    .eq("league_key", league_key).eq("market", market)
                    .gte("snapshot_date", cutoff).order("snapshot_date", desc=False).execute())
            rows = resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_performance_history error: %s", exc)
            return pd.DataFrame()
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def save_ensemble_weights(self, weight_dc: float, weight_xgb: float) -> None:
        try:
            self._client.table("ensemble_weights_override").update({"is_active": 0}).eq("is_active", 1).execute()
            self._insert("ensemble_weights_override", {"set_at": datetime.now(timezone.utc).isoformat(),
                                                        "weight_dc": weight_dc, "weight_xgb": weight_xgb, "is_active": 1})
        except Exception as exc:
            logger.error("Supabase save_ensemble_weights error: %s", exc)

    def get_active_ensemble_weights(self) -> tuple:
        try:
            resp = (self._client.table("ensemble_weights_override").select("weight_dc,weight_xgb")
                    .eq("is_active", 1).order("set_at", desc=True).limit(1).execute())
            rows = resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_active_ensemble_weights error: %s", exc)
            return (ENSEMBLE_WEIGHT_DC, ENSEMBLE_WEIGHT_XGB)
        if not rows:
            return (ENSEMBLE_WEIGHT_DC, ENSEMBLE_WEIGHT_XGB)
        return (rows[0]["weight_dc"], rows[0]["weight_xgb"])

    def log_feature_importance(self, league_key: str, market: str, importance: dict) -> None:
        self._insert("feature_importance_log", {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "league_key": league_key, "market": market, "importance_json": json.dumps(importance)})

    def get_feature_importance_history(self, league_key: str, market: str, weeks: int = 8) -> list:
        try:
            resp = (self._client.table("feature_importance_log").select("logged_at,importance_json")
                    .eq("league_key", league_key).eq("market", market)
                    .order("logged_at", desc=True).limit(weeks).execute())
            rows = resp.data if resp.data else []
        except Exception as exc:
            logger.error("Supabase get_feature_importance_history error: %s", exc)
            return []
        result = []
        for row in rows:
            try:
                result.append({"logged_at": row["logged_at"], "importance": json.loads(row["importance_json"])})
            except Exception as exc:
                logger.warning("Supabase importance parse error: %s", exc)
        return result

    def __repr__(self) -> str:
        return f"_SupabaseBackend(url={SUPABASE_URL[:30]!r}...)"


class DatabaseManager:
    """Singleton router delegating to _SQLiteBackend or _SupabaseBackend."""

    _instance: "DatabaseManager | None" = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        if STORAGE_MODE == "supabase" and SUPABASE_URL and SUPABASE_KEY:
            try:
                self._backend = _SupabaseBackend()
                logger.info("DatabaseManager using Supabase backend.")
            except Exception as exc:
                logger.warning("Supabase init failed (%s). Falling back to SQLite.", exc)
                self._backend = _SQLiteBackend()
        else:
            self._backend = _SQLiteBackend()
            logger.info("DatabaseManager using SQLite backend at %s", SQLITE_PATH)

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        """Return (or create) the singleton DatabaseManager instance."""
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def log_prediction(self, prediction: dict) -> str: return self._backend.log_prediction(prediction)
    def resolve_prediction(self, match_id: str, market: str, actual_ht_home: int, actual_ht_away: int) -> None: return self._backend.resolve_prediction(match_id, market, actual_ht_home, actual_ht_away)
    def get_pending_predictions(self, older_than_hours: float = 2.0) -> list: return self._backend.get_pending_predictions(older_than_hours)
    def get_resolved_prediction_count(self, league_key: str) -> int: return self._backend.get_resolved_prediction_count(league_key)
    def get_training_samples(self, league_key: str, market: str, limit: int = 5000) -> pd.DataFrame: return self._backend.get_training_samples(league_key, market, limit)
    def get_accuracy_stats(self, league_key: str = None, market: str = None, days: int = 30) -> dict: return self._backend.get_accuracy_stats(league_key, market, days)
    def get_brier_score(self, league_key: str, market: str, window: int = BRIER_WINDOW) -> float: return self._backend.get_brier_score(league_key, market, window)
    def get_feature_drift(self, league_key: str, market: str) -> dict: return self._backend.get_feature_drift(league_key, market)
    def cache_match(self, match_id: str, league_key: str, source: str, data: dict) -> None: return self._backend.cache_match(match_id, league_key, source, data)
    def get_cached_match(self, match_id: str, source: str) -> "dict | None": return self._backend.get_cached_match(match_id, source)
    def is_cache_fresh(self, match_id: str, source: str, ttl_seconds: int) -> bool: return self._backend.is_cache_fresh(match_id, source, ttl_seconds)
    def track_api_call(self, source_name: str, success: bool, error: str = "") -> None: return self._backend.track_api_call(source_name, success, error)
    def get_api_usage(self, source_name: str) -> dict: return self._backend.get_api_usage(source_name)
    def log_model_update(self, event: dict) -> None: return self._backend.log_model_update(event)
    def get_model_update_log(self, limit: int = 20) -> list: return self._backend.get_model_update_log(limit)
    def save_team_ratings(self, ratings: list) -> None: return self._backend.save_team_ratings(ratings)
    def get_team_ratings(self, league_key: str) -> pd.DataFrame: return self._backend.get_team_ratings(league_key)
    def log_performance_snapshot(self, snapshot: dict) -> None: return self._backend.log_performance_snapshot(snapshot)
    def get_performance_history(self, league_key: str, market: str, days: int = 90) -> pd.DataFrame: return self._backend.get_performance_history(league_key, market, days)
    def save_ensemble_weights(self, weight_dc: float, weight_xgb: float) -> None: return self._backend.save_ensemble_weights(weight_dc, weight_xgb)
    def get_active_ensemble_weights(self) -> tuple: return self._backend.get_active_ensemble_weights()
    def log_feature_importance(self, league_key: str, market: str, importance: dict) -> None: return self._backend.log_feature_importance(league_key, market, importance)
    def get_feature_importance_history(self, league_key: str, market: str, weeks: int = 8) -> list: return self._backend.get_feature_importance_history(league_key, market, weeks)

    def __repr__(self) -> str: return f"DatabaseManager(backend={self._backend!r})"
