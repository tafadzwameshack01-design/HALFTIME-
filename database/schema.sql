-- HalfTime Oracle — Database Schema
-- Compatible with both SQLite and PostgreSQL (Supabase).
-- Only cross-compatible types: TEXT, INTEGER, REAL, TIMESTAMP.

CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY,
    match_id            TEXT    NOT NULL,
    home_team           TEXT    NOT NULL,
    away_team           TEXT    NOT NULL,
    league_key          TEXT    NOT NULL,
    match_date          TEXT,
    kickoff_utc         TEXT,
    market              TEXT    NOT NULL,
    predicted_prob      REAL,
    predicted_outcome   INTEGER,
    confidence_label    TEXT,
    dixon_coles_prob    REAL,
    xgb_prob            REAL,
    sgd_adjustment      REAL,
    ensemble_prob       REAL,
    features_json       TEXT,
    actual_ht_home      INTEGER,
    actual_ht_away      INTEGER,
    actual_outcome      INTEGER,
    is_correct          INTEGER,
    created_at          TEXT    DEFAULT (datetime('now')),
    resolved_at         TEXT,
    pipeline_type       TEXT
);

CREATE TABLE IF NOT EXISTS model_performance (
    id                  INTEGER PRIMARY KEY,
    snapshot_date       TEXT,
    league_key          TEXT,
    market              TEXT,
    total_predictions   INTEGER,
    correct_predictions INTEGER,
    accuracy            REAL,
    brier_score         REAL,
    avg_confidence      REAL,
    created_at          TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS api_usage (
    id              INTEGER PRIMARY KEY,
    source_name     TEXT    NOT NULL,
    usage_date      TEXT    NOT NULL,
    call_count      INTEGER DEFAULT 0,
    error_count     INTEGER DEFAULT 0,
    last_call_at    TEXT,
    last_error      TEXT
);

CREATE TABLE IF NOT EXISTS match_cache (
    match_id    TEXT    NOT NULL,
    league_key  TEXT,
    source      TEXT    NOT NULL,
    data_json   TEXT,
    cached_at   TEXT    DEFAULT (datetime('now')),
    PRIMARY KEY (match_id, source)
);

CREATE TABLE IF NOT EXISTS feature_importance_log (
    id              INTEGER PRIMARY KEY,
    logged_at       TEXT    DEFAULT (datetime('now')),
    league_key      TEXT,
    market          TEXT,
    importance_json TEXT
);

CREATE TABLE IF NOT EXISTS online_learning_log (
    id                  INTEGER PRIMARY KEY,
    updated_at          TEXT    DEFAULT (datetime('now')),
    league_key          TEXT,
    market              TEXT,
    trigger             TEXT,
    samples_added       INTEGER,
    new_brier_score     REAL,
    old_brier_score     REAL,
    retrained           INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS team_ratings (
    id              INTEGER PRIMARY KEY,
    team_id         TEXT,
    team_name       TEXT,
    league_key      TEXT,
    attack_param    REAL,
    defense_param   REAL,
    home_advantage  REAL,
    last_updated    TEXT    DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ensemble_weights_override (
    id          INTEGER PRIMARY KEY,
    set_at      TEXT    DEFAULT (datetime('now')),
    weight_dc   REAL    NOT NULL,
    weight_xgb  REAL    NOT NULL,
    is_active   INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_predictions_league    ON predictions(league_key);
CREATE INDEX IF NOT EXISTS idx_predictions_market    ON predictions(market);
CREATE INDEX IF NOT EXISTS idx_predictions_match     ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_resolved  ON predictions(resolved_at);
CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence_label);
CREATE INDEX IF NOT EXISTS idx_cache_cached_at       ON match_cache(cached_at);
CREATE INDEX IF NOT EXISTS idx_api_usage_source_date ON api_usage(source_name, usage_date);
CREATE INDEX IF NOT EXISTS idx_team_ratings_league   ON team_ratings(league_key);
CREATE INDEX IF NOT EXISTS idx_feature_importance_lm ON feature_importance_log(league_key, market, logged_at);
CREATE INDEX IF NOT EXISTS idx_online_learning_league ON online_learning_log(league_key, updated_at);
