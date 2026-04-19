# CONTRACT: config.py
# Single source of truth. All constants, paths, and league definitions.
# No classes. No functions. Only constants.

import os
from dotenv import load_dotenv

load_dotenv()

# Base directory — absolute, Streamlit-Cloud-safe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Storage ───────────────────────────────────────────────────────────────────
STORAGE_MODE = "supabase" if os.getenv("SUPABASE_URL") else "sqlite"
SQLITE_PATH = os.path.join(BASE_DIR, "data", "predictions.db")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "")
MODEL_DIR = os.path.join(BASE_DIR, "models", "saved")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "data")
LOG_FILE = os.path.join(BASE_DIR, "data", "app.log")
RESOLVER_LOG = os.path.join(BASE_DIR, "data", "resolver.log")
SGD_STATE_PATH = os.path.join(BASE_DIR, "models", "saved", "sgd_state.joblib")

# ── API Keys ──────────────────────────────────────────────────────────────────
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY", "")
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ── League Registry ───────────────────────────────────────────────────────────
LEAGUES = {
    "bundesliga": {
        "name": "Bundesliga",
        "country": "Germany",
        "espn_slug": "ger.1",
        "api_football_id": 78,
        "api_football_season": 2024,
        "football_data_code": "BL1",
        "openligadb_shortcut": "bl1",
        "openligadb_season": 2024,
        "understat_name": "Bundesliga",
        "fbref_url_segment": "20-Bundesliga",
        "timezone": "Europe/Berlin",
        "avg_ht_goals_over_05": 0.847,
        "avg_ht_goals_over_15": 0.421,
        "avg_ht_goals_over_25": 0.142,
        "active": True,
        "flag": "🇩🇪",
    },
    "eredivisie": {
        "name": "Eredivisie",
        "country": "Netherlands",
        "espn_slug": "ned.1",
        "api_football_id": 88,
        "api_football_season": 2024,
        "football_data_code": "DED",
        "openligadb_shortcut": None,
        "openligadb_season": None,
        "understat_name": "Eredivisie",
        "fbref_url_segment": "23-Eredivisie",
        "timezone": "Europe/Amsterdam",
        "avg_ht_goals_over_05": 0.871,
        "avg_ht_goals_over_15": 0.448,
        "avg_ht_goals_over_25": 0.156,
        "active": True,
        "flag": "🇳🇱",
    },
    "belgian_pro": {
        "name": "Belgian Pro League",
        "country": "Belgium",
        "espn_slug": "bel.1",
        "api_football_id": 144,
        "api_football_season": 2024,
        "football_data_code": "BSA",
        "openligadb_shortcut": None,
        "openligadb_season": None,
        "understat_name": None,
        "fbref_url_segment": "37-Belgian-First-Division-A",
        "timezone": "Europe/Brussels",
        "avg_ht_goals_over_05": 0.831,
        "avg_ht_goals_over_15": 0.398,
        "avg_ht_goals_over_25": 0.129,
        "active": True,
        "flag": "🇧🇪",
    },
    "super_lig": {
        "name": "Süper Lig",
        "country": "Turkey",
        "espn_slug": "tur.1",
        "api_football_id": 203,
        "api_football_season": 2024,
        "football_data_code": None,
        "openligadb_shortcut": None,
        "openligadb_season": None,
        "understat_name": None,
        "fbref_url_segment": "26-Super-Lig",
        "timezone": "Europe/Istanbul",
        "avg_ht_goals_over_05": 0.812,
        "avg_ht_goals_over_15": 0.379,
        "avg_ht_goals_over_25": 0.118,
        "active": True,
        "flag": "🇹🇷",
    },
    "championship": {
        "name": "EFL Championship",
        "country": "England",
        "espn_slug": "eng.2",
        "api_football_id": 40,
        "api_football_season": 2024,
        "football_data_code": "ELC",
        "openligadb_shortcut": None,
        "openligadb_season": None,
        "understat_name": None,
        "fbref_url_segment": "10-Championship",
        "timezone": "Europe/London",
        "avg_ht_goals_over_05": 0.798,
        "avg_ht_goals_over_15": 0.361,
        "avg_ht_goals_over_25": 0.109,
        "active": True,
        "flag": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    },
}

ACTIVE_LEAGUE_KEYS: list = [k for k, v in LEAGUES.items() if v["active"]]

# ── Markets ───────────────────────────────────────────────────────────────────
MARKETS: list = [
    "HT_over_0.5",
    "HT_under_0.5",
    "HT_over_1.5",
    "HT_under_1.5",
    "HT_over_2.5",
    "HT_under_2.5",
]

CONFIDENCE_THRESHOLDS: dict = {
    "HT_over_0.5":  0.88,
    "HT_under_0.5": 0.75,
    "HT_over_1.5":  0.75,
    "HT_under_1.5": 0.75,
    "HT_over_2.5":  0.75,
    "HT_under_2.5": 0.80,
}

# ── Dixon-Coles ───────────────────────────────────────────────────────────────
HT_POISSON_RATE_FACTOR: float = 0.38
DC_MAX_GOALS: int = 5
DC_TAU_SCORELINES: list = [(0, 0), (1, 0), (0, 1), (1, 1)]
DC_RHO: float = -0.13
DC_MIN_MATCHES: int = 60

# ── Online Learning ───────────────────────────────────────────────────────────
DECAY_LAMBDA: float = 0.995
RETRAIN_EVERY_N: int = 10
SGD_LEARNING_RATE: float = 0.01
MIN_TRAINING_MATCHES: int = 200
BRIER_WINDOW: int = 50
BRIER_DEGRADATION_THRESHOLD: float = 0.05

# ── Ensemble ──────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHT_DC: float = 0.35
ENSEMBLE_WEIGHT_XGB: float = 0.65
PLATT_MIN_SAMPLES: int = 50

# ── Features ──────────────────────────────────────────────────────────────────
PREMATCH_FEATURE_COUNT: int = 31
INPLAY_FEATURE_COUNT: int = 42

PREMATCH_FEATURE_NAMES: list = [
    "home_ht_goals_scored_avg_l10",
    "away_ht_goals_scored_avg_l10",
    "home_ht_goals_conceded_avg_l10",
    "away_ht_goals_conceded_avg_l10",
    "h2h_ht_goals_avg",
    "h2h_ht_over05_rate",
    "h2h_ht_over15_rate",
    "h2h_ht_over25_rate",
    "league_avg_ht_goals_over_05",
    "league_avg_ht_goals_over_15",
    "league_avg_ht_goals_over_25",
    "home_xg_synthetic",
    "away_xg_synthetic",
    "xg_differential",
    "home_attack_param",
    "away_attack_param",
    "home_defense_param",
    "away_defense_param",
    "home_advantage_param",
    "dc_ht_over05_prob",
    "dc_ht_over15_prob",
    "dc_ht_over25_prob",
    "home_days_rest",
    "away_days_rest",
    "home_form_pts_per_game_l5",
    "away_form_pts_per_game_l5",
    "home_shots_on_target_avg_l10",
    "away_shots_on_target_avg_l10",
    "home_corners_avg_l10",
    "away_corners_avg_l10",
    "day_of_week",
]

INPLAY_FEATURE_NAMES: list = PREMATCH_FEATURE_NAMES + [
    "current_minute",
    "current_ht_goals_total",
    "live_home_shots",
    "live_away_shots",
    "live_home_possession_pct",
    "live_home_dangerous_attacks",
    "live_away_dangerous_attacks",
    "live_xg_home",
    "live_xg_away",
    "goals_per_minute_pace",
    "pressure_index",
]

assert len(PREMATCH_FEATURE_NAMES) == PREMATCH_FEATURE_COUNT
assert len(INPLAY_FEATURE_NAMES) == INPLAY_FEATURE_COUNT

# ── Caching ───────────────────────────────────────────────────────────────────
CACHE_TTL_HISTORICAL: int = 21600
CACHE_TTL_LIVE: int = 60
SEED_SEASONS: list = [2022, 2023, 2024]

# ── Scraper ───────────────────────────────────────────────────────────────────
SCRAPER_HEADERS: dict = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}
SCRAPER_DELAY_SECONDS: float = 3.0

# ── API Endpoints ─────────────────────────────────────────────────────────────
ESPN_BASE: str = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_TIMEOUT: int = 10
API_FOOTBALL_BASE: str = "https://v3.football.api-sports.io"
API_FOOTBALL_DAILY_LIMIT: int = 100
FOOTBALL_DATA_BASE: str = "https://api.football-data.org/v4"
OPENLIGADB_BASE: str = "https://api.openligadb.de"
ODDS_API_BASE: str = "https://api.the-odds-api.com/v4"
ODDS_API_MONTHLY_LIMIT: int = 500

# ── Resolver ──────────────────────────────────────────────────────────────────
RESOLVER_INTERVAL_MINUTES: int = 30
RESOLVER_MATCH_GRACE_HOURS: float = 2.0
RESOLVER_CACHE_KEY: str = "last_resolver_run_ts"

# ── UI Colors ─────────────────────────────────────────────────────────────────
COLOR_BG: str = "#0a0e1a"
COLOR_SURFACE: str = "#111827"
COLOR_BORDER: str = "#1f2937"
COLOR_HIGH: str = "#00ff88"
COLOR_MEDIUM: str = "#ffaa00"
COLOR_NO_BET: str = "#ef4444"
COLOR_TEXT: str = "#f9fafb"
COLOR_MUTED: str = "#6b7280"
COLOR_ACCENT: str = "#3b82f6"

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGB_PARAMS: dict = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "early_stopping_rounds": 30,
    "random_state": 42,
    "use_label_encoder": False,
}
XGB_EVAL_SPLIT: float = 0.10
