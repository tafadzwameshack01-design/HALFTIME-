import streamlit as st

st.set_page_config(
    page_title="HalfTime Oracle",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import logging
import os
import traceback

from config import *


# ── CSS ───────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject full dark-theme CSS with Google Fonts."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500&display=swap');

    :root {{
        --color-bg:      {COLOR_BG};
        --color-surface: {COLOR_SURFACE};
        --color-border:  {COLOR_BORDER};
        --color-high:    {COLOR_HIGH};
        --color-medium:  {COLOR_MEDIUM};
        --color-nobet:   {COLOR_NO_BET};
        --color-text:    {COLOR_TEXT};
        --color-muted:   {COLOR_MUTED};
        --color-accent:  {COLOR_ACCENT};
    }}

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: var(--color-bg);
        color: var(--color-text);
    }}
    h1, h2, h3, h4 {{ font-family: 'Rajdhani', sans-serif; }}

    section[data-testid="stSidebar"] {{
        background-color: var(--color-surface) !important;
        border-right: 1px solid var(--color-border);
    }}

    .stButton > button {{
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        border-radius: 8px;
    }}

    .metric-card {{
        background: var(--color-surface);
        border: 1px solid var(--color-border);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }}
    .metric-value {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 28px;
        font-weight: 700;
        color: var(--color-high);
    }}
    .metric-label {{
        font-size: 12px;
        color: var(--color-muted);
        margin-top: 2px;
    }}

    .sidebar-title {{
        font-family: 'Rajdhani', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--color-high);
        margin-bottom: 4px;
    }}
    .sidebar-subtitle {{
        font-size: 11px;
        color: var(--color-muted);
        margin-bottom: 16px;
    }}

    div[data-testid="stDataFrame"] {{ background: var(--color-surface); }}
    </style>
    """, unsafe_allow_html=True)


# ── System Initialisation ─────────────────────────────────────────────────────

@st.cache_resource
def initialize_system() -> dict:
    """
    Initialise all system components. Cached across Streamlit reruns.

    Returns:
        dict: All system objects keyed by short name.

    Raises:
        Exception: Propagated to caller which shows a friendly error card.
    """
    # Create required directories
    for d in [DATA_DIR, MODEL_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("app")
    logger.info("HalfTime Oracle initialising…")

    # ── Database ──────────────────────────────────────────────────────────────
    from database.db_manager import DatabaseManager
    db = DatabaseManager.get_instance()

    # ── Data Sources ──────────────────────────────────────────────────────────
    from data_sources.source_registry import SourceRegistry
    from data_sources.espn_api import ESPNApi
    from data_sources.openligadb import OpenLigaDB
    from data_sources.football_data_org import FootballDataOrg
    from data_sources.api_football import ApiFootball
    from data_sources.understat_scraper import UnderstatScraper
    from data_sources.fbref_scraper import FBrefScraper
    from data_sources.odds_api import OddsApi

    registry = SourceRegistry(db)
    espn     = ESPNApi(registry)
    openliga = OpenLigaDB(registry)
    fdorg    = FootballDataOrg(registry)
    apif     = ApiFootball(registry)
    understat = UnderstatScraper(registry)
    fbref    = FBrefScraper(registry)
    odds     = OddsApi(registry)

    # ── Feature Engineer ──────────────────────────────────────────────────────
    from utils.feature_engineering import FeatureEngineer
    fe = FeatureEngineer(db)

    # ── Per-league models ─────────────────────────────────────────────────────
    from models.dixon_coles import DixonColesModel
    from models.synthetic_xg import SyntheticXGEstimator
    from models.xgb_classifier import XGBHalfTimeClassifier
    from models.online_learner import OnlineLearner
    from models.ensemble import EnsemblePredictor

    dc_map:       dict = {}
    xg_map:       dict = {}
    xgb_map:      dict = {}
    ol_map:       dict = {}
    ensemble_map: dict = {}

    for lk in ACTIVE_LEAGUE_KEYS:
        # Dixon-Coles
        dc = DixonColesModel(lk)
        dc.load()
        dc_map[lk] = dc

        # Synthetic xG
        xg = SyntheticXGEstimator(lk)
        xg.load()
        xg_map[lk] = xg

        # XGBoost classifier
        xgb = XGBHalfTimeClassifier(lk)
        xgb.load()
        xgb_map[lk] = xgb

        # Online learner
        ol = OnlineLearner(lk, db, xgb, fe)
        ol_map[lk] = ol

        # Ensemble predictor
        ep = EnsemblePredictor(lk, dc, xgb, ol, fe, db)
        ensemble_map[lk] = ep

    # ── Pipelines ─────────────────────────────────────────────────────────────
    from pipelines.result_resolver import ResultResolver
    from pipelines.prematch_pipeline import PreMatchPipeline
    from pipelines.inplay_pipeline import InPlayPipeline

    resolver = ResultResolver(db, espn, ol_map)
    prematch = PreMatchPipeline(db, registry, espn, fdorg, apif, ensemble_map, fe)
    inplay   = InPlayPipeline(db, espn, ensemble_map, fe)

    logger.info("HalfTime Oracle initialised successfully.")

    return {
        "db":                db,
        "registry":          registry,
        "espn":              espn,
        "openliga":          openliga,
        "fdorg":             fdorg,
        "apif":              apif,
        "understat":         understat,
        "fbref":             fbref,
        "odds":              odds,
        "fe":                fe,
        "dc_map":            dc_map,
        "xg_map":            xg_map,
        "xgb_map":           xgb_map,
        "online_learner_map": ol_map,
        "ensemble_map":      ensemble_map,
        "resolver":          resolver,
        "prematch_pipeline": prematch,
        "inplay_pipeline":   inplay,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(system: dict) -> None:
    """Render the application sidebar."""
    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚡ HalfTime Oracle</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">AI Football Prediction System</div>', unsafe_allow_html=True)
        st.divider()

        # League multiselect
        league_options = {LEAGUES[lk]["name"]: lk for lk in ACTIVE_LEAGUE_KEYS}
        selected_names = st.multiselect(
            "Active Leagues",
            options=list(league_options.keys()),
            default=list(league_options.keys()),
            key="sidebar_leagues",
        )
        st.session_state["active_leagues"] = [league_options[n] for n in selected_names]

        # Market checkboxes
        st.markdown("**Markets**")
        selected_markets = []
        for market in MARKETS:
            if st.checkbox(market.replace("_", " "), value=True, key=f"mkt_{market}"):
                selected_markets.append(market)
        st.session_state["active_markets"] = selected_markets

        st.divider()

        # Confidence thresholds (read-only display)
        st.markdown("**Confidence Thresholds**")
        for market, threshold in CONFIDENCE_THRESHOLDS.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;font-size:12px;"
                f"color:{COLOR_MUTED};margin:2px 0;'>"
                f"<span>{market.replace('_',' ')}</span>"
                f"<span style='color:{COLOR_TEXT};'>{threshold*100:.0f}%</span></div>",
                unsafe_allow_html=True
            )

        st.divider()

        # Source health dots
        st.markdown("**Data Sources**")
        registry = system.get("registry")
        if registry:
            health = registry.get_all_health()
            dot_map = {"GREEN": COLOR_HIGH, "AMBER": COLOR_MEDIUM, "RED": COLOR_NO_BET}
            emoji_map = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}
            for source, status in health.items():
                st.markdown(
                    f"<div style='font-size:12px;color:{COLOR_MUTED};margin:2px 0;'>"
                    f"{emoji_map.get(status,'⚪')} {source}</div>",
                    unsafe_allow_html=True
                )

        st.divider()

        # Model readiness per league
        st.markdown("**Model Readiness**")
        prematch = system.get("prematch_pipeline")
        if prematch:
            for lk in ACTIVE_LEAGUE_KEYS:
                ready, _ = prematch.is_model_ready(lk)
                flag = LEAGUES[lk].get("flag", "")
                name = LEAGUES[lk]["name"]
                icon = "✅" if ready else "🔄"
                st.markdown(
                    f"<div style='font-size:12px;color:{COLOR_MUTED};margin:2px 0;'>"
                    f"{icon} {flag} {name}</div>",
                    unsafe_allow_html=True
                )


# ── Top Metrics Bar ───────────────────────────────────────────────────────────

def render_top_metrics(system: dict) -> None:
    """Render 4-column top metrics bar."""
    db = system.get("db")
    espn = system.get("espn")

    # Count today's matches across all leagues
    today_matches = 0
    if espn:
        for lk in ACTIVE_LEAGUE_KEYS:
            today_matches += len(espn.get_todays_fixtures(lk))

    # 30-day accuracy
    stats = db.get_accuracy_stats(days=30) if db else {}
    accuracy = stats.get("accuracy", 0.0) * 100
    total_preds = stats.get("total", 0)

    # Average Brier score across markets and leagues
    brier_scores = []
    if db:
        for lk in ACTIVE_LEAGUE_KEYS[:2]:  # sample first 2 to avoid too many DB calls
            for market in MARKETS[:3]:
                b = db.get_brier_score(lk, market)
                if b < 0.25:
                    brier_scores.append(b)
    avg_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0.25

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Today's Matches", str(today_matches), COLOR_ACCENT),
        ("Predictions (30d)", str(total_preds), COLOR_MEDIUM),
        ("Accuracy (30d)", f"{accuracy:.1f}%", COLOR_HIGH),
        ("Avg Brier Score", f"{avg_brier:.4f}", COLOR_MUTED if avg_brier >= 0.25 else COLOR_HIGH),
    ]
    for col, (label, value, colour) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{colour};">{value}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ── Main ──────────────────────────────────────────────────────────────────────

inject_css()

# Initialise system with error boundary
try:
    system = initialize_system()
    st.session_state["system"] = system
except Exception as exc:
    tb = traceback.format_exc()
    st.error(
        f"⚠️ **HalfTime Oracle failed to initialise.**\n\n"
        f"**Error:** `{exc}`\n\n"
        f"Check your `.env` file and ensure all dependencies are installed.\n\n"
        f"<details><summary>Full traceback</summary><pre>{tb}</pre></details>",
        icon="🚨",
    )
    st.stop()

# Run result resolver on each page load (Streamlit-Cloud-safe — no threads)
try:
    system["resolver"].maybe_run_resolver()
except Exception as exc:
    logging.getLogger("app").warning("Resolver error on page load: %s", exc)

render_sidebar(system)

# Home page content
st.markdown(
    f"<h1 style='background:linear-gradient(90deg,{COLOR_HIGH},{COLOR_ACCENT});"
    f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
    f"font-family:Rajdhani,sans-serif;font-size:42px;margin-bottom:4px;'>"
    f"⚡ HalfTime Oracle</h1>",
    unsafe_allow_html=True
)
st.markdown(
    f"<p style='color:{COLOR_MUTED};font-size:15px;margin-top:0;'>"
    f"Advanced Football Halftime Over/Under AI — Bloom · Silver · Benter · Dixon-Coles</p>",
    unsafe_allow_html=True
)

st.divider()
render_top_metrics(system)
st.divider()

# Quick navigation cards
col1, col2, col3, col4 = st.columns(4)
nav_items = [
    ("📋 Pre-Match", "Today's fixtures & predictions", "pages/1_Pre_Match_Predictions"),
    ("📡 Live",      "Real-time match dashboard",      "pages/2_Live_Dashboard"),
    ("📊 Analytics", "Performance & calibration",      "pages/3_Performance_Analytics"),
    ("🔬 Insights",  "Model internals & controls",     "pages/4_Model_Insights"),
]
for col, (title, desc, _) in zip([col1, col2, col3, col4], nav_items):
    with col:
        st.markdown(
            f'<div class="metric-card" style="cursor:pointer;">'
            f'<div style="font-family:Rajdhani,sans-serif;font-size:18px;'
            f'font-weight:700;color:{COLOR_TEXT};">{title}</div>'
            f'<div class="metric-label">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown(
    f"<p style='color:{COLOR_MUTED};font-size:12px;margin-top:24px;'>"
    f"Use the sidebar to navigate between pages. "
    f"Data sources: ESPN (free) · OpenLigaDB (free) · API-Football (free tier) · "
    f"football-data.org (free tier) · Understat · FBref · The Odds API (free tier).</p>",
    unsafe_allow_html=True
)
