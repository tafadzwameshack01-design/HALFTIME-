# CONTRACT: pages/4_Model_Insights.py
# NO class imports. Only: streamlit, plotly, pandas, numpy, os, from config import *
# Functions: inject_css, render_page, render_weight_sliders,
#            render_learning_log, render_source_health,
#            render_team_ratings, render_goal_distributions,
#            render_model_inventory, render_force_retrain

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime

from config import *

if "system" not in st.session_state:
    st.error("⚠ System not initialized. Return to the main app page.")
    st.stop()

sys = st.session_state["system"]


def inject_css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    h1, h2, h3 {{ font-family: 'Rajdhani', sans-serif; }}
    .health-dot {{ display:inline-block; width:12px; height:12px; border-radius:50%; margin-right:6px; }}
    </style>
    """, unsafe_allow_html=True)


def render_weight_sliders(db) -> None:
    """Render ensemble weight sliders with save button."""
    st.markdown("### ⚖️ Ensemble Weights")
    current_dc, current_xgb = db.get_active_ensemble_weights()

    dc_weight = st.slider(
        "Dixon-Coles Weight", min_value=0.10, max_value=0.90,
        value=float(current_dc), step=0.05, key="dc_weight_slider"
    )
    xgb_weight = round(1.0 - dc_weight, 2)
    st.markdown(
        f"XGBoost Weight: **{xgb_weight:.2f}** *(auto-calculated)*",
    )

    if st.button("💾 Save Weights", type="primary"):
        db.save_ensemble_weights(dc_weight, xgb_weight)
        st.success(f"Weights saved: DC={dc_weight:.2f}, XGB={xgb_weight:.2f}")


def render_learning_log(db) -> None:
    """Render the last 20 online learning update events."""
    st.markdown("### 🧠 Online Learning Log")
    log = db.get_model_update_log(limit=20)
    if not log:
        st.info("No learning events recorded yet.")
        return

    df = pd.DataFrame(log)
    display_cols = [c for c in ["updated_at", "league_key", "market", "trigger",
                                 "samples_added", "old_brier_score",
                                 "new_brier_score", "retrained"] if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def render_source_health(registry) -> None:
    """Render API source health indicators."""
    st.markdown("### 📡 Data Source Health")
    health = registry.get_all_health()
    colour_map = {"GREEN": COLOR_HIGH, "AMBER": COLOR_MEDIUM, "RED": COLOR_NO_BET}
    emoji_map  = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}

    rows = []
    for source, status in health.items():
        usage = sys.get("db").get_api_usage(source)
        rows.append({
            "Source":      source,
            "Status":      f"{emoji_map.get(status, '⚪')} {status}",
            "Calls Today": usage.get("call_count", 0),
            "Errors":      usage.get("error_count", 0),
            "Last Call":   (usage.get("last_call_at") or "—")[:19],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_team_ratings(db) -> None:
    """Render Dixon-Coles team ratings for a selected league."""
    st.markdown("### 🏆 Dixon-Coles Team Ratings")
    league_names = {LEAGUES[lk]["name"]: lk for lk in ACTIVE_LEAGUE_KEYS}
    selected_name = st.selectbox("Select League", list(league_names.keys()), key="ratings_league")
    lk = league_names[selected_name]

    df = db.get_team_ratings(lk)
    if df.empty:
        st.info(f"No team ratings stored for {selected_name} yet.")
        return

    display_cols = [c for c in ["team_name", "attack_param", "defense_param", "home_advantage", "last_updated"] if c in df.columns]
    st.dataframe(df[display_cols].sort_values("attack_param", ascending=False),
                 use_container_width=True, hide_index=True)


def render_goal_distributions(db) -> None:
    """Render HT goal distribution histograms per league."""
    st.markdown("### ⚽ HT Goal Distributions")
    cols = st.columns(len(ACTIVE_LEAGUE_KEYS))
    for i, lk in enumerate(ACTIVE_LEAGUE_KEYS):
        league_name = LEAGUES[lk]["name"]
        df = db.get_training_samples(lk, "HT_over_1.5", limit=1000)
        with cols[i]:
            if df.empty or "actual_ht_home" not in df.columns:
                st.caption(f"{league_name}: no data")
                continue
            totals = (df["actual_ht_home"].fillna(0).astype(float) +
                      df["actual_ht_away"].fillna(0).astype(float)).dropna()
            fig = go.Figure(go.Histogram(
                x=totals, nbinsx=8,
                marker_color=COLOR_ACCENT,
                marker_line=dict(color=COLOR_BORDER, width=1),
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
                height=180, title=dict(text=league_name, font=dict(color=COLOR_TEXT, size=11)),
                margin=dict(l=5, r=5, t=30, b=5),
                xaxis=dict(gridcolor=COLOR_BORDER),
                yaxis=dict(gridcolor=COLOR_BORDER),
            )
            st.plotly_chart(fig, use_container_width=True)


def render_model_inventory() -> None:
    """List all saved model .joblib files with size and modification time."""
    st.markdown("### 📁 Model File Inventory")
    if not os.path.exists(MODEL_DIR):
        st.info("Model directory not yet created.")
        return

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]
    if not files:
        st.info("No model files found yet.")
        return

    rows = []
    for fname in sorted(files):
        fpath = os.path.join(MODEL_DIR, fname)
        stat = os.stat(fpath)
        rows.append({
            "File":     fname,
            "Size":     f"{stat.st_size / 1024:.1f} KB",
            "Modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_force_retrain() -> None:
    """Force retrain buttons with confirmation checkbox guard."""
    st.markdown("### 🔁 Force Actions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Force Retrain by League**")
        for lk in ACTIVE_LEAGUE_KEYS:
            league_name = LEAGUES[lk]["name"]
            confirm = st.checkbox(f"Confirm retrain: {league_name}", key=f"confirm_{lk}")
            if confirm:
                if st.button(f"🔁 Retrain {league_name}", key=f"retrain_{lk}"):
                    ol_map = sys.get("online_learner_map", {})
                    ol = ol_map.get(lk)
                    if ol:
                        with st.spinner(f"Retraining {league_name}…"):
                            ol.force_retrain(lk)
                        st.success(f"Retrain complete for {league_name}.")
                    else:
                        st.error(f"OnlineLearner not found for {league_name}.")

    with col2:
        st.markdown("**Force Resolve Pending**")
        if st.button("🔍 Resolve All Pending Predictions", type="secondary"):
            resolver = sys.get("resolver")
            if resolver:
                with st.spinner("Resolving pending predictions…"):
                    count = resolver.resolve_pending_predictions()
                st.success(f"Resolved {count} predictions.")
            else:
                st.error("ResultResolver not available.")


def render_page() -> None:
    inject_css()
    st.markdown(
        f"<h1 style='font-family:Rajdhani,sans-serif;color:{COLOR_MEDIUM};'>🔬 Model Insights</h1>",
        unsafe_allow_html=True
    )

    db = sys.get("db")
    registry = sys.get("registry")
    if not db:
        st.error("Database not available.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Weights", "Learning Log", "Source Health",
        "Team Ratings", "Goal Distributions", "Files & Actions"
    ])

    with tab1:
        render_weight_sliders(db)
    with tab2:
        render_learning_log(db)
    with tab3:
        if registry:
            render_source_health(registry)
        else:
            st.info("Registry not available.")
    with tab4:
        render_team_ratings(db)
    with tab5:
        render_goal_distributions(db)
    with tab6:
        render_model_inventory()
        st.divider()
        render_force_retrain()


render_page()
