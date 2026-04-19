# CONTRACT: pages/1_Pre_Match_Predictions.py
# NO class imports. Only: streamlit, plotly, pandas, numpy, from config import *
# All system objects from st.session_state["system"]
# Functions: inject_css, render_page, render_league_pills,
#            render_cold_start_banner, render_prediction_card,
#            render_card_expander

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from config import *

# ── Guard ─────────────────────────────────────────────────────────────────────
if "system" not in st.session_state:
    st.error("⚠ System not initialized. Return to the main app page.")
    st.stop()

sys = st.session_state["system"]


def inject_css() -> None:
    """Inject custom dark-theme CSS with Google Fonts."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    h1, h2, h3 {{ font-family: 'Rajdhani', sans-serif; }}
    .pred-card {{
        background: {COLOR_SURFACE}; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0;
        border-left: 5px solid {COLOR_MUTED};
    }}
    .pred-card.high  {{ border-left-color: {COLOR_HIGH}; }}
    .pred-card.medium {{ border-left-color: {COLOR_MEDIUM}; }}
    .pred-card.nobet {{ border-left-color: {COLOR_NO_BET}; opacity: 0.55; }}
    .badge {{
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-family: 'Rajdhani', sans-serif; font-size: 13px; font-weight: 700;
        margin-left: 8px;
    }}
    .badge-high   {{ background: {COLOR_HIGH};   color: #000; }}
    .badge-medium {{ background: {COLOR_MEDIUM}; color: #000; }}
    .badge-nobet  {{ background: {COLOR_NO_BET}; color: #fff; }}
    .prob-bar-bg  {{ background: {COLOR_BORDER}; border-radius: 4px; height: 8px; width: 100%; }}
    .prob-bar     {{ height: 8px; border-radius: 4px; background: {COLOR_HIGH}; }}
    </style>
    """, unsafe_allow_html=True)


def render_league_pills() -> list:
    """Render horizontal league filter pills. Returns list of selected league keys."""
    if "selected_leagues" not in st.session_state:
        st.session_state["selected_leagues"] = list(ACTIVE_LEAGUE_KEYS)

    st.markdown("**Leagues**")
    cols = st.columns(len(ACTIVE_LEAGUE_KEYS))
    selected = []
    for i, lk in enumerate(ACTIVE_LEAGUE_KEYS):
        league = LEAGUES[lk]
        label = f"{league['flag']} {league['name']}"
        is_active = lk in st.session_state["selected_leagues"]
        with cols[i]:
            if st.button(label, key=f"pill_{lk}",
                         type="primary" if is_active else "secondary"):
                if is_active:
                    st.session_state["selected_leagues"] = [
                        x for x in st.session_state["selected_leagues"] if x != lk
                    ]
                else:
                    st.session_state["selected_leagues"].append(lk)
                st.rerun()
        if lk in st.session_state["selected_leagues"]:
            selected.append(lk)
    return selected


def render_cold_start_banner(league_key: str) -> None:
    """Show cold-start progress bar if model is warming up."""
    pipeline = sys.get("prematch_pipeline")
    if pipeline is None:
        return
    ready, reason = pipeline.is_model_ready(league_key)
    if not ready:
        progress = pipeline.get_training_progress(league_key)
        league_name = LEAGUES[league_key]["name"]
        st.warning(f"🔄 **{league_name}**: {reason}")
        st.progress(float(progress.get("pct", 0.0)),
                    text=f"{progress['count']}/{progress['required']} resolved matches")


def render_prediction_card(pred: dict, market_key: str) -> None:
    """Render a single prediction as an HTML card with expandable details."""
    markets = pred.get("markets", {})
    result  = markets.get(market_key, {})
    if not result:
        return

    label    = result.get("confidence_label", "NO_BET")
    prob     = result.get("calibrated_prob", 0.0)
    should   = result.get("should_predict", False)

    card_cls = {"HIGH": "high", "MEDIUM": "medium"}.get(label, "nobet")
    badge_cls = {"HIGH": "badge-high", "MEDIUM": "badge-medium"}.get(label, "badge-nobet")
    badge_txt = label.replace("_", " ")

    home = pred.get("home_team", "?")
    away = pred.get("away_team", "?")
    lk   = pred.get("league_key", "")
    flag = LEAGUES.get(lk, {}).get("flag", "")
    time_str = pred.get("kickoff_utc", "")[:16].replace("T", " ") if pred.get("kickoff_utc") else "--:--"
    bar_width = int(prob * 100)

    statistical_label = "" if pred.get("model_ready", True) else " <small>(Statistical only)</small>"

    st.markdown(f"""
    <div class="pred-card {card_cls}">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
          <span style="font-family:'Rajdhani',sans-serif; font-size:17px; font-weight:700;">
            {flag} {home} <span style="color:{COLOR_MUTED};">vs</span> {away}
          </span>
          <span class="badge {badge_cls}">{badge_txt}</span>
          {statistical_label}
        </div>
        <div style="color:{COLOR_MUTED}; font-size:13px;">{time_str} UTC</div>
      </div>
      <div style="margin-top:10px;">
        <div style="font-size:13px; color:{COLOR_MUTED}; margin-bottom:4px;">
          {market_key.replace("_"," ")} — {prob*100:.1f}%
        </div>
        <div class="prob-bar-bg"><div class="prob-bar" style="width:{bar_width}%;"></div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Expandable details
    with st.expander(f"Details: {home} vs {away} — {market_key}"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Market Probabilities**")
            for mkt, res in markets.items():
                p = res.get("calibrated_prob", 0.0)
                lbl = res.get("confidence_label", "NO_BET")
                colour = COLOR_HIGH if lbl == "HIGH" else COLOR_MEDIUM if lbl == "MEDIUM" else COLOR_NO_BET
                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;margin:3px 0;'>"
                    f"<span style='color:{COLOR_MUTED};font-size:13px;'>{mkt.replace('_',' ')}</span>"
                    f"<span style='color:{colour};font-weight:700;'>{p*100:.1f}%</span></div>",
                    unsafe_allow_html=True
                )

        with col2:
            st.markdown("**Model Breakdown**")
            dc_p   = result.get("dixon_coles_prob", 0.0)
            xgb_p  = result.get("xgb_prob", 0.0)
            raw_p  = result.get("raw_ensemble_prob", 0.0)
            cal_p  = result.get("calibrated_prob", 0.0)

            fig = go.Figure(go.Bar(
                x=[dc_p, xgb_p, raw_p, cal_p],
                y=["Dixon-Coles", "XGBoost", "Raw Ensemble", "Calibrated"],
                orientation="h",
                marker_color=[COLOR_ACCENT, COLOR_MEDIUM, COLOR_MUTED, COLOR_HIGH],
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
                plot_bgcolor=COLOR_SURFACE, height=180,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(range=[0, 1], tickformat=".0%"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # xG comparison
        features = pred.get("features", {})
        home_xg = features.get("home_xg_synthetic", 0.0)
        away_xg = features.get("away_xg_synthetic", 0.0)
        if home_xg or away_xg:
            fig2 = go.Figure(go.Bar(
                x=[home, away],
                y=[home_xg, away_xg],
                marker_color=[COLOR_HIGH, COLOR_ACCENT],
            ))
            fig2.update_layout(
                template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
                plot_bgcolor=COLOR_SURFACE, height=160,
                title=dict(text="Synthetic xG", font=dict(color=COLOR_TEXT, size=12)),
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)


def render_page() -> None:
    """Main page render function."""
    inject_css()

    # Gradient title
    st.markdown(
        f"<h1 style='background:linear-gradient(90deg,{COLOR_HIGH},{COLOR_ACCENT});"
        f"-webkit-background-clip:text;-webkit-text-fill-color:transparent;"
        f"font-family:Rajdhani,sans-serif;font-size:36px;'>⚡ Pre-Match Predictions</h1>",
        unsafe_allow_html=True
    )

    selected_leagues = render_league_pills()
    st.divider()

    # Cold start banners
    for lk in selected_leagues:
        render_cold_start_banner(lk)

    # Market tabs
    tab_labels = ["⚽ HT 0.5", "⚽ HT 1.5", "⚽ HT 2.5"]
    market_map = {
        "⚽ HT 0.5": "HT_over_0.5",
        "⚽ HT 1.5": "HT_over_1.5",
        "⚽ HT 2.5": "HT_over_2.5",
    }
    tabs = st.tabs(tab_labels)

    # Run predictions button
    col_btn, col_info = st.columns([2, 6])
    with col_btn:
        run_clicked = st.button("▶ Run Today's Predictions", type="primary")

    if run_clicked:
        pipeline = sys.get("prematch_pipeline")
        if pipeline:
            with st.spinner("Fetching fixtures and running predictions…"):
                predictions = pipeline.run(league_keys=selected_leagues)
                st.session_state["predictions"] = predictions
                st.success(f"✓ Generated predictions for {len(predictions)} fixtures.")
        else:
            st.error("PreMatchPipeline not available.")

    predictions = st.session_state.get("predictions", [])

    # Filter to selected leagues
    filtered = [p for p in predictions if p.get("league_key") in selected_leagues]

    if not filtered:
        st.info("No predictions yet. Select leagues and click ▶ Run Today's Predictions.")
        return

    for tab, tab_label in zip(tabs, tab_labels):
        market_key = market_map[tab_label]
        with tab:
            shown = 0
            for pred in filtered:
                markets = pred.get("markets", {})
                result  = markets.get(market_key, {})
                label   = result.get("confidence_label", "NO_BET")
                if label == "NO_BET":
                    continue
                render_prediction_card(pred, market_key)
                shown += 1

            if shown == 0:
                st.info(f"No predictions above confidence threshold for {market_key.replace('_',' ')}.")

            # Show NO_BET suppressed count
            no_bet_count = sum(
                1 for p in filtered
                if p.get("markets", {}).get(market_key, {}).get("confidence_label") == "NO_BET"
            )
            if no_bet_count:
                st.caption(f"🚫 {no_bet_count} matches suppressed (below confidence threshold).")


render_page()
