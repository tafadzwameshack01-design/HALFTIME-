# CONTRACT: pages/3_Performance_Analytics.py
# NO class imports. Only: streamlit, plotly, pandas, numpy, from config import *
# Functions: inject_css, render_page, render_metrics_row,
#            render_accuracy_heatmap, render_brier_timeline,
#            render_calibration_curve, render_streak,
#            render_confusion_matrix, render_roi_simulation

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
    </style>
    """, unsafe_allow_html=True)


def render_metrics_row(db, league_filter, market_filter, days) -> None:
    """Render top-level metric cards."""
    stats = db.get_accuracy_stats(
        league_key=league_filter if league_filter != "All" else None,
        market=market_filter if market_filter != "All" else None,
        days=days,
    )
    brier = 0.25
    if league_filter != "All" and market_filter != "All":
        brier = db.get_brier_score(league_filter, market_filter)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", stats.get("total", 0))
    col2.metric("Accuracy", f"{stats.get('accuracy', 0)*100:.1f}%")
    col3.metric("Brier Score", f"{brier:.4f}")
    col4.metric("NO BET Suppressed", f"{stats.get('suppression_rate', 0)*100:.1f}%")


def render_accuracy_heatmap(db) -> None:
    """Render leagues × markets accuracy heatmap."""
    st.markdown("### Accuracy by League & Market")
    z_data = []
    y_labels = []
    for lk in ACTIVE_LEAGUE_KEYS:
        row = []
        y_labels.append(LEAGUES[lk]["name"])
        for market in MARKETS:
            s = db.get_accuracy_stats(league_key=lk, market=market, days=90)
            row.append(round(s.get("accuracy", 0.0) * 100, 1))
        z_data.append(row)

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=[m.replace("_", " ") for m in MARKETS],
        y=y_labels,
        colorscale=[[0, COLOR_NO_BET], [0.5, COLOR_MEDIUM], [1, COLOR_HIGH]],
        text=[[f"{v:.1f}%" for v in row] for row in z_data],
        texttemplate="%{text}",
        showscale=True,
        zmin=0, zmax=100,
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
        height=280, margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_brier_timeline(db, league_filter, days) -> None:
    """Render Brier score timeline per market."""
    st.markdown("### Brier Score Timeline")
    fig = go.Figure()
    colours = [COLOR_HIGH, COLOR_MEDIUM, COLOR_ACCENT, COLOR_NO_BET, COLOR_MUTED, COLOR_TEXT]
    lk = league_filter if league_filter != "All" else ACTIVE_LEAGUE_KEYS[0]

    for i, market in enumerate(MARKETS):
        df = db.get_performance_history(lk, market, days=days)
        if df.empty or "snapshot_date" not in df.columns or "brier_score" not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["snapshot_date"].tolist(),
            y=df["brier_score"].tolist(),
            mode="lines+markers",
            name=market.replace("_", " "),
            line=dict(color=colours[i % len(colours)], width=2),
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
        height=300, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(gridcolor=COLOR_BORDER, color=COLOR_MUTED),
        yaxis=dict(gridcolor=COLOR_BORDER, color=COLOR_MUTED, title="Brier Score"),
        legend=dict(font=dict(color=COLOR_TEXT)),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_calibration_curve(db, league_filter, market_filter) -> None:
    """Render the Silver calibration test chart."""
    st.markdown("### Calibration Curve (Silver Test)")
    lk = league_filter if league_filter != "All" else ACTIVE_LEAGUE_KEYS[0]
    mkt = market_filter if market_filter != "All" else MARKETS[0]

    df = db.get_training_samples(lk, mkt, limit=2000)
    if df.empty or "ensemble_prob" not in df.columns or "actual_outcome" not in df.columns:
        st.info("Insufficient resolved data for calibration curve.")
        return

    df = df.dropna(subset=["ensemble_prob", "actual_outcome"])
    from utils.calibration import CalibrationCurve
    cal_df = CalibrationCurve.compute(
        df["ensemble_prob"].tolist(),
        df["actual_outcome"].astype(int).tolist()
    )
    fig = CalibrationCurve.plot(cal_df)
    st.plotly_chart(fig, use_container_width=True)


def render_streak(db, league_filter, market_filter, days) -> None:
    """Render win/loss streak horizontal bar."""
    st.markdown("### Prediction Streak")
    lk = league_filter if league_filter != "All" else ACTIVE_LEAGUE_KEYS[0]
    mkt = market_filter if market_filter != "All" else MARKETS[0]

    df = db.get_training_samples(lk, mkt, limit=500)
    if df.empty or "is_correct" not in df.columns:
        st.info("No resolved predictions for streak chart.")
        return

    outcomes = df["is_correct"].dropna().astype(int).tail(50).tolist()
    colours = [COLOR_HIGH if o == 1 else COLOR_NO_BET for o in outcomes]
    fig = go.Figure(go.Bar(
        x=list(range(1, len(outcomes) + 1)),
        y=[1] * len(outcomes),
        marker_color=colours,
        hovertext=["✓" if o == 1 else "✗" for o in outcomes],
    ))
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
        height=120, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(visible=False),
        xaxis=dict(title="Last 50 predictions", gridcolor=COLOR_BORDER),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix(db, league_filter, market_filter) -> None:
    """Render confusion matrix for a market."""
    st.markdown("### Confusion Matrix")
    lk = league_filter if league_filter != "All" else ACTIVE_LEAGUE_KEYS[0]
    mkt = market_filter if market_filter != "All" else MARKETS[0]

    df = db.get_training_samples(lk, mkt, limit=2000)
    if df.empty or "predicted_outcome" not in df.columns or "actual_outcome" not in df.columns:
        st.info("Insufficient data for confusion matrix.")
        return

    df = df.dropna(subset=["predicted_outcome", "actual_outcome"])
    from sklearn.metrics import confusion_matrix  # stdlib-compatible import
    try:
        cm = confusion_matrix(
            df["actual_outcome"].astype(int),
            df["predicted_outcome"].astype(int),
            labels=[0, 1]
        )
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[[0, COLOR_SURFACE], [1, COLOR_HIGH]],
            text=cm, texttemplate="%{text}",
        ))
        fig.update_layout(
            template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
            height=250, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not render confusion matrix: {exc}")


def render_roi_simulation(db, league_filter, market_filter) -> None:
    """Simulate flat-stake P&L on HIGH confidence predictions."""
    st.markdown("### ROI Simulation (1 unit, avg odds 1.85)")
    lk = league_filter if league_filter != "All" else ACTIVE_LEAGUE_KEYS[0]
    mkt = market_filter if market_filter != "All" else MARKETS[0]

    df = db.get_training_samples(lk, mkt, limit=2000)
    if df.empty or "confidence_label" not in df.columns or "is_correct" not in df.columns:
        st.info("Insufficient data for ROI simulation.")
        return

    high_bets = df[df["confidence_label"] == "HIGH"].dropna(subset=["is_correct"])
    if high_bets.empty:
        st.info("No HIGH confidence resolved predictions yet.")
        return

    pnl = []
    running = 0.0
    for correct in high_bets["is_correct"].astype(int).tolist():
        running += (0.85 if correct == 1 else -1.0)
        pnl.append(round(running, 4))

    fig = go.Figure(go.Scatter(
        y=pnl, mode="lines",
        line=dict(color=COLOR_HIGH if pnl[-1] >= 0 else COLOR_NO_BET, width=2),
        fill="tozeroy",
        fillcolor=f"{COLOR_HIGH}22" if pnl[-1] >= 0 else f"{COLOR_NO_BET}22",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_MUTED)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Bet #", gridcolor=COLOR_BORDER),
        yaxis=dict(title="P&L (units)", gridcolor=COLOR_BORDER),
    )
    st.plotly_chart(fig, use_container_width=True)
    final = pnl[-1] if pnl else 0.0
    roi = (final / len(pnl) * 100) if pnl else 0.0
    col1, col2 = st.columns(2)
    col1.metric("Total P&L", f"{final:+.2f} units")
    col2.metric("ROI per bet", f"{roi:.2f}%")


def render_page() -> None:
    inject_css()
    st.markdown(
        f"<h1 style='font-family:Rajdhani,sans-serif;color:{COLOR_ACCENT};'>📊 Performance Analytics</h1>",
        unsafe_allow_html=True
    )

    db = sys.get("db")
    if not db:
        st.error("Database not available.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        days = st.radio("Period", [7, 30, 90], horizontal=True, index=1)
    with col2:
        league_filter = st.selectbox("League", ["All"] + [LEAGUES[lk]["name"] for lk in ACTIVE_LEAGUE_KEYS])
        if league_filter != "All":
            league_filter = next((lk for lk in ACTIVE_LEAGUE_KEYS if LEAGUES[lk]["name"] == league_filter), "All")
    with col3:
        market_filter = st.selectbox("Market", ["All"] + MARKETS)

    st.divider()
    render_metrics_row(db, league_filter, market_filter, days)
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        render_accuracy_heatmap(db)
    with col_b:
        render_brier_timeline(db, league_filter, days)

    st.divider()
    col_c, col_d = st.columns(2)
    with col_c:
        render_calibration_curve(db, league_filter, market_filter)
    with col_d:
        render_streak(db, league_filter, market_filter, days)

    st.divider()
    col_e, col_f = st.columns(2)
    with col_e:
        render_confusion_matrix(db, league_filter, market_filter)
    with col_f:
        render_roi_simulation(db, league_filter, market_filter)


render_page()
