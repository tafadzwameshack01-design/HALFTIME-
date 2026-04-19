# CONTRACT: pages/2_Live_Dashboard.py
# NO class imports. Only: streamlit, plotly, pandas, numpy, time, from config import *
# All system objects from st.session_state["system"]
# Functions: inject_css, render_live_matches, render_live_card,
#            render_kicking_off_soon, render_at_halftime, render_page

import streamlit as st
import time
import plotly.graph_objects as go
from datetime import datetime, timezone

from config import *

if "system" not in st.session_state:
    st.error("⚠ System not initialized. Return to the main app page.")
    st.stop()

sys = st.session_state["system"]


def inject_css() -> None:
    """Inject live dashboard CSS."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    .live-card {{
        background: {COLOR_SURFACE}; border-radius: 10px;
        padding: 16px 20px; margin: 8px 0;
        border: 1px solid {COLOR_BORDER};
    }}
    .live-score {{
        font-family: 'Rajdhani', sans-serif; font-size: 32px;
        font-weight: 700; text-align: center; color: {COLOR_HIGH};
    }}
    .final-call-banner {{
        background: {COLOR_NO_BET}22; border: 2px solid {COLOR_NO_BET};
        border-radius: 8px; padding: 12px 20px; text-align: center;
        font-family: 'Rajdhani', sans-serif; font-size: 22px;
        font-weight: 700; color: {COLOR_NO_BET}; margin: 8px 0;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_live_card(match: dict) -> None:
    """Render a single live match card with score, minute, and probability gauge."""
    home = match.get("home_team_name", "?")
    away = match.get("away_team_name", "?")
    home_score = match.get("home_score", 0)
    away_score = match.get("away_score", 0)
    minute = match.get("current_minute", 0)
    match_id = match.get("match_id", "")
    league_key = match.get("league_key", "")

    inplay = sys.get("inplay_pipeline")
    prediction = {}
    if inplay and match_id:
        live_data = {
            "home_score": home_score, "away_score": away_score,
            "home_shots": match.get("home_shots", 0),
            "away_shots": match.get("away_shots", 0),
            "home_possession": match.get("home_possession", 50),
            "home_dangerous_attacks": match.get("home_dangerous_attacks", 0),
            "away_dangerous_attacks": match.get("away_dangerous_attacks", 0),
        }
        prediction = inplay.compute_live_prediction(match_id, league_key, minute, live_data)

    with st.container():
        st.markdown(f'<div class="live-card">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([3, 2, 3])

        with col1:
            st.markdown(f"**{home}**")
            st.caption(f"{LEAGUES.get(league_key, {}).get('flag', '')} {LEAGUES.get(league_key, {}).get('name', '')}")

        with col2:
            st.markdown(
                f'<div class="live-score">{home_score} – {away_score}</div>',
                unsafe_allow_html=True
            )
            # Minute progress bar
            pct = min(minute / 45.0, 1.0)
            st.progress(pct, text=f"⏱ {minute}'")

        with col3:
            st.markdown(f"**{away}**")

        # Final call banner at minute 40+
        if minute >= 40:
            markets = prediction.get("markets", {})
            best_market = "HT_over_1.5"
            best_result = markets.get(best_market, {})
            prob = best_result.get("calibrated_prob", 0.0)
            label = best_result.get("confidence_label", "NO_BET")
            if label != "NO_BET":
                st.markdown(
                    f'<div class="final-call-banner">⚡ FINAL CALL — '
                    f'{best_market.replace("_"," ")} — {prob*100:.1f}%</div>',
                    unsafe_allow_html=True
                )

        # Probability gauge for HT Over 1.5
        markets = prediction.get("markets", {})
        result15 = markets.get("HT_over_1.5", {})
        prob15 = result15.get("calibrated_prob", 0.5)

        if prob15 != 0.5:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob15 * 100,
                number={"suffix": "%", "font": {"color": COLOR_TEXT}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": COLOR_MUTED},
                    "bar": {"color": COLOR_HIGH if prob15 >= 0.75 else COLOR_MEDIUM if prob15 >= 0.65 else COLOR_MUTED},
                    "bgcolor": COLOR_SURFACE,
                    "bordercolor": COLOR_BORDER,
                    "threshold": {
                        "line": {"color": COLOR_NO_BET, "width": 2},
                        "thickness": 0.75,
                        "value": CONFIDENCE_THRESHOLDS.get("HT_over_1.5", 75) * 100,
                    },
                },
                title={"text": "HT Over 1.5", "font": {"color": COLOR_MUTED, "size": 12}},
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
                height=180, margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Probability trend sparkline
        if sys.get("inplay_pipeline") and match_id:
            history = sys["inplay_pipeline"].get_probability_history(match_id, "HT_over_1.5")
            if len(history) >= 2:
                minutes_list = [h["minute"] for h in history]
                probs_list   = [h["prob"] * 100 for h in history]
                fig2 = go.Figure(go.Scatter(
                    x=minutes_list, y=probs_list,
                    mode="lines+markers",
                    line=dict(color=COLOR_HIGH, width=2),
                    marker=dict(size=4, color=COLOR_HIGH),
                    fill="tozeroy", fillcolor=f"{COLOR_HIGH}22",
                ))
                fig2.update_layout(
                    template="plotly_dark", paper_bgcolor=COLOR_SURFACE,
                    height=120, margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="Minute", color=COLOR_MUTED, gridcolor=COLOR_BORDER),
                    yaxis=dict(range=[0, 100], title="%", color=COLOR_MUTED),
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


def render_kicking_off_soon() -> None:
    """Render matches kicking off within the next 120 minutes."""
    espn = sys.get("espn")
    if not espn:
        return

    st.markdown(f"### 🕐 Kicking Off Soon")
    found = False
    for lk in ACTIVE_LEAGUE_KEYS:
        fixtures = espn.get_todays_fixtures(lk)
        for fix in fixtures:
            kickoff_str = fix.get("kickoff_utc", "")
            if not kickoff_str:
                continue
            try:
                kickoff = datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
                mins = (kickoff - datetime.now(timezone.utc)).total_seconds() / 60.0
                if 0 <= mins <= 120:
                    home = fix.get("home_team_name", "?")
                    away = fix.get("away_team_name", "?")
                    flag = LEAGUES.get(lk, {}).get("flag", "")
                    st.markdown(
                        f"**{flag} {home} vs {away}** — "
                        f"<span style='color:{COLOR_MEDIUM};'>in {int(mins)} min</span>",
                        unsafe_allow_html=True
                    )
                    found = True
            except Exception:
                continue

    if not found:
        st.caption("No matches kicking off in the next 2 hours.")


def render_at_halftime() -> None:
    """Render section for matches currently at halftime."""
    espn = sys.get("espn")
    if not espn:
        return

    st.markdown("### ⏸ At Half Time")
    found = False
    for lk in ACTIVE_LEAGUE_KEYS:
        events = espn.get_scoreboard(lk)
        for event in events:
            status = event.get("status", {})
            state = status.get("type", {}).get("name", "")
            if "halftime" in state.lower() or "half time" in state.lower():
                comps = event.get("competitions", [])
                if not comps:
                    continue
                competitors = comps[0].get("competitors", [])
                if len(competitors) < 2:
                    continue
                home_comp = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
                away_comp = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
                home_name = home_comp.get("team", {}).get("displayName", "?")
                away_name = away_comp.get("team", {}).get("displayName", "?")
                home_score = home_comp.get("score", 0)
                away_score = away_comp.get("score", 0)
                flag = LEAGUES.get(lk, {}).get("flag", "")
                st.markdown(
                    f"**{flag} {home_name} vs {away_name}** — "
                    f"<span style='color:{COLOR_HIGH};font-weight:700;'>{home_score} – {away_score} HT</span>",
                    unsafe_allow_html=True
                )
                found = True

    if not found:
        st.caption("No matches currently at half time.")


def render_live_matches() -> None:
    """Render all live dashboard sections."""
    st.markdown(f"### 🔴 Live Now")
    live_pipeline = sys.get("inplay_pipeline")

    if live_pipeline:
        live_matches = live_pipeline.get_live_matches(ACTIVE_LEAGUE_KEYS)
        if live_matches:
            for match in live_matches:
                render_live_card(match)
        else:
            st.info("No matches currently live.")
    else:
        st.warning("In-play pipeline not available.")

    st.divider()
    render_kicking_off_soon()
    st.divider()
    render_at_halftime()
    st.divider()

    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    st.caption(f"Last updated: {now_str} — refreshes every {CACHE_TTL_LIVE}s")


def render_page() -> None:
    """Main live dashboard page."""
    inject_css()
    st.markdown(
        f"<h1 style='font-family:Rajdhani,sans-serif;color:{COLOR_HIGH};'>📡 Live Dashboard</h1>",
        unsafe_allow_html=True
    )

    # Streamlit 1.32.0 compatible refresh — no st.fragment
    placeholder = st.empty()
    while True:
        with placeholder.container():
            render_live_matches()
        time.sleep(CACHE_TTL_LIVE)
        placeholder.empty()


render_page()
