"""
Streamlit Dashboard: Project DiMaggio.

Displays:
  - Streak progress bar
  - Daily picks table with P(Hit) gauges
  - SHAP explanation panel (waterfall-style)
  - Weather & park environment card per venue
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import date

st.set_page_config(
    page_title="Project DiMaggio",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .prob-high  { color: #00ff88; font-size: 2rem; font-weight: 700; }
    .prob-med   { color: #ffd700; font-size: 2rem; font-weight: 700; }
    .prob-low   { color: #ff6b6b; font-size: 2rem; font-weight: 700; }

    .stProgress > div > div > div { border-radius: 8px; }
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar Controls ───────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a6/National_Baseball_Hall_of_Fame.svg/200px-National_Baseball_Hall_of_Fame.svg.png", width=80)
st.sidebar.title("⚾ Project DiMaggio")
st.sidebar.markdown("*Targeting the 57-game streak*")
st.sidebar.markdown("---")

game_date   = st.sidebar.date_input("📅 Game Date", value=date.today())
streak_len  = st.sidebar.slider("🔥 Current Streak", min_value=0, max_value=57, value=0)
dd_budget   = st.sidebar.number_input("2️⃣  Double Downs Left", min_value=0, max_value=10, value=5)
savers_left = st.sidebar.number_input("🛡️  Streak Savers Left", min_value=0, max_value=5, value=2)
min_p       = st.sidebar.slider("📊 Min P(Hit) Filter", 0.60, 0.99, 0.70, 0.01)

run_button = st.sidebar.button("🚀 Get Picks", type="primary", use_container_width=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; color:#00d4ff; margin-bottom:0'>⚾ Project DiMaggio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa; margin-top:0'>ML-Powered Beat the Streak Engine</p>", unsafe_allow_html=True)
st.markdown("---")

# ── Streak Progress ────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🔥 Current Streak", f"{streak_len} games")
with col2:
    phase = "Aggressive" if streak_len <= 10 else "Opportunistic" if streak_len <= 40 else "Ultra-Conservative"
    st.metric("📋 Phase", phase)
with col3:
    from config import get_threshold
    st.metric("🎯 Required P(Hit)", f">{get_threshold(streak_len):.0%}")
with col4:
    pct = streak_len / 57
    st.metric("📈 Milestone Progress", f"{pct:.1%}")

streak_bar = st.progress(min(streak_len / 57, 1.0))
for milestone, label in [(10, "10"), (20, "20"), (30, "30"), (40, "40"), (50, "50"), (57, "57 🏆")]:
    if streak_len >= milestone:
        st.success(f"✅ Game {label} milestone reached!")

st.markdown("---")

# ── Main Panel ─────────────────────────────────────────────────────────────────
if run_button:
    date_str = str(game_date)

    with st.spinner("⚙️ Fetching predictions..."):
        try:
            import requests
            resp = requests.get(
                "http://127.0.0.1:8000/picks",
                params={
                    "date": date_str,
                    "streak_len": streak_len,
                    "double_down_budget": dd_budget,
                    "savers_remaining": savers_left,
                    "min_prob": min_p,
                },
                timeout=30,
            )
            data = resp.json()
        except Exception as exc:
            st.error(f"❌ API connection failed: {exc}\n\nMake sure the API is running: `uvicorn api.main:app --reload`")
            st.stop()

    picks_list = data.get("picks", [])
    rl_rec     = data.get("rl_recommendation", {})

    # ── RL Strategy Banner ─────────────────────────────────────────────────────
    if rl_rec:
        action  = rl_rec.get("action_name", "pick_only").upper().replace("_", " ")
        st.info(f"🤖 **RL Strategic Advisor**: **{action}** recommended today. Q-values: `{rl_rec.get('q_values', {})}`")

    if not picks_list:
        st.warning(f"⚠️ No qualified picks for {date_str} with P > {min_p:.0%}. Try lowering the filter or check if the pipeline has been run.")
        st.stop()

    # ── Picks Table ────────────────────────────────────────────────────────────
    st.subheader(f"🎯 Today's Picks — {date_str}")
    picks_df = pd.DataFrame(picks_list)

    # P(Hit) Gauge charts
    gauge_cols = st.columns(min(len(picks_list), 3))
    for i, (col, pick) in enumerate(zip(gauge_cols, picks_list)):
        p = pick["p_hit"]
        color = "#00ff88" if p >= 0.85 else "#ffd700" if p >= 0.75 else "#ff6b6b"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=p * 100,
            title={"text": f"Batter {pick['batter_id']}<br><small>{pick.get('away_team','?')} @ {pick.get('home_team','?')}</small>"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0, 70],  "color": "#2d2d44"},
                    {"range": [70, 85], "color": "#3d3d5c"},
                    {"range": [85, 100],"color": "#4d4d7c"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": get_threshold(streak_len) * 100,
                },
            },
            number={"suffix": "%", "font": {"color": color, "size": 28}},
        ))
        fig.update_layout(
            height=260,
            paper_bgcolor="#0f0f1a",
            font={"color": "white"},
            margin=dict(l=20, r=20, t=60, b=20),
        )
        with col:
            st.plotly_chart(fig, use_container_width=True)
            if pick["double_down"]:
                st.markdown("<p style='text-align:center; color:#ff6b6b; font-weight:700'>🎲 DOUBLE DOWN</p>", unsafe_allow_html=True)

    # Full picks table
    display_df = picks_df[["batter_id", "game_date", "p_hit", "home_team", "away_team", "stand", "double_down"]].copy()
    display_df["p_hit"] = display_df["p_hit"].apply(lambda p: f"{p:.1%}")
    display_df.columns = ["Batter ID", "Date", "P(Hit)", "Home", "Away", "Stand", "Double Down"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── SHAP Explanation Panel ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Pick Explanation (SHAP)")

    if picks_list:
        top_pick    = picks_list[0]
        explain_btn = st.button(f"Explain Pick: Batter {top_pick['batter_id']}")

        if explain_btn:
            with st.spinner("Running SHAP analysis..."):
                try:
                    exp_resp = requests.get(
                        "http://127.0.0.1:8000/explain",
                        params={"batter_id": top_pick["batter_id"], "date": date_str},
                        timeout=60,
                    )
                    exp_data = exp_resp.json()
                except Exception as exc:
                    st.error(f"SHAP failed: {exc}")
                    st.stop()

            st.success(f"**Summary**: {exp_data.get('explanation_text', 'N/A')}")

            feats = exp_data.get("top_features", [])
            if feats:
                feat_df = pd.DataFrame(feats)
                fig_shap = px.bar(
                    feat_df.sort_values("shap_value"),
                    x="shap_value", y="feature",
                    orientation="h",
                    color="shap_value",
                    color_continuous_scale=["#ff6b6b", "#f0f0f0", "#00ff88"],
                    labels={"shap_value": "SHAP Value", "feature": "Feature"},
                    title="SHAP Feature Importance (Waterfall)",
                )
                fig_shap.update_layout(
                    paper_bgcolor="#0f0f1a",
                    plot_bgcolor="#1a1a2e",
                    font={"color": "white"},
                    height=350,
                )
                st.plotly_chart(fig_shap, use_container_width=True)

    # ── Park Environment Panel ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🌤️ Park & Environment Conditions")

    try:
        from physics.park_factors import PARK_FACTORS, compute_env_composite

        env_rows = []
        for team, info in PARK_FACTORS.items():
            env = compute_env_composite(team, 72.0, 50.0, 1013.25)
            env_rows.append({
                "Team": team,
                "Stadium": info["stadium"],
                "HR Factor": f"{info['hr_factor']:.2f}",
                "BABIP Factor": f"{info['babip_factor']:.2f}",
                "Env Score": f"{env['env_composite']:+.3f}",
                "Air Density": f"{env['air_density']:.3f}",
                "COR": f"{env['cor_adjustment']:.4f}",
            })
        env_df = pd.DataFrame(env_rows).sort_values("Env Score", ascending=False)
        st.dataframe(env_df, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Environment panel unavailable: {exc}")

else:
    st.markdown("""
    <div style="text-align:center; padding:80px; color:#666;">
        <h2>⚾ Ready to pick your winners</h2>
        <p>Select a date and click <strong>Get Picks</strong> to generate today's recommendations.</p>
        <p style="font-size:0.85rem;">Make sure the data pipeline has been run and the API server is started:<br>
        <code>uvicorn api.main:app --reload</code></p>
    </div>
    """, unsafe_allow_html=True)
