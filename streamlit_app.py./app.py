import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

BASE = os.path.dirname(__file__)

# ---------------- Load model & metadata ----------------
@st.cache_resource
def load_artifacts():
    with open(os.path.join(BASE, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(BASE, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return model, meta

model, meta = load_artifacts()
TEAMS = meta["teams"]
VENUES = meta["venues"]

# ---------------- Dark / IPL Theme CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: #fafafa;
}
h1, h2, h3 { color: #FFD700; }
.card {
    background: linear-gradient(135deg, #1c2333, #2a3350);
    padding: 1.2rem;
    border-radius: 12px;
    border: 1px solid #FFD700;
    text-align: center;
    margin-bottom: 0.6rem;
}
.card h2 { margin: 0; font-size: 2rem; }
.card p { margin: 0; color: #b0b0b0; }
.win-badge {
    font-size: 1.4rem;
    font-weight: bold;
    padding: 0.6rem 1.2rem;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.title("🏏 IPL Match Winner Predictor")
st.caption("AI-powered live win-probability predictor using XGBoost — IPL Data Analytics & ML Project")

st.markdown("---")

# ---------------- Input Section ----------------
col1, col2, col3 = st.columns(3)
with col1:
    batting_team = st.selectbox("🏏 Batting Team (Chasing)", TEAMS)
with col2:
    bowling_team = st.selectbox("🎯 Bowling Team", [t for t in TEAMS if t != batting_team])
with col3:
    venue = st.selectbox("📍 Venue", VENUES)

st.markdown("### Match Situation")
c1, c2, c3, c4 = st.columns(4)
with c1:
    target = st.number_input("🎯 Target Score", min_value=1, max_value=300, value=180)
with c2:
    current_score = st.number_input("🏃 Current Score", min_value=0, max_value=300, value=90)
with c3:
    overs = st.number_input("⏱️ Overs Completed", min_value=0.0, max_value=19.5, value=10.0, step=0.1)
with c4:
    wickets = st.number_input("❌ Wickets Fallen", min_value=0, max_value=10, value=3)

predict_btn = st.button("🔮 Predict Winner", use_container_width=True, type="primary")

st.markdown("---")

if predict_btn:
    balls_bowled = int(overs) * 6 + round((overs - int(overs)) * 10)
    runs_left = target - current_score
    balls_left = 120 - balls_bowled
    wickets_left = 10 - wickets
    crr = current_score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    if runs_left <= 0:
        bat_win_prob, bowl_win_prob = 1.0, 0.0
    elif wickets_left <= 0 or balls_left <= 0:
        bat_win_prob, bowl_win_prob = 0.0, 1.0
    else:
        input_df = pd.DataFrame([{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "target_score": target,
            "current_score": current_score,
            "overs_completed": overs,
            "current_wickets": wickets,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "crr": crr,
            "rrr": rrr
        }])
        bat_win_prob = float(model.predict_proba(input_df)[0][1])
        bowl_win_prob = 1 - bat_win_prob

    # ---------------- Result Cards ----------------
    st.markdown("## 🏆 Prediction Result")
    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown(f"<div class='card'><p>Runs Needed</p><h2>{runs_left}</h2></div>", unsafe_allow_html=True)
    with rc2:
        st.markdown(f"<div class='card'><p>Balls Remaining</p><h2>{balls_left}</h2></div>", unsafe_allow_html=True)
    with rc3:
        st.markdown(f"<div class='card'><p>Current RR</p><h2>{crr:.2f}</h2></div>", unsafe_allow_html=True)
    with rc4:
        st.markdown(f"<div class='card'><p>Required RR</p><h2>{rrr:.2f}</h2></div>", unsafe_allow_html=True)

    winner = batting_team if bat_win_prob >= bowl_win_prob else bowling_team
    win_color = "#1f8b4c" if bat_win_prob >= bowl_win_prob else "#c0392b"

    st.markdown(
        f"<div class='win-badge' style='background-color:{win_color};'>"
        f"🏆 Predicted Winner: {winner} "
        f"({max(bat_win_prob, bowl_win_prob)*100:.1f}% Win Probability)</div>",
        unsafe_allow_html=True
    )

    st.markdown("### 📊 Win Probability Breakdown")
    fig = go.Figure(go.Bar(
        x=[bat_win_prob * 100, bowl_win_prob * 100],
        y=[batting_team, bowling_team],
        orientation="h",
        marker=dict(color=["#FFD700", "#3498DB"]),
        text=[f"{bat_win_prob*100:.1f}%", f"{bowl_win_prob*100:.1f}%"],
        textposition="auto"
    ))
    fig.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", xaxis_title="Win Probability (%)",
        xaxis_range=[0, 100], height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pie chart
    fig2 = go.Figure(go.Pie(
        labels=[batting_team, bowling_team],
        values=[bat_win_prob, bowl_win_prob],
        hole=0.5,
        marker=dict(colors=["#FFD700", "#3498DB"])
    ))
    fig2.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=350,
        title="Win Probability Share"
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Set the match situation above and click **Predict Winner**.")

st.markdown("---")
st.caption("Built with ❤️ using Python, Scikit-Learn, XGBoost & Streamlit | IPL Data Analytics & Match Winner Prediction System")
