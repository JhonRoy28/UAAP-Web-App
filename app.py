from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# Define model path
MODEL_PATH = Path(__file__).resolve().parent / "uaap_volleyball_model.pkl"

# Page Configuration
st.set_page_config(
    page_title="UAAP Volleyball Match Predictor",
    page_icon="🏐",
    layout="centered"
)

# Clean Header Section
st.markdown("<h1 style='text-align: center;'>🏐 UAAP Season 87 Volleyball Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Predict match outcomes using pre-match team momentum and machine learning.</p>", unsafe_allow_html=True)
st.divider()

# 1. Load Model with Caching
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

# 2. Input Controls Layout
st.subheader("📊 Match Pre-Conditions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Team A")
    team_name = st.selectbox("Select Team A", ["NU", "DLSU", "UST", "FEU", "Adamson", "UP", "Ateneo", "UE"], index=0)
    team_wins = st.slider("Wins Before Match", min_value=0, max_value=14, value=5, key="t_wins")
    team_streak = st.slider("Win Streak (negative for loss streak)", min_value=-15, max_value=15, value=2, key="t_streak")

with col2:
    st.markdown("### Team B (Opponent)")
    opp_name = st.selectbox("Select Opponent", ["NU", "DLSU", "UST", "FEU", "Adamson", "UP", "Ateneo", "UE"], index=1)
    opp_wins = st.slider("Wins Before Match", min_value=0, max_value=14, value=4, key="o_wins")
    opp_streak = st.slider("Win Streak (negative for loss streak)", min_value=-15, max_value=15, value=1, key="o_streak")

st.markdown("")
# Tournament Round selector
round_val = st.selectbox(
    "Tournament Round", 
    [1, 2, "Final Four", "Finals"], 
    format_func=lambda x: f"Round {x}" if isinstance(x, int) else x
)

st.divider()

# 3. Prediction Action & Output
if st.button("Predict Match Outcome", type="primary", use_container_width=True):
    try:
        # Build DataFrame with exact feature names and types expected by model
        input_data = pd.DataFrame([{
            'Round': round_val,
            'Team_Wins_Before': team_wins,
            'Opp_Wins_Before': opp_wins,
            'Team_Win_Streak': team_streak,
            'Opp_Win_Streak': opp_streak
        }])

        pred = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]  # [prob_loss, prob_win]

        win_prob = probs[1] * 100
        loss_prob = probs[0] * 100

        st.subheader("🎯 Prediction Results")
        
        # Visual metric cards side-by-side
        mcol1, mcol2 = st.columns(2)
        mcol1.metric(label=f"{team_name} Win Probability", value=f"{win_prob:.1f}%")
        mcol2.metric(label=f"{opp_name} Win Probability", value=f"{loss_prob:.1f}%")

        if pred == 1:
            st.success(f"🏆 **{team_name}** is predicted to **WIN** against {opp_name}!")
        else:
            st.warning(f"⚠️ **{team_name}** is predicted to **LOSE** against {opp_name}.")

    except Exception as err:
        st.error(f"Prediction failed: {err}")