import importlib

# Import dynamically so the app can start and report a clear error if Streamlit
# is not installed in the active environment.
st = importlib.import_module("streamlit")

# Import dynamically so the app can start and report a clear error if pandas
# is not installed in the active environment.
pd = importlib.import_module("pandas")

# Import dynamically so the app can start and report a clear error if the
# optional model-loading dependency is not installed in the active environment.
joblib = importlib.import_module("joblib")

st.set_page_config(page_title="UAAP Volleyball Match Predictor", page_icon="🏐")
st.title("🏐 UAAP Season 87 Volleyball Match Outcome Predictor")
st.markdown("Predict match outcomes using pre-match team features and Bench's trained Logistic Regression pipeline.")

# 1. Load Model with caching
@st.cache_resource
def load_model():
    return joblib.load('uaap_volleyball_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

# 2. Input Controls (Sidebar or Main)
st.header("Match Pre-Conditions")
col1, col2 = st.columns(2)

with col1:
    team_name = st.text_input("Team A Name", "NU Lady Bulldogs")
    team_wins = st.slider("Team A Wins Before Match", min_value=0, max_value=14, value=5)
    team_streak = st.slider("Team A Current Win Streak", min_value=0, max_value=14, value=2)

with col2:
    opp_name = st.text_input("Team B (Opponent) Name", "DLSU Lady Spikers")
    opp_wins = st.slider("Opponent Wins Before Match", min_value=0, max_value=14, value=4)
    opp_streak = st.slider("Opponent Current Win Streak", min_value=0, max_value=14, value=1)

round_val = st.selectbox("Tournament Round", ["Round 1", "Round 2", "Final Four", "Finals"])

# 3. Prediction Action
if st.button("Predict Match Outcome", type="primary"):
    try:
        # Build DataFrame with exact feature names expected by pipeline
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

        st.subheader("Prediction Result")
        if pred == 1:
            st.success(f"**{team_name}** is predicted to **WIN** against {opp_name}!")
        else:
            st.warning(f"**{team_name}** is predicted to **LOSE** against {opp_name}.")

        st.write(f"Estimated Win Probability: **{win_prob:.1f}%** (Loss: **{loss_prob:.1f}%**)")

    except Exception as err:
        st.error(f"Prediction failed: {err}")
