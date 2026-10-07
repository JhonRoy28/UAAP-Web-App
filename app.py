import os

import joblib
import pandas as pd
import streamlit as st

# set_page_config must be the first Streamlit call in the script
st.set_page_config(
    page_title="UAAP Volleyball Match Predictor",
    page_icon="🏐",
    layout="centered",
)

MODEL_FILE = "uaap_volleyball_model.pkl"
DATA_FILE = "UAAP_Season87_Womens_Volleyball_Full__1_.csv"

FEATURES = [
    "Round",
    "Team_Wins_Before",
    "Opp_Wins_Before",
    "Team_Win_Streak",
    "Opp_Win_Streak",
]

ROUND_LABELS = ["First Round", "Second Round", "Final Four", "Finals"]

# How Round was turned into numbers when the model was trained.
# Ask Bench. If the model reads the text labels directly, leave this as None.
# Example if Bench used 1 to 4 in order: {"First Round": 1, "Second Round": 2, "Final Four": 3, "Finals": 4}
ROUND_MAP = None


@st.cache_resource
def load_model():
    """Load the trained model once and reuse it on every rerun."""
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_dataset():
    """Load the dataset if it exists. The app still works without it."""
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return None


try:
    st.title("🏐 UAAP Volleyball Match Predictor")
    st.write(
        "Set the pre-match stats for both teams, then click **Predict** "
        "to see the estimated chance of a win or a loss."
    )

    model = load_model()
    data = load_dataset()

    if data is not None:
        with st.expander("Preview the dataset"):
            st.dataframe(data.head(20), use_container_width=True)
    else:
        st.info(f"Dataset file '{DATA_FILE}' was not found, so the preview is hidden.")

    st.subheader("Match setup")
    match_round = st.select_slider("Round", options=ROUND_LABELS, value="First Round")

    col_team, col_opp = st.columns(2)

    with col_team:
        st.markdown("**Your team**")
        team_wins = st.slider("Team wins before match", 0, 13, 6)
        team_streak = st.slider("Team win streak", -13, 5, 0)

    with col_opp:
        st.markdown("**Opponent**")
        opp_wins = st.slider("Opponent wins before match", 0, 13, 6)
        opp_streak = st.slider("Opponent win streak", -13, 5, 0)

    if st.button("Predict", type="primary"):
        round_value = ROUND_MAP[match_round] if ROUND_MAP else match_round

        inputs = pd.DataFrame(
            [[round_value, team_wins, opp_wins, team_streak, opp_streak]],
            columns=FEATURES,
        )

        # Match the column order the model was trained with, if it recorded it
        if hasattr(model, "feature_names_in_"):
            inputs = inputs[list(model.feature_names_in_)]

        probabilities = model.predict_proba(inputs)[0]
        classes = list(model.classes_)

        # Match_Outcome uses 1 for a win and 0 for a loss
        win_index = classes.index(1) if 1 in classes else len(classes) - 1
        win_prob = probabilities[win_index] * 100
        loss_prob = 100 - win_prob

        st.subheader("Prediction")
        res_win, res_loss = st.columns(2)
        res_win.metric("Win probability", f"{win_prob:.1f}%")
        res_loss.metric("Loss probability", f"{loss_prob:.1f}%")
        st.progress(int(round(win_prob)))

        if win_prob >= 50:
            st.success("The model favors your team to win this match.")
        else:
            st.warning("The model favors the opponent in this match.")

except FileNotFoundError:
    st.warning(
        f"Could not find '{MODEL_FILE}'. Make sure it is in the same folder as app.py."
    )
except Exception as error:
    st.warning(
        "Something went wrong while running the predictor. "
        "Please check your inputs and try again."
    )
    st.caption(f"Details: {error}")
