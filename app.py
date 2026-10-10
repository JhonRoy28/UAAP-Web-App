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
    "Team",
    "Opponent",
    "Round",
    "Team_Wins_Before",
    "Opp_Wins_Before",
    "Team_Win_Streak",
    "Opp_Win_Streak",
]

TEAMS = ["Adamson", "Ateneo", "DLSU", "FEU", "NU", "UE", "UP", "UST"]
ROUND_LABELS = ["First Round", "Second Round", "Final Four", "Finals"]

st.markdown(
    """
    <style>
    .block-container { max-width: 760px; padding-top: 2.5rem; }
    div.stButton > button {
        width: 100%;
        padding: 0.75rem 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        "Estimate who wins a UAAP Season 87 women's volleyball match "
        "in three quick steps."
    )

    model = load_model()
    data = load_dataset()

    # Step 1: teams
    with st.container(border=True):
        st.subheader("1. Pick the teams")
        col_a, col_b = st.columns(2)
        team = col_a.selectbox("Team", TEAMS, index=1, key="team")
        opponent = col_b.selectbox(
            "Opponent",
            [t for t in TEAMS if t != team],
            index=0,
            key="opponent",
        )

    # Step 2: match stats
    with st.container(border=True):
        st.subheader("2. Set the match stats")
        match_round = st.select_slider(
            "Round", options=ROUND_LABELS, value="First Round", key="round"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**{team}**")
            team_wins = st.slider("Wins before the match", 0, 13, 6, key="team_wins")
            team_streak = st.slider(
                "Win streak (negative means losing streak)",
                -13, 5, 0, key="team_streak",
            )
        with col_b:
            st.markdown(f"**{opponent}**")
            opp_wins = st.slider("Wins before the match", 0, 13, 6, key="opp_wins")
            opp_streak = st.slider(
                "Win streak (negative means losing streak)",
                -13, 5, 0, key="opp_streak",
            )

    # Step 3: prediction
    with st.container(border=True):
        st.subheader("3. See the prediction")
        clicked = st.button("Predict the winner", type="primary")

        if clicked:
            inputs = pd.DataFrame(
                [[team, opponent, match_round, team_wins, opp_wins, team_streak, opp_streak]],
                columns=FEATURES,
            )

            # Match the column order the model was trained with, if it recorded it
            if hasattr(model, "feature_names_in_"):
                inputs = inputs[list(model.feature_names_in_)]

            probabilities = model.predict_proba(inputs)[0]
            classes = list(model.classes_)

            # Match_Outcome uses 1 for a win and 0 for a loss
            win_index = classes.index(1) if 1 in classes else len(classes) - 1
            team_prob = probabilities[win_index] * 100
            opp_prob = 100 - team_prob

            res_a, res_b = st.columns(2)
            res_a.metric(team, f"{team_prob:.1f}%")
            res_b.metric(opponent, f"{opp_prob:.1f}%")
            st.progress(int(round(team_prob)))

            if abs(team_prob - 50) < 5:
                st.info("Too close to call. The model sees this as an even match.")
            elif team_prob > 50:
                st.success(f"{team} is favored to win ({team_prob:.1f}% chance).")
            else:
                st.success(f"{opponent} is favored to win ({opp_prob:.1f}% chance).")

            st.caption(
                "This is an estimate learned from Season 87 results only. "
                "It is not a guarantee."
            )
        else:
            st.caption("Click the button to see each team's chance of winning.")

    if data is not None:
        with st.expander("See the Season 87 data used to train the model"):
            st.dataframe(data)

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