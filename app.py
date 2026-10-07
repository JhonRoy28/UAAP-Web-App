"""
UAAP Season 87 Volleyball Match Outcome Predictor
-------------------------------------------------
A professional Streamlit dashboard predicting match outcomes based on pre-match momentum.
"""

from __future__ import annotations

import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UAAP S87 Volleyball Predictor",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Configuration & Mapping
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "uaap_volleyball_model.pkl"

FEATURE_COLUMNS = {
    "team_wins": "Team_Wins_Before",
    "opp_wins": "Opp_Wins_Before",
    "team_streak": "Team_Win_Streak",
    "opp_streak": "Opp_Win_Streak",
    "round": "Round",
}

ROUND_OPTIONS = ["Round 1", "Round 2", "Final Four", "Finals"]

ROUND_AS_TEXT = False
ROUND_ENCODING = {
    "Round 1": 1,
    "Round 2": 2,
    "Final Four": "Final Four",
    "Finals": "Finals"
}

WIN_LABELS = {"1", "1.0", "true", "win", "w", "won"}
TOSS_UP_MARGIN = 0.10

COLOR_TEAM_A = "#2F80ED"
COLOR_OPPONENT = "#F2994A"

# ---------------------------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(135deg, #0b1f3a 0%, #17407a 55%, #2f6fc4 100%);
        border-radius: 20px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 12px 32px rgba(11, 31, 58, 0.28);
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: "🏐";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 6rem;
        opacity: 0.15;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.14);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.28);
        border-radius: 999px;
        padding: 0.2rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
    }
    .hero p {
        color: rgba(255, 255, 255, 0.82);
        font-size: 1.02rem;
        margin: 0.6rem 0 0 0;
        max-width: 640px;
    }
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.65;
        margin: 1.4rem 0 0.6rem 0;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.07);
    }
    .team-tag {
        display: inline-block;
        border-radius: 8px;
        padding: 0.25rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #ffffff;
        margin-bottom: 0.4rem;
    }
    .tag-a { background: #2F80ED; }
    .tag-b { background: #F2994A; }
    .matchup {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1.4rem;
        flex-wrap: wrap;
        padding: 1.1rem 1rem;
        border-radius: 16px;
        background: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.22);
        margin-bottom: 1rem;
    }
    .matchup .team {
        font-size: 1.5rem;
        font-weight: 800;
    }
    .matchup .team-a { color: #2F80ED; }
    .matchup .team-b { color: #F2994A; }
    .matchup .vs {
        font-size: 0.85rem;
        font-weight: 800;
        background: rgba(127, 127, 127, 0.2);
        border-radius: 999px;
        padding: 0.3rem 0.8rem;
    }
    .matchup .round {
        width: 100%;
        text-align: center;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        opacity: 0.6;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1d5fd1 0%, #2f80ed 100%);
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1.5rem;
        box-shadow: 0 8px 20px rgba(47, 128, 237, 0.38);
    }
    .prob-bar {
        display: flex;
        height: 38px;
        border-radius: 999px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
        margin: 0.4rem 0 0.3rem 0;
    }
    .prob-bar .seg {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.8;
    }
    .footer-note {
        text-align: center;
        font-size: 0.8rem;
        opacity: 0.55;
        margin-top: 2rem;
    }
    #MainMenu, footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading & Helper Functions
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model(path: str):
    return joblib.load(path)

def clean_name(raw: str, fallback: str) -> str:
    name = (raw or "").strip()
    return name if name else fallback

def describe_streak(value: int) -> str:
    if value > 0:
        return f"On a {value}-game winning streak"
    if value < 0:
        return f"On a {abs(value)}-game losing streak"
    return "No active streak"

def build_feature_frame(inputs: dict, model) -> pd.DataFrame:
    round_value = inputs["round"] if ROUND_AS_TEXT else ROUND_ENCODING[inputs["round"]]
    row = {
        FEATURE_COLUMNS["team_wins"]: inputs["team_wins"],
        FEATURE_COLUMNS["opp_wins"]: inputs["opp_wins"],
        FEATURE_COLUMNS["team_streak"]: inputs["team_streak"],
        FEATURE_COLUMNS["opp_streak"]: inputs["opp_streak"],
        FEATURE_COLUMNS["round"]: round_value,
    }
    frame = pd.DataFrame([row])
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None:
        frame = frame[list(expected)]
    return frame

def predict_win_probability(model, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        win_index = next(
            (i for i, c in enumerate(classes) if str(c).strip().lower() in WIN_LABELS),
            len(classes) - 1,
        )
        return float(probabilities[win_index])
    prediction = model.predict(frame)[0]
    return 1.0 if str(prediction).strip().lower() in WIN_LABELS else 0.0

def render_probability_bar(team_a: str, team_b: str, p_a: float) -> None:
    p_b = 1.0 - p_a
    pct_a, pct_b = p_a * 100, p_b * 100
    label_a = f"{pct_a:.1f}%" if pct_a >= 12 else ""
    label_b = f"{pct_b:.1f}%" if pct_b >= 12 else ""
    st.markdown(
        f"""
        <div class="prob-bar">
            <div class="seg" style="width:{pct_a:.2f}%; background:{COLOR_TEAM_A};">{label_a}</div>
            <div class="seg" style="width:{pct_b:.2f}%; background:{COLOR_OPPONENT};">{label_b}</div>
        </div>
        <div class="prob-labels">
            <span>{html.escape(team_a)}</span>
            <span>{html.escape(team_b)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="badge">UAAP Season 87 · Volleyball</span>
        <h1>Match Outcome Predictor</h1>
        <p>Enter each team's pre-match momentum and see who the model favors based on historical Season 87 data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load Model
try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("The model could not be loaded.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Matchup Preview Banner
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Matchup Preview</div>', unsafe_allow_html=True)
banner_slot = st.empty()

# ---------------------------------------------------------------------------
# Input Cards with Informational Tooltips (?)
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Pre-Match Momentum</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    with st.container(border=True):
        st.markdown('<span class="team-tag tag-a">Team A</span>', unsafe_allow_html=True)
        team_a_raw = st.text_input(
            "Team name",
            value="NU Lady Bulldogs",
            key="team_a_name",
            help="Enter the official school or team name for Team A."
        )
        team_a_wins = st.slider(
            "Wins before match",
            min_value=0, max_value=14, value=7,
            key="team_a_wins",
            help="Total number of games Team A has won prior to entering this match."
        )
        team_a_streak = st.slider(
            "Win streak",
            min_value=-10, max_value=10, value=2,
            key="team_a_streak",
            help="Positive numbers indicate consecutive wins (e.g., +2). Negative numbers indicate consecutive losses (e.g., -1)."
        )
        st.caption(describe_streak(team_a_streak))

with col_b:
    with st.container(border=True):
        st.markdown('<span class="team-tag tag-b">Opponent</span>', unsafe_allow_html=True)
        team_b_raw = st.text_input(
            "Team name",
            value="DLSU Lady Spikers",
            key="team_b_name",
            help="Enter the official school or team name for the opposing team."
        )
        team_b_wins = st.slider(
            "Wins before match",
            min_value=0, max_value=14, value=5,
            key="team_b_wins",
            help="Total number of games the opponent has won prior to entering this match."
        )
        team_b_streak = st.slider(
            "Win streak",
            min_value=-10, max_value=10, value=1,
            key="team_b_streak",
            help="Opponent's current momentum streak (positive for wins, negative for losses)."
        )
        st.caption(describe_streak(team_b_streak))

st.markdown('<div class="section-title">Tournament Stage</div>', unsafe_allow_html=True)
with st.container(border=True):
    selected_round = st.selectbox(
        "Tournament round",
        options=ROUND_OPTIONS,
        key="tournament_round",
        help="Select whether this match takes place in Round 1, Round 2, the Final Four, or the Finals."
    )

team_a_name = clean_name(team_a_raw, "Team A")
team_b_name = clean_name(team_b_raw, "Opponent")

banner_slot.markdown(
    f"""
    <div class="matchup">
        <span class="team team-a">{html.escape(team_a_name)}</span>
        <span class="vs">VS</span>
        <span class="team team-b">{html.escape(team_b_name)}</span>
        <span class="round">{html.escape(selected_round)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Prediction Action & Results
# ---------------------------------------------------------------------------
st.write("")
predict_clicked = st.button("🏐  Predict Match Outcome", type="primary")

if predict_clicked:
    if team_a_name.lower() == team_b_name.lower():
        st.warning("Both teams have the same name. Please double-check the matchup.")

    inputs = {
        "team_wins": team_a_wins,
        "opp_wins": team_b_wins,
        "team_streak": team_a_streak,
        "opp_streak": team_b_streak,
        "round": selected_round,
    }

    try:
        with st.spinner("Crunching the numbers..."):
            features = build_feature_frame(inputs, model)
            p_a = predict_win_probability(model, features)
        p_b = 1.0 - p_a

    except Exception as exc:
        st.error("Something went wrong while making the prediction.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

    else:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        with st.container(border=True):
            gap = abs(p_a - 0.5)
            if gap < TOSS_UP_MARGIN:
                st.warning(f"**Too close to call.** The model sees {team_a_name} vs {team_b_name} as nearly even.", icon="⚖️")
            elif p_a > 0.5:
                st.success(f"**{team_a_name} is predicted to WIN** against {team_b_name}.", icon="🏆")
            else:
                st.error(f"**{team_a_name} is predicted to LOSE** to {team_b_name}.", icon="⚠️")

            m1, m2, m3 = st.columns(3)
            m1.metric(label=f"{team_a_name} win chance", value=f"{p_a * 100:.1f}%", delta=f"{(p_a - p_b) * 100:+.1f} pts")
            m2.metric(label=f"{team_b_name} win chance", value=f"{p_b * 100:.1f}%", delta=f"{(p_b - p_a) * 100:+.1f} pts")
            m3.metric(label="Model confidence", value=f"{max(p_a, p_b) * 100:.1f}%")

            render_probability_bar(team_a_name, team_b_name, p_a)

        with st.expander("See the inputs used for this prediction"):
            summary = pd.DataFrame({
                "Stat": ["Wins before match", "Win streak", "Round"],
                team_a_name: [team_a_wins, team_a_streak, selected_round],
                team_b_name: [team_b_wins, team_b_streak, selected_round],
            })
            st.dataframe(summary, hide_index=True)

st.markdown('<div class="footer-note">Predictions come from a machine learning model trained on Season 87 UAAP results.</div>', unsafe_allow_html=True)