"""
UAAP Season 87 Volleyball Match Outcome Predictor
-------------------------------------------------
A Streamlit dashboard that predicts whether a team will win or lose against
an opponent, based on pre-match momentum stats.

Run with:
    streamlit run app.py

Required files in the same folder:
    app.py
    uaap_volleyball_model.pkl
"""

from __future__ import annotations

import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UAAP S87 Volleyball Predictor",
    page_icon="🏐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Configuration
# EDIT THIS SECTION if your model was trained with different column names or
# a different encoding for the tournament round.
# ---------------------------------------------------------------------------
MODEL_PATH = Path(__file__).parent / "uaap_volleyball_model.pkl"

# Left side: our internal key. Right side: the column name your model expects.
FEATURE_COLUMNS = {
    "team_wins": "Team_Wins_Before",
    "opp_wins": "Opp_Wins_Before",
    "team_streak": "Team_Win_Streak",
    "opp_streak": "Opp_Win_Streak",
    "round": "Round",
}

ROUND_OPTIONS = ["Round 1", "Round 2", "Final Four", "Finals"]

# How the round is sent to the model.
# False -> numeric codes from ROUND_ENCODING (label encoding)
# True  -> the round text exactly as shown (use this if your model is a
#          Pipeline that handles text categories itself)
ROUND_AS_TEXT = True
ROUND_ENCODING = {"Round 1": 0, "Round 2": 1, "Final Four": 2, "Finals": 3}

# Class labels that mean "Team A wins". Compared in lowercase text form.
WIN_LABELS = {"1", "1.0", "true", "win", "w", "won"}

# A prediction is called a toss-up when Team A's win chance is within this
# distance of 50%.
TOSS_UP_MARGIN = 0.10

SCHOOL_COLORS = {
    "Adamson": "#0057A8",
    "Ateneo": "#003DA5",
    "DLSU": "#007A33",
    "FEU": "#006B3F",
    "NU": "#003DA5",
    "UE": "#C8102E",
    "UP": "#7B1113",
    "UST": "#FDB81E",
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    /* Page width and spacing */
    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }
    [data-testid="stHeader"] {
        background: #ffffff;
    }

    /* Hero header */
    .hero {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-top: 8px solid #003da5;
        border-image: linear-gradient(
            90deg,
            #0057a8 0%,
            #003da5 14%,
            #007a33 28%,
            #006b3f 42%,
            #c8102e 56%,
            #7b1113 70%,
            #fdb81e 84%,
            #0057a8 100%
        ) 1;
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
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
        border-radius: 999px;
        padding: 0.2rem 0.85rem;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero h1 {
        color: #111827;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.15;
        margin: 0;
        padding: 0;
    }
    .hero p {
        color: #4b5563;
        font-size: 1.02rem;
        margin: 0.6rem 0 0 0;
        max-width: 640px;
    }

    /* Section titles */
    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.65;
        margin: 1.4rem 0 0.6rem 0;
    }

    /* Bordered cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.07);
    }

    /* Team headers inside the input cards */
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
    .tag-a, .tag-b { color: #ffffff; }

    /* Matchup banner */
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
        letter-spacing: -0.01em;
    }
    .matchup .team { font-weight: 800; }
    .matchup .vs {
        font-size: 0.85rem;
        font-weight: 800;
        background: rgba(127, 127, 127, 0.2);
        border-radius: 999px;
        padding: 0.3rem 0.8rem;
        letter-spacing: 0.1em;
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

    /* Call-to-action button */
    .stButton > button {
        width: 100%;
        background: #003da5;
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        border: none;
        border-radius: 14px;
        padding: 0.85rem 1.5rem;
        box-shadow: 0 8px 20px rgba(47, 128, 237, 0.38);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover {
        color: #ffffff;
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(47, 128, 237, 0.5);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Probability split bar */
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
        white-space: nowrap;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.8;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        padding: 0.2rem 0.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-weight: 800;
    }

    /* Footer */
    .footer-note {
        text-align: center;
        font-size: 0.8rem;
        opacity: 0.55;
        margin-top: 2rem;
    }

    /* Hide default Streamlit chrome for a cleaner look */
    #MainMenu, footer { visibility: hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading prediction model...")
def load_model(path: str):
    """Load the trained model once and reuse it across reruns."""
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def clean_name(raw: str, fallback: str) -> str:
    """Trim the team name and fall back to a default if it is empty."""
    name = (raw or "").strip()
    return name if name else fallback


def school_text_color(school: str) -> str:
    return "#111827" if school == "UST" else "#ffffff"


def describe_streak(value: int) -> str:
    """Turn a streak number into plain words for the slider caption."""
    if value > 0:
        return f"On a {value}-game winning streak"
    if value < 0:
        return f"On a {abs(value)}-game losing streak"
    return "No active streak"


def build_feature_frame(inputs: dict, model) -> pd.DataFrame:
    """
    Build the one-row DataFrame the model expects.
    If the model remembers its training column names, we check against them
    and reorder our columns to match.
    """
    round_value = (
        inputs["round"] if ROUND_AS_TEXT else ROUND_ENCODING[inputs["round"]]
    )

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
        expected = list(expected)
        missing = [col for col in expected if col not in frame.columns]
        if missing:
            raise ValueError(
                "The model expects different column names than this app sends.\n"
                f"Model expects: {expected}\n"
                f"App sends: {list(frame.columns)}\n"
                "Update FEATURE_COLUMNS near the top of app.py to match."
            )
        frame = frame[expected]

    return frame


def predict_win_probability(model, frame: pd.DataFrame) -> float:
    """Return the probability (0 to 1) that Team A wins."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        classes = list(getattr(model, "classes_", [0, 1]))

        # Find which class means "win". Default to the last class.
        win_index = next(
            (i for i, c in enumerate(classes) if str(c).strip().lower() in WIN_LABELS),
            len(classes) - 1,
        )
        return float(probabilities[win_index])

    # Fallback for models without probabilities: a hard 0% or 100%.
    prediction = model.predict(frame)[0]
    return 1.0 if str(prediction).strip().lower() in WIN_LABELS else 0.0


def render_probability_bar(
    team_a: str,
    team_b: str,
    p_a: float,
    color_a: str,
    color_b: str,
) -> None:
    """Draw a two-color split bar showing both win probabilities."""
    p_b = 1.0 - p_a
    pct_a, pct_b = p_a * 100, p_b * 100

    label_a = f"{pct_a:.1f}%" if pct_a >= 12 else ""
    label_b = f"{pct_b:.1f}%" if pct_b >= 12 else ""

    st.markdown(
        f"""
        <div class="prob-bar">
            <div class="seg" style="width:{pct_a:.2f}%; background:{color_a};">{label_a}</div>
            <div class="seg" style="width:{pct_b:.2f}%; background:{color_b};">{label_b}</div>
        </div>
        <div class="prob-labels">
            <span>{html.escape(team_a)}</span>
            <span>{html.escape(team_b)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <span class="badge">UAAP Season 87 · Volleyball</span>
        <h1>Match Outcome Predictor</h1>
        <p>Enter each team's pre-match momentum and see who the model favors.
        Wins before the match, current streaks, and the tournament round all
        feed into the prediction.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load the model (stop the app cleanly if it fails)
# ---------------------------------------------------------------------------
try:
    model = load_model(str(MODEL_PATH))
except FileNotFoundError:
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. "
        "Place it in the same folder as `app.py` and refresh the page."
    )
    st.stop()
except Exception as exc:  # noqa: BLE001 - show any load problem to the user
    st.error("The model could not be loaded.")
    with st.expander("Technical details"):
        st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Matchup banner placeholder (filled in after the inputs so it updates live)
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Matchup Preview</div>', unsafe_allow_html=True)
banner_slot = st.empty()

# ---------------------------------------------------------------------------
# Input cards: Team A on the left, Opponent on the right
# ---------------------------------------------------------------------------
st.markdown('<div class="section-title">Pre-Match Momentum</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    with st.container(border=True):
        team_a_school = st.selectbox(
            "Team A school",
            options=list(SCHOOL_COLORS),
            index=list(SCHOOL_COLORS).index("NU"),
            key="team_a_school",
        )
        team_a_color = SCHOOL_COLORS[team_a_school]
        st.markdown(
            f'<span class="team-tag tag-a" style="background:{team_a_color}; '
            f'color:{school_text_color(team_a_school)};">{team_a_school}</span>',
            unsafe_allow_html=True,
        )
        team_a_raw = st.text_input(
            "Team name",
            value=team_a_school,
            key="team_a_name",
        )
        team_a_wins = st.slider(
            "Wins before match",
            min_value=0,
            max_value=14,
            value=7,
            key="team_a_wins",
            help="Number of games Team A has won so far in the season.",
        )
        team_a_streak = st.slider(
            "Win streak",
            min_value=0,
            max_value=14,
            value=2,
            key="team_a_streak",
            help="Number of consecutive wins before the match.",
        )
        st.caption(describe_streak(team_a_streak))

with col_b:
    with st.container(border=True):
        team_b_school = st.selectbox(
            "Opponent school",
            options=list(SCHOOL_COLORS),
            index=list(SCHOOL_COLORS).index("DLSU"),
            key="team_b_school",
        )
        team_b_color = SCHOOL_COLORS[team_b_school]
        st.markdown(
            f'<span class="team-tag tag-b" style="background:{team_b_color}; '
            f'color:{school_text_color(team_b_school)};">{team_b_school}</span>',
            unsafe_allow_html=True,
        )
        team_b_raw = st.text_input(
            "Team name",
            value=team_b_school,
            key="team_b_name",
        )
        team_b_wins = st.slider(
            "Wins before match",
            min_value=0,
            max_value=14,
            value=7,
            key="team_b_wins",
            help="Number of games the opponent has won so far in the season.",
        )
        team_b_streak = st.slider(
            "Win streak",
            min_value=0,
            max_value=14,
            value=1,
            key="team_b_streak",
            help="Number of consecutive wins before the match.",
        )
        st.caption(describe_streak(team_b_streak))

# Tournament round card
st.markdown('<div class="section-title">Tournament Stage</div>', unsafe_allow_html=True)
with st.container(border=True):
    selected_round = st.selectbox(
        "Tournament round",
        options=ROUND_OPTIONS,
        key="tournament_round",
        help="Later rounds usually mean higher stakes.",
    )

# Resolve the display names and fill in the live banner
team_a_name = clean_name(team_a_raw, "Team A")
team_b_name = clean_name(team_b_raw, "Opponent")

banner_slot.markdown(
    f"""
    <div class="matchup">
        <span class="team" style="color:{team_a_color};">{html.escape(team_a_name)}</span>
        <span class="vs">VS</span>
        <span class="team" style="color:{team_b_color};">{html.escape(team_b_name)}</span>
        <span class="round">{html.escape(selected_round)}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
st.write("")
predict_clicked = st.button("🏐  Predict Match Outcome", type="primary")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if predict_clicked:
    # Basic input checks before calling the model
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

    except Exception as exc:  # noqa: BLE001 - never crash the app on a bad prediction
        st.error("Something went wrong while making the prediction.")
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")

    else:
        st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

        with st.container(border=True):
            # Headline verdict
            gap = abs(p_a - 0.5)
            if gap < TOSS_UP_MARGIN:
                st.warning(
                    f"**Too close to call.** The model sees "
                    f"{team_a_name} vs {team_b_name} as nearly even.",
                    icon="⚖️",
                )
            elif p_a > 0.5:
                st.success(
                    f"**{team_a_name} is predicted to WIN** against {team_b_name}.",
                    icon="🏆",
                )
            else:
                st.error(
                    f"**{team_a_name} is predicted to LOSE** to {team_b_name}.",
                    icon="⚠️",
                )

            # Probability metrics
            m1, m2, m3 = st.columns(3)
            m1.metric(
                label=f"{team_a_name} win chance",
                value=f"{p_a * 100:.1f}%",
                delta=f"{(p_a - p_b) * 100:+.1f} pts vs opponent",
            )
            m2.metric(
                label=f"{team_b_name} win chance",
                value=f"{p_b * 100:.1f}%",
                delta=f"{(p_b - p_a) * 100:+.1f} pts vs {team_a_name}",
            )
            m3.metric(
                label="Model confidence",
                value=f"{max(p_a, p_b) * 100:.1f}%",
                help="The probability of the model's favored outcome.",
            )

            # Visual split bar
            render_probability_bar(
                team_a_name,
                team_b_name,
                p_a,
                team_a_color,
                team_b_color,
            )

        # Show exactly what was sent to the model
        with st.expander("See the inputs used for this prediction"):
            summary = pd.DataFrame(
                {
                    "Stat": ["Wins before match", "Win streak", "Round"],
                    team_a_name: [team_a_wins, team_a_streak, selected_round],
                    team_b_name: [team_b_wins, team_b_streak, selected_round],
                }
            )
            st.dataframe(summary, hide_index=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="footer-note">Predictions come from a machine learning model '
    "trained on past UAAP results. They show likelihood, not certainty.</div>",
    unsafe_allow_html=True,
)