# UAAP Volleyball Match Predictor

A Streamlit web app that estimates the chance of a win or a loss in a UAAP Season 87 Women's Volleyball match, using only information available before the match starts.

**Live app:** https://uaap-web-app-7kncnzfkla9utmegq7tctj.streamlit.app/
**Repository:** https://github.com/JhonRoy28/UAAP-Web-App

This is the final group project for the Machine Learning course at National University (Instructor: Dr. Armida P. Salazar).

---

## What the app does

You pick two teams, choose the round, and set each team's pre-match record. The app then shows each team's estimated chance of winning and says which team the model favors. If the two chances are within 5 percentage points of each other, it says the match is too close to call.

## Model used

| Item | Details |
|---|---|
| Task | Binary classification (`Match_Outcome`: 1 = win, 0 = loss, from the selected team's side) |
| Selected model | Logistic Regression (`random_state=42`) |
| Compared with | K-Nearest Neighbors (K=5) and Random Forest |
| Preprocessing | `StandardScaler` on the four numeric features, `OneHotEncoder` on `Team`, `Opponent`, and `Round`, all inside one scikit-learn `Pipeline` |
| Saved file | `uaap_volleyball_model.pkl` |
| Validation | 5-fold stratified cross-validation on the 96 training records (mean accuracy 62.47%) |
| Final test | 24 held-out records (accuracy 66.67%) |

The app uses the same saved pipeline that was trained and evaluated in the paper.

## Inputs and outputs

| Input | What it means | Allowed values |
|---|---|---|
| Team | The team you want the prediction for | Adamson, Ateneo, DLSU, FEU, NU, UE, UP, UST |
| Opponent | The opposing team | Any team except the one selected above |
| Round | Stage of the tournament | First Round, Second Round, Final Four, Finals |
| Wins before the match | Matches the team has won so far this season | 0 to 13 |
| Win streak | Consecutive wins going into the match. A negative number means a losing streak | -13 to 5 |

**Output:** the win probability (%) for each team, a progress bar, and a plain sentence naming the favored team.

The slider ranges come from the values that appear in the Season 87 data.

## Dataset

- **File:** `UAAP_Season87_Womens_Volleyball_Full__1_.csv`
- **Size:** 120 records, representing 60 actual matches (28 first-round, 28 second-round, 2 Final Four, 2 Finals).
- **Structure:** every match appears twice, once from each team's side (dual-perspective).
- **Columns:** `Team`, `Opponent`, `Round`, `Team_Wins_Before`, `Opp_Wins_Before`, `Team_Win_Streak`, `Opp_Win_Streak`, `Match_Outcome`.
- **No data leakage:** all features use only results from earlier matches. Scores and set scores from the match itself are never used as inputs.
- **Source:** match schedules and results reported by Rappler, "Team standings, results: UAAP Season 87 volleyball" (J. Payo, May 2025). The data is publicly reported and used for non-commercial academic purposes only.

## Run it on your computer

You need **Python 3.12** and Git. Python 3.14 is not supported yet by `scikit-learn==1.6.1` (see Troubleshooting).

**Windows (PowerShell):**

```
git clone https://github.com/JhonRoy28/UAAP-Web-App.git
cd UAAP-Web-App
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

**macOS / Linux:**

```
git clone https://github.com/JhonRoy28/UAAP-Web-App.git
cd UAAP-Web-App
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser, usually at http://localhost:8501.

## Files in this repository

```
UAAP-Web-App/
  app.py                                         Streamlit application
  requirements.txt                               Python dependencies
  uaap_volleyball_model.pkl                      Saved model pipeline
  UAAP_Season87_Womens_Volleyball_Full__1_.csv   Training dataset
  README.md                                      This file
```

`app.py` expects the `.pkl` file to be in the same folder. The CSV is optional: if it is missing, the app still runs and only hides the data preview.

## Limitations

- **Small dataset.** Only 60 real matches from one season, so the numbers are rough estimates and not reliable forecasts.
- **Doubled records.** Each match appears twice, so the records are not fully independent. A random split can place the two sides of the same match in different sets, which can make test scores look better than they really are.
- **Team names carry weight.** The model learned team strength from Season 87 results. A team that lost all season stays an underdog even when you enter strong stats, and unusual stat combinations the model never saw give less trustworthy results.
- **Limited features.** Only team records, win streaks, and the round are used. Lineups, injuries, and in-match performance are not captured.
- **One season only.** The model has not been tested on other seasons.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails while building `scikit-learn` | You are probably on Python 3.14. Use Python 3.12 and a fresh virtual environment as shown above. |
| "Could not find uaap_volleyball_model.pkl" | Put the `.pkl` file in the same folder as `app.py`, with exactly that name. |
| The dataset preview does not show | Check that the CSV is in the same folder with exactly the name `UAAP_Season87_Womens_Volleyball_Full__1_.csv`, then restart the app. |
| The live app shows a sleeping page | Click the button on the page to wake it up and wait about a minute. |
| The app shows a yellow warning | Change an input and try again. If it keeps happening, restart the app. |

## Authors

[Add group member names here]