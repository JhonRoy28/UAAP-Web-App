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
| Selected model | Random Forest Classifier (`random_state=42`, default settings, no tuning) |
| Compared with | Logistic Regression and K-Nearest Neighbors (K=5), under the same conditions |
| Preprocessing | Inside one scikit-learn `Pipeline`: median or most-frequent imputation (a safeguard, the data has no missing values), `StandardScaler` on the four numeric features, and `OneHotEncoder` on `Team`, `Opponent`, and `Round` |
| Saved file | `uaap_volleyball_model.pkl` |
| Split | Chronological 80/20: the first 96 records (48 matches) for training and the last 24 records (12 matches) for testing. Both records of a match always stay together. |
| Validation | 5-fold time-series cross-validation on the training records (mean accuracy 63.75%) |
| Final test | 24 held-out records from the last 12 matches of the season (accuracy 66.67%, macro F1 0.667) |

Model comparison:

| Model | Mean CV accuracy | Test accuracy |
|---|---|---|
| Random Forest (selected) | 63.75% | 66.67% |
| Logistic Regression | 55.00% | 75.00% |
| K-Nearest Neighbors | 55.00% | 58.33% |

The model was selected by cross-validation accuracy only. The test set was used once, after the selection. Because the test set has only 12 matches, the differences between models are small and uncertain.

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
- **Structure:** every match appears twice, once from each team's side (dual-perspective). The rows are in date order with the two rows of each match next to each other, so do not shuffle the file.
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

## Retraining the model

The training notebook is in `notebooks/UAAP_Chronological_Model_Training.ipynb`. It runs in Google Colab or on a normal computer with Jupyter. It reads the CSV, applies the chronological split, runs the time-series cross-validation, compares the three algorithms, and saves the winner as `uaap_volleyball_model.pkl` plus a results table (`uaap_chronological_model_results.csv`). It needs `scikit-learn==1.6.1` and the notebook checks for it.

## Files in this repository

```
UAAP-Web-App/
  app.py                                         Streamlit application
  requirements.txt                               Python dependencies
  uaap_volleyball_model.pkl                      Saved model pipeline (Random Forest)
  UAAP_Season87_Womens_Volleyball_Full__1_.csv   Training dataset
  notebooks/
    UAAP_Chronological_Model_Training.ipynb      Training and evaluation notebook
    uaap_chronological_model_results.csv         Model comparison results
  README.md                                      This file
```

`app.py` expects the `.pkl` file to be in the same folder. The CSV is optional: if it is missing, the app still runs and only hides the data preview.

## Limitations

- **Small dataset.** Only 60 real matches from one season, so the numbers are rough estimates and not reliable forecasts.
- **Small test set.** The test set has only 12 matches, so one match changes the accuracy by about 8 percentage points. Cross-validation results also varied a lot between folds, so the ranking of the models may change with more data.
- **Unseen rounds.** The Final Four and Finals matches appear only in the test set, so the model never saw those rounds during training.
- **Team names carry weight.** The model learned team strength from Season 87 results. A team that lost all season stays an underdog even when you enter strong stats, and unusual stat combinations the model never saw give less trustworthy results.
- **Limited features.** Only team records, win streaks, and the round are used. Lineups, injuries, and in-match performance are not captured.
- **No tuning, one season.** No hyperparameter tuning was done, and the model has not been tested on other seasons.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails while building `scikit-learn` | You are probably on Python 3.14. Use Python 3.12 and a fresh virtual environment as shown above. |
| "Could not find uaap_volleyball_model.pkl" | Put the `.pkl` file in the same folder as `app.py`, with exactly that name. |
| The dataset preview does not show | Check that the CSV is in the same folder with exactly the name `UAAP_Season87_Womens_Volleyball_Full__1_.csv`, then restart the app. |
| The live app shows a sleeping page | Click the button on the page to wake it up and wait about a minute. |
| The app shows a yellow warning | Change an input and try again. If it keeps happening, restart the app. |

## Authors

GAMBOA, JHON ROY B
DE CHAVEZ, LEONA
CULUBONG, BENCH MATTHEW C.