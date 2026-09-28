"""The three prediction models, one per contest target."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import KILL_FEATURES, LENGTH_FEATURES, WINNER_FEATURES

SEED = 440


def fit_winner(games: pd.DataFrame, features=WINNER_FEATURES):
    """Logistic regression on red-minus-blue differences (the report's model)."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(games[features], games["winner_red"].astype(int))
    return model


def predict_winner(model, games: pd.DataFrame, features=WINNER_FEATURES) -> np.ndarray:
    return model.predict_proba(games[features])[:, 1]


def fit_kills(players: pd.DataFrame, features=KILL_FEATURES):
    """XGBoost with an absolute-error objective, since the contest scores kills by MAE."""
    model = XGBRegressor(objective="reg:absoluteerror", n_estimators=400, max_depth=5,
                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=-1)
    model.fit(players[features], players["kills"])
    return model


def predict_kills(model, players: pd.DataFrame, features=KILL_FEATURES) -> np.ndarray:
    return np.clip(model.predict(players[features]), 0, None)


def fit_length(games: pd.DataFrame, features=LENGTH_FEATURES):
    """XGBoost with squared error, since the contest scores game length by RMSE."""
    model = XGBRegressor(objective="reg:squarederror", n_estimators=300, max_depth=3,
                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=SEED, n_jobs=-1)
    model.fit(games[features], games["gamelength"])
    return model


def predict_length(model, games: pd.DataFrame, features=LENGTH_FEATURES) -> np.ndarray:
    return model.predict(games[features])
