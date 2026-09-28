"""The contest's scoring rule.

score = 1/3 * 2 * (AUC(winner) - 0.5)
      + 1/3 * (1 - MAE(kills) / 2.83)
      + 1/3 * (1 - RMSE(gamelength) / 339.15)

The constants make a coin toss, all-zero kills, and the mean game length each
score 0 on the official test set. Reported as a percentage.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score

KILLS_MAE_CONST = 2.83
LENGTH_RMSE_CONST = 339.15


def contest_score(win_true, win_prob, kills_true, kills_pred, len_true, len_pred) -> dict:
    auc = roc_auc_score(win_true, win_prob)
    mae = mean_absolute_error(np.ravel(kills_true), np.ravel(kills_pred))
    rmse = float(np.sqrt(np.mean((np.asarray(len_true) - np.asarray(len_pred)) ** 2)))
    parts = {
        "winner": 100 / 3 * 2 * (auc - 0.5),
        "kills": 100 / 3 * (1 - mae / KILLS_MAE_CONST),
        "length": 100 / 3 * (1 - rmse / LENGTH_RMSE_CONST),
    }
    return {"score": sum(parts.values()), "auc": auc, "kills_mae": mae, "length_rmse": rmse,
            **{f"part_{k}": v for k, v in parts.items()}}
