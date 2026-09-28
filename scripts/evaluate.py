"""Chronological hold-out evaluation with the contest's scoring rule.

Trains on the first 85% of 2018–2022 games and scores on the last 15%
(most of 2022), using features frozen at the end of the training window,
exactly as they would be for the 2023 test set.

    python scripts/evaluate.py --train data/train.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lolpred import models  # noqa: E402
from lolpred.data import chronological_split, load_train, train_to_games  # noqa: E402
from lolpred.features import FeatureBuilder, WINNER_FEATURES  # noqa: E402
from lolpred.metrics import contest_score  # noqa: E402

BURN_IN_END = pd.Timestamp("2018-07-01")  # early games have uninformative Elo; used for state only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    games = train_to_games(load_train(args.train))
    fit_games, hold_games = chronological_split(games, args.holdout_frac)
    hold_games = [g for g in hold_games if g.complete]
    print(f"train games: {len(fit_games)}  ({fit_games[0].date.date()} → {fit_games[-1].date.date()})")
    print(f"holdout games: {len(hold_games)}  ({hold_games[0].date.date()} → {hold_games[-1].date.date()})")

    fb = FeatureBuilder()
    g_tr, p_tr = fb.stream(fit_games)
    g_ho, p_ho = fb.frozen(hold_games)
    dates = {g.gid: g.date for g in fit_games}
    keep = g_tr["gid"].map(dates) >= BURN_IN_END
    g_tr = g_tr[keep.values]
    p_tr = p_tr[p_tr["gid"].isin(set(g_tr["gid"]))]

    # ground truth for the holdout
    y_win = g_ho["winner_red"].astype(int).values
    y_len = g_ho["gamelength"].values
    y_k = p_ho["kills"].values

    results = {}

    # --- baselines: the "zero point" of each score component ---
    results["baseline (coin toss, zero kills, mean length)"] = contest_score(
        y_win, np.full(len(y_win), 0.5), y_k, np.zeros_like(y_k), y_len,
        np.full(len(y_len), g_tr["gamelength"].mean()))
    results["baseline (position-mean kills, league-mean length)"] = contest_score(
        y_win, np.full(len(y_win), 0.5), y_k, p_ho["pos_mean_k"].round().values,
        y_len, g_ho["league_len"].values)

    # --- winner ablation: which features carry the signal? ---
    ablation = {}
    feature_sets = {
        "Elo only": ["elo_diff"],
        "+ champion win rate": ["elo_diff", "champ_wr_diff"],
        "+ on-main (report's model)": ["elo_diff", "champ_wr_diff", "on_main_diff"],
        "+ team Elo (v2)": WINNER_FEATURES,
    }
    from sklearn.metrics import roc_auc_score
    for name, feats in feature_sets.items():
        m = models.fit_winner(g_tr, feats)
        ablation[name] = float(roc_auc_score(y_win, models.predict_winner(m, g_ho, feats)))

    # --- full v2 pipeline ---
    m_win = models.fit_winner(g_tr)
    m_k = models.fit_kills(p_tr)
    m_len = models.fit_length(g_tr)
    p_win = models.predict_winner(m_win, g_ho)
    p_k = models.predict_kills(m_k, p_ho)
    p_len = models.predict_length(m_len, g_ho)
    results["v2: logistic regression + XGBoost"] = contest_score(y_win, p_win, y_k, p_k, y_len, p_len)
    results["v2, kills rounded to integers"] = contest_score(y_win, p_win, y_k, np.round(p_k), y_len, p_len)

    # --- reliability: win rate by predicted-probability decile ---
    cal = (pd.DataFrame({"p": p_win, "y": y_win})
           .assign(bin=lambda d: pd.qcut(d["p"], 10, labels=False, duplicates="drop"))
           .groupby("bin").agg(pred=("p", "mean"), actual=("y", "mean"), n=("y", "size")))

    summary = {
        "train_games_used": int(len(g_tr)),
        "holdout_games": int(len(g_ho)),
        "holdout_range": [str(hold_games[0].date.date()), str(hold_games[-1].date.date())],
        "holdout_unseen_player_frac": float(1 - g_ho["frac_seen"].mean()),
        "scores": results,
        "winner_auc_ablation": ablation,
        "calibration": cal.round(4).to_dict(orient="records"),
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))

    print("\n=== Contest score on holdout (%) ===")
    for k, v in results.items():
        print(f"{v['score']:6.2f}  {k}   [AUC {v['auc']:.3f} | kills MAE {v['kills_mae']:.3f} | "
              f"length RMSE {v['length_rmse']:.1f}]")
    print("\n=== Winner AUC ablation ===")
    for k, v in ablation.items():
        print(f"{v:.4f}  {k}")
    make_figures(out, ablation, cal, results)


def make_figures(out: Path, ablation: dict, cal: pd.DataFrame, results: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    names, vals = list(ablation), list(ablation.values())
    ax.barh(names[::-1], vals[::-1], color="#4C72B0")
    ax.set_xlim(0.5, max(vals) + 0.03)
    ax.axvline(0.5, color="grey", lw=1, ls="--")
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Holdout ROC AUC (winner)")
    ax.set_title("Winner model: feature ablation")

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--", label="perfect calibration")
    ax.plot(cal["pred"], cal["actual"], "o-", color="#DD8452", label="v2 logistic regression")
    ax.set_xlabel("Predicted P(Red wins)")
    ax.set_ylabel("Observed Red win rate")
    ax.set_title("Calibration on holdout (deciles)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "winner_model.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
