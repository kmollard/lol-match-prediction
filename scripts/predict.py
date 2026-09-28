"""Train on all 2018–2022 games and write contest predictions for 2023 (Xte.csv).

    python scripts/predict.py --train data/train.csv --test data/Xte.csv --out submission.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lolpred import models  # noqa: E402
from lolpred.data import load_test_games, load_train, train_to_games  # noqa: E402
from lolpred.features import FeatureBuilder  # noqa: E402

BURN_IN_END = pd.Timestamp("2018-07-01")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.csv")
    ap.add_argument("--test", default="data/Xte.csv")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    games = train_to_games(load_train(args.train))
    test_games = load_test_games(args.test)

    fb = FeatureBuilder()
    g_tr, p_tr = fb.stream(games)
    g_te, p_te = fb.frozen(test_games)          # 2023: frozen end-of-2022 state
    dates = {g.gid: g.date for g in games}
    g_tr = g_tr[(g_tr["gid"].map(dates) >= BURN_IN_END).values]
    p_tr = p_tr[p_tr["gid"].isin(set(g_tr["gid"]))]

    win = models.predict_winner(models.fit_winner(g_tr), g_te)
    length = models.predict_length(models.fit_length(g_tr), g_te)
    p_te["k_pred"] = np.round(models.predict_kills(models.fit_kills(p_tr), p_te)).astype(int)

    kills_wide = (p_te.assign(col=lambda d: "k" + d["slot"].astype(str) + "_" + d["side"])
                  .pivot(index="gid", columns="col", values="k_pred"))
    cols = [f"k{i}_{s}" for s in ("Red", "Blue") for i in range(1, 6)]
    sub = pd.DataFrame({"gid": g_te["gid"], "winner_Red": win})
    sub = sub.join(kills_wide[cols], on="gid")
    sub["gamelength"] = np.round(length).astype(int)
    sub.to_csv(args.out, index=False)
    print(f"wrote {len(sub)} rows → {args.out}")


if __name__ == "__main__":
    main()
