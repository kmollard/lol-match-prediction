"""Loading and reshaping the contest data.

train.csv is player-level (one row per player per game); Xte.csv is game-level
(one row per game with 5 player slots per side). Everything downstream works on
a common "roster" representation: one dict per game with both sides' players.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SIDES = ("Red", "Blue")
POSITIONS = ("top", "jng", "mid", "bot", "sup")


@dataclass
class Player:
    pid: str
    champion: str
    position: str
    tid: str
    kills: float | None = None  # None for test games (unknown)


@dataclass
class Game:
    gid: str
    date: pd.Timestamp
    lid: str
    red: list[Player] = field(default_factory=list)
    blue: list[Player] = field(default_factory=list)
    winner_red: int | None = None
    gamelength: float | None = None

    @property
    def complete(self) -> bool:
        return len(self.red) == 5 and len(self.blue) == 5


def load_train(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["position"] = df["position"].str.lower()
    df["side"] = df["side"].str.title()
    return df


def train_to_games(df: pd.DataFrame) -> list[Game]:
    """Player rows -> chronologically ordered list of Game objects."""
    df = df.sort_values(["date", "gid", "side", "position"])
    games: list[Game] = []
    for gid, g in df.groupby("gid", sort=False):
        first = g.iloc[0]
        game = Game(gid=gid, date=first["date"], lid=first["lid"],
                    gamelength=float(first["gamelength"]))
        red_rows = g[g["side"] == "Red"]
        if len(red_rows):
            game.winner_red = int(red_rows["result"].max())
        else:  # only Blue rows present
            game.winner_red = int(1 - g["result"].max())
        for side, bucket in (("Red", game.red), ("Blue", game.blue)):
            for r in g[g["side"] == side].itertuples(index=False):
                bucket.append(Player(r.pid, r.champion, r.position, r.tid, float(r.k)))
        games.append(game)
    games.sort(key=lambda x: (x.date, x.gid))
    return games


def load_test_games(path: str | Path) -> list[Game]:
    """Xte.csv (wide, game-level) -> list of Game objects without labels."""
    x = pd.read_csv(path, parse_dates=["date"])
    games = []
    for r in x.itertuples(index=False):
        r = r._asdict()
        game = Game(gid=r["gid"], date=r["date"], lid=r["lid"])
        for side, bucket in (("Red", game.red), ("Blue", game.blue)):
            tid = r[f"tid_{side}"]
            for i in range(1, 6):
                bucket.append(Player(r[f"pid{i}_{side}"], r[f"champion{i}_{side}"],
                                     str(r[f"position{i}_{side}"]).lower(), tid))
        games.append(game)
    return games


def chronological_split(games: list[Game], holdout_frac: float = 0.15):
    """Last `holdout_frac` of games (by date) become the validation set."""
    cut = int(len(games) * (1 - holdout_frac))
    return games[:cut], games[cut:]
