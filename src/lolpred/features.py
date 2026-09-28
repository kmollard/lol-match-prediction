"""Leak-free feature engineering.

Every feature for a game is computed from games that happened *before* it.
`FeatureBuilder.stream()` walks training games in date order: it emits the
features for each game from the current state, then updates the state with
that game's result. For held-out / 2023 games, `frozen()` emits features from
the final state without updating, which mirrors the real contest (no 2023
results are available at prediction time).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from .data import Game, POSITIONS

ELO_INIT = 1500.0
ELO_K = 24.0
# Bayesian smoothing strengths (pseudo-counts) for win-rate / kill averages
CHAMP_WR_PRIOR = 20
KILL_PRIOR = 10
LEN_PRIOR = 20


def _expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


class FeatureBuilder:
    def __init__(self):
        self.player_elo: dict[str, float] = defaultdict(lambda: ELO_INIT)
        self.team_elo: dict[str, float] = defaultdict(lambda: ELO_INIT)
        self.player_games: Counter = Counter()
        self.player_champs: dict[str, Counter] = defaultdict(Counter)
        self.champ_wins: Counter = Counter()
        self.champ_games: Counter = Counter()
        # kills: sums and counts at several granularities
        self.kill_sum = defaultdict(float)
        self.kill_n = Counter()
        # game length by league and by champion
        self.len_sum = defaultdict(float)
        self.len_n = Counter()
        self.global_len = (0.0, 0)
        self.red_wins = (0, 0)

    # ---------- helpers reading the current state ----------
    def _smoothed(self, key, prior_mean, strength, sums=None, counts=None):
        sums = self.kill_sum if sums is None else sums
        counts = self.kill_n if counts is None else counts
        n = counts[key]
        return (sums[key] + prior_mean * strength) / (n + strength)

    def _pos_mean_kills(self, pos):
        n = self.kill_n[("pos", pos)]
        return self.kill_sum[("pos", pos)] / n if n else 2.8

    def _global_len(self):
        s, n = self.global_len
        return s / n if n else 1923.0

    def _is_on_main(self, p) -> int:
        c = self.player_champs.get(p.pid)
        if not c:
            return 0
        return int(c.most_common(1)[0][0] == p.champion)

    def _champ_wr(self, champ) -> float:
        return (self.champ_wins[champ] + 0.5 * CHAMP_WR_PRIOR) / (self.champ_games[champ] + CHAMP_WR_PRIOR)

    def _side_summary(self, players):
        elos = [self.player_elo[p.pid] for p in players]
        seen = [self.player_games[p.pid] > 0 for p in players]
        tid = players[0].tid if players else None
        return {
            "elo": float(np.mean(elos)) if elos else ELO_INIT,
            "team_elo": self.team_elo[tid] if tid else ELO_INIT,
            "champ_wr": float(np.mean([self._champ_wr(p.champion) for p in players])) if players else 0.5,
            "on_main": sum(self._is_on_main(p) for p in players),
            "frac_seen": float(np.mean(seen)) if seen else 0.0,
            "exp_games": float(np.mean([np.log1p(self.player_games[p.pid]) for p in players])) if players else 0.0,
        }

    # ---------- feature rows ----------
    def game_features(self, g: Game) -> dict:
        r, b = self._side_summary(g.red), self._side_summary(g.blue)
        league_len = self._smoothed(("lid", g.lid), self._global_len(), LEN_PRIOR, self.len_sum, self.len_n)
        champ_len = [self._smoothed(("champ", p.champion), self._global_len(), LEN_PRIOR, self.len_sum, self.len_n)
                     for p in g.red + g.blue]
        return {
            "gid": g.gid,
            "elo_diff": r["elo"] - b["elo"],
            "team_elo_diff": r["team_elo"] - b["team_elo"],
            "champ_wr_diff": r["champ_wr"] - b["champ_wr"],
            "on_main_diff": r["on_main"] - b["on_main"],
            "elo_sum": r["elo"] + b["elo"] - 2 * ELO_INIT,
            "abs_elo_diff": abs(r["elo"] - b["elo"]),
            "frac_seen": (r["frac_seen"] + b["frac_seen"]) / 2,
            "exp_games": (r["exp_games"] + b["exp_games"]) / 2,
            "league_len": league_len,
            "champ_len": float(np.mean(champ_len)) if champ_len else self._global_len(),
            "winner_red": g.winner_red,
            "gamelength": g.gamelength,
        }

    def player_features(self, g: Game) -> list[dict]:
        rows = []
        r, b = self._side_summary(g.red), self._side_summary(g.blue)
        for side, players, own, opp in (("Red", g.red, r, b), ("Blue", g.blue, b, r)):
            for slot, p in enumerate(players, start=1):
                pos_mean = self._pos_mean_kills(p.position)
                rows.append({
                    "gid": g.gid, "side": side, "slot": slot,
                    "pos_code": POSITIONS.index(p.position) if p.position in POSITIONS else -1,
                    "is_red": int(side == "Red"),
                    "pos_mean_k": pos_mean,
                    "champ_pos_k": self._smoothed(("cp", p.champion, p.position), pos_mean, KILL_PRIOR),
                    "player_k": self._smoothed(("pid", p.pid), pos_mean, KILL_PRIOR),
                    "player_pos_k": self._smoothed(("pp", p.pid, p.position), pos_mean, KILL_PRIOR),
                    "league_pos_k": self._smoothed(("lp", g.lid, p.position), pos_mean, KILL_PRIOR),
                    "player_n": np.log1p(self.player_games[p.pid]),
                    "own_elo": own["elo"], "opp_elo": opp["elo"],
                    "elo_edge": own["elo"] - opp["elo"],
                    "team_elo_edge": own["team_elo"] - opp["team_elo"],
                    "champ_wr_edge": own["champ_wr"] - opp["champ_wr"],
                    "on_main": self._is_on_main(p),
                    "kills": p.kills,
                })
        return rows

    # ---------- state update ----------
    def update(self, g: Game) -> None:
        if g.winner_red is None:
            return
        res_red = float(g.winner_red)
        # player Elo: every player moves by the team-level surprise
        if g.red and g.blue:
            r_mean = np.mean([self.player_elo[p.pid] for p in g.red])
            b_mean = np.mean([self.player_elo[p.pid] for p in g.blue])
            delta = ELO_K * (res_red - _expected(r_mean, b_mean))
            for p in g.red:
                self.player_elo[p.pid] += delta
            for p in g.blue:
                self.player_elo[p.pid] -= delta
            tr, tb = g.red[0].tid, g.blue[0].tid
            tdelta = ELO_K * (res_red - _expected(self.team_elo[tr], self.team_elo[tb]))
            self.team_elo[tr] += tdelta
            self.team_elo[tb] -= tdelta
        for side_players, won in ((g.red, res_red), (g.blue, 1 - res_red)):
            for p in side_players:
                self.player_games[p.pid] += 1
                self.player_champs[p.pid][p.champion] += 1
                self.champ_games[p.champion] += 1
                self.champ_wins[p.champion] += won
                if p.kills is not None:
                    for key in (("pos", p.position), ("cp", p.champion, p.position), ("pid", p.pid),
                                ("pp", p.pid, p.position), ("lp", g.lid, p.position)):
                        self.kill_sum[key] += p.kills
                        self.kill_n[key] += 1
                if g.gamelength is not None:
                    self.len_sum[("champ", p.champion)] += g.gamelength
                    self.len_n[("champ", p.champion)] += 1
        if g.gamelength is not None:
            self.len_sum[("lid", g.lid)] += g.gamelength
            self.len_n[("lid", g.lid)] += 1
            s, n = self.global_len
            self.global_len = (s + g.gamelength, n + 1)

    # ---------- drivers ----------
    def stream(self, games: list[Game]):
        """Features for each game from the past only, then update. Returns (game_df, player_df)."""
        grows, prows = [], []
        for g in games:
            if g.complete:
                grows.append(self.game_features(g))
                prows.extend(self.player_features(g))
            self.update(g)
        return pd.DataFrame(grows), pd.DataFrame(prows)

    def frozen(self, games: list[Game]):
        """Features from the current (frozen) state; no updates."""
        grows = [self.game_features(g) for g in games]
        prows = [row for g in games for row in self.player_features(g)]
        return pd.DataFrame(grows), pd.DataFrame(prows)


WINNER_FEATURES = ["elo_diff", "team_elo_diff", "champ_wr_diff", "on_main_diff"]
LENGTH_FEATURES = ["league_len", "champ_len", "abs_elo_diff", "elo_sum", "champ_wr_diff",
                   "frac_seen", "exp_games"]
KILL_FEATURES = ["pos_code", "is_red", "pos_mean_k", "champ_pos_k", "player_k", "player_pos_k",
                 "league_pos_k", "player_n", "own_elo", "opp_elo", "elo_edge", "team_elo_edge",
                 "champ_wr_edge", "on_main"]
