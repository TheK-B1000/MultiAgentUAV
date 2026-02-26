from __future__ import annotations

import math
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class OpponentChoice:
    key: str
    policy_name: str


class LeagueController:
    """
    Lightweight league + curriculum controller for Pyquaticus RLlib training.
    Supports:
      - CURRICULUM_LEAGUE
      - CURRICULUM_NO_LEAGUE
      - SELF_PLAY
    """

    def __init__(self, config: Dict):
        self.cfg = config
        self.mode = str(config.get("mode", "CURRICULUM_LEAGUE")).upper()
        self.seed = int(config.get("seed", 42))
        self.rng = random.Random(self.seed)

        cur = config.get("curriculum", {})
        phase_list = cur.get("phases", [{"name": "OP1"}, {"name": "OP2"}, {"name": "OP3"}])
        self.phases: List[Dict] = [{**p, "name": str(p.get("name", "OP3")).upper()} for p in phase_list]
        self.phase_idx = 0
        self.phase_episode_count = 0
        self.phase_recent_results: Dict[str, List[float]] = {p["name"]: [] for p in self.phases}
        self.winrate_window = int(cur.get("winrate_window", 50))

        # League/snapshot config
        lg = config.get("league", {})
        self.k_factor = float(lg.get("k_factor", 32.0))
        self.matchmaking_tau = float(lg.get("matchmaking_tau", 250.0))
        self.anchor_op3_prob = float(lg.get("anchor_op3_prob", 0.30))
        self.species_prob = float(lg.get("species_prob", 0.30))
        self.snapshot_prob = float(lg.get("snapshot_prob", 0.40))
        self.snapshot_every_iters = int(lg.get("snapshot_every_iters", 10))
        self.max_snapshots = int(lg.get("max_snapshots", 5))
        self.min_games_vs_op3_for_league = int(lg.get("min_games_vs_op3", 20))
        self.min_wr_vs_op3_for_league = float(lg.get("min_winrate_vs_op3", 0.70))

        self.league_mode = False

        # Elo state
        self.learner_key = "__LEARNER__"
        self.ratings: Dict[str, float] = {self.learner_key: 1200.0}
        self._ensure_rating("SCRIPTED:OP1")
        self._ensure_rating("SCRIPTED:OP2")
        self._ensure_rating("SCRIPTED:OP3")
        self._ensure_rating("SPECIES:RUSHER")
        self._ensure_rating("SPECIES:CAMPER")
        self._ensure_rating("SPECIES:BALANCED")
        self.snapshots: List[str] = []
        self.snapshot_paths: List[str] = []  # same order as snapshots, for deleting oldest

        # Tracking for OP3 gate
        self.op3_results: List[float] = []

        # Lifetime W/L/D for logging
        self.total_wins: int = 0
        self.total_losses: int = 0
        self.total_draws: int = 0

    @property
    def phase_name(self) -> str:
        return self.phases[min(self.phase_idx, len(self.phases) - 1)]["name"]

    @property
    def learner_rating(self) -> float:
        return float(self.ratings.get(self.learner_key, 1200.0))

    def _ensure_rating(self, key: str) -> None:
        if key not in self.ratings:
            self.ratings[key] = 1200.0

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** (-(ra - rb) / 400.0))

    def _weighted_pick(self, keys: List[str]) -> str:
        target = self.learner_rating
        weights = []
        for key in keys:
            r = float(self.ratings.get(key, 1200.0))
            dist = abs(r - target)
            weights.append(math.exp(-dist / max(1e-6, self.matchmaking_tau)) + 1e-3)
        total = sum(weights)
        if total <= 0:
            return self.rng.choice(keys)
        pick = self.rng.random() * total
        acc = 0.0
        for key, w in zip(keys, weights):
            acc += w
            if acc >= pick:
                return key
        return keys[-1]

    def _key_to_policy_name(self, key: str) -> str:
        if key.startswith("SCRIPTED:"):
            tag = key.split(":", 1)[1].lower()
            return f"scripted_{tag}_policy"
        if key.startswith("SPECIES:"):
            tag = key.split(":", 1)[1].lower()
            if tag == "rusher":
                return "species_rusher_policy"
            if tag == "camper":
                return "species_camper_policy"
            return "species_balanced_policy"
        if key.startswith("SNAPSHOT:"):
            return "snapshot_policy"
        return "scripted_op3_policy"

    def select_opponent(self) -> OpponentChoice:
        mode = self.mode
        if mode == "SELF_PLAY":
            if self.snapshots:
                snap_key = self._weighted_pick(self.snapshots)
                return OpponentChoice(key=snap_key, policy_name="snapshot_policy")
            return OpponentChoice(key="SCRIPTED:OP3", policy_name="scripted_op3_policy")

        if (mode == "CURRICULUM_NO_LEAGUE") or (mode == "CURRICULUM_LEAGUE" and not self.league_mode):
            key = f"SCRIPTED:{self.phase_name}"
            return OpponentChoice(key=key, policy_name=self._key_to_policy_name(key))

        # CURRICULUM_LEAGUE in league mode
        r = self.rng.random()
        if r < self.anchor_op3_prob:
            key = "SCRIPTED:OP3"
        elif r < (self.anchor_op3_prob + self.species_prob):
            key = self._weighted_pick(["SPECIES:RUSHER", "SPECIES:CAMPER", "SPECIES:BALANCED"])
        else:
            if self.snapshots:
                key = self._weighted_pick(self.snapshots)
            else:
                key = "SCRIPTED:OP3"
        return OpponentChoice(key=key, policy_name=self._key_to_policy_name(key))

    def record_episode(self, opponent_key: str, blue_score: float, red_score: float) -> None:
        actual = 0.5
        if blue_score > red_score:
            actual = 1.0
            self.total_wins += 1
        elif blue_score < red_score:
            actual = 0.0
            self.total_losses += 1
        else:
            self.total_draws += 1

        self.phase_episode_count += 1
        ph = self.phase_name
        rec = self.phase_recent_results.setdefault(ph, [])
        rec.append(actual)
        if len(rec) > self.winrate_window:
            del rec[0]

        if opponent_key == "SCRIPTED:OP3":
            self.op3_results.append(actual)
            if len(self.op3_results) > max(self.winrate_window, self.min_games_vs_op3_for_league):
                del self.op3_results[0]

        # Elo update
        self._ensure_rating(opponent_key)
        lr = self.learner_rating
        orr = float(self.ratings.get(opponent_key, 1200.0))
        exp = self._expected(lr, orr)
        self.ratings[self.learner_key] = max(0.0, lr + self.k_factor * (actual - exp))
        # Only snapshots are non-anchored in this simplified variant.
        if opponent_key.startswith("SNAPSHOT:"):
            self.ratings[opponent_key] = max(
                0.0,
                orr + self.k_factor * ((1.0 - actual) - (1.0 - exp)),
            )

    def maybe_advance_phase(self) -> bool:
        if self.phase_idx >= len(self.phases) - 1:
            return False
        phase_cfg = self.phases[self.phase_idx]
        ph = phase_cfg["name"]
        min_eps = int(phase_cfg.get("min_episodes", 0))
        min_wr = float(phase_cfg.get("min_winrate", 0.0))
        rec = self.phase_recent_results.get(ph, [])
        wr = (sum(rec) / len(rec)) if rec else 0.0
        if self.phase_episode_count >= min_eps and wr >= min_wr:
            self.phase_idx += 1
            self.phase_episode_count = 0
            return True
        return False

    def maybe_enable_league_mode(self) -> bool:
        if self.mode != "CURRICULUM_LEAGUE" or self.league_mode or self.phase_name != "OP3":
            return False
        if len(self.op3_results) < self.min_games_vs_op3_for_league:
            return False
        wr = sum(self.op3_results) / max(1, len(self.op3_results))
        if wr >= self.min_wr_vs_op3_for_league:
            self.league_mode = True
            return True
        return False

    def add_snapshot(self, snapshot_path: str) -> str:
        # Cap at max_snapshots: drop oldest and delete its checkpoint from disk.
        while len(self.snapshots) >= self.max_snapshots:
            old_key = self.snapshots.pop(0)
            old_path = self.snapshot_paths.pop(0)
            self.ratings.pop(old_key, None)
            try:
                if os.path.isdir(old_path):
                    shutil.rmtree(old_path, ignore_errors=True)
                elif os.path.isfile(old_path):
                    os.remove(old_path)
            except OSError:
                pass
        key = f"SNAPSHOT:{len(self.snapshots):04d}"
        self.snapshots.append(key)
        self.snapshot_paths.append(snapshot_path)
        self._ensure_rating(key)
        self.ratings[key] = self.learner_rating
        return key
