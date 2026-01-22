from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional


def elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_a - r_b) / 400.0))


@dataclass
class OpponentSpec:
    kind: str  # "SCRIPTED" | "SPECIES" | "SNAPSHOT"
    key: str
    rating: float


class EloLeague:
    def __init__(
        self,
        *,
        seed: int = 42,
        k_factor: float = 32.0,
        matchmaking_tau: float = 250.0,
        scripted_floor: float = 0.20,
        species_prob: float = 0.20,
        snapshot_prob: float = 0.60,
    ) -> None:
        self.rng = random.Random(int(seed))
        self.k = float(k_factor)
        self.tau = float(matchmaking_tau)
        self.scripted_floor = float(scripted_floor)
        self.species_prob = float(species_prob)
        self.snapshot_prob = float(snapshot_prob)

        self.learner_key = "__LEARNER__"
        self.ratings: Dict[str, float] = {self.learner_key: 1200.0}
        self.snapshots: List[str] = []
        self.species_keys = ["SPECIES:RUSHER", "SPECIES:CAMPER", "SPECIES:BALANCED"]
        for key in [
            "SCRIPTED:OP1",
            "SCRIPTED:OP2",
            "SCRIPTED:OP3",
            "SCRIPTED:OP3_EASY",
            "SCRIPTED:OP3_HARD",
        ] + self.species_keys:
            self.ratings.setdefault(key, 1200.0)

    @property
    def learner_rating(self) -> float:
        return float(self.ratings.get(self.learner_key, 1200.0))

    def set_learner_rating(self, rating: float) -> None:
        self.ratings[self.learner_key] = max(0.0, float(rating))

    def add_snapshot(self, key: str) -> None:
        if key not in self.snapshots:
            self.snapshots.append(key)
        self.ratings.setdefault(key, self.learner_rating)

    def get_rating(self, key: str) -> float:
        return float(self.ratings.get(key, 1200.0))

    def update_elo(self, opponent_key: str, actual_score: float) -> None:
        lr = self.learner_rating
        opp_r = self.get_rating(opponent_key)
        exp = elo_expected(lr, opp_r)

        learner_new = max(0.0, lr + self.k * (float(actual_score) - exp))
        opp_new = max(0.0, opp_r + self.k * ((1.0 - float(actual_score)) - (1.0 - exp)))

        self.set_learner_rating(learner_new)
        self.ratings[opponent_key] = float(opp_new)

    def _weighted_pick(self, keys: List[str], target_rating: float) -> str:
        weights = []
        for key in keys:
            r = self.get_rating(key)
            dist = abs(r - float(target_rating))
            weights.append(math.exp(-dist / max(1e-6, self.tau)) + 1e-3)
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

    def sample_curriculum(self, phase: str) -> OpponentSpec:
        phase = str(phase).upper()
        key = f"SCRIPTED:{phase}"
        return OpponentSpec(kind="SCRIPTED", key=phase, rating=self.get_rating(key))

    def sample_league(self, target_rating: Optional[float] = None) -> OpponentSpec:
        target = self.learner_rating if target_rating is None else float(target_rating)
        r = self.rng.random()
        if r < self.scripted_floor or (not self.snapshots):
            key = "SCRIPTED:OP3_HARD" if "SCRIPTED:OP3_HARD" in self.ratings else "SCRIPTED:OP3"
            tag = key.split(":", 1)[1]
            return OpponentSpec(kind="SCRIPTED", key=tag, rating=self.get_rating(key))

        r -= self.scripted_floor
        if r < self.species_prob:
            key = self._weighted_pick(self.species_keys, target)
            tag = key.split(":", 1)[1]
            return OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(key))

        if self.snapshots:
            key = self._weighted_pick(self.snapshots, target)
            return OpponentSpec(kind="SNAPSHOT", key=key, rating=self.get_rating(key))

        key = "SCRIPTED:OP3"
        return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))

    def sample_species(self, target_rating: Optional[float] = None) -> OpponentSpec:
        target = self.learner_rating if target_rating is None else float(target_rating)
        key = self._weighted_pick(self.species_keys, target)
        tag = key.split(":", 1)[1]
        return OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(key))

    def sample_snapshot(self, target_rating: Optional[float] = None) -> OpponentSpec:
        if not self.snapshots:
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating("SCRIPTED:OP3"))
        target = self.learner_rating if target_rating is None else float(target_rating)
        key = self._weighted_pick(self.snapshots, target)
        return OpponentSpec(kind="SNAPSHOT", key=key, rating=self.get_rating(key))
