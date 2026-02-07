from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .episode_result import path_to_snapshot_key


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
        # OP3 anchor + Species Elo: after qualification, league = anchor OP3 + species only (no snapshots)
        anchor_op3_prob: float = 0.40,  # 40% scripted OP3 anchor
        # species_prob for league is implied 1 - anchor_op3_prob (60% species)
        species_rusher_bias: float = 0.40,  # When sampling species, force RUSHER with this prob (0 = uniform)
        # Legacy stability mix (unused when using anchor+species; kept for compat)
        stability_scripted_prob: float = 0.70,
        stability_snapshot_prob: float = 0.20,
        stability_species_prob: float = 0.10,
        use_stability_mix: bool = False,  # False = use anchor+species; True = legacy stability mix
        # Opponent switching frequency cap
        min_episodes_per_opponent: int = 3,  # Don't switch opponents more than once per N episodes
    ) -> None:
        self.rng = random.Random(int(seed))
        self.k = float(k_factor)
        self.tau = float(matchmaking_tau)
        self.scripted_floor = float(scripted_floor)
        self.species_prob = float(species_prob)
        self.snapshot_prob = float(snapshot_prob)
        self.anchor_op3_prob = max(0.0, min(1.0, float(anchor_op3_prob)))
        self.species_rusher_bias = max(0.0, min(1.0, float(species_rusher_bias)))
        self.use_stability_mix = bool(use_stability_mix)
        self.stability_scripted_prob = float(stability_scripted_prob)
        self.stability_snapshot_prob = float(stability_snapshot_prob)
        self.stability_species_prob = float(stability_species_prob)
        self.min_episodes_per_opponent = max(1, int(min_episodes_per_opponent))
        
        # Track last opponent and episode count for switching cap (structured to avoid path colon bug)
        self._last_kind: Optional[str] = None
        self._last_key: Optional[str] = None
        self._episodes_with_current_opponent: int = 0

        self.learner_key = "__LEARNER__"
        self.ratings: Dict[str, float] = {self.learner_key: 1200.0}
        self.snapshots: List[str] = []
        self.species_keys = ["SPECIES:RUSHER", "SPECIES:CAMPER", "SPECIES:BALANCED"]
        for key in [
            "SCRIPTED:OP1",
            "SCRIPTED:OP2",
            "SCRIPTED:OP3",
        ] + self.species_keys:
            self.ratings.setdefault(key, 1200.0)

    @property
    def learner_rating(self) -> float:
        return float(self.ratings.get(self.learner_key, 1200.0))

    def set_learner_rating(self, rating: float) -> None:
        self.ratings[self.learner_key] = max(0.0, float(rating))

    def add_snapshot(self, path: str) -> None:
        """Register a snapshot by full path; ratings are keyed by stable ID for portability."""
        if path not in self.snapshots:
            self.snapshots.append(path)
        self.ratings.setdefault(path_to_snapshot_key(path), self.learner_rating)

    def latest_snapshot_key(self) -> Optional[str]:
        if not self.snapshots:
            return None
        return str(self.snapshots[-1])

    def get_rating(self, key: str) -> float:
        return float(self.ratings.get(key, 1200.0))

    def update_elo(self, opponent_key: str, actual_score: float) -> None:
        """
        Update learner Elo from match result. Opponent rating is updated only for
        SNAPSHOT opponents; SCRIPTED and SPECIES ratings are anchored (fixed) so
        Elo stays comparable over time and does not inflate.
        """
        lr = self.learner_rating
        opp_r = self.get_rating(opponent_key)
        exp = elo_expected(lr, opp_r)

        learner_new = max(0.0, lr + self.k * (float(actual_score) - exp))
        self.set_learner_rating(learner_new)

        # Rating anchors: do not update SCRIPTED or SPECIES ratings; only update SNAPSHOT
        if opponent_key.startswith("SCRIPTED:") or opponent_key.startswith("SPECIES:"):
            return
        opp_new = max(0.0, opp_r + self.k * ((1.0 - float(actual_score)) - (1.0 - exp)))
        self.ratings[opponent_key] = float(opp_new)

    def _weighted_pick(
        self,
        keys: List[str],
        target_rating: float,
        key_to_rating: Optional[Callable[[str], float]] = None,
    ) -> str:
        if key_to_rating is None:
            key_to_rating = self.get_rating
        weights = []
        for key in keys:
            r = key_to_rating(key)
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

    def sample_league(self, target_rating: Optional[float] = None, phase: Optional[str] = None) -> OpponentSpec:
        """
        Sample opponent from league: OP3 anchor + Species Elo (no snapshots).
        After qualification, league is 40% scripted OP3 (anchor) + 60% species (matchmade).
        min_episodes_per_opponent stickiness is preserved.
        """
        target = self.learner_rating if target_rating is None else float(target_rating)
        phase = str(phase).upper() if phase else "OP3"

        # Cap opponent switching frequency (stickiness)
        if (self._last_kind is not None and self._last_key is not None and
            self._episodes_with_current_opponent < self.min_episodes_per_opponent):
            self._episodes_with_current_opponent += 1
            return self._reconstruct_last_opponent()

        self._episodes_with_current_opponent = 1

        if self.use_stability_mix:
            opp_spec = self._sample_stability_mix(target, phase)
        else:
            # OP3 anchor + species only
            r = self.rng.random()
            if r < self.anchor_op3_prob:
                opp_spec = OpponentSpec(
                    kind="SCRIPTED", key="OP3", rating=self.get_rating("SCRIPTED:OP3")
                )
            else:
                # Species (with optional rusher bias)
                if self.species_rusher_bias > 0 and self.rng.random() < self.species_rusher_bias:
                    key = "SPECIES:RUSHER"
                else:
                    key = self._weighted_pick(self.species_keys, target)
                tag = key.split(":", 1)[1]
                opp_spec = OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(key))

        self._last_kind = opp_spec.kind
        self._last_key = opp_spec.key
        return opp_spec
    
    def _sample_stability_mix(self, target_rating: float, phase: str) -> OpponentSpec:
        """
        Sprint A: Sample with stability mix (70% scripted, 20% snapshot, 10% species).
        """
        r = self.rng.random()
        
        # 70% scripted opponents (OP1/OP2/OP3 depending on phase)
        if r < self.stability_scripted_prob:
            # Select scripted opponent based on phase
            if phase == "OP1":
                scripted_tag = "OP1"
            elif phase == "OP2":
                scripted_tag = "OP2"
            else:  # OP3 or default
                scripted_tag = "OP3"
            key = f"SCRIPTED:{scripted_tag}"
            return OpponentSpec(kind="SCRIPTED", key=scripted_tag, rating=self.get_rating(key))
        
        r -= self.stability_scripted_prob
        
        # 20% snapshot opponents: sample by rating closeness to learner (not uniform)
        # So snapshots aren't "free wins" from weak recent checkpoints.
        if r < self.stability_snapshot_prob and self.snapshots:
            path = self._weighted_pick(
                self.snapshots,
                target_rating,
                key_to_rating=lambda p: self.get_rating(path_to_snapshot_key(p)),
            )
            return OpponentSpec(kind="SNAPSHOT", key=path, rating=self.get_rating(path_to_snapshot_key(path)))
        
        r -= self.stability_snapshot_prob
        
        # Species variants (RUSHER bias to remediate weakness)
        if r < self.stability_species_prob:
            if self.species_rusher_bias > 0 and self.rng.random() < self.species_rusher_bias:
                key = "SPECIES:RUSHER"
            else:
                key = self._weighted_pick(self.species_keys, target_rating)
            tag = key.split(":", 1)[1]
            return OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(key))
        
        # Fallback: scripted (shouldn't reach here, but safety)
        key = "SCRIPTED:OP3"
        return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating(key))
    
    def _reconstruct_last_opponent(self) -> OpponentSpec:
        """Reconstruct last opponent spec from structured _last_kind/_last_key (no string parsing)."""
        if self._last_kind is None or self._last_key is None:
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating("SCRIPTED:OP3"))
        kind = self._last_kind
        key = self._last_key
        if kind == "SNAPSHOT":
            rating = self.get_rating(path_to_snapshot_key(key))
        elif kind in ("SCRIPTED", "SPECIES"):
            rating = self.get_rating(f"{kind}:{key}")
        else:
            rating = self.learner_rating
        return OpponentSpec(kind=kind, key=key, rating=rating)
    
    def reset_opponent_tracking(self) -> None:
        """Reset opponent tracking (e.g., on episode reset)."""
        self._last_kind = None
        self._last_key = None
        self._episodes_with_current_opponent = 0

    def sample_species(self, target_rating: Optional[float] = None) -> OpponentSpec:
        target = self.learner_rating if target_rating is None else float(target_rating)
        key = self._weighted_pick(self.species_keys, target)
        tag = key.split(":", 1)[1]
        return OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(key))

    def sample_snapshot(self, target_rating: Optional[float] = None) -> OpponentSpec:
        if not self.snapshots:
            return OpponentSpec(kind="SCRIPTED", key="OP3", rating=self.get_rating("SCRIPTED:OP3"))
        target = self.learner_rating if target_rating is None else float(target_rating)
        path = self._weighted_pick(
            self.snapshots, target, key_to_rating=lambda p: self.get_rating(path_to_snapshot_key(p))
        )
        return OpponentSpec(kind="SNAPSHOT", key=path, rating=self.get_rating(path_to_snapshot_key(path)))
