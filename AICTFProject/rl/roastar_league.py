from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set, Tuple

from .episode_result import path_to_snapshot_key
from .league import EloLeague, OpponentSpec


class ROAStarLeague(EloLeague):
    """
    Adapted ROA-Star baseline, stage 1: PFSP (prioritized fictitious self-play)
    opponent sampling in place of EloLeague's Elo-distance matchmaking.

    Reuses EloLeague's SCRIPTED/SPECIES/SNAPSHOT category split, Elo bookkeeping,
    and snapshot registration unchanged; only the within-category opponent choice
    is replaced with a win-rate-weighted pick: weight ~ (1 - win_rate)^p, so the
    learner is pushed toward opponents it currently struggles against rather than
    ones near its own Elo rating.
    """

    def __init__(
        self,
        *,
        pfsp_p: float = 2.0,
        pfsp_floor: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.pfsp_p = float(pfsp_p)
        self.pfsp_floor = float(pfsp_floor)
        # opponent_key -> (wins, games), keyed the same way as self.ratings
        # (e.g. "SCRIPTED:OP3", "SPECIES:RUSHER", or a snapshot key).
        self.win_rate_stats: Dict[str, Tuple[float, int]] = {}
        # Snapshot paths that came from exploiter training rather than plain
        # self-play, tracked purely as metadata (no change to how they're loaded
        # or executed by BatchedCTFCore).
        self.exploiter_snapshots: Set[str] = set()

    def record_result(self, opponent_key: str, actual_score: float) -> None:
        """
        Update win-rate stats for an opponent. Call alongside update_elo() with
        the same opponent_key and actual_score so Elo and PFSP stats stay in sync.
        actual_score follows EloLeague's convention: 1.0 = win, 0.5 = draw, 0.0 = loss.
        """
        wins, games = self.win_rate_stats.get(opponent_key, (0.0, 0))
        self.win_rate_stats[opponent_key] = (wins + float(actual_score), games + 1)

    def win_rate(self, opponent_key: str) -> Optional[float]:
        """Return the learner's win rate against opponent_key, or None if unplayed."""
        wins, games = self.win_rate_stats.get(opponent_key, (0.0, 0))
        if games <= 0:
            return None
        return wins / games

    def register_exploiter_snapshot(self, path: str) -> None:
        """Register an exploiter-trained checkpoint into the same snapshot pool
        used for self-play snapshots. Execution is identical (BatchedCTFCore can't
        tell the difference); only PFSP sampling/reporting treats it distinctly."""
        self.add_snapshot(path)
        self.exploiter_snapshots.add(path)

    def _pfsp_weight(self, opponent_key: str) -> float:
        wr = self.win_rate(opponent_key)
        if wr is None:
            # Unplayed opponents get full weight so they get sampled early.
            return 1.0
        return max(self.pfsp_floor, (1.0 - wr) ** self.pfsp_p)

    def _weighted_pick_pfsp(
        self,
        keys: List[str],
        key_to_stats_key: Optional[Callable[[str], str]] = None,
    ) -> str:
        if key_to_stats_key is None:
            key_to_stats_key = lambda k: k
        weights = [self._pfsp_weight(key_to_stats_key(k)) for k in keys]
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

    def sample_league_pfsp(
        self,
        phase: Optional[str] = None,
        enable_snapshots: bool = False,
    ) -> OpponentSpec:
        """
        PFSP-based counterpart to EloLeague.sample_league(): same category split
        (SCRIPTED anchor / SPECIES / SNAPSHOT) and opponent-switch stickiness, but
        within SPECIES/SNAPSHOT the choice is weighted by (1 - win_rate)^p instead
        of Elo-rating distance.
        """
        phase = str(phase).upper() if phase else "OP3"

        if (
            self._last_kind is not None
            and self._last_key is not None
            and self._episodes_with_current_opponent < self.min_episodes_per_opponent
        ):
            self._episodes_with_current_opponent += 1
            return self._reconstruct_last_opponent()

        self._episodes_with_current_opponent = 1

        categories: List[Tuple[str, float]] = []
        scripted_tag = phase if phase in ("OP1", "OP2", "OP3") else "OP3"
        categories.append(("SCRIPTED", max(0.0, float(self.anchor_op3_prob))))
        if self.species_prob > 0.0:
            categories.append(("SPECIES", max(0.0, float(self.species_prob))))
        if enable_snapshots and self.snapshots and self.snapshot_prob > 0.0:
            categories.append(("SNAPSHOT", max(0.0, float(self.snapshot_prob))))

        total_weight = sum(weight for _, weight in categories)
        if total_weight <= 0.0:
            chosen_kind = "SCRIPTED"
        else:
            pick = self.rng.random() * total_weight
            acc = 0.0
            chosen_kind = "SCRIPTED"
            for kind, weight in categories:
                acc += weight
                if acc >= pick:
                    chosen_kind = kind
                    break

        if chosen_kind == "SCRIPTED":
            key = f"SCRIPTED:{scripted_tag}"
            opp_spec = OpponentSpec(kind="SCRIPTED", key=scripted_tag, rating=self.get_rating(key))
        elif chosen_kind == "SPECIES":
            species_key = self._weighted_pick_pfsp(self.species_keys)
            tag = species_key.split(":", 1)[1]
            opp_spec = OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(species_key))
        else:
            path = self._weighted_pick_pfsp(
                self.snapshots,
                key_to_stats_key=path_to_snapshot_key,
            )
            opp_spec = OpponentSpec(
                kind="SNAPSHOT",
                key=path,
                rating=self.get_rating(path_to_snapshot_key(path)),
            )

        self._last_kind = opp_spec.kind
        self._last_key = opp_spec.key
        return opp_spec
