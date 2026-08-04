"""Fictitious play and double oracle leagues -- the two missing rungs of the
empirical-game-theoretic baseline ladder (self-play -> FP -> DO), as standardized
by VGC-Bench (Angliss et al., AAMAS 2026).

Both classes extend :class:`~rl.roastar_league.ROAStarLeague` so they inherit the
SCRIPTED / SPECIES / SNAPSHOT category split, opponent-switch stickiness, Elo
bookkeeping, win-rate stats and state serialization unchanged. Exactly as
ROAStarLeague changed only the *within-category* opponent choice relative to
EloLeague, these change only that same rule:

    EloLeague              Elo-distance matchmaking      exp(-|r_opp - r_self|/tau)
    ROAStarLeague (PFSP)   prioritized fictitious play   (1 - win_rate)^p
    FictitiousPlayLeague   fictitious play               uniform over the pool
    DoubleOracleLeague     double oracle / PSRO          meta-Nash over the pool

Keeping the category mix identical across all four is what makes the comparison a
controlled one: any difference in the resulting policy is attributable to the
opponent-selection rule and nothing else.

Note on the double oracle's *oracle* step: in PSRO terms the main PPO agent is
the best-response oracle. It trains against opponents drawn from the meta-Nash
over the current pool, and each periodic snapshot registered via
:meth:`add_snapshot` is that oracle's output entering the population. No separate
best-response training loop is required.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .episode_result import path_to_snapshot_key
from .league import OpponentSpec
from .roastar_league import ROAStarLeague


class _CategorySamplerMixin:
    """Category draw + stickiness shared by the FP and DO sampling rules.

    Factored out so both leagues provably use the same SCRIPTED/SPECIES/SNAPSHOT
    mix as EloLeague and ROAStarLeague, and differ only in ``pick``.
    """

    def _sample_with_rule(self, *, phase, enable_snapshots, pick):
        """Shared category draw + stickiness; ``pick`` decides within a category.

        ``pick(keys, category)`` receives the candidate keys for the drawn
        category ("SPECIES" or "SNAPSHOT") and returns the chosen one.
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
        chosen_kind = "SCRIPTED"
        if total_weight > 0.0:
            draw = self.rng.random() * total_weight
            acc = 0.0
            for kind, weight in categories:
                acc += weight
                if acc >= draw:
                    chosen_kind = kind
                    break

        if chosen_kind == "SCRIPTED":
            opp_spec = OpponentSpec(
                kind="SCRIPTED",
                key=scripted_tag,
                rating=self.get_rating(f"SCRIPTED:{scripted_tag}"),
            )
        elif chosen_kind == "SPECIES":
            species_key = pick(self.species_keys, "SPECIES")
            tag = str(species_key).split(":", 1)[1]
            opp_spec = OpponentSpec(kind="SPECIES", key=tag, rating=self.get_rating(species_key))
        else:
            path = pick(self.snapshots, "SNAPSHOT")
            opp_spec = OpponentSpec(
                kind="SNAPSHOT",
                key=path,
                rating=self.get_rating(path_to_snapshot_key(path)),
            )

        self._last_kind = opp_spec.kind
        self._last_key = opp_spec.key
        return opp_spec


class FictitiousPlayLeague(_CategorySamplerMixin, ROAStarLeague):
    """Fictitious play: learn against a uniform distribution over the pool.

    VGC-Bench's FP baseline "maintains a pool of the agent's past checkpoints, and
    the agent learns against a uniform distribution of its past policies". The
    uniform draw is the whole method -- there is no win-rate or rating weighting,
    which is precisely what distinguishes it from PFSP and from Elo matchmaking.
    """

    def sample_league_fp(
        self,
        phase: Optional[str] = None,
        enable_snapshots: bool = False,
    ) -> OpponentSpec:
        """Uniform-within-category counterpart to :meth:`sample_league_pfsp`."""
        return self._sample_with_rule(
            phase=phase,
            enable_snapshots=enable_snapshots,
            pick=lambda keys, _category: self.rng.choice(list(keys)),
        )


def solve_zero_sum_nash(payoff: np.ndarray, *, max_iters: int = 20_000) -> np.ndarray:
    """Row player's maximin strategy for a two-player zero-sum matrix game.

    ``payoff[i, j]`` is the row player's expected score when row plays ``i`` and
    column plays ``j``, on a [0, 1] scale (1 = row wins). Solves the standard LP

        max_x  v   s.t.   sum_i payoff[i, j] * x_i >= v  for all j,
                          sum_i x_i = 1,  x >= 0

    with SciPy when available, falling back to fictitious-play iteration (which
    converges to a Nash equilibrium in zero-sum games, Robinson 1951) so the
    league never hard-depends on SciPy being installed.
    """
    mat = np.asarray(payoff, dtype=float)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        raise ValueError(f"payoff must be a non-empty 2-D matrix, got shape {mat.shape}")
    n_rows = int(mat.shape[0])
    if n_rows == 1:
        return np.ones(1, dtype=float)

    try:
        from scipy.optimize import linprog
    except ImportError:
        return _fictitious_play_nash(mat, max_iters=max_iters)

    n_cols = int(mat.shape[1])
    # Variables: [x_0 .. x_{n_rows-1}, v]; minimize -v.
    c = np.zeros(n_rows + 1, dtype=float)
    c[-1] = -1.0
    # For each column j:  v - sum_i payoff[i, j] * x_i <= 0
    a_ub = np.zeros((n_cols, n_rows + 1), dtype=float)
    a_ub[:, :n_rows] = -mat.T
    a_ub[:, -1] = 1.0
    b_ub = np.zeros(n_cols, dtype=float)
    a_eq = np.zeros((1, n_rows + 1), dtype=float)
    a_eq[0, :n_rows] = 1.0
    b_eq = np.array([1.0], dtype=float)
    bounds = [(0.0, 1.0)] * n_rows + [(None, None)]

    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return _fictitious_play_nash(mat, max_iters=max_iters)
    x = np.asarray(res.x[:n_rows], dtype=float)
    x = np.clip(x, 0.0, None)
    total = float(x.sum())
    if not np.isfinite(total) or total <= 0.0:
        return _fictitious_play_nash(mat, max_iters=max_iters)
    return x / total


def _fictitious_play_nash(payoff: np.ndarray, *, max_iters: int = 20_000) -> np.ndarray:
    """SciPy-free fallback: fictitious play on the payoff matrix."""
    n_rows, n_cols = payoff.shape
    row_counts = np.zeros(n_rows, dtype=float)
    col_counts = np.zeros(n_cols, dtype=float)
    row_payoff_sum = np.zeros(n_rows, dtype=float)
    col_payoff_sum = np.zeros(n_cols, dtype=float)
    for _ in range(int(max_iters)):
        # Row maximizes its payoff; column minimizes it.
        i = int(np.argmax(row_payoff_sum)) if row_counts.sum() > 0 else 0
        j = int(np.argmin(col_payoff_sum)) if col_counts.sum() > 0 else 0
        row_counts[i] += 1.0
        col_counts[j] += 1.0
        row_payoff_sum += payoff[:, j]
        col_payoff_sum += payoff[i, :]
    total = float(row_counts.sum())
    if total <= 0.0:
        return np.full(n_rows, 1.0 / n_rows, dtype=float)
    return row_counts / total


class DoubleOracleLeague(_CategorySamplerMixin, ROAStarLeague):
    """Double oracle / PSRO: sample opponents from the pool's meta-Nash.

    VGC-Bench's DO baseline "derives the Nash equilibrium distribution from a
    maintained empirical payoff matrix between all agents in the pool, and uses
    that distribution to sample opponents", solving an LP because the game is
    two-player, zero-sum and symmetric-payoff. This class does the same.

    Building the empirical payoff matrix from a single-learner training loop needs
    one documented approximation. Only learner-vs-opponent games are ever played,
    so the learner's running results fill the ``__LEARNER__`` row; when a snapshot
    is registered, the learner's current row is *frozen* into that snapshot's row,
    because the snapshot is exactly the learner at that moment. Unobserved cells
    fall back to the zero-sum reflection ``1 - payoff[j, i]`` when the mirror was
    observed, and to 0.5 (an even matchup) otherwise.
    """

    def __init__(self, *, nash_temperature: float = 1.0, min_games_for_payoff: int = 5, **kwargs) -> None:
        super().__init__(**kwargs)
        self.min_games_for_payoff = max(1, int(min_games_for_payoff))
        self.nash_temperature = float(nash_temperature)
        # (row_key, col_key) -> (score_sum, games), score from the row player's view.
        self.payoff_stats: Dict[Tuple[str, str], Tuple[float, int]] = {}

    # -- payoff bookkeeping -------------------------------------------------

    def record_result(self, opponent_key: str, actual_score: float) -> None:
        """Record into both the PFSP win-rate stats and the learner's payoff row."""
        super().record_result(opponent_key, actual_score)
        self._add_payoff(self.learner_key, str(opponent_key), float(actual_score))

    def _add_payoff(self, row_key: str, col_key: str, score: float) -> None:
        total, games = self.payoff_stats.get((row_key, col_key), (0.0, 0))
        self.payoff_stats[(row_key, col_key)] = (total + float(score), games + 1)

    def add_snapshot(self, path: str) -> None:
        """Register a snapshot, freezing the learner's current payoff row into it."""
        snapshot_key = path_to_snapshot_key(path)
        for (row_key, col_key), (total, games) in list(self.payoff_stats.items()):
            if row_key == self.learner_key and games > 0:
                self.payoff_stats.setdefault((snapshot_key, col_key), (total, games))
        super().add_snapshot(path)

    def pool_keys(self, *, enable_snapshots: bool = True) -> List[str]:
        """Population members the meta-game is defined over."""
        keys = ["SCRIPTED:OP1", "SCRIPTED:OP2", "SCRIPTED:OP3"] + list(self.species_keys)
        if enable_snapshots:
            keys.extend(path_to_snapshot_key(p) for p in self.snapshots)
        seen: Dict[str, None] = {}
        for key in keys:
            seen.setdefault(key, None)
        return list(seen)

    def payoff_matrix(self, keys: Sequence[str]) -> np.ndarray:
        """Empirical payoff matrix over ``keys`` (row player's score in [0, 1])."""
        n = len(keys)
        mat = np.full((n, n), 0.5, dtype=float)
        for i, row_key in enumerate(keys):
            for j, col_key in enumerate(keys):
                if i == j:
                    mat[i, j] = 0.5
                    continue
                total, games = self.payoff_stats.get((row_key, col_key), (0.0, 0))
                if games >= self.min_games_for_payoff:
                    mat[i, j] = total / float(games)
                    continue
                rev_total, rev_games = self.payoff_stats.get((col_key, row_key), (0.0, 0))
                if rev_games >= self.min_games_for_payoff:
                    mat[i, j] = 1.0 - (rev_total / float(rev_games))
        return mat

    def meta_nash(self, *, enable_snapshots: bool = True) -> Dict[str, float]:
        """Meta-Nash distribution over the pool, as ``{pool_key: probability}``."""
        keys = self.pool_keys(enable_snapshots=enable_snapshots)
        if not keys:
            return {}
        probs = solve_zero_sum_nash(self.payoff_matrix(keys))
        return {key: float(p) for key, p in zip(keys, probs)}

    # -- sampling -----------------------------------------------------------

    def sample_league_do(
        self,
        phase: Optional[str] = None,
        enable_snapshots: bool = False,
    ) -> OpponentSpec:
        """Meta-Nash-within-category counterpart to :meth:`sample_league_pfsp`.

        The Nash is solved over the *full* pool and then conditioned on the drawn
        category, so the SCRIPTED/SPECIES/SNAPSHOT mix stays identical to the Elo,
        PFSP and FP baselines and only the within-category rule differs. That is a
        controlled-comparison choice; the conditioned marginal is not itself
        claimed to be the exact meta-Nash.
        """
        nash = self.meta_nash(enable_snapshots=enable_snapshots)

        def pick(keys: Sequence[str], category: str) -> str:
            candidates = list(keys)
            if not candidates:
                raise ValueError(f"no candidates available for category {category}")
            if category == "SNAPSHOT":
                weights = [max(0.0, nash.get(path_to_snapshot_key(k), 0.0)) for k in candidates]
            else:
                weights = [max(0.0, nash.get(k, 0.0)) for k in candidates]
            total = float(sum(weights))
            if total <= 0.0:
                return self.rng.choice(candidates)
            draw = self.rng.random() * total
            acc = 0.0
            for key, weight in zip(candidates, weights):
                acc += weight
                if acc >= draw:
                    return key
            return candidates[-1]

        return self._sample_with_rule(phase=phase, enable_snapshots=enable_snapshots, pick=pick)

    # -- serialization ------------------------------------------------------

    def to_dict(self):
        state = super().to_dict()
        state["payoff_stats"] = [
            [row_key, col_key, total, games]
            for (row_key, col_key), (total, games) in sorted(self.payoff_stats.items())
        ]
        return state

    def load_state_dict(self, state) -> None:
        super().load_state_dict(state)
        for entry in state.get("payoff_stats", []) or []:
            row_key, col_key, total, games = entry
            self.payoff_stats[(str(row_key), str(col_key))] = (float(total), int(games))
