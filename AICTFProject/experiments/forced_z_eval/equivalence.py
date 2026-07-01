"""Equivalence checks for forced-z evaluation optimizations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from experiments.forced_z_eval.io import CellEpisodes
from experiments.forced_z_eval.protocol import ForcedZProtocol
from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES

# Exact equality (discrete / categorical).
EXACT_INT_KEYS = (
    "episode_index",
    "episode_seed",
    "fixed_latent_id",
    "success",
    "blue_score",
    "red_score",
    "steps",
    "win_margin",
    "collision_free",
    "decision_steps",
    "blue_stuck_steps",
    "red_stuck_steps",
    "blue_blocked_movement_events",
    "blue_repeated_blocked_movement_events",
    "collisions_per_episode",
    "collision_events_per_episode",
    "obstacle_collision_events_per_episode",
    "blue_obstacle_collision_events",
    "blue_route_switches",
    "blue_upper_lane_steps",
    "blue_lower_lane_steps",
    "blue_neutral_lane_steps",
    "blue_movement_attempts",
    "blue_successful_movement_steps",
    "near_misses_per_episode",
    "blue_attack_upper_crossings",
    "blue_attack_lower_crossings",
    "blue_return_upper_crossings",
    "blue_return_lower_crossings",
    "blue_intercept_upper_crossings",
    "blue_intercept_lower_crossings",
)

EXACT_STR_KEYS = (
    "opponent",
    "outcome",
    "episode_start_phase",
)

FLOAT_KEYS = (
    "return",
    "zone_coverage",
    "time_to_first_score",
    "mean_inter_robot_dist",
    "reward_total",
    "reward_sparse_points",
    "reward_offense",
    "reward_terminal",
    "reward_team",
    "reward_sparse",
)

BEHAVIOR_KEYS = tuple(f"behavior_{name}" for name in BEHAVIOR_TELEMETRY_NAMES)

TERMINAL_KEYS = EXACT_INT_KEYS + EXACT_STR_KEYS + FLOAT_KEYS + BEHAVIOR_KEYS

# Strict — do not loosen; stale hidden state should fail.
STRICT_FLOAT_ATOL = 1e-6
STRICT_FLOAT_RTOL = 1e-6

DEFAULT_FLOAT_ATOL = STRICT_FLOAT_ATOL
DEFAULT_FLOAT_RTOL = STRICT_FLOAT_RTOL


@dataclass
class EpisodeMismatch:
    opponent: str
    map_name: str
    latent_z: int
    episode_index: int
    episode_seed: int
    field: str
    left: Any
    right: Any
    comparison: str = "fresh_vs_reuse"


@dataclass
class EquivalenceReport:
    passed: bool
    episodes_compared: int
    comparison: str
    mismatches: list[EpisodeMismatch] = field(default_factory=list)

    def summary(self) -> str:
        if self.passed:
            return f"PASS [{self.comparison}] — {self.episodes_compared} episodes matched by (z, episode_seed)"
        lines = [
            f"FAIL [{self.comparison}] — {len(self.mismatches)} mismatch(es) "
            f"across {self.episodes_compared} episodes"
        ]
        for mm in self.mismatches[:25]:
            lines.append(
                f"  {mm.opponent} z={mm.latent_z} {mm.map_name} seed={mm.episode_seed} "
                f"field={mm.field}: left={mm.left!r} right={mm.right!r}"
            )
        if len(self.mismatches) > 25:
            lines.append(f"  ... and {len(self.mismatches) - 25} more")
        lines.append("")
        lines.append(decision_tree_hint(self))
        return "\n".join(lines)


def _float_equal(a: float, b: float, *, atol: float, rtol: float) -> bool:
    if a != a and b != b:
        return True
    if a != a or b != b:
        return False
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))


def compare_episode(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    atol: float = STRICT_FLOAT_ATOL,
    rtol: float = STRICT_FLOAT_RTOL,
) -> list[tuple[str, Any, Any]]:
    diffs: list[tuple[str, Any, Any]] = []
    for key in EXACT_INT_KEYS:
        if key not in left and key not in right:
            continue
        lv = int(left.get(key, -999999))
        rv = int(right.get(key, -999999))
        if lv != rv:
            diffs.append((key, lv, rv))
    for key in EXACT_STR_KEYS:
        if key not in left and key not in right:
            continue
        lv = str(left.get(key, ""))
        rv = str(right.get(key, ""))
        if lv != rv:
            diffs.append((key, lv, rv))
    for key in FLOAT_KEYS + BEHAVIOR_KEYS:
        if key not in left and key not in right:
            continue
        lv = float(left.get(key, float("nan")))
        rv = float(right.get(key, float("nan")))
        if not _float_equal(lv, rv, atol=atol, rtol=rtol):
            diffs.append((key, lv, rv))
    return diffs


def _episodes_by_seed(eps: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for ep in eps:
        seed = int(ep.get("episode_seed", -1))
        if seed in out:
            raise ValueError(f"Duplicate episode_seed {seed} in cell episodes")
        out[seed] = ep
    return out


def compare_forced_z_cells(
    left: CellEpisodes,
    right: CellEpisodes,
    *,
    opponents: list[str],
    maps: list[str],
    latents: tuple[int, ...],
    comparison: str = "fresh_vs_reuse",
    atol: float = STRICT_FLOAT_ATOL,
    rtol: float = STRICT_FLOAT_RTOL,
) -> EquivalenceReport:
    """Compare episodes keyed by (opponent, map, z, episode_seed), not completion order."""
    mismatches: list[EpisodeMismatch] = []
    compared = 0
    for opponent in opponents:
        for map_name in maps:
            for z in latents:
                key = (opponent, z, map_name)
                left_eps = left.get(key, [])
                right_eps = right.get(key, [])
                if len(left_eps) != len(right_eps):
                    mismatches.append(
                        EpisodeMismatch(
                            opponent=opponent,
                            map_name=map_name,
                            latent_z=int(z),
                            episode_index=-1,
                            episode_seed=-1,
                            field="episode_count",
                            left=len(left_eps),
                            right=len(right_eps),
                            comparison=comparison,
                        )
                    )
                try:
                    left_by_seed = _episodes_by_seed(left_eps)
                    right_by_seed = _episodes_by_seed(right_eps)
                except ValueError as exc:
                    mismatches.append(
                        EpisodeMismatch(
                            opponent=opponent,
                            map_name=map_name,
                            latent_z=int(z),
                            episode_index=-1,
                            episode_seed=-1,
                            field="duplicate_episode_seed",
                            left=str(exc),
                            right="",
                            comparison=comparison,
                        )
                    )
                    continue
                all_seeds = sorted(set(left_by_seed) | set(right_by_seed))
                for seed in all_seeds:
                    if seed not in left_by_seed or seed not in right_by_seed:
                        mismatches.append(
                            EpisodeMismatch(
                                opponent=opponent,
                                map_name=map_name,
                                latent_z=int(z),
                                episode_index=-1,
                                episode_seed=int(seed),
                                field="missing_episode_seed",
                                left=seed in left_by_seed,
                                right=seed in right_by_seed,
                                comparison=comparison,
                            )
                        )
                        continue
                    compared += 1
                    le = left_by_seed[seed]
                    re = right_by_seed[seed]
                    for field_name, lv, rv in compare_episode(le, re, atol=atol, rtol=rtol):
                        mismatches.append(
                            EpisodeMismatch(
                                opponent=opponent,
                                map_name=map_name,
                                latent_z=int(z),
                                episode_index=int(le.get("episode_index", -1)),
                                episode_seed=int(seed),
                                field=field_name,
                                left=lv,
                                right=rv,
                                comparison=comparison,
                            )
                        )
    return EquivalenceReport(
        passed=not mismatches,
        episodes_compared=compared,
        comparison=comparison,
        mismatches=mismatches,
    )


def decision_tree_hint(report: EquivalenceReport) -> str:
    if report.passed:
        return "Decision: All exact and tolerance comparisons pass → approve reuse_block as canonical."
    fields = {m.field for m in report.mismatches}
    if fields <= {"episode_index"} or fields == {"episode_count"}:
        return "Decision: Only row order/count differs → canonical sorting defect."
    if "episode_seed" in fields or "missing_episode_seed" in fields:
        return "Decision: Episode seed mismatch → reset or RNG leakage; fix deterministic protocol first."
    score_fields = {"blue_score", "red_score", "success", "outcome", "win_margin"}
    float_behavior = set(FLOAT_KEYS) | set(BEHAVIOR_KEYS)
    if fields & score_fields and not fields & float_behavior:
        return "Decision: Scores/outcome differ → environment or opponent state not fully reset."
    if fields & float_behavior and not fields & score_fields:
        return "Decision: Scores match but returns/behavior differ → telemetry or temporal-state leakage."
    if report.comparison.startswith("order_"):
        return "Decision: Differences depend on latent order → cross-z state contamination."
    return "Decision: Mixed mismatches — inspect per-field diffs above."


def annotate_expected_seeds(cells: CellEpisodes, protocol: ForcedZProtocol) -> None:
    for opp_idx, opponent in enumerate(protocol.opponents):
        for map_idx, map_name in enumerate(protocol.maps):
            cell_seed = protocol.cell_seed(opp_idx, map_idx)
            for z in protocol.latents:
                for ep_idx, ep in enumerate(cells.get((opponent, z, map_name), [])):
                    ep.setdefault("episode_index", ep_idx)
                    ep.setdefault("episode_seed", protocol.episode_seed(cell_seed, ep_idx))
                    ep.setdefault("fixed_latent_id", int(z))
                    ep.setdefault("opponent", opponent)


__all__ = [
    "BEHAVIOR_KEYS",
    "EquivalenceReport",
    "STRICT_FLOAT_ATOL",
    "STRICT_FLOAT_RTOL",
    "TERMINAL_KEYS",
    "annotate_expected_seeds",
    "compare_episode",
    "compare_forced_z_cells",
    "decision_tree_hint",
]
