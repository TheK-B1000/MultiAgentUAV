"""Seven-dimensional forced-z behavior vectors and pairwise distance telemetry.

Maps rollout ``BEHAVIOR_TELEMETRY_NAMES`` (full env geometry) or macro-action
probabilities (counterfactual forced-z on replayed observations) into a
fixed repertoire diagnostic basis. Components are normalized to comparable
``[0, 1]`` scales before pairwise L2 distance. Used to separate *actor
intervention* (action JSD) from *behavioral realization* (team-level tactic
separation).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from macro_actions import MacroAction
from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES, N_TELEMETRY
from rl.gate_telemetry import phase_a_actor_pair_telemetry_from_actor_gate_details

FORCED_Z_BEHAVIOR_VECTOR_NAMES: tuple[str, ...] = (
    "attack_lane_preference",
    "return_lane_preference",
    "interception_pressure",
    "escort_allocation",
    "team_spread",
    "objective_entry_timing",
    "role_allocation",
)

N_BEHAVIOR_VECTOR: int = len(FORCED_Z_BEHAVIOR_VECTOR_NAMES)

# Canonical normalization bounds per source (lo, hi). Distances use clipped linear maps.
BEHAVIOR_COMPONENT_BOUNDS_MACRO: dict[str, tuple[float, float]] = {
    "attack_lane_preference": (0.0, 1.0),
    "return_lane_preference": (0.0, 1.0),
    "interception_pressure": (0.0, 1.25),
    "escort_allocation": (0.0, 1.0),
    "team_spread": (0.0, 0.45),
    "objective_entry_timing": (0.0, 1.0),
    "role_allocation": (0.0, 1.0),
}

BEHAVIOR_COMPONENT_BOUNDS_TELEMETRY: dict[str, tuple[float, float]] = {
    "attack_lane_preference": (0.0, 1.0),
    "return_lane_preference": (0.0, 1.0),
    "interception_pressure": (0.0, 1.0),
    "escort_allocation": (0.0, 1.0),
    "team_spread": (0.0, 1.5),
    "objective_entry_timing": (0.0, 1.0),
    "role_allocation": (0.0, 1.0),
}

DEFAULT_BEHAVIOR_PAIR_THRESHOLD: float = 0.35
DEFAULT_BEHAVIOR_PAIR_FLOOR_FRACTION: float = 0.5
DEFAULT_MIN_PAIRS_ABOVE: int = 5
DEFAULT_OPPORTUNITY_MIN_SAMPLES_PER_Z: int = 3
DEFAULT_OPPORTUNITY_BEST_MARGIN_FLOOR: float = 0.05
OPPORTUNITY_MAX_CELLS_REPORTED: int = 4
PHASE_A_TREND_WINDOW: int = 20
PHASE_A_STATS_MAX_STALENESS_STEPS: int = 131_072

CF_REGIME_UNDERPOWERED = 0
CF_REGIME_PRODUCTIVE = 1
CF_REGIME_DESTRUCTIVE = 2
CF_REGIME_COSMETIC = 3

INTERVENTION_QUADRANT_COLLAPSE = 0
INTERVENTION_QUADRANT_COSMETIC = 1
INTERVENTION_QUADRANT_MISMATCH = 2
INTERVENTION_QUADRANT_GENUINE = 3


def _idx(name: str) -> int:
    return BEHAVIOR_TELEMETRY_NAMES.index(name)


def _bounds_for_source(source: str) -> dict[str, tuple[float, float]]:
    if source == "telemetry":
        return BEHAVIOR_COMPONENT_BOUNDS_TELEMETRY
    return BEHAVIOR_COMPONENT_BOUNDS_MACRO


def behavior_vector_from_telemetry_row(row: np.ndarray | torch.Tensor) -> np.ndarray:
    """Map one ``BEHAVIOR_TELEMETRY_NAMES`` row to the raw 7-dim behavior vector."""
    arr = np.asarray(row, dtype=np.float64).reshape(-1)
    if arr.shape[0] < N_TELEMETRY:
        pad = np.zeros((N_TELEMETRY,), dtype=np.float64)
        pad[: arr.shape[0]] = arr
        arr = pad
    n_agents = 4.0
    attack_macro = float(arr[_idx("num_attackers")]) / n_agents
    return np.asarray(
        [
            float(arr[_idx("avg_blue_to_enemy_flag")]),
            float(arr[_idx("avg_blue_to_own_flag")]),
            float(
                0.5 * (arr[_idx("intercept_pressure")] + arr[_idx("n_intercept_near_enemy_carrier")] / n_agents)
            ),
            float(arr[_idx("carrier_escort_count")] / n_agents),
            float(arr[_idx("team_spread")]),
            float(attack_macro),
            float(arr[_idx("attack_defense_ratio")]),
        ],
        dtype=np.float64,
    )


def behavior_vector_from_macro_probs(mean_macro: torch.Tensor | np.ndarray) -> np.ndarray:
    """Raw 7-dim proxy vector from mean macro probabilities."""
    m = np.asarray(
        mean_macro.detach().cpu().numpy() if isinstance(mean_macro, torch.Tensor) else mean_macro,
        dtype=np.float64,
    ).reshape(-1)
    if m.size < 5:
        pad = np.zeros((5,), dtype=np.float64)
        pad[: m.size] = m
        m = pad
    get_flag = float(m[int(MacroAction.GET_FLAG)])
    go_home = float(m[int(MacroAction.GO_HOME)])
    go_to = float(m[int(MacroAction.GO_TO)])
    place_mine = float(m[int(MacroAction.PLACE_MINE)])
    attack_denom = max(get_flag + go_home, 1e-6)
    return np.asarray(
        [
            get_flag,
            go_home,
            place_mine + 0.25 * go_to,
            go_to * max(0.0, 1.0 - get_flag),
            float(np.std(m)),
            get_flag / max(get_flag + go_to, 1e-6),
            get_flag / attack_denom,
        ],
        dtype=np.float64,
    )


def component_scale_and_validity(
    raw_vectors: list[np.ndarray],
    *,
    source: str,
) -> dict[str, float]:
    """Report per-component scale (hi-lo) and validity across all z vectors."""
    bounds = _bounds_for_source(source)
    out: dict[str, float] = {}
    stacked = np.stack(raw_vectors, axis=0) if raw_vectors else np.zeros((0, N_BEHAVIOR_VECTOR))
    for dim_idx, name in enumerate(FORCED_Z_BEHAVIOR_VECTOR_NAMES):
        lo, hi = bounds[name]
        scale = float(max(hi - lo, 1e-6))
        out[f"behavior_component_scale_{name}"] = scale
        if stacked.shape[0] == 0:
            out[f"behavior_component_valid_{name}"] = 0.0
            continue
        col = stacked[:, dim_idx]
        finite = np.isfinite(col)
        in_range = (col >= lo - 1e-3) & (col <= hi + 1e-3)
        out[f"behavior_component_valid_{name}"] = float(finite.all() and in_range.all())
    return out


def normalize_behavior_vectors(
    raw_vectors: list[np.ndarray],
    *,
    source: str,
) -> list[np.ndarray]:
    """Map raw vectors to comparable ``[0, 1]`` components using canonical bounds."""
    bounds = _bounds_for_source(source)
    normalized: list[np.ndarray] = []
    for raw in raw_vectors:
        out = np.zeros((N_BEHAVIOR_VECTOR,), dtype=np.float64)
        for dim_idx, name in enumerate(FORCED_Z_BEHAVIOR_VECTOR_NAMES):
            lo, hi = bounds[name]
            scale = max(hi - lo, 1e-6)
            out[dim_idx] = float(np.clip((float(raw[dim_idx]) - lo) / scale, 0.0, 1.0))
        normalized.append(out)
    return normalized


def pairwise_behavior_distances(
    vectors: list[np.ndarray],
    *,
    pair_count: int | None = None,
    pair_threshold: float = DEFAULT_BEHAVIOR_PAIR_THRESHOLD,
    pair_floor_fraction: float = DEFAULT_BEHAVIOR_PAIR_FLOOR_FRACTION,
    min_pairs_above: int = DEFAULT_MIN_PAIRS_ABOVE,
    already_normalized: bool = False,
    source: str = "macro",
) -> tuple[dict[str, float], list[float]]:
    """Pairwise L2 on normalized behavior vectors with min / count-above-threshold."""
    k = len(vectors)
    if k < 2:
        return {}, []
    work = vectors if already_normalized else normalize_behavior_vectors(vectors, source=source)
    out: dict[str, float] = {}
    aggregate: list[float] = []
    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            diff = work[i] - work[j]
            dist = float(np.linalg.norm(diff))
            aggregate.append(dist)
            if pair_count is None or pair_idx < int(pair_count):
                out[f"forced_z_behavior_pair_distance_{pair_idx}"] = dist
                for dim_idx, name in enumerate(FORCED_Z_BEHAVIOR_VECTOR_NAMES):
                    out[f"forced_z_pair_{name}_distance_{pair_idx}"] = float(abs(diff[dim_idx]))
            pair_idx += 1
    if aggregate:
        out["forced_z_behavior_pair_distance_mean"] = float(np.mean(aggregate))
        out["forced_z_behavior_pair_distance_max"] = float(np.max(aggregate))
        out["forced_z_behavior_pair_distance_min"] = float(np.min(aggregate))
        floor = float(pair_threshold) * float(pair_floor_fraction)
        above = sum(1 for d in aggregate if d >= float(pair_threshold))
        out["forced_z_behavior_pairs_above_threshold"] = float(above)
        out["phase_a_behavior_pairs_above_threshold"] = float(above)
        out["phase_a_behavior_weakest_pair_distance"] = float(min(aggregate))
        out["phase_a_behavior_pair_gate_pass"] = float(
            above >= int(min_pairs_above) and min(aggregate) >= floor
        )
    else:
        out["forced_z_behavior_pair_distance_mean"] = 0.0
        out["forced_z_behavior_pair_distance_max"] = 0.0
        out["forced_z_behavior_pair_distance_min"] = 0.0
        out["forced_z_behavior_pairs_above_threshold"] = 0.0
        out["phase_a_behavior_pairs_above_threshold"] = 0.0
        out["phase_a_behavior_weakest_pair_distance"] = 0.0
        out["phase_a_behavior_pair_gate_pass"] = 0.0
    return out, aggregate


def per_z_vector_telemetry(vectors: list[np.ndarray], *, normalized: bool = True) -> dict[str, float]:
    out: dict[str, float] = {}
    suffix = "" if normalized else "_raw"
    for z_id, vec in enumerate(vectors):
        for dim_idx, name in enumerate(FORCED_Z_BEHAVIOR_VECTOR_NAMES):
            out[f"forced_z{z_id}_behavior_{name}{suffix}"] = float(vec[dim_idx])
    return out


def build_behavior_distance_profile(
    raw_vectors: list[np.ndarray],
    *,
    source: str,
    pair_count: int,
    latent_k: int,
    pair_threshold: float = DEFAULT_BEHAVIOR_PAIR_THRESHOLD,
) -> dict[str, float]:
    """Full forced-z behavior profile: scales, normalized per-z vectors, pairwise stats."""
    out = component_scale_and_validity(raw_vectors, source=source)
    normalized = normalize_behavior_vectors(raw_vectors, source=source)
    out.update(per_z_vector_telemetry(normalized, normalized=True))
    pair_stats, _ = pairwise_behavior_distances(
        normalized,
        pair_count=pair_count,
        pair_threshold=pair_threshold,
        already_normalized=True,
    )
    out.update(pair_stats)
    represented = len(raw_vectors)
    all_z = float(represented >= int(latent_k))
    all_components = all(
        float(out.get(f"behavior_component_valid_{name}", 0.0)) >= 0.5
        for name in FORCED_Z_BEHAVIOR_VECTOR_NAMES
    )
    out["forced_z_behavior_all_z_represented"] = all_z
    out["forced_z_behavior_components_valid"] = float(all_components)
    return out


def actor_pair_stats_from_update(
    stats: dict[str, float],
    *,
    margin: float,
    pair_count: int = 6,
    min_pairs_above: int = DEFAULT_MIN_PAIRS_ABOVE,
    floor_fraction: float = DEFAULT_BEHAVIOR_PAIR_FLOOR_FRACTION,
) -> dict[str, float]:
    """Actor-intervention pair coverage from CF-batch or forced-z macro JSD pairs."""
    cf_keys = [f"cf_batch_pair_jsd_{i}" for i in range(int(pair_count))]
    cf_pairs_present = all(key in stats for key in cf_keys)
    use_cf = float(stats.get("cf_batch_evidence_valid", 0.0) or 0.0) >= 0.5 or cf_pairs_present
    prefix = "cf_batch_pair_jsd_" if use_cf else "forced_z_pair_jsd_"
    pairs = [float(stats.get(f"{prefix}{i}", 0.0) or 0.0) for i in range(int(pair_count))]
    if not use_cf and not any(p > 0.0 for p in pairs):
        prefix = "forced_z_pair_jsd_"
        pairs = [float(stats.get(f"{prefix}{i}", 0.0) or 0.0) for i in range(int(pair_count))]
    if not pairs:
        return {
            "phase_a_actor_pairs_above_margin": 0.0,
            "phase_a_actor_weakest_pair_jsd": 0.0,
            "phase_a_actor_pair_gate_pass": 0.0,
        }
    above = sum(1 for p in pairs if p >= float(margin))
    weakest = float(min(pairs))
    floor = float(margin) * float(floor_fraction)
    return {
        "phase_a_actor_pairs_above_margin": float(above),
        "phase_a_actor_weakest_pair_jsd": weakest,
        "phase_a_actor_pair_gate_pass": float(above >= int(min_pairs_above) and weakest >= floor),
    }


@dataclass
class PhaseABehaviorTrendTracker:
    """Rolling window for Phase A actor JSD and behavior-distance slopes."""

    window: int = PHASE_A_TREND_WINDOW
    _actor_jsd: deque[float] = field(default_factory=deque)
    _behavior_dist: deque[float] = field(default_factory=deque)
    actor_valid_updates: int = 0
    behavior_valid_updates: int = 0
    last_behavior_step: int = -1
    last_actor_step: int = -1
    last_global_step: int = -1

    def __post_init__(self) -> None:
        self._actor_jsd = deque(maxlen=int(self.window))
        self._behavior_dist = deque(maxlen=int(self.window))

    @staticmethod
    def _slope(values: deque[float]) -> float:
        if len(values) < 2:
            return 0.0
        y = np.asarray(values, dtype=np.float64)
        if float(np.std(y)) < 1e-12:
            return 0.0
        x = np.arange(len(y), dtype=np.float64)
        return float(np.polyfit(x, y, 1)[0])

    def record(
        self,
        *,
        global_step: int,
        actor_jsd: float,
        behavior_dist: float,
        actor_valid: bool,
        behavior_valid: bool,
    ) -> None:
        if actor_valid:
            self._actor_jsd.append(float(actor_jsd))
            self.actor_valid_updates += 1
            self.last_actor_step = int(global_step)
        if behavior_valid:
            self._behavior_dist.append(float(behavior_dist))
            self.behavior_valid_updates += 1
            self.last_behavior_step = int(global_step)
        self.last_global_step = int(global_step)

    def telemetry(self) -> dict[str, float]:
        return {
            "phase_a_actor_jsd_slope_20": self._slope(self._actor_jsd),
            "phase_a_behavior_distance_slope_20": self._slope(self._behavior_dist),
            "phase_a_actor_jsd_valid_updates": float(self.actor_valid_updates),
            "phase_a_behavior_valid_updates": float(self.behavior_valid_updates),
        }


def classify_intervention_quadrant(
    actor_jsd: float,
    behavior_distance: float,
    *,
    actor_jsd_threshold: float = 0.01,
    behavior_distance_threshold: float = 0.05,
) -> tuple[int, str]:
    high_jsd = float(actor_jsd) >= float(actor_jsd_threshold)
    high_beh = float(behavior_distance) >= float(behavior_distance_threshold)
    if not high_jsd and not high_beh:
        return INTERVENTION_QUADRANT_COLLAPSE, "collapse"
    if high_jsd and not high_beh:
        return INTERVENTION_QUADRANT_COSMETIC, "cosmetic"
    if not high_jsd and high_beh:
        return INTERVENTION_QUADRANT_MISMATCH, "measurement_mismatch"
    return INTERVENTION_QUADRANT_GENUINE, "genuine_latent_control"


def classify_cf_training_regime(
    *,
    cf_to_ppo_ratio: float,
    competence_min: float,
    behavior_distance: float,
    behavior_slope: float,
    actor_slope: float,
    ratio_floor: float = 0.02,
    ratio_ceiling: float = 2.0,
    competence_floor: float = 0.35,
    behavior_slope_floor: float = 0.001,
    actor_jsd_floor: float = 0.01,
    actor_jsd: float = 0.0,
) -> tuple[int, str]:
    ratio = float(cf_to_ppo_ratio)
    comp = float(competence_min)
    beh = float(behavior_distance)
    b_slope = float(behavior_slope)
    a_slope = float(actor_slope)
    jsd = float(actor_jsd)

    if ratio < ratio_floor and b_slope < behavior_slope_floor:
        return CF_REGIME_UNDERPOWERED, "underpowered"
    if jsd >= actor_jsd_floor and b_slope < behavior_slope_floor:
        return CF_REGIME_COSMETIC, "cosmetic"
    if ratio > ratio_ceiling and comp < competence_floor:
        return CF_REGIME_DESTRUCTIVE, "destructive"
    if b_slope >= behavior_slope_floor and a_slope >= behavior_slope_floor and comp >= competence_floor:
        return CF_REGIME_PRODUCTIVE, "productive"
    if beh > 0.0 and comp >= competence_floor and b_slope >= 0.0:
        return CF_REGIME_PRODUCTIVE, "productive"
    return CF_REGIME_UNDERPOWERED, "underpowered"


def phase_a_diagnostic_telemetry(
    stats: dict[str, float],
    *,
    trend_tracker: PhaseABehaviorTrendTracker | None = None,
    global_step: int = 0,
    actor_gate_details: dict[str, Any] | None = None,
    actor_jsd_threshold: float = 0.01,
    behavior_distance_threshold: float = 0.05,
    actor_jsd_margin: float = 0.01,
    ratio_floor: float = 0.02,
    ratio_ceiling: float = 2.0,
    competence_floor: float = 0.35,
    behavior_slope_floor: float = 0.001,
) -> dict[str, float]:
    """Augment one update row with Phase A repertoire diagnostic fields."""
    actor_jsd = float(
        stats.get("latent_actor_z_separation_jsd", stats.get("forced_z_macro_jsd_mean", 0.0)) or 0.0
    )
    cf_jsd = float(stats.get("latent_actor_z_separation_jsd", 0.0) or 0.0)
    if stats.get("cf_batch_pair_jsd_0") is not None:
        cf_pairs = [
            float(stats.get(f"cf_batch_pair_jsd_{i}", 0.0) or 0.0)
            for i in range(6)
            if f"cf_batch_pair_jsd_{i}" in stats
        ]
        if cf_pairs:
            actor_jsd = max(actor_jsd, float(np.mean(cf_pairs)))

    beh_mean = float(stats.get("forced_z_behavior_pair_distance_mean", 0.0) or 0.0)
    beh_min = float(stats.get("forced_z_behavior_pair_distance_min", beh_mean) or beh_mean)

    tracker = trend_tracker or PhaseABehaviorTrendTracker()
    behavior_valid = float(stats.get("forced_z_behavior_components_valid", 0.0)) >= 0.5 and float(
        stats.get("forced_z_behavior_all_z_represented", 0.0)
    ) >= 0.5
    actor_valid = float(stats.get("pairwise_profile_available", 0.0)) >= 0.5 or float(
        stats.get("cf_batch_evidence_valid", 0.0)
    ) >= 0.5
    tracker.record(
        global_step=int(global_step),
        actor_jsd=actor_jsd,
        behavior_dist=beh_mean,
        actor_valid=bool(actor_valid),
        behavior_valid=bool(behavior_valid),
    )
    trend = tracker.telemetry()
    actor_slope = float(trend["phase_a_actor_jsd_slope_20"])
    behavior_slope = float(trend["phase_a_behavior_distance_slope_20"])

    comp_vals = [float(stats.get(f"cf_competence_z{z}", 0.0) or 0.0) for z in range(4)]
    competence_min = min(comp_vals) if comp_vals else 0.0
    cf_ratio = float(stats.get("cf_to_ppo_grad_ratio", 0.0) or 0.0)

    quad_code, quad_label = classify_intervention_quadrant(
        actor_jsd,
        beh_min,
        actor_jsd_threshold=actor_jsd_threshold,
        behavior_distance_threshold=behavior_distance_threshold,
    )
    regime_code, regime_label = classify_cf_training_regime(
        cf_to_ppo_ratio=cf_ratio,
        competence_min=competence_min,
        behavior_distance=beh_mean,
        behavior_slope=behavior_slope,
        actor_slope=actor_slope,
        ratio_floor=ratio_floor,
        ratio_ceiling=ratio_ceiling,
        competence_floor=competence_floor,
        behavior_slope_floor=behavior_slope_floor,
        actor_jsd_floor=actor_jsd_threshold,
        actor_jsd=actor_jsd,
    )

    actor_pairs = (
        phase_a_actor_pair_telemetry_from_actor_gate_details(actor_gate_details)
        if actor_gate_details is not None
        else actor_pair_stats_from_update(stats, margin=actor_jsd_margin)
    )
    competence_floor_pass = competence_min >= competence_floor
    cf_ratio_in_band = ratio_floor <= cf_ratio <= ratio_ceiling
    actor_trending_up = actor_slope >= behavior_slope_floor
    behavior_trending_up = behavior_slope >= behavior_slope_floor
    behavior_pair_gate = float(stats.get("phase_a_behavior_pair_gate_pass", 0.0)) >= 0.5
    actor_pair_gate = float(actor_pairs.get("phase_a_actor_pair_gate_pass", 0.0)) >= 0.5
    corridor_viable = (
        competence_floor_pass
        and cf_ratio_in_band
        and actor_trending_up
        and behavior_trending_up
        and behavior_pair_gate
        and actor_pair_gate
    )

    out = {
        "phase_a_stats_source_step": float(global_step),
        "phase_a_actor_jsd_mean": actor_jsd,
        "phase_a_cf_actor_jsd_mean": cf_jsd,
        "phase_a_behavior_distance_mean": beh_mean,
        "phase_a_behavior_distance_min": beh_min,
        "phase_a_intervention_quadrant": float(quad_code),
        "phase_a_intervention_quadrant_name": float(
            {
                "collapse": 0.0,
                "cosmetic": 1.0,
                "measurement_mismatch": 2.0,
                "genuine_latent_control": 3.0,
            }.get(quad_label, -1.0)
        ),
        "phase_a_cf_regime": float(regime_code),
        "phase_a_cf_regime_name": float(
            {
                "underpowered": 0.0,
                "productive": 1.0,
                "destructive": 2.0,
                "cosmetic": 3.0,
            }.get(regime_label, -1.0)
        ),
        "phase_a_competence_min": competence_min,
        "phase_a_cf_ratio": cf_ratio,
        "phase_a_competence_floor_pass": float(competence_floor_pass),
        "phase_a_cf_ratio_in_band": float(cf_ratio_in_band),
        "phase_a_actor_intervention_trending_up": float(actor_trending_up),
        "phase_a_behavioral_realization_trending_up": float(behavior_trending_up),
        "phase_a_corridor_viable": float(corridor_viable),
        "phase_a_behavior_measurement_valid": float(behavior_valid),
    }
    out.update(actor_pairs)
    out.update(trend)
    return out


PHASE_A_DIAGNOSTIC_STAT_KEYS: tuple[str, ...] = (
    "phase_a_stats_source_step",
    "phase_a_actor_jsd_mean",
    "phase_a_actor_jsd_slope_20",
    "phase_a_behavior_distance_mean",
    "phase_a_behavior_distance_min",
    "phase_a_behavior_distance_slope_20",
    "phase_a_intervention_quadrant",
    "phase_a_intervention_quadrant_name",
    "phase_a_cf_regime",
    "phase_a_cf_regime_name",
    "phase_a_competence_min",
    "phase_a_cf_ratio",
    "phase_a_competence_floor_pass",
    "phase_a_cf_ratio_in_band",
    "phase_a_actor_intervention_trending_up",
    "phase_a_behavioral_realization_trending_up",
    "phase_a_corridor_viable",
    "phase_a_behavior_measurement_valid",
    "phase_a_behavior_valid_updates",
    "phase_a_actor_jsd_valid_updates",
    "phase_a_actor_pairs_above_margin",
    "phase_a_actor_weakest_pair_jsd",
    "phase_a_actor_pair_gate_pass",
    "phase_a_behavior_pairs_above_threshold",
    "phase_a_behavior_weakest_pair_distance",
    "phase_a_behavior_pair_gate_pass",
    "forced_z_behavior_pair_distance_mean",
    "forced_z_behavior_pair_distance_min",
    "opportunity_cell_count",
    "opportunity_eligible_cell_count",
    "opportunity_fork_fraction",
    "opportunity_fork_fraction_valid",
    "opportunity_homogeneous_fraction",
    "opportunity_best_z_unique",
    "opportunity_measurement_valid",
)


def phase_a_stats_snapshot(
    stats: dict[str, float],
    *,
    gate_step: int,
    max_staleness_steps: int = PHASE_A_STATS_MAX_STALENESS_STEPS,
) -> dict[str, Any]:
    """Subset of training stats for gate JSON / stdout; rejects stale or invalid rows."""
    out: dict[str, Any] = {}
    for key in PHASE_A_DIAGNOSTIC_STAT_KEYS:
        if key in stats:
            out[key] = float(stats[key])

    source = int(stats.get("phase_a_stats_source_step", -1))
    stale = source < 0 or (int(gate_step) - source) > int(max_staleness_steps)
    behavior_valid = float(stats.get("phase_a_behavior_measurement_valid", 0.0)) >= 0.5
    opportunity_valid = float(stats.get("opportunity_measurement_valid", 0.0)) >= 0.5
    all_z = float(stats.get("forced_z_behavior_all_z_represented", 0.0)) >= 0.5
    components_valid = float(stats.get("forced_z_behavior_components_valid", 0.0)) >= 0.5

    snapshot_usable = not stale and behavior_valid and all_z and components_valid
    out["phase_a_stats_source_step"] = float(source)
    out["phase_a_stats_gate_step"] = float(gate_step)
    out["phase_a_stats_stale"] = float(stale)
    out["phase_a_behavior_measurement_valid"] = float(behavior_valid and not stale)
    out["phase_a_opportunity_measurement_valid"] = float(opportunity_valid)
    out["phase_a_snapshot_usable"] = float(snapshot_usable)

    quad = int(stats.get("phase_a_intervention_quadrant", -1))
    quad_names = ("collapse", "cosmetic", "measurement_mismatch", "genuine_latent_control")
    if 0 <= quad < len(quad_names):
        out["phase_a_intervention_quadrant_label"] = quad_names[quad]
    regime = int(stats.get("phase_a_cf_regime", -1))
    regime_names = ("underpowered", "productive", "destructive", "cosmetic")
    if 0 <= regime < len(regime_names):
        out["phase_a_cf_regime_label"] = regime_names[regime]
    if not snapshot_usable:
        out["phase_a_intervention_quadrant_label"] = "not_run"
        out["phase_a_cf_regime_label"] = "not_run"
    return out


def _opportunity_key(
    opponent_id: int,
    pressure_bucket: int,
    role_bucket: int,
    spread_bucket: int,
    phase_id: int,
    blue_ahead: bool,
) -> tuple[int, ...]:
    return (
        int(opponent_id),
        int(pressure_bucket),
        int(role_bucket),
        int(spread_bucket),
        int(phase_id),
        1 if blue_ahead else 0,
    )


def opportunity_conditioned_z_returns(
    buffer: Any,
    *,
    latent_k: int,
    max_opportunities: int = 128,
    forced_steps_only: bool = False,
    min_samples_per_z: int = DEFAULT_OPPORTUNITY_MIN_SAMPLES_PER_Z,
    best_margin_floor: float = DEFAULT_OPPORTUNITY_BEST_MARGIN_FLOOR,
    max_cells_reported: int = OPPORTUNITY_MAX_CELLS_REPORTED,
) -> dict[str, float]:
    """Observational opportunity fork stats with per-cell support and margin gates."""
    length = int(buffer.pos)
    if length <= 0 or "returns" not in buffer.fields or "z" not in buffer.fields:
        return {"opportunity_measurement_valid": 0.0}
    required = (
        "opponent_id",
        "pressure_bucket_id",
        "role_bucket_id",
        "spread_bucket_id",
        "phase_id",
        "blue_ahead",
    )
    if any(f not in buffer.fields for f in required):
        return {"opportunity_measurement_valid": 0.0}

    def _fork_stats(*, forced_mask: np.ndarray | None) -> dict[str, float]:
        rets = buffer.fields["returns"][:length].reshape(-1).float().cpu().numpy()
        z = buffer.fields["z"][:length].reshape(-1).long().cpu().numpy()
        opp = buffer.fields["opponent_id"][:length].reshape(-1).long().cpu().numpy()
        pb = buffer.fields["pressure_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        rb = buffer.fields["role_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        sb = buffer.fields["spread_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        ph = buffer.fields["phase_id"][:length].reshape(-1).long().cpu().numpy()
        ahead = buffer.fields["blue_ahead"][:length].reshape(-1).float().cpu().numpy() > 0.5

        cells: dict[tuple[int, ...], dict[int, list[float]]] = {}
        for idx in range(rets.shape[0]):
            if forced_mask is not None and not bool(forced_mask[idx]):
                continue
            zk = int(z[idx])
            if zk < 0 or zk >= int(latent_k):
                continue
            key = _opportunity_key(
                int(opp[idx]), int(pb[idx]), int(rb[idx]), int(sb[idx]), int(ph[idx]), bool(ahead[idx])
            )
            cells.setdefault(key, {}).setdefault(zk, []).append(float(rets[idx]))

        if not cells:
            return {
                "opportunity_cell_count": 0.0,
                "opportunity_eligible_cell_count": 0.0,
                "opportunity_fork_fraction": 0.0,
                "opportunity_fork_fraction_valid": 0.0,
                "opportunity_homogeneous_fraction": 1.0,
                "opportunity_best_z_unique": 0.0,
                "opportunity_measurement_valid": 0.0,
            }

        cell_records: list[dict[str, Any]] = []
        eligible_best_z: list[int] = []

        for cell_id, (_key, by_z) in enumerate(cells.items()):
            record: dict[str, Any] = {"cell_id": cell_id, "counts": {}, "means": {}, "ses": {}}
            for zk in range(int(latent_k)):
                vals = by_z.get(zk, [])
                n = len(vals)
                record["counts"][zk] = n
                if n > 0:
                    arr = np.asarray(vals, dtype=np.float64)
                    record["means"][zk] = float(np.mean(arr))
                    record["ses"][zk] = float(np.std(arr) / max(1.0, np.sqrt(float(n))))
                else:
                    record["means"][zk] = 0.0
                    record["ses"][zk] = 0.0

            supported = all(int(record["counts"].get(zk, 0)) >= int(min_samples_per_z) for zk in range(int(latent_k)))
            record["eligible"] = supported
            if supported:
                ranked = sorted(
                    ((zk, record["means"][zk]) for zk in range(int(latent_k))),
                    key=lambda t: t[1],
                    reverse=True,
                )
                best_z, best_mean = ranked[0]
                second_z, second_mean = ranked[1]
                margin = float(best_mean - second_mean)
                uncertainty = float(record["ses"][best_z] + record["ses"][second_z])
                record["best_margin"] = margin
                record["best_z"] = int(best_z)
                record["margin_ambiguous"] = margin < max(float(best_margin_floor), uncertainty)
                if not record["margin_ambiguous"]:
                    eligible_best_z.append(int(best_z))
            else:
                record["best_margin"] = 0.0
                record["best_z"] = -1
                record["margin_ambiguous"] = True
            cell_records.append(record)

        cell_records.sort(
            key=lambda r: sum(int(r["counts"].get(zk, 0)) for zk in range(int(latent_k))),
            reverse=True,
        )

        out: dict[str, float] = {
            "opportunity_cell_count": float(min(len(cells), max_opportunities)),
            "opportunity_eligible_cell_count": float(sum(1 for r in cell_records if r["eligible"])),
            "opportunity_measurement_valid": float(len(cells) > 0),
        }

        for rep_idx in range(int(max_cells_reported)):
            if rep_idx >= len(cell_records):
                for zk in range(int(latent_k)):
                    out[f"opportunity_cell_{rep_idx}_count_z{zk}"] = 0.0
                    out[f"opportunity_cell_{rep_idx}_return_mean_z{zk}"] = 0.0
                    out[f"opportunity_cell_{rep_idx}_return_se_z{zk}"] = 0.0
                out[f"opportunity_cell_{rep_idx}_best_margin"] = 0.0
                out[f"opportunity_cell_{rep_idx}_eligible"] = 0.0
                continue
            rec = cell_records[rep_idx]
            for zk in range(int(latent_k)):
                out[f"opportunity_cell_{rep_idx}_count_z{zk}"] = float(rec["counts"].get(zk, 0))
                out[f"opportunity_cell_{rep_idx}_return_mean_z{zk}"] = float(rec["means"].get(zk, 0.0))
                out[f"opportunity_cell_{rep_idx}_return_se_z{zk}"] = float(rec["ses"].get(zk, 0.0))
            out[f"opportunity_cell_{rep_idx}_best_margin"] = float(rec.get("best_margin", 0.0))
            out[f"opportunity_cell_{rep_idx}_eligible"] = float(rec.get("eligible", False) and not rec.get("margin_ambiguous", True))

        all_best: list[int] = []
        for rec in cell_records:
            if rec.get("eligible") and not rec.get("margin_ambiguous"):
                all_best.append(int(rec["best_z"]))

        if len(all_best) >= 2:
            total_pairs = len(all_best) * (len(all_best) - 1) // 2
            fork_cells = sum(
                1
                for i in range(len(all_best))
                for j in range(i + 1, len(all_best))
                if all_best[i] != all_best[j]
            )
            fork_fraction_valid = float(fork_cells / max(1, total_pairs))
        else:
            fork_fraction_valid = 0.0

        raw_best: list[int] = []
        for rec in cell_records:
            if int(rec.get("best_z", -1)) >= 0:
                raw_best.append(int(rec["best_z"]))

        if len(raw_best) >= 2:
            total_raw = len(raw_best) * (len(raw_best) - 1) // 2
            fork_raw = sum(
                1 for i in range(len(raw_best)) for j in range(i + 1, len(raw_best)) if raw_best[i] != raw_best[j]
            )
            fork_fraction = float(fork_raw / max(1, total_raw))
        else:
            fork_fraction = 0.0

        out["opportunity_fork_fraction"] = fork_fraction
        out["opportunity_fork_fraction_valid"] = fork_fraction_valid
        out["opportunity_homogeneous_fraction"] = 1.0 - fork_fraction_valid if all_best else 1.0
        out["opportunity_best_z_unique"] = float(len(set(all_best))) if all_best else float(len(set(raw_best)))
        return out

    out = _fork_stats(forced_mask=None)
    if forced_steps_only and "z_forced" in buffer.fields:
        forced = buffer.fields["z_forced"][:length].reshape(-1).bool().cpu().numpy()
        out = _fork_stats(forced_mask=forced)
    elif "z_forced" in buffer.fields:
        forced = buffer.fields["z_forced"][:length].reshape(-1).bool().cpu().numpy()
        if bool(forced.any()):
            forced_stats = _fork_stats(forced_mask=forced)
            out["opportunity_fork_fraction_forced"] = float(forced_stats.get("opportunity_fork_fraction", 0.0))
            out["opportunity_fork_fraction_valid_forced"] = float(
                forced_stats.get("opportunity_fork_fraction_valid", 0.0)
            )
            out["opportunity_best_z_unique_forced"] = float(forced_stats.get("opportunity_best_z_unique", 0.0))
    return out
