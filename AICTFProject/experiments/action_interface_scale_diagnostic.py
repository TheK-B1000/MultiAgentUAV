"""Outcome-blind 2v2/4v4 action-interface scale diagnostic.

Implements ACTION_INTERFACE_SCALE_DIAGNOSTIC_SPEC.json and its provenance
amendment. The production macro interface is read-only. Scripted source traces
are collected once, then I0-I3 are replayed against the same exogenous context.

Scientific delta: measure whether spatial projection, temporal commitment, or
the residual full macro-execution contract has a BREACH-specific 2v2-to-4v4
interaction on certified/historical Pole-B traces. Classification: DIAGNOSTIC.
There is no parent latent preset and no resolved PPOConfig delta.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_lock import RunLock  # noqa: E402
from experiments.sds_genome import apply_genome_to_core  # noqa: E402
from experiments.teacher_action_adapter import Adapted, adapt  # noqa: E402
from gpu_env import BatchedCTFCore, GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from macro_actions import MacroAction  # noqa: E402


SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_SPEC.json"
AMENDMENT = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_INTEGRITY_AMENDMENT.json"
CONTRACT_RESULT = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_CONTRACT_RESULT.json"
TRACE_MANIFEST = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_TRACE_MANIFEST.json"
TICK_ROWS = SD / "action_interface_scale_diagnostic_agent_tick_rows.csv"
EPISODE_ROWS = SD / "action_interface_scale_diagnostic_episode_rows.csv"
RESULT = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_RESULT.json"
LOCK = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC.run.lock"
POLE_B_CANDIDATE = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"
REGRESSION_GUARD = SD / "SIZE_NORMALIZED_POLES_REGRESSION_GUARD.json"

LABEL = "ACTION_INTERFACE_SCALE_DIAGNOSTIC"
EXPERIMENT_ID = LABEL

MECHANISM_AMENDMENT_ID = "ACTION_INTERFACE_SCALE_DIAGNOSTIC_MECHANISM_AMENDMENT_V1"
MECHANISM_AMENDMENT = SD / f"{MECHANISM_AMENDMENT_ID}.json"

INTERVENTION_ID = "COMMITMENT_INTERRUPTIBILITY_INTERVENTION_V1"
INTERVENTION_SPEC = SD / f"{INTERVENTION_ID}_SPEC.json"
INTERVENTION_AMENDMENT = SD / f"{INTERVENTION_ID}_IMPLEMENTATION_AMENDMENT.json"

# INTERRUPT_CONDITION_R, frozen operating point. Not tunable after results.
R_THETA_DEG = 45.0
R_K_TICKS = 2
INTERVENTION_ARMS = (
    "I0_CONTINUOUS_ORACLE",
    "I1_W50_NO_COMMIT",
    "I2_W50_CURRENT_COMMIT",
    "I2R_W50_INTERRUPTIBLE_COMMIT",
    "I3_FULL_CURRENT_MACRO",
    "I3R_FULL_MACRO_INTERRUPTIBLE",
)
R_ARMS = frozenset({"I2R_W50_INTERRUPTIBLE_COMMIT", "I3R_FULL_MACRO_INTERRUPTIBLE"})
R_ENABLED = True
LEGACY_OF_R_ARM = {
    "I2R_W50_INTERRUPTIBLE_COMMIT": "I2_W50_CURRENT_COMMIT",
    "I3R_FULL_MACRO_INTERRUPTIBLE": "I3_FULL_CURRENT_MACRO",
}

# Sealed outputs of the original run. Read-only under every record id; the
# mechanism amendment re-verifies their digests after it finishes.
SEALED_OUTPUTS = (
    SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_RESULT.json",
    SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_TRACE_MANIFEST.json",
    SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_CONTRACT_RESULT.json",
    SD / "action_interface_scale_diagnostic_agent_tick_rows.csv",
    SD / "action_interface_scale_diagnostic_episode_rows.csv",
)
SEALED_EPISODE_ROWS = SD / "action_interface_scale_diagnostic_episode_rows.csv"
ARMS = (
    "I0_CONTINUOUS_ORACLE",
    "I1_W50_NO_COMMIT",
    "I2_W50_CURRENT_COMMIT",
    "I3_FULL_CURRENT_MACRO",
)
STYLES = {
    "GUARD": "BLUE_ONE_DEFENDER_V2",
    "BREACH": "BLUE_BOTH_ATTACK_V2",
}
SCALES = (2, 4)
HORIZON = 240
N_SEEDS = 16
ANGLE_THRESHOLD_DEG = 15.0
TARGET_JUMP_THRESHOLD_CELLS = 1.0
FLOAT_TOL = 1e-6
CONTRACT_TOL = 1e-5
MATERIAL_BURDEN = 0.10
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 7

COMPONENTS = {
    "C_spatial": ("I1_W50_NO_COMMIT", "I0_CONTINUOUS_ORACLE"),
    "C_commit": ("I2_W50_CURRENT_COMMIT", "I1_W50_NO_COMMIT"),
    "C_macro_residual": ("I3_FULL_CURRENT_MACRO", "I2_W50_CURRENT_COMMIT"),
}

FORBIDDEN_FIELDS = {
    "blue_score", "red_score", "win", "reward", "return", "value", "advantage"
}

TICK_FIELDS = [
    "scale", "style", "seed", "tick", "agent", "arm", "trace_hash",
    "alive", "carrying", "tagged", "eligible_motion", "environment_reset",
    "raw_target_x", "raw_target_y", "effective_target_x", "effective_target_y",
    "position_x", "position_y", "velocity_x", "velocity_y",
    "velocity_error_norm", "position_error_cells", "target_endpoint_error_cells",
    "direction_error_degrees", "switch_request", "blocked_switch",
    "stale_target_tick", "decision_boundary", "adapted_category",
    "committed_macro", "committed_target_idx", "commit_ticks_left",
    "forced_home_carrying", "tagged_redirect", "speed_cap_cps",
]

EPISODE_FIELDS = [
    "scale", "style", "seed", "arm", "trace_hash", "n_ticks", "n_agents",
    "eligible_agent_ticks", "E_v", "E_x_rmse_cells", "E_x_p90_cells",
    "mean_target_endpoint_error_cells", "mean_direction_error_degrees",
    "switch_requests", "blocked_switch_fraction", "mean_decision_lag_ticks",
    "p95_decision_lag_ticks", "censored_switch_fraction", "stale_target_fraction",
    "semantic_execution_fraction", "forced_home_carrying_fraction",
    "tagged_redirect_fraction", "macro_early_end_fraction",
    "r_fire_count", "r_eligible_ticks", "interruption_rate", "mean_commit_run_length",
]

PROTECTED_ARTIFACTS = [
    SD / "ACTION_INTERFACE_DECOMP_SPEC.json",
    SD / "ACTION_INTERFACE_DECOMP_SEALED_READING.json",
    SD / "PROJECTION_DIVERGENCE_TRACE_RESULT.json",
    SD / "REPAIRED_GO_TO_H1_PROJECTED_SPEC.json",
    SD / "PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json",
    SD / "PYQUATICUS_PORT_CONTRACT_RESULT.json",
]


def bind_mechanism_amendment_outputs() -> None:
    """Redirect every output path to the mechanism-amendment record id.

    refuse_if_result_exists=true on the original record, so the conformance
    rerun must not write to any sealed path.
    """
    global CONTRACT_RESULT, TRACE_MANIFEST, TICK_ROWS, EPISODE_ROWS, RESULT, LOCK
    global LABEL, PROTECTED_ARTIFACTS

    stem = MECHANISM_AMENDMENT_ID
    CONTRACT_RESULT = SD / f"{stem}_CONTRACT_RESULT.json"
    TRACE_MANIFEST = SD / f"{stem}_TRACE_MANIFEST.json"
    TICK_ROWS = SD / "action_interface_scale_diagnostic_mechanism_amendment_v1_agent_tick_rows.csv"
    EPISODE_ROWS = SD / "action_interface_scale_diagnostic_mechanism_amendment_v1_episode_rows.csv"
    RESULT = SD / f"{stem}_RESULT.json"
    LOCK = SD / f"{stem}.run.lock"
    LABEL = stem
    PROTECTED_ARTIFACTS = list(PROTECTED_ARTIFACTS) + [p for p in SEALED_OUTPUTS]


def bind_intervention_outputs(*, repair_enabled: bool) -> None:
    """Redirect outputs to COMMITMENT_INTERRUPTIBILITY_INTERVENTION_V1 and add the R arms.

    ``repair_enabled=False`` runs the same six-arm matrix with R never firing, which
    is the executable form of the disabled-equals-legacy contract.
    """
    global CONTRACT_RESULT, TRACE_MANIFEST, TICK_ROWS, EPISODE_ROWS, RESULT, LOCK
    global LABEL, PROTECTED_ARTIFACTS, ARMS, R_ENABLED

    suffix = "" if repair_enabled else "_REPAIR_OFF"
    stem = INTERVENTION_ID + suffix
    lower = stem.lower()
    CONTRACT_RESULT = SD / f"{stem}_CONTRACT_RESULT.json"
    TRACE_MANIFEST = SD / f"{stem}_TRACE_MANIFEST.json"
    TICK_ROWS = SD / f"{lower}_agent_tick_rows.csv"
    EPISODE_ROWS = SD / f"{lower}_episode_rows.csv"
    RESULT = SD / f"{stem}_RESULT.json"
    LOCK = SD / f"{stem}.run.lock"
    LABEL = stem
    ARMS = INTERVENTION_ARMS
    R_ENABLED = repair_enabled
    PROTECTED_ARTIFACTS = list(PROTECTED_ARTIFACTS) + [p for p in SEALED_OUTPUTS] + [
        SD / f"{MECHANISM_AMENDMENT_ID}_RESULT.json",
        SD / "action_interface_scale_diagnostic_mechanism_amendment_v1_episode_rows.csv",
    ]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros(2, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _angle_deg(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lu, ru = _unit(lhs), _unit(rhs)
    if float(np.linalg.norm(lu)) <= 1e-8 or float(np.linalg.norm(ru)) <= 1e-8:
        return 0.0
    cosine = float(np.clip(np.dot(lu, ru), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def nearest_waypoint(target: np.ndarray, waypoints: np.ndarray) -> tuple[int, np.ndarray]:
    """Deterministic Euclidean-nearest W50 projection with lowest-index ties."""
    distances = np.sum((waypoints - np.asarray(target)[None, :]) ** 2, axis=1)
    index = int(np.argmin(distances))
    return index, np.asarray(waypoints[index], dtype=np.float64).copy()


def i2_boundary_schedule(horizon: int, commit_ticks: int = 4) -> list[int]:
    left = 0
    boundaries: list[int] = []
    for tick in range(int(horizon)):
        if left <= 0:
            boundaries.append(tick)
            left = int(commit_ticks)
        left = max(0, left - 1)
    return boundaries


def _bootstrap(values: Iterable[float]) -> dict[str, Any]:
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "lcb95": None, "ucb95": None, "n": 0}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, x.size, size=(BOOTSTRAP_SAMPLES, x.size))
    draws = x[indices].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "mean": float(x.mean()),
        "lcb95": float(lo),
        "ucb95": float(hi),
        "n": int(x.size),
    }


def _base_cfg(scale: int, seed: int) -> GPUFieldConfig:
    import experiments.strategic_demand_searcher as sds

    return GPUFieldConfig(
        n_envs=1,
        max_blue_agents=int(scale),
        max_red_agents=int(scale),
        map_set="train",
        map_layout="map_a",
        max_decision_steps=HORIZON,
        score_limit=1_000_000,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=int(seed),
        obstacle_obs_channel=True,
        tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True,
        train_domain_randomization=False,
        **sds.RULESET,
    )


def _resolved_pole_b(scale: int):
    from experiments.pole_attestation import resolve_pole_genome

    candidate = str(POLE_B_CANDIDATE) if int(scale) == 4 else None
    return resolve_pole_genome("B", int(scale), candidate)


def _pre_env_pole_evidence(scale: int, genome) -> dict[str, Any]:
    from experiments.pole_attestation import (
        assert_resolved_matches_certification,
        governing_certification,
    )

    if int(scale) == 4:
        verdict, cert_path = governing_certification(4)
        attestation = assert_resolved_matches_certification(
            "B", 4, cert_path, genome, is_smoke=False
        )
        return {
            "label": "CERTIFIED_POLE_B3_3",
            "governing_verdict": verdict,
            "governing_record": cert_path.name,
            "attestation": attestation,
        }

    guard = _json(REGRESSION_GUARD)
    if guard.get("VERDICT") != "PASS" or guard.get("passed") != "7/7":
        raise RuntimeError("2v2 size-normalized pole regression guard is not PASS 7/7")
    if str(genome.base_opponent) != "OP7" or dict(genome.overlay or {}):
        raise RuntimeError(
            f"2v2 historical control drifted: base={genome.base_opponent!r} "
            f"overlay={dict(genome.overlay or {})!r}"
        )
    return {
        "label": "HISTORICAL_CANONICAL_POLE_B_CONTROL",
        "formal_rule14_certification": "UNAVAILABLE_MISSING_RECORD",
        "regression_guard": REGRESSION_GUARD.name,
        "regression_guard_sha256": _sha256(REGRESSION_GUARD),
        "genome_id": str(genome.genome_id),
        "base_opponent": str(genome.base_opponent),
        "overlay": dict(genome.overlay or {}),
    }


def _make_source_env(scale: int, style: str, seed: int):
    from experiments.pole_attestation import attest_live_pole
    from experiments.train_specialist_scale import assert_live_pole_matches_team_size

    genome = _resolved_pole_b(scale)
    evidence = _pre_env_pole_evidence(scale, genome)
    env = GPUCTFVecEnv(_base_cfg(scale, seed))
    core = env.core
    opponent = str(genome.base_opponent)
    env.env_method("set_phase", opponent)
    env.env_method("set_next_opponent", "SCRIPTED", opponent)
    apply_genome_to_core(core, genome)
    core.blue_scripted = True
    core.set_blue_style(style)
    env.reset()
    apply_genome_to_core(core, genome)
    core.drain_tag_events()

    if int(scale) == 4:
        evidence["live"] = attest_live_pole(
            core,
            "B",
            4,
            evidence["attestation"],
            context=LABEL,
            is_smoke=False,
        )
    else:
        live = assert_live_pole_matches_team_size(env, "B", 2)
        resolved = core._bt_resolved_profile_tensors()
        observed = {
            "min_alive_for_defender": int(resolved["min_alive_for_defender"].flatten()[0]),
            "defender_zone_frac": float(resolved["defender_zone_frac"].flatten()[0]),
            "threat_radius": float(resolved["threat_radius"].flatten()[0]),
        }
        expected = {
            "min_alive_for_defender": 2,
            "defender_zone_frac": 0.05,
            "threat_radius": 12.0,
        }
        if any(abs(observed[key] - value) > 1e-6 for key, value in expected.items()):
            env.close()
            raise RuntimeError(f"2v2 live Pole-B profile drift: {observed} != {expected}")
        evidence["live"] = {**live, "observed_profile": observed}
    return env, core, evidence


def capture_source_intent(core: BatchedCTFCore) -> np.ndarray:
    """Capture the native scripted target before adapter/commit/macro overrides."""
    target_x, target_y = core._get_scripted_targets("blue")
    return torch.stack([target_x[0], target_y[0]], dim=1).detach().cpu().numpy().astype(np.float64)


@dataclass
class ArmState:
    name: str
    core: BatchedCTFCore
    x: torch.Tensor
    y: torch.Tensor
    heading: torch.Tensor
    speed: torch.Tensor
    i2_target: np.ndarray | None = None
    i2_target_indices: np.ndarray | None = None
    i2_ticks_left: np.ndarray | None = None
    prior_tagged: np.ndarray | None = None
    prior_alive: np.ndarray | None = None
    pending_switch: np.ndarray | None = None
    lag_samples: list[list[int]] = field(default_factory=list)
    censored_by_agent: list[int] = field(default_factory=list)
    early_end_count: int = 0
    commit_count: int = 0
    # INTERRUPT_CONDITION_R bookkeeping (R arms only; inert elsewhere)
    r_streak: np.ndarray | None = None
    r_fire_count: int = 0
    r_eligible_ticks: int = 0
    run_lengths: list[int] = field(default_factory=list)
    current_run: np.ndarray | None = None


def _new_arm_state(name: str, scale: int, seed: int, source_core: BatchedCTFCore) -> ArmState:
    core = BatchedCTFCore(_base_cfg(scale, seed))
    x = source_core.blue_x.detach().clone()
    y = source_core.blue_y.detach().clone()
    heading = source_core.blue_heading.detach().clone()
    speed = source_core.blue_speed.detach().clone()
    return ArmState(
        name=name,
        core=core,
        x=x,
        y=y,
        heading=heading,
        speed=speed,
        i2_target=np.zeros((scale, 2), dtype=np.float64),
        i2_target_indices=np.zeros(scale, dtype=np.int64),
        i2_ticks_left=np.zeros(scale, dtype=np.int32),
        prior_tagged=source_core.blue_tagged[0].detach().cpu().numpy().astype(bool),
        prior_alive=source_core.blue_alive[0].detach().cpu().numpy().astype(bool),
        pending_switch=np.full(scale, -1, dtype=np.int32),
        lag_samples=[[] for _ in range(scale)],
        censored_by_agent=[0 for _ in range(scale)],
        r_streak=np.zeros(scale, dtype=np.int32),
        current_run=np.zeros(scale, dtype=np.int32),
    )


def _copy_context(
    state: ArmState,
    source_core: BatchedCTFCore,
    *,
    red_after_x: torch.Tensor | None = None,
    red_after_y: torch.Tensor | None = None,
) -> None:
    core = state.core
    for attr in (
        "blue_flag_home", "blue_flag_pos", "red_flag_home", "red_flag_pos",
        "blue_carrying", "blue_tagged", "blue_alive", "red_carrying",
        "red_tagged", "red_alive", "rt_blue_speed_scale", "rt_current_strength_cps",
        "rt_drift_sigma_cells",
    ):
        setattr(core, attr, getattr(source_core, attr).detach().clone())
    core.blue_x = state.x
    core.blue_y = state.y
    core.blue_heading = state.heading
    core.blue_speed = state.speed
    core.red_x = (red_after_x if red_after_x is not None else source_core.red_x).detach().clone()
    core.red_y = (red_after_y if red_after_y is not None else source_core.red_y).detach().clone()
    core.red_heading = source_core.red_heading.detach().clone()
    core.red_speed = source_core.red_speed.detach().clone()


def _sync_environment_resets(
    state: ArmState,
    source_core: BatchedCTFCore,
) -> np.ndarray:
    tagged = source_core.blue_tagged[0].detach().cpu().numpy().astype(bool)
    alive = source_core.blue_alive[0].detach().cpu().numpy().astype(bool)
    prior_tagged = np.asarray(state.prior_tagged, dtype=bool)
    prior_alive = np.asarray(state.prior_alive, dtype=bool)
    reset = (tagged & ~prior_tagged) | (alive & ~prior_alive)
    if bool(reset.any()):
        mask = torch.as_tensor(reset, dtype=torch.bool, device=state.x.device).reshape(1, -1)
        state.x = torch.where(mask, source_core.blue_x, state.x)
        state.y = torch.where(mask, source_core.blue_y, state.y)
        state.heading = torch.where(mask, source_core.blue_heading, state.heading)
        state.speed = torch.where(mask, source_core.blue_speed, state.speed)
    state.prior_tagged = tagged
    state.prior_alive = alive
    return reset


def _common_target_postprocess(
    state: ArmState,
    targets: np.ndarray,
    *,
    apply_tagged_redirect: bool = True,
) -> np.ndarray:
    core = state.core
    target = torch.as_tensor(targets, dtype=torch.float32, device=core.device).reshape(1, -1, 2)
    tx, ty = target[..., 0], target[..., 1]
    if apply_tagged_redirect:
        tx, ty, _, _ = core._redirect_tagged_to_home(tx, ty, tx.clone(), ty.clone())
    tx, ty = core._route_targets_around_obstacles(state.x, state.y, tx, ty)
    return torch.stack([tx[0], ty[0]], dim=1).detach().cpu().numpy().astype(np.float64)


def _full_macro_request(
    state: ArmState,
    raw_targets: np.ndarray,
    waypoints: np.ndarray,
) -> tuple[np.ndarray, list[Adapted]]:
    actions = np.zeros((raw_targets.shape[0], 2), dtype=np.int64)
    adapted: list[Adapted] = []
    for agent, target in enumerate(raw_targets):
        item = adapt(state.core, float(target[0]), float(target[1]), agent, waypoints)
        adapted.append(item)
        actions[agent, 0] = int(item.macro) if item.macro is not None else int(MacroAction.GO_TO)
        actions[agent, 1] = int(item.target_idx) if item.target_idx is not None else 0
    return actions, adapted


def _interrupt_condition_r(
    state: ArmState,
    requested_action: np.ndarray,
    committed_action: np.ndarray,
    desired_vector: np.ndarray,
    committed_vector: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    """INTERRUPT_CONDITION_R at the frozen operating point (theta=45, k=2).

    R1 committed action identity differs from the action the adapter requests now,
    R2 divergence >= theta, R3 both sustained for k consecutive ticks. Evaluated
    only mid-commitment; a natural boundary is not an interruption.
    """
    fires = np.zeros(active.shape[0], dtype=bool)
    if not R_ENABLED:
        return fires
    for i in range(active.shape[0]):
        if not active[i]:
            state.r_streak[i] = 0
            continue
        state.r_eligible_ticks += 1
        r1 = tuple(int(v) for v in requested_action[i]) != tuple(int(v) for v in committed_action[i])
        r2 = _angle_deg(desired_vector[i], committed_vector[i]) >= R_THETA_DEG
        state.r_streak[i] = state.r_streak[i] + 1 if (r1 and r2) else 0
        if state.r_streak[i] >= R_K_TICKS:
            fires[i] = True
            state.r_streak[i] = 0
            state.r_fire_count += 1
    return fires


def _track_run_lengths(state: ArmState, boundary: np.ndarray, active: np.ndarray) -> None:
    for i in range(boundary.shape[0]):
        if boundary[i]:
            if state.current_run[i] > 0:
                state.run_lengths.append(int(state.current_run[i]))
            state.current_run[i] = 1 if active[i] else 0
        elif active[i]:
            state.current_run[i] += 1


def _arm_positions(state: ArmState) -> np.ndarray:
    return torch.stack([state.x[0], state.y[0]], dim=1).detach().cpu().numpy().astype(np.float64)


def _resolve_arm_targets(
    state: ArmState,
    raw_targets: np.ndarray,
    waypoints: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = raw_targets.shape[0]
    if state.name == "I1_W50_NO_COMMIT":
        indices = np.zeros(n, dtype=np.int64)
        targets = np.zeros_like(raw_targets, dtype=np.float64)
        for i in range(n):
            indices[i], targets[i] = nearest_waypoint(raw_targets[i], waypoints)
        return _common_target_postprocess(state, targets), {
            "boundary": np.ones(n, dtype=bool),
            "category": ["WAYPOINT"] * n,
            "macro": np.full(n, int(MacroAction.GO_TO), dtype=np.int64),
            "target_idx": indices,
            "ticks_left": np.zeros(n, dtype=np.int32),
            "forced_home": np.zeros(n, dtype=bool),
            "tagged_redirect": state.core.blue_tagged[0].detach().cpu().numpy().astype(bool),
        }

    if state.name in ("I2_W50_CURRENT_COMMIT", "I2R_W50_INTERRUPTIBLE_COMMIT"):
        assert state.i2_ticks_left is not None
        assert state.i2_target is not None
        assert state.i2_target_indices is not None
        alive_now = state.core.blue_alive[0].detach().cpu().numpy().astype(bool)
        if state.name == "I2R_W50_INTERRUPTIBLE_COMMIT":
            arm_pos = _arm_positions(state)
            requested = np.zeros((n, 2), dtype=np.int64)
            committed = np.zeros((n, 2), dtype=np.int64)
            desired_vector = np.zeros((n, 2), dtype=np.float64)
            committed_vector = np.zeros((n, 2), dtype=np.float64)
            for i in range(n):
                index_now, _ = nearest_waypoint(raw_targets[i], waypoints)
                requested[i] = (int(MacroAction.GO_TO), int(index_now))
                committed[i] = (int(MacroAction.GO_TO), int(state.i2_target_indices[i]))
                desired_vector[i] = raw_targets[i] - arm_pos[i]
                committed_vector[i] = state.i2_target[i] - arm_pos[i]
            fires = _interrupt_condition_r(
                state, requested, committed, desired_vector, committed_vector,
                (state.i2_ticks_left > 0) & alive_now,
            )
            state.i2_ticks_left = np.where(fires, 0, state.i2_ticks_left)
        boundary = state.i2_ticks_left <= 0
        _track_run_lengths(state, boundary, alive_now)
        for i in range(n):
            if boundary[i]:
                index, target = nearest_waypoint(raw_targets[i], waypoints)
                state.i2_target_indices[i] = index
                state.i2_target[i] = target
                state.i2_ticks_left[i] = int(state.core.cfg.macro_commit_go_to_ticks)
        targets = _common_target_postprocess(state, state.i2_target)
        during = state.i2_ticks_left.copy()
        state.i2_ticks_left = np.maximum(0, state.i2_ticks_left - 1)
        return targets, {
            "boundary": boundary,
            "category": ["WAYPOINT"] * n,
            "macro": np.full(n, int(MacroAction.GO_TO), dtype=np.int64),
            "target_idx": state.i2_target_indices.copy(),
            "ticks_left": during,
            "forced_home": np.zeros(n, dtype=bool),
            "tagged_redirect": state.core.blue_tagged[0].detach().cpu().numpy().astype(bool),
        }

    if state.name not in ("I3_FULL_CURRENT_MACRO", "I3R_FULL_MACRO_INTERRUPTIBLE"):
        raise ValueError(f"unsupported shadow arm {state.name}")

    actions, desired = _full_macro_request(state, raw_targets, waypoints)
    if state.name == "I3R_FULL_MACRO_INTERRUPTIBLE":
        core = state.core
        alive_now = core.blue_alive[0].detach().cpu().numpy().astype(bool)
        ticks_left_now = core.blue_commit_ticks_left[0].detach().cpu().numpy().astype(np.int32)
        committed = np.stack([
            core.blue_commit_macro[0].detach().cpu().numpy().astype(np.int64),
            core.blue_commit_target[0].detach().cpu().numpy().astype(np.int64),
        ], axis=1)
        held_tx, held_ty = core._build_targets_from_action(
            core.blue_commit_macro, core.blue_commit_target, side="blue",
        )
        held = torch.stack([held_tx[0], held_ty[0]], dim=1).detach().cpu().numpy().astype(np.float64)
        arm_pos = _arm_positions(state)
        fires = _interrupt_condition_r(
            state, actions, committed,
            raw_targets - arm_pos, held - arm_pos,
            (ticks_left_now > 0) & alive_now,
        )
        if fires.any():
            mask = torch.as_tensor(fires, dtype=torch.bool, device=core.device).reshape(1, -1)
            core.blue_commit_ticks_left = torch.where(
                mask, torch.zeros_like(core.blue_commit_ticks_left), core.blue_commit_ticks_left,
            )

    boundary = (state.core.blue_commit_ticks_left[0] <= 0).detach().cpu().numpy().astype(bool)
    _track_run_lengths(
        state, boundary, state.core.blue_alive[0].detach().cpu().numpy().astype(bool),
    )
    macro, target_idx = state.core._advance_blue_macros(
        torch.as_tensor(actions.reshape(-1), dtype=torch.int64, device=state.core.device)
    )
    tx, ty = state.core._build_targets_from_action(macro, target_idx, side="blue")
    tx, ty, _, _ = state.core._redirect_tagged_to_home(tx, ty, tx.clone(), ty.clone())
    tx, ty = state.core._route_targets_around_obstacles(state.x, state.y, tx, ty)
    target = torch.stack([tx[0], ty[0]], dim=1).detach().cpu().numpy().astype(np.float64)
    committed_macro = macro[0].detach().cpu().numpy().astype(np.int64)
    state.commit_count += int(boundary.sum())
    return target, {
        "boundary": boundary,
        "category": [item.category for item in desired],
        "macro": committed_macro,
        "target_idx": target_idx[0].detach().cpu().numpy().astype(np.int64),
        "ticks_left": state.core.blue_commit_ticks_left[0].detach().cpu().numpy().astype(np.int32),
        "forced_home": state.core.blue_carrying[0].detach().cpu().numpy().astype(bool),
        "tagged_redirect": state.core.blue_tagged[0].detach().cpu().numpy().astype(bool),
    }


def _integrate_shadow(
    state: ArmState,
    targets: np.ndarray,
    speed_cap: torch.Tensor,
    red_before_x: torch.Tensor,
    red_before_y: torch.Tensor,
    red_after_x: torch.Tensor,
    red_after_y: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    core = state.core
    previous_x = state.x.clone()
    previous_y = state.y.clone()
    previous_red_x = red_before_x.clone()
    previous_red_y = red_before_y.clone()
    target = torch.as_tensor(targets, dtype=torch.float32, device=core.device).reshape(1, -1, 2)
    x, y, heading, speed, _oob, _yaw = core._integrate_side(
        state.x,
        state.y,
        state.heading,
        state.speed,
        core.blue_alive,
        target[..., 0],
        target[..., 1],
        speed_cap=speed_cap,
    )
    x, y, speed, _ = core._revert_obstacle_hits(
        previous_x, previous_y, x, y, speed, core.blue_alive
    )
    core.blue_x, core.blue_y = x, y
    core.blue_heading, core.blue_speed = heading, speed
    core.red_x, core.red_y = red_after_x.clone(), red_after_y.clone()
    pre_guard_x, pre_guard_y = core.blue_x.clone(), core.blue_y.clone()
    core._apply_avoid_collision_guard(
        previous_x, previous_y, previous_red_x, previous_red_y
    )
    core.blue_x, core.blue_y, core.blue_speed, _ = core._revert_obstacle_hits(
        pre_guard_x,
        pre_guard_y,
        core.blue_x,
        core.blue_y,
        core.blue_speed,
        core.blue_alive,
    )
    state.x, state.y = core.blue_x, core.blue_y
    state.heading, state.speed = core.blue_heading, core.blue_speed
    before = torch.stack([previous_x[0], previous_y[0]], dim=1).detach().cpu().numpy()
    after = torch.stack([state.x[0], state.y[0]], dim=1).detach().cpu().numpy()
    return before.astype(np.float64), after.astype(np.float64)


def _finish_i3_commit(
    state: ArmState,
    source_carry_before: np.ndarray,
    source_carry_after: np.ndarray,
) -> np.ndarray:
    core = state.core
    macro = core.blue_commit_macro
    decoded = core._decode_targets(core.blue_commit_target)
    distance = torch.sqrt(
        (state.x - decoded[..., 0]) ** 2 + (state.y - decoded[..., 1]) ** 2 + 1e-8
    )
    success = (macro == int(MacroAction.GO_TO)) & (
        distance <= float(core.cfg.macro_arrival_radius_cells)
    )
    grabbed = torch.as_tensor(
        (~source_carry_before) & source_carry_after,
        dtype=torch.bool,
        device=core.device,
    ).reshape(1, -1)
    captured = torch.as_tensor(
        source_carry_before & (~source_carry_after),
        dtype=torch.bool,
        device=core.device,
    ).reshape(1, -1)
    success = success | ((macro == int(MacroAction.GET_FLAG)) & grabbed)
    success = success | ((macro == int(MacroAction.GO_HOME)) & captured)
    core.blue_commit_success = core.blue_commit_success | success
    core.blue_commit_ticks_left = torch.clamp(core.blue_commit_ticks_left - 1, min=0)
    ended = (
        core.blue_commit_success
        | (core.blue_commit_ticks_left <= 0)
        | (~core.blue_alive)
        | core.blue_tagged
    )
    ended_early = ended & (core.blue_commit_ticks_left > 0)
    state.early_end_count += int(ended_early.sum().item())
    core.blue_commit_ticks_left = torch.where(
        ended, torch.zeros_like(core.blue_commit_ticks_left), core.blue_commit_ticks_left
    )
    core.blue_commit_success = torch.where(
        ended, torch.zeros_like(core.blue_commit_success), core.blue_commit_success
    )
    return ended_early[0].detach().cpu().numpy().astype(bool)


def _source_effective_targets(core: BatchedCTFCore, raw: np.ndarray) -> np.ndarray:
    target = torch.as_tensor(raw, dtype=torch.float32, device=core.device).reshape(1, -1, 2)
    tx, ty = target[..., 0], target[..., 1]
    tx, ty, _, _ = core._redirect_tagged_to_home(tx, ty, tx.clone(), ty.clone())
    tx, ty = core._route_targets_around_obstacles(core.blue_x, core.blue_y, tx, ty)
    return torch.stack([tx[0], ty[0]], dim=1).detach().cpu().numpy().astype(np.float64)


def _mean_agent(values: list[list[float]], *, rms: bool = False) -> float:
    per_agent = []
    for agent_values in values:
        if not agent_values:
            continue
        arr = np.asarray(agent_values, dtype=np.float64)
        per_agent.append(float(np.sqrt(np.mean(arr * arr))) if rms else float(np.mean(arr)))
    return float(np.mean(per_agent)) if per_agent else 0.0


def _p_agent(values: list[list[float]], percentile: float) -> float:
    per_agent = [float(np.percentile(v, percentile)) for v in values if v]
    return float(np.mean(per_agent)) if per_agent else 0.0


def _aggregate_episode(
    rows: list[dict[str, Any]],
    state: ArmState | None,
    *,
    scale: int,
    style: str,
    seed: int,
    arm: str,
    trace_hash: str,
) -> dict[str, Any]:
    by_agent: dict[int, list[dict[str, Any]]] = {i: [] for i in range(scale)}
    for row in rows:
        by_agent[int(row["agent"])].append(row)

    def eligible_values(field: str) -> list[list[float]]:
        return [
            [float(row[field]) for row in by_agent[i] if int(row["eligible_motion"]) == 1]
            for i in range(scale)
        ]

    switch_by_agent = [
        [row for row in by_agent[i] if int(row["switch_request"]) == 1]
        for i in range(scale)
    ]
    blocked_fraction = _mean_agent([
        [float(row["blocked_switch"]) for row in requests]
        for requests in switch_by_agent
    ])
    stale_fraction = _mean_agent([
        [float(row["stale_target_tick"]) for row in by_agent[i]]
        for i in range(scale)
    ])
    semantic_fraction = _mean_agent([
        [float(str(row["adapted_category"]) not in ("", "WAYPOINT", "ORACLE")) for row in by_agent[i]]
        for i in range(scale)
    ])
    forced_fraction = _mean_agent([
        [float(row["forced_home_carrying"]) for row in by_agent[i]]
        for i in range(scale)
    ])
    tagged_fraction = _mean_agent([
        [float(row["tagged_redirect"]) for row in by_agent[i]]
        for i in range(scale)
    ])

    lag_samples = state.lag_samples if state is not None else [[0] * len(v) for v in switch_by_agent]
    lag_mean = _mean_agent([[float(v) for v in values] for values in lag_samples])
    lag_p95 = _p_agent([[float(v) for v in values] for values in lag_samples], 95.0)
    censored = state.censored_by_agent if state is not None else [0] * scale
    requests_count = [len(values) for values in switch_by_agent]
    censored_fraction = float(np.mean([
        censored[i] / max(1, requests_count[i]) for i in range(scale)
    ]))
    commit_count = state.commit_count if state is not None else 0
    early_fraction = (
        float(state.early_end_count / max(1, commit_count)) if state is not None else 0.0
    )
    return {
        "scale": f"{scale}v{scale}",
        "style": style,
        "seed": seed,
        "arm": arm,
        "trace_hash": trace_hash,
        "n_ticks": HORIZON,
        "n_agents": scale,
        "eligible_agent_ticks": sum(
            int(row["eligible_motion"]) for values in by_agent.values() for row in values
        ),
        "E_v": _mean_agent(eligible_values("velocity_error_norm")),
        "E_x_rmse_cells": _mean_agent(eligible_values("position_error_cells"), rms=True),
        "E_x_p90_cells": _p_agent(eligible_values("position_error_cells"), 90.0),
        "mean_target_endpoint_error_cells": _mean_agent(
            eligible_values("target_endpoint_error_cells")
        ),
        "mean_direction_error_degrees": _mean_agent(
            eligible_values("direction_error_degrees")
        ),
        "switch_requests": sum(requests_count),
        "blocked_switch_fraction": blocked_fraction,
        "mean_decision_lag_ticks": lag_mean,
        "p95_decision_lag_ticks": lag_p95,
        "censored_switch_fraction": censored_fraction,
        "stale_target_fraction": stale_fraction,
        "semantic_execution_fraction": semantic_fraction,
        "forced_home_carrying_fraction": forced_fraction,
        "tagged_redirect_fraction": tagged_fraction,
        "macro_early_end_fraction": early_fraction,
        "r_fire_count": (state.r_fire_count if state is not None else 0),
        "r_eligible_ticks": (state.r_eligible_ticks if state is not None else 0),
        "interruption_rate": (
            float(state.r_fire_count / state.r_eligible_ticks)
            if state is not None and state.r_eligible_ticks > 0 else 0.0
        ),
        "mean_commit_run_length": (
            float(np.mean(state.run_lengths))
            if state is not None and state.run_lengths else 0.0
        ),
    }


def _update_switch_tracking(
    state: ArmState,
    tick: int,
    switch_request: np.ndarray,
    direction_error: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    assert state.pending_switch is not None
    blocked = np.zeros_like(switch_request, dtype=bool)
    for i in range(switch_request.size):
        if switch_request[i]:
            if state.pending_switch[i] >= 0:
                state.censored_by_agent[i] += 1
            state.pending_switch[i] = tick
        adopted = direction_error[i] <= ANGLE_THRESHOLD_DEG
        if switch_request[i] and not adopted:
            blocked[i] = True
        if state.pending_switch[i] >= 0 and adopted:
            state.lag_samples[i].append(int(tick - state.pending_switch[i]))
            state.pending_switch[i] = -1
    return blocked, state.pending_switch >= 0


def run_source_trace(scale: int, style_label: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    style = STYLES[style_label]
    env, core, pole_evidence = _make_source_env(scale, style, seed)
    waypoints = core._macro_targets.detach().cpu().numpy().astype(np.float64)
    arms = {
        name: _new_arm_state(name, scale, seed, core)
        for name in ARMS
        if name != "I0_CONTINUOUS_ORACLE"
    }
    tick_rows: list[dict[str, Any]] = []
    arm_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ARMS}
    trace_hasher = hashlib.sha256()
    previous_raw: np.ndarray | None = None
    previous_source_position: np.ndarray | None = None

    try:
        for tick in range(HORIZON):
            raw = capture_source_intent(core)
            source_effective = _source_effective_targets(core, raw)
            source_before = torch.stack([core.blue_x[0], core.blue_y[0]], dim=1).detach().cpu().numpy().astype(np.float64)
            source_heading_before = core.blue_heading[0].detach().cpu().numpy().astype(np.float64)
            source_speed_before = core.blue_speed[0].detach().cpu().numpy().astype(np.float64)
            source_alive_before = core.blue_alive[0].detach().cpu().numpy().astype(bool)
            source_tagged_before = core.blue_tagged[0].detach().cpu().numpy().astype(bool)
            source_carry_before = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
            red_before_x = core.red_x.detach().clone()
            red_before_y = core.red_y.detach().clone()
            speed_cap = (
                torch.full_like(core.blue_speed, float(core.cfg.max_speed_cps))
                * core.rt_blue_speed_scale.reshape(core.B, 1).expand_as(core.blue_speed)
            )
            trace_hasher.update(np.asarray(raw, dtype=np.float32).tobytes())
            trace_hasher.update(source_alive_before.tobytes())
            trace_hasher.update(source_tagged_before.tobytes())
            trace_hasher.update(source_carry_before.tobytes())

            env.step_async(env.action_space.sample() * 0)
            _unused_obs, _unused_learning_signal, done, _unused_info = env.step_wait()
            source_after = torch.stack([core.blue_x[0], core.blue_y[0]], dim=1).detach().cpu().numpy().astype(np.float64)
            source_alive_after = core.blue_alive[0].detach().cpu().numpy().astype(bool)
            source_tagged_after = core.blue_tagged[0].detach().cpu().numpy().astype(bool)
            source_carry_after = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
            red_after_x = core.red_x.detach().clone()
            red_after_y = core.red_y.detach().clone()

            if bool(np.asarray(done).any()) and tick != HORIZON - 1:
                raise RuntimeError(
                    f"source trace terminated at tick {tick + 1}, expected {HORIZON}; "
                    "outcome-blind exposure contract failed"
                )

            source_velocity = (source_after - source_before) / float(core.dt)
            desired_vectors = raw - source_before
            if previous_raw is None or previous_source_position is None:
                switch_request = np.ones(scale, dtype=bool)
            else:
                previous_vectors = previous_raw - previous_source_position
                switch_request = np.asarray([
                    _angle_deg(previous_vectors[i], desired_vectors[i]) >= ANGLE_THRESHOLD_DEG
                    or float(np.linalg.norm(raw[i] - previous_raw[i])) >= TARGET_JUMP_THRESHOLD_CELLS
                    for i in range(scale)
                ], dtype=bool)
            previous_raw = raw.copy()
            previous_source_position = source_before.copy()
            environment_reset = (
                (~source_alive_before & source_alive_after)
                | (~source_tagged_before & source_tagged_after)
            )
            eligible = (
                source_alive_before
                & source_alive_after
                & (~source_tagged_before)
                & (~source_tagged_after)
                & (~environment_reset)
            )

            trace_hash_live = trace_hasher.hexdigest()
            i0_direction_error = np.asarray([
                _angle_deg(desired_vectors[i], source_effective[i] - source_before[i])
                for i in range(scale)
            ])
            for i in range(scale):
                row = {
                    "scale": f"{scale}v{scale}", "style": style_label, "seed": seed,
                    "tick": tick, "agent": i, "arm": "I0_CONTINUOUS_ORACLE",
                    "trace_hash": trace_hash_live,
                    "alive": int(source_alive_before[i]),
                    "carrying": int(source_carry_before[i]),
                    "tagged": int(source_tagged_before[i]),
                    "eligible_motion": int(eligible[i]),
                    "environment_reset": int(environment_reset[i]),
                    "raw_target_x": raw[i, 0], "raw_target_y": raw[i, 1],
                    "effective_target_x": source_effective[i, 0],
                    "effective_target_y": source_effective[i, 1],
                    "position_x": source_after[i, 0], "position_y": source_after[i, 1],
                    "velocity_x": source_velocity[i, 0], "velocity_y": source_velocity[i, 1],
                    "velocity_error_norm": 0.0, "position_error_cells": 0.0,
                    "target_endpoint_error_cells": float(np.linalg.norm(source_effective[i] - raw[i])),
                    "direction_error_degrees": i0_direction_error[i],
                    "switch_request": int(switch_request[i]), "blocked_switch": 0,
                    "stale_target_tick": 0, "decision_boundary": 1,
                    "adapted_category": "ORACLE", "committed_macro": -1,
                    "committed_target_idx": -1, "commit_ticks_left": 0,
                    "forced_home_carrying": 0,
                    "tagged_redirect": int(source_tagged_before[i]),
                    "speed_cap_cps": float(speed_cap[0, i]),
                }
                arm_rows["I0_CONTINUOUS_ORACLE"].append(row)
                tick_rows.append(row)

            for arm_name, state in arms.items():
                _sync_environment_resets(state, core)
                _copy_context(
                    state,
                    core,
                    red_after_x=red_after_x,
                    red_after_y=red_after_y,
                )
                arm_before = torch.stack([state.x[0], state.y[0]], dim=1).detach().cpu().numpy().astype(np.float64)
                targets, meta = _resolve_arm_targets(state, raw, waypoints)
                before, after = _integrate_shadow(
                    state,
                    targets,
                    speed_cap,
                    red_before_x,
                    red_before_y,
                    red_after_x,
                    red_after_y,
                )
                ended_early = np.zeros(scale, dtype=bool)
                if arm_name in ("I3_FULL_CURRENT_MACRO", "I3R_FULL_MACRO_INTERRUPTIBLE"):
                    ended_early = _finish_i3_commit(
                        state, source_carry_before, source_carry_after
                    )
                arm_velocity = (after - before) / float(core.dt)
                velocity_error = np.linalg.norm(arm_velocity - source_velocity, axis=1) / float(core.cfg.max_speed_cps)
                position_error = np.linalg.norm(after - source_after, axis=1)
                target_error = np.linalg.norm(targets - raw, axis=1)
                direction_error = np.asarray([
                    _angle_deg(desired_vectors[i], targets[i] - arm_before[i])
                    for i in range(scale)
                ])
                blocked, _pending = _update_switch_tracking(
                    state, tick, switch_request, direction_error
                )
                stale = (~np.asarray(meta["boundary"], dtype=bool)) & (
                    direction_error >= ANGLE_THRESHOLD_DEG
                )
                for i in range(scale):
                    row = {
                        "scale": f"{scale}v{scale}", "style": style_label, "seed": seed,
                        "tick": tick, "agent": i, "arm": arm_name,
                        "trace_hash": trace_hash_live,
                        "alive": int(source_alive_before[i]),
                        "carrying": int(source_carry_before[i]),
                        "tagged": int(source_tagged_before[i]),
                        "eligible_motion": int(eligible[i]),
                        "environment_reset": int(environment_reset[i]),
                        "raw_target_x": raw[i, 0], "raw_target_y": raw[i, 1],
                        "effective_target_x": targets[i, 0],
                        "effective_target_y": targets[i, 1],
                        "position_x": after[i, 0], "position_y": after[i, 1],
                        "velocity_x": arm_velocity[i, 0], "velocity_y": arm_velocity[i, 1],
                        "velocity_error_norm": velocity_error[i],
                        "position_error_cells": position_error[i],
                        "target_endpoint_error_cells": target_error[i],
                        "direction_error_degrees": direction_error[i],
                        "switch_request": int(switch_request[i]),
                        "blocked_switch": int(blocked[i]),
                        "stale_target_tick": int(stale[i]),
                        "decision_boundary": int(meta["boundary"][i]),
                        "adapted_category": meta["category"][i],
                        "committed_macro": int(meta["macro"][i]),
                        "committed_target_idx": int(meta["target_idx"][i]),
                        "commit_ticks_left": int(meta["ticks_left"][i]),
                        "forced_home_carrying": int(meta["forced_home"][i]),
                        "tagged_redirect": int(meta["tagged_redirect"][i]),
                        "speed_cap_cps": float(speed_cap[0, i]),
                    }
                    if ended_early[i]:
                        row["commit_ticks_left"] = -abs(int(row["commit_ticks_left"]))
                    arm_rows[arm_name].append(row)
                    tick_rows.append(row)

        for state in arms.values():
            assert state.pending_switch is not None
            for i in range(scale):
                if state.pending_switch[i] >= 0:
                    state.censored_by_agent[i] += 1

        trace_hash = trace_hasher.hexdigest()
        for row in tick_rows:
            row["trace_hash"] = trace_hash
        episodes = [
            _aggregate_episode(
                arm_rows[arm],
                arms.get(arm),
                scale=scale,
                style=style_label,
                seed=seed,
                arm=arm,
                trace_hash=trace_hash,
            )
            for arm in ARMS
        ]
        manifest = {
            "scale": f"{scale}v{scale}",
            "style": style_label,
            "seed": seed,
            "ticks": HORIZON,
            "agents": scale,
            "trace_hash": trace_hash,
            "tick_rows": len(tick_rows),
            "expected_tick_rows": HORIZON * scale * len(ARMS),
            "pole_evidence": pole_evidence,
        }
        if manifest["tick_rows"] != manifest["expected_tick_rows"]:
            raise RuntimeError(f"G8 row count failure: {manifest}")
        return tick_rows, episodes, manifest
    finally:
        env.close()


def _synthetic_scale_error(n: int) -> float:
    core = BatchedCTFCore(_base_cfg(n, 99_900_001))
    waypoints = core._macro_targets.detach().cpu().numpy().astype(np.float64)
    target = np.tile(np.asarray([[13.2, 7.4]], dtype=np.float64), (n, 1))
    origin = np.tile(np.asarray([[2.0, 2.0]], dtype=np.float64), (n, 1))
    projected = np.stack([nearest_waypoint(value, waypoints)[1] for value in target])
    return float(np.mean([
        _angle_deg(target[i] - origin[i], projected[i] - origin[i]) for i in range(n)
    ]))


PARITY_SEEDS = (19500001, 19500002, 19500003, 19500004)
# I0 must stay in the iteration set: run_source_trace unconditionally appends an
# I0 row regardless of ARMS. It carries no comparison weight (see PARITY_COMPARE_ARMS).
PARITY_ARMS = ("I0_CONTINUOUS_ORACLE", "I1_W50_NO_COMMIT", "I2_W50_CURRENT_COMMIT", "I3_FULL_CURRENT_MACRO")
PARITY_COMPARE_ARMS = ("I1_W50_NO_COMMIT", "I2_W50_CURRENT_COMMIT", "I3_FULL_CURRENT_MACRO")


def _g9_legacy_arm_parity() -> dict[str, Any]:
    """G9: prove adding the R arms did not silently alter I1/I2/I3.

    Reuses a subset of the block already SPENT by ACTION_INTERFACE_SCALE_DIAGNOSTIC
    (19500001-19500016), the same precedent the mechanism amendment itself used to
    verify legacy reproduction. This is a code-parity self-test at contract time,
    not new statistical evidence: no registry entry is read or written, and none of
    these rows enter the intervention's own episode_rows.csv.

    Added after discovering that comparing against the amendment's rows by seed
    failed to find any matching key (the intervention necessarily draws a fresh,
    non-overlapping seed block per Rule 9), so the in-run legacy-equivalence check
    was vacuously true (n_compared=0) rather than actually verifying anything.
    """
    if not AMENDMENT_EPISODE_ROWS.is_file():
        return {"pass": False, "reason": f"reference missing: {AMENDMENT_EPISODE_ROWS}"}

    reference: dict[tuple[str, str, int, str], dict[str, str]] = {}
    with AMENDMENT_EPISODE_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) in PARITY_SEEDS and row["arm"] in PARITY_COMPARE_ARMS:
                reference[(row["scale"], row["style"], int(row["seed"]), row["arm"])] = row

    expected = len(PARITY_SEEDS) * len(SCALES) * len(STYLES) * len(PARITY_COMPARE_ARMS)
    if len(reference) != expected:
        return {
            "pass": False,
            "reason": f"reference incomplete: found {len(reference)}, expected {expected}",
        }

    global ARMS
    saved_arms = ARMS
    ARMS = PARITY_ARMS
    try:
        worst: dict[str, float] = {name: 0.0 for name in LEGACY_EQUIVALENCE_FIELDS}
        compared = 0
        for scale in SCALES:
            for style in STYLES:
                for seed in PARITY_SEEDS:
                    _tick_rows, episode_rows, _manifest = run_source_trace(scale, style, seed)
                    for row in episode_rows:
                        if row["arm"] not in PARITY_COMPARE_ARMS:
                            continue
                        key = (row["scale"], row["style"], row["seed"], row["arm"])
                        ref = reference[key]
                        compared += 1
                        for field_name in LEGACY_EQUIVALENCE_FIELDS:
                            worst[field_name] = max(
                                worst[field_name],
                                abs(float(row[field_name]) - float(ref[field_name])),
                            )
    finally:
        ARMS = saved_arms

    max_delta = max(worst.values()) if worst else 0.0
    return {
        "pass": bool(compared == expected and max_delta == 0.0),
        "n_compared": compared,
        "n_expected": expected,
        "max_abs_delta_by_field": worst,
        "max_abs_delta": max_delta,
        "reference_seeds": list(PARITY_SEEDS),
        "reference": AMENDMENT_EPISODE_ROWS.name,
        "note": (
            "Reuses ACTION_INTERFACE_SCALE_DIAGNOSTIC's already-spent seeds "
            "19500001-19500004 for a deterministic code-parity check only. "
            "Not new statistical evidence; no registry interaction."
        ),
    }


def run_contracts() -> dict[str, Any]:
    for path in (SPEC, AMENDMENT, REGRESSION_GUARD):
        if not path.is_file():
            raise RuntimeError(f"required frozen artifact missing: {path}")
    spec = _json(SPEC)
    amendment = _json(AMENDMENT)
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise RuntimeError("diagnostic spec is not frozen")
    if not str(amendment.get("status", "")).startswith("FROZEN"):
        raise RuntimeError("integrity amendment is not frozen")

    protected = {}
    g0_pass = True
    for path in PROTECTED_ARTIFACTS:
        exists = path.is_file()
        protected[path.name] = _sha256(path) if exists else None
        g0_pass = g0_pass and exists

    core = BatchedCTFCore(_base_cfg(2, 99_900_001))
    core.blue_scripted = True
    core.set_blue_style(STYLES["GUARD"])
    raw = capture_source_intent(core)
    g1_pass = raw.shape == (2, 2) and np.isfinite(raw).all()

    # G2: exact production integration identity on an isolated deterministic fixture.
    x = torch.tensor([[2.0, 4.0]], dtype=torch.float32)
    y = torch.tensor([[2.0, 14.0]], dtype=torch.float32)
    heading = torch.zeros_like(x)
    speed = torch.zeros_like(x)
    alive = torch.ones_like(x, dtype=torch.bool)
    tx = torch.tensor([[12.0, 12.0]], dtype=torch.float32)
    ty = torch.tensor([[2.0, 14.0]], dtype=torch.float32)
    cap = torch.full_like(x, float(core.cfg.max_speed_cps))
    direct = core._integrate_side(x, y, heading, speed, alive, tx, ty, speed_cap=cap)[:4]
    replay = core._integrate_side(x.clone(), y.clone(), heading.clone(), speed.clone(), alive, tx, ty, speed_cap=cap)[:4]
    g2_error = max(float(torch.max(torch.abs(a - b))) for a, b in zip(direct, replay))
    g2_pass = g2_error <= CONTRACT_TOL

    w = core._macro_targets.detach().cpu().numpy().astype(np.float64)
    probe = np.asarray([13.2, 7.4])
    index, snapped = nearest_waypoint(probe, w)
    expected_index = int(np.argmin(np.sum((w - probe[None, :]) ** 2, axis=1)))
    g3_pass = index == expected_index and np.array_equal(snapped, w[expected_index])

    schedule = i2_boundary_schedule(12, int(core.cfg.macro_commit_go_to_ticks))
    g4_pass = schedule == [0, 4, 8]

    macros = torch.tensor([[int(m) for m in MacroAction]], dtype=torch.int64)
    targets = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    target_core = BatchedCTFCore(
        GPUFieldConfig(
            n_envs=1, max_blue_agents=5, max_red_agents=5,
            map_set="train", map_layout="map_a", device="cpu", seed=99_900_002,
        )
    )
    tx1, ty1 = target_core._build_targets_from_action(macros, targets, side="blue")
    tx2, ty2 = target_core._build_targets_from_action(macros.clone(), targets.clone(), side="blue")
    ticks = target_core._macro_commit_ticks(macros)
    g5_pass = (
        torch.equal(tx1, tx2)
        and torch.equal(ty1, ty2)
        and ticks[0].tolist() == [4, 3, 4, 2, 4]
    )

    n2_error = _synthetic_scale_error(2)
    n4_error = _synthetic_scale_error(4)
    g6_pass = abs(n2_error - n4_error) <= FLOAT_TOL

    g9_result = _g9_legacy_arm_parity() if LABEL.startswith(INTERVENTION_ID) else None

    fields = set(TICK_FIELDS) | set(EPISODE_FIELDS)
    g7_pass = not bool(fields & FORBIDDEN_FIELDS)
    synthetic_hash = hashlib.sha256(b"synthetic-four-tick-trace").hexdigest()
    synthetic_counts = {arm: 4 * 2 for arm in ARMS}
    g8_pass = len(set(synthetic_counts.values())) == 1 and len(synthetic_hash) == 64

    gates = {
        "G0_PRIOR_ARTIFACT_PROTECTION": {"pass": g0_pass, "sha256": protected},
        "G1_SOURCE_INTENT": {"pass": g1_pass, "shape": list(raw.shape)},
        "G2_I0_FIDELITY": {"pass": g2_pass, "max_abs_error": g2_error},
        "G3_I1_ISOLATION": {"pass": g3_pass, "probe_index": index},
        "G4_I2_ISOLATION": {"pass": g4_pass, "boundaries": schedule},
        "G5_I3_PRODUCTION_PARITY": {"pass": g5_pass, "macro_ticks": ticks[0].tolist()},
        "G6_SCALE_NEUTRAL_INTEGRITY": {
            "pass": g6_pass, "N2_error": n2_error, "N4_error": n4_error,
        },
        "G7_OUTCOME_BLIND_SCHEMA": {
            "pass": g7_pass,
            "forbidden_intersection": sorted(fields & FORBIDDEN_FIELDS),
        },
        "G8_TRACE_COMPLETENESS": {
            "pass": g8_pass,
            "synthetic_counts": synthetic_counts,
            "synthetic_trace_hash": synthetic_hash,
            "live_recheck_required": True,
        },
    }
    if g9_result is not None:
        gates["G9_LEGACY_ARM_PARITY"] = g9_result
    overall = all(bool(value["pass"]) for value in gates.values())
    implements = [SPEC.name, AMENDMENT.name]
    if LABEL == MECHANISM_AMENDMENT_ID:
        implements.append(MECHANISM_AMENDMENT.name)
    return {
        "record_id": f"{LABEL}_CONTRACT_RESULT",
        "status": "FROZEN_CONTRACT_RESULT",
        "utc": _now(),
        "implements": implements,
        "classification": "DIAGNOSTIC",
        "gates": gates,
        "overall_pass": overall,
        "source_episodes_unlocked": overall,
        "production_interface_modified": False,
        "claim_boundary": "Contracts validate execution paths only. They contain no team outcome evidence.",
    }


def _append_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _allocate_seeds() -> tuple[list[int], dict[str, Any]]:
    import experiments.seed_registry as registry

    doc = registry.load()
    existing = next(
        (item for item in doc["blocks"] if item["experiment_id"] == EXPERIMENT_ID),
        None,
    )
    if existing is None:
        lo = registry.next_free(N_SEEDS, "exploratory")
        existing = registry.allocate(
            EXPERIMENT_ID,
            lo,
            lo + N_SEEDS - 1,
            "exploratory",
            "Outcome-blind I0-I3 action-interface scale diagnostic",
            spec=str(SPEC.relative_to(ROOT)),
            status="RESERVED",
        )
    if (
        existing["seed_class"] != "exploratory"
        or int(existing["n"]) != N_SEEDS
        or existing["status"] not in ("RESERVED", "SPENT")
    ):
        raise RuntimeError(f"Rule-9 allocation mismatch: {existing}")
    seeds = list(range(int(existing["lo"]), int(existing["hi"]) + 1))
    return seeds, existing


AMENDMENT_EPISODE_ROWS = SD / "action_interface_scale_diagnostic_mechanism_amendment_v1_episode_rows.csv"

LEGACY_EQUIVALENCE_FIELDS = (
    "E_v", "E_x_rmse_cells", "blocked_switch_fraction", "stale_target_fraction",
    "mean_decision_lag_ticks", "p95_decision_lag_ticks", "switch_requests",
    "semantic_execution_fraction", "forced_home_carrying_fraction",
)


def _legacy_equivalence(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """G_L1: the shared arms must be untouched by adding the R arms.

    Compares I1/I2/I3 in this run against the mechanism-amendment run, which
    predates every intervention edit. Any drift means the refactor changed a
    legacy arm and the manipulation contrast is not interpretable.
    """
    if not AMENDMENT_EPISODE_ROWS.is_file():
        return {"status": "AMENDMENT_ROWS_MISSING", "comparable": False}

    reference: dict[tuple[str, str, int, str], dict[str, str]] = {}
    with AMENDMENT_EPISODE_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference[(row["scale"], row["style"], int(row["seed"]), row["arm"])] = row

    worst: dict[str, float] = {field_name: 0.0 for field_name in LEGACY_EQUIVALENCE_FIELDS}
    compared = 0
    for row in episodes:
        arm = row["arm"]
        if arm not in ("I1_W50_NO_COMMIT", "I2_W50_CURRENT_COMMIT", "I3_FULL_CURRENT_MACRO"):
            continue
        key = (row["scale"], row["style"], int(row["seed"]), arm)
        ref = reference.get(key)
        if ref is None:
            continue
        compared += 1
        for field_name in LEGACY_EQUIVALENCE_FIELDS:
            if field_name not in ref:
                continue
            delta = abs(float(row[field_name]) - float(ref[field_name]))
            worst[field_name] = max(worst[field_name], delta)

    if compared == 0:
        # The intervention necessarily draws a seed block disjoint from the
        # amendment's (Rule 9), so this per-seed lookup is EXPECTED to miss and
        # must not be reported as a pass: a vacuous comparison is not a check.
        # Real legacy-arm parity is established by G9_LEGACY_ARM_PARITY in
        # run_contracts(), which reuses a shared seed subset for exactly this.
        return {
            "status": "NOT_COMPARABLE_NO_SEED_OVERLAP",
            "comparable": False,
            "n_compared": 0,
            "pass": None,
            "reading": (
                "This run's seeds do not overlap the amendment's, so no row-level "
                "comparison is possible here by construction. This field is not the "
                "legacy-arm-parity check -- see CONTRACT_RESULT.gates.G9_LEGACY_ARM_PARITY, "
                "which verifies parity on a shared seed subset."
            ),
            "reference": AMENDMENT_EPISODE_ROWS.name,
        }

    max_delta = max(worst.values()) if worst else 0.0
    return {
        "status": "LEGACY_ARMS_UNCHANGED" if max_delta == 0.0 else "LEGACY_ARM_DRIFT",
        "comparable": True,
        "n_compared": compared,
        "max_abs_delta_by_field": worst,
        "max_abs_delta": max_delta,
        "pass": bool(max_delta == 0.0),
        "reference": AMENDMENT_EPISODE_ROWS.name,
    }


def _repair_off_equivalence(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """G_L1 second half: with R disabled, I2R must equal I2 and I3R must equal I3."""
    table = {
        (row["scale"], row["style"], int(row["seed"]), row["arm"]): row
        for row in episodes
    }
    worst = 0.0
    offenders: list[str] = []
    compared = 0
    for (scale, style, seed, arm), row in table.items():
        if arm not in LEGACY_OF_R_ARM:
            continue
        legacy = table.get((scale, style, seed, LEGACY_OF_R_ARM[arm]))
        if legacy is None:
            continue
        compared += 1
        for field_name in LEGACY_EQUIVALENCE_FIELDS:
            delta = abs(float(row[field_name]) - float(legacy[field_name]))
            if delta > 0.0:
                offenders.append(f"{scale}|{style}|{seed}|{arm}|{field_name}|{delta:g}")
            worst = max(worst, delta)
    return {
        "status": "REPAIR_OFF_EQUALS_LEGACY" if worst == 0.0 else "REPAIR_OFF_DIVERGES",
        "n_compared": compared,
        "max_abs_delta": worst,
        "offenders_sample": offenders[:8],
        "pass": bool(worst == 0.0),
    }


def _observational_equivalence(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """S2: the added instrumentation must not move trajectories.

    Digest equality is not asserted: the sealed emitter hashes a different
    payload definition, so only E_v agreement is meaningful.
    """
    if not SEALED_EPISODE_ROWS.is_file():
        return {"status": "SEALED_ROWS_MISSING", "comparable": False}

    sealed: dict[tuple[str, str, int, str], float] = {}
    with SEALED_EPISODE_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["scale"], row["style"], int(row["seed"]), row["arm"])
            sealed[key] = float(row["E_v"])

    diffs: list[float] = []
    missing: list[str] = []
    for row in episodes:
        key = (row["scale"], row["style"], int(row["seed"]), row["arm"])
        if key not in sealed:
            missing.append("|".join(str(part) for part in key))
            continue
        diffs.append(abs(float(row["E_v"]) - sealed[key]))

    if not diffs:
        return {"status": "NO_OVERLAP", "comparable": False, "missing_keys": missing[:8]}

    max_abs = float(np.max(diffs))
    equivalent = bool(max_abs <= 1e-9 and not missing)
    return {
        "status": "OBSERVATIONAL_EQUIVALENT" if equivalent else "IMPLEMENTATION_DIVERGENCE",
        "comparable": True,
        "n_compared": len(diffs),
        "n_missing_in_sealed": len(missing),
        "missing_keys_sample": missing[:8],
        "max_abs_E_v_difference": max_abs,
        "mean_abs_E_v_difference": float(np.mean(diffs)),
        "tolerance": 1e-9,
        "reading": (
            "Sealed E_v independently reproduced; instrumentation is observational."
            if equivalent else
            "Sealed and conformant emitters disagree on E_v. Reportable implementation "
            "divergence: neither run is invalidated by this check alone."
        ),
    }


_SEMANTIC_EXEMPT = ("", "WAYPOINT", "ORACLE")


def _macro_residual_sign_probe(tick_rows_path: Path) -> dict[str, Any]:
    """S1: establish from execution traces whether C_macro_residual < 0 is real.

    Partitions paired I2/I3 agent-ticks by I3 execution state. Concentration in
    named production events means the sign is attributed; concentration in the
    identical-target partition means an accounting defect.
    """
    paired: dict[tuple[str, str, int, int, int], dict[str, dict[str, Any]]] = {}
    with tick_rows_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arm = row["arm"]
            if arm not in ("I2_W50_CURRENT_COMMIT", "I3_FULL_CURRENT_MACRO"):
                continue
            if int(row["eligible_motion"]) != 1:
                continue
            key = (
                row["scale"], row["style"], int(row["seed"]),
                int(row["tick"]), int(row["agent"]),
            )
            paired.setdefault(key, {})[arm] = {
                "err": float(row["velocity_error_norm"]),
                "tx": float(row["effective_target_x"]),
                "ty": float(row["effective_target_y"]),
                "forced_home": int(row["forced_home_carrying"]),
                "tagged": int(row["tagged_redirect"]),
                "category": str(row["adapted_category"]),
            }

    buckets: dict[str, dict[str, list[float]]] = {}
    for (scale, style, _seed, _tick, _agent), arms in paired.items():
        i2 = arms.get("I2_W50_CURRENT_COMMIT")
        i3 = arms.get("I3_FULL_CURRENT_MACRO")
        if i2 is None or i3 is None:
            continue
        same_target = (
            abs(i3["tx"] - i2["tx"]) <= 1e-6 and abs(i3["ty"] - i2["ty"]) <= 1e-6
        )
        if same_target:
            part = "a_identical_effective_target"
        elif i3["forced_home"]:
            part = "b_forced_home_carrying"
        elif i3["tagged"]:
            part = "c_tagged_redirect"
        elif i3["category"] not in _SEMANTIC_EXEMPT:
            part = "d_semantic_macro_target"
        else:
            part = "e_commit_horizon_residual"
        cell = buckets.setdefault(f"{scale}_{style}", {})
        cell.setdefault(f"{part}|i2", []).append(i2["err"])
        cell.setdefault(f"{part}|i3", []).append(i3["err"])

    report: dict[str, Any] = {}
    for cell, series in buckets.items():
        parts = sorted({name.split("|")[0] for name in series})
        total = sum(len(series.get(f"{p}|i3", [])) for p in parts)
        cell_report = {}
        for part in parts:
            i2_vals = series.get(f"{part}|i2", [])
            i3_vals = series.get(f"{part}|i3", [])
            if not i3_vals:
                continue
            delta = float(np.mean(i3_vals) - np.mean(i2_vals))
            share = len(i3_vals) / max(1, total)
            cell_report[part] = {
                "agent_ticks": len(i3_vals),
                "tick_share": share,
                "mean_E_v_tick_I2": float(np.mean(i2_vals)),
                "mean_E_v_tick_I3": float(np.mean(i3_vals)),
                "mean_delta_I3_minus_I2": delta,
                "contribution_to_C_macro_residual": share * delta,
            }
        report[cell] = cell_report

    primary = report.get("4v4_BREACH", {})
    negative_parts = {
        name: item["contribution_to_C_macro_residual"]
        for name, item in primary.items()
        if item["contribution_to_C_macro_residual"] < 0.0
    }
    total_negative = sum(negative_parts.values())
    identical_share = negative_parts.get("a_identical_effective_target", 0.0)
    named_event_share = sum(
        value for name, value in negative_parts.items()
        if name in ("b_forced_home_carrying", "c_tagged_redirect", "d_semantic_macro_target")
    )
    if not negative_parts:
        verdict = "NO_NEGATIVE_CONTRIBUTION_IN_PRIMARY_CELL"
    elif abs(identical_share) > abs(named_event_share):
        verdict = "SUSPECT_ACCOUNTING"
    else:
        verdict = "ATTRIBUTED_TO_NAMED_PRODUCTION_EVENTS"
    return {
        "partitions": report,
        "primary_cell": "4v4_BREACH",
        "negative_contribution_total": total_negative,
        "negative_from_identical_target_partition": identical_share,
        "negative_from_named_production_events": named_event_share,
        "VERDICT": verdict,
        "reading_rule": (
            "SUSPECT_ACCOUNTING means the two arms command the same effective target "
            "yet report different error, which is an arm or accounting defect rather "
            "than a physical compensation. This probe changes no threshold or decision."
        ),
    }


PRIMARY_RHO_METRICS = ("blocked_switch_fraction", "p95_decision_lag_ticks")
SECONDARY_RHO_METRICS = ("stale_target_fraction", "mean_decision_lag_ticks")
RHO_PRIMARY_MIN = 0.50
RHO_PRIMARY_LCB_MIN = 0.25
RHO_SECONDARY_MIN = 0.30
INTERRUPTION_RATE_CEILING = 0.40
RUN_LENGTH_FLOOR_RATIO = 0.70
RHO_STALE_CEILING = 0.95


def _analyze_intervention(
    episodes: list[dict[str, Any]], manifests: list[dict[str, Any]]
) -> dict[str, Any]:
    """Manipulation check per COMMITMENT_INTERRUPTIBILITY_INTERVENTION_V1_SPEC."""
    table = {
        (row["scale"], row["style"], int(row["seed"]), row["arm"]): row
        for row in episodes
    }
    seeds = sorted({int(row["seed"]) for row in episodes})

    def rho_series(scale: str, style: str, metric: str) -> list[float]:
        values = []
        for seed in seeds:
            legacy = float(table[(scale, style, seed, "I2_W50_CURRENT_COMMIT")][metric])
            floor = float(table[(scale, style, seed, "I1_W50_NO_COMMIT")][metric])
            repaired = float(table[(scale, style, seed, "I2R_W50_INTERRUPTIBLE_COMMIT")][metric])
            excess = legacy - floor
            if abs(excess) < 1e-12:
                continue
            values.append((legacy - repaired) / excess)
        return values

    cells: dict[str, Any] = {}
    for scale in SCALES:
        for style in STYLES:
            key = f"{scale}v{scale}_{style}"
            entry: dict[str, Any] = {"rho": {}, "raw": {}}
            for metric in PRIMARY_RHO_METRICS + SECONDARY_RHO_METRICS:
                entry["rho"][metric] = _bootstrap(rho_series(f"{scale}v{scale}", style, metric))
                entry["raw"][metric] = {
                    arm: float(np.mean([
                        float(table[(f"{scale}v{scale}", style, seed, arm)][metric])
                        for seed in seeds
                    ]))
                    for arm in ARMS if arm != "I0_CONTINUOUS_ORACLE"
                }
            entry["interruption_rate_I2R"] = float(np.mean([
                float(table[(f"{scale}v{scale}", style, seed, "I2R_W50_INTERRUPTIBLE_COMMIT")]["interruption_rate"])
                for seed in seeds
            ]))
            legacy_run = float(np.mean([
                float(table[(f"{scale}v{scale}", style, seed, "I2_W50_CURRENT_COMMIT")]["mean_commit_run_length"])
                for seed in seeds
            ]))
            repaired_run = float(np.mean([
                float(table[(f"{scale}v{scale}", style, seed, "I2R_W50_INTERRUPTIBLE_COMMIT")]["mean_commit_run_length"])
                for seed in seeds
            ]))
            entry["mean_commit_run_length"] = {"I2": legacy_run, "I2R": repaired_run}
            entry["run_length_ratio"] = (repaired_run / legacy_run) if legacy_run > 0 else 0.0
            cells[key] = entry

    primary = cells["4v4_BREACH"]
    gate_detail = {}
    primary_pass = True
    for metric in PRIMARY_RHO_METRICS:
        summary = primary["rho"][metric]
        ok = bool(
            summary["mean"] is not None
            and summary["mean"] >= RHO_PRIMARY_MIN
            and summary["lcb95"] is not None
            and summary["lcb95"] > RHO_PRIMARY_LCB_MIN
        )
        gate_detail[metric] = {
            "summary": summary, "point_min": RHO_PRIMARY_MIN,
            "lcb95_min": RHO_PRIMARY_LCB_MIN, "pass": ok,
        }
        primary_pass = primary_pass and ok

    secondary_detail = {}
    secondary_contradicts = False
    for metric in SECONDARY_RHO_METRICS:
        summary = primary["rho"][metric]
        ok = bool(
            summary["mean"] is not None
            and summary["mean"] >= RHO_SECONDARY_MIN
            and summary["lcb95"] is not None
            and summary["lcb95"] > 0.0
        )
        secondary_detail[metric] = {
            "summary": summary, "point_min": RHO_SECONDARY_MIN, "pass": ok,
        }
        secondary_contradicts = secondary_contradicts or not ok

    stale_rho = primary["rho"]["stale_target_fraction"]["mean"]
    guards = {
        "G_P1_non_firing_ticks_bit_identical": {
            "discharged_by": "construction plus the REPAIR_OFF run",
            "detail": (
                "R only zeroes the commit counter for firing agents; every other "
                "statement is the shared legacy code path, so a non-firing tick "
                "executes identical code. Trajectory-level comparison is impossible "
                "after the first interruption and is therefore not claimed."
            ),
            "pass": None,
        },
        "G_P2_interruption_rate_ceiling": {
            "value": primary["interruption_rate_I2R"],
            "ceiling": INTERRUPTION_RATE_CEILING,
            "pass": bool(primary["interruption_rate_I2R"] <= INTERRUPTION_RATE_CEILING),
        },
        "G_P3_run_length_floor": {
            "value": primary["run_length_ratio"],
            "floor": RUN_LENGTH_FLOOR_RATIO,
            "pass": bool(primary["run_length_ratio"] >= RUN_LENGTH_FLOOR_RATIO),
        },
        "G_P4_no_total_recovery": {
            "value": stale_rho,
            "ceiling": RHO_STALE_CEILING,
            "pass": bool(stale_rho is not None and stale_rho <= RHO_STALE_CEILING),
        },
    }
    guards_pass = all(item["pass"] for item in guards.values() if item["pass"] is not None)

    if not R_ENABLED:
        verdict = "LEGACY_EQUIVALENCE_RUN"
        nxt = (
            "Contract run only. R never fired, so every rho is 0 by construction and "
            "no manipulation verdict is expressed or implied."
        )
    elif not guards_pass and primary_pass:
        verdict = "MANIPULATION_DEGENERATE"
        nxt = "STOP. The intervention reproduced h=1 by another route and inherits that negative result."
    elif primary_pass and not secondary_contradicts:
        verdict = "MANIPULATION_CONFIRMED"
        nxt = "Freeze a separate PPO crossover spec. Confirmation is not specialization recovery."
    elif primary_pass and secondary_contradicts:
        verdict = "MANIPULATION_CONFIRMED_WITH_SECONDARY_CONTRADICTION"
        nxt = "Do not proceed on this alone. A secondary signal contradicts the primary gates; adjudicate before any training."
    else:
        verdict = "MANIPULATION_INSUFFICIENT"
        nxt = "STOP. Conditional interruption at theta=45, k=2 does not move the localized mechanism."

    return {
        "record_id": f"{LABEL}_RESULT",
        "utc": _now(),
        "implements": [INTERVENTION_SPEC.name, INTERVENTION_AMENDMENT.name, CONTRACT_RESULT.name],
        "classification": "OUTCOME_BLIND_MANIPULATION_CHECK",
        "outcome_blind": True,
        "repair_enabled": R_ENABLED,
        "operating_point": {"theta_degrees": R_THETA_DEG, "k_consecutive_ticks": R_K_TICKS},
        "n_seeds": len(seeds),
        "n_source_traces": len(manifests),
        "primary_cell": "4v4_BREACH",
        "PRIMARY_GATE": gate_detail,
        "SECONDARY": secondary_detail,
        "PERSISTENCE_GUARD": guards,
        "cells": cells,
        "VERDICT": verdict,
        "NEXT": nxt,
        "claim_boundary": (
            "Outcome-blind manipulation check on scripted traces. It shows only whether "
            "the localized commitment mechanism is movable. It is not a win-rate result, "
            "not a causal claim about 4v4 specialization, and not training authorization."
        ),
        "PPO": "OFF",
        "GPU": "OFF",
    }


def _analyze(episodes: list[dict[str, Any]], manifests: list[dict[str, Any]]) -> dict[str, Any]:
    table = {
        (row["scale"], row["style"], int(row["seed"]), row["arm"]): row
        for row in episodes
    }
    seeds = sorted({int(row["seed"]) for row in episodes})
    expected_keys = {
        (f"{scale}v{scale}", style, seed, arm)
        for scale in SCALES for style in STYLES for seed in seeds for arm in ARMS
    }
    missing = sorted(expected_keys - set(table))
    if missing:
        raise RuntimeError(f"G8 incomplete episode matrix: {missing[:8]}")

    component_rows: dict[str, dict[tuple[str, str, int], float]] = {}
    for component, (upper, lower) in COMPONENTS.items():
        values = {}
        for scale in SCALES:
            for style in STYLES:
                for seed in seeds:
                    key = (f"{scale}v{scale}", style, seed)
                    values[key] = float(table[(*key, upper)]["E_v"]) - float(table[(*key, lower)]["E_v"])
        component_rows[component] = values

    components: dict[str, Any] = {}
    candidates = []
    for component, values in component_rows.items():
        cells = {
            f"{scale}v{scale}_{style}": _bootstrap(
                values[(f"{scale}v{scale}", style, seed)] for seed in seeds
            )
            for scale in SCALES for style in STYLES
        }
        scale_interaction = []
        style_scale_interaction = []
        for seed in seeds:
            c4 = np.mean([values[("4v4", style, seed)] for style in STYLES])
            c2 = np.mean([values[("2v2", style, seed)] for style in STYLES])
            scale_interaction.append(float(c4 - c2))
            style_scale_interaction.append(float(
                (values[("4v4", "BREACH", seed)] - values[("4v4", "GUARD", seed)])
                - (values[("2v2", "BREACH", seed)] - values[("2v2", "GUARD", seed)])
            ))
        S = _bootstrap(scale_interaction)
        J = _bootstrap(style_scale_interaction)
        breach_4 = cells["4v4_BREACH"]["mean"]

        if component == "C_spatial":
            support_values = [
                float(table[("4v4", "BREACH", seed, "I1_W50_NO_COMMIT")]["mean_target_endpoint_error_cells"])
                - float(table[("4v4", "BREACH", seed, "I0_CONTINUOUS_ORACLE")]["mean_target_endpoint_error_cells"])
                for seed in seeds
            ]
            support = _bootstrap(support_values)
            support_pass = bool(support["mean"] is not None and support["mean"] > 1e-6)
            support_name = "W50_endpoint_error_increment"
        elif component == "C_commit":
            support_values = [
                float(table[("4v4", "BREACH", seed, "I2_W50_CURRENT_COMMIT")]["stale_target_fraction"])
                - float(table[("4v4", "BREACH", seed, "I1_W50_NO_COMMIT")]["stale_target_fraction"])
                for seed in seeds
            ]
            support = _bootstrap(support_values)
            support_pass = bool(support["mean"] is not None and support["mean"] > 0.0)
            support_name = "stale_target_fraction_increment"
        else:
            support_values = [
                float(table[("4v4", "BREACH", seed, "I3_FULL_CURRENT_MACRO")]["semantic_execution_fraction"])
                + float(table[("4v4", "BREACH", seed, "I3_FULL_CURRENT_MACRO")]["forced_home_carrying_fraction"])
                + float(table[("4v4", "BREACH", seed, "I3_FULL_CURRENT_MACRO")]["macro_early_end_fraction"])
                for seed in seeds
            ]
            support = _bootstrap(support_values)
            support_pass = bool(support["mean"] is not None and support["mean"] > 0.0)
            support_name = "named_macro_execution_event_burden"

        localized = bool(
            J["lcb95"] is not None
            and J["lcb95"] > 0.0
            and breach_4 is not None
            and breach_4 >= MATERIAL_BURDEN
            and support_pass
        )
        if localized:
            candidates.append(component)
        components[component] = {
            "cells": cells,
            "S_k_scale_interaction": S,
            "J_k_BREACH_by_scale_interaction": J,
            "material_4v4_BREACH_threshold": MATERIAL_BURDEN,
            "material_4v4_BREACH_pass": bool(breach_4 is not None and breach_4 >= MATERIAL_BURDEN),
            "mechanism_support": {
                "name": support_name,
                "summary": support,
                "pass": support_pass,
            },
            "B_RELEVANT_SCALE_BINDER_CANDIDATE": localized,
        }

    general_only = [
        name for name, item in components.items()
        if item["S_k_scale_interaction"]["lcb95"] is not None
        and item["S_k_scale_interaction"]["lcb95"] > 0.0
        and not item["B_RELEVANT_SCALE_BINDER_CANDIDATE"]
    ]
    if len(candidates) == 1:
        verdict = "ONE_B_RELEVANT_SCALE_BINDER_CANDIDATE"
        next_step = "Freeze a separate outcome-blind component intervention. No training is authorized."
    elif len(candidates) > 1:
        verdict = "MULTIPLE_B_RELEVANT_COMPONENTS_NO_WINNER"
        next_step = "Freeze separate component interventions or a factorial contract. Do not combine repairs or train."
    elif general_only:
        verdict = "GENERAL_INTERFACE_COST_ONLY_NO_B_LOCALIZATION"
        next_step = "Do not launch B training. General scale cost does not explain Pole-B specialization failure."
    else:
        verdict = "NO_SCALE_LOCALIZATION_ACTION_INTERFACE_DEMOTED"
        next_step = "Stop action-interface repair work for the current B failure and move upstream."

    protected_after = {
        path.name: (_sha256(path) if path.is_file() else None)
        for path in PROTECTED_ARTIFACTS
    }
    contract = _json(CONTRACT_RESULT)
    protected_before = contract["gates"]["G0_PRIOR_ARTIFACT_PROTECTION"]["sha256"]
    g0_post_pass = protected_after == protected_before
    g8_post_pass = all(
        int(item["tick_rows"]) == int(item["expected_tick_rows"])
        and int(item["ticks"]) == HORIZON
        for item in manifests
    )
    return {
        "record_id": "ACTION_INTERFACE_SCALE_DIAGNOSTIC_RESULT",
        "status": "FROZEN_RESULT" if g0_post_pass and g8_post_pass else "INTEGRITY_REQUIRED",
        "utc": _now(),
        "implements": [SPEC.name, AMENDMENT.name, CONTRACT_RESULT.name],
        "classification": "EXPLORATORY_MECHANISTIC_DIAGNOSTIC",
        "outcome_blind": True,
        "n_seeds": len(seeds),
        "n_source_traces": len(manifests),
        "n_episode_arm_rows": len(episodes),
        "components": components,
        "localized_candidates": candidates,
        "general_scale_costs_without_B_localization": general_only,
        "VERDICT": verdict if g0_post_pass and g8_post_pass else "INTEGRITY_REQUIRED",
        "NEXT": next_step if g0_post_pass and g8_post_pass else "Stop and audit artifact integrity.",
        "post_run_integrity": {
            "G0_prior_artifacts_unchanged": g0_post_pass,
            "G8_trace_matrix_complete": g8_post_pass,
            "protected_sha256_after": protected_after,
        },
        "claim_boundary": (
            "Scale-conditioned historical 2v2 canonical control versus certified 4v4 Pole B. "
            "Not a pure causal effect of N, not an outcome result, and not evidence that PPO can learn a repair."
        ),
        "production_interface_modified": False,
        "PPO": "OFF",
        "GPU": "OFF",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts-only", action="store_true")
    parser.add_argument(
        "--mechanism-amendment",
        action="store_true",
        help="Run the spec-conformance rerun under ACTION_INTERFACE_SCALE_DIAGNOSTIC_MECHANISM_AMENDMENT_V1.",
    )
    parser.add_argument(
        "--intervention",
        action="store_true",
        help="Run the COMMITMENT_INTERRUPTIBILITY_INTERVENTION_V1 manipulation check.",
    )
    parser.add_argument(
        "--repair-off",
        action="store_true",
        help="With --intervention: run the same matrix with R disabled (legacy-equivalence contract).",
    )
    args = parser.parse_args()

    global EXPERIMENT_ID

    amendment_mode = bool(args.mechanism_amendment)
    intervention_mode = bool(args.intervention)
    if amendment_mode and intervention_mode:
        raise SystemExit("REFUSING: --mechanism-amendment and --intervention are exclusive")

    if amendment_mode:
        if not MECHANISM_AMENDMENT.is_file():
            raise SystemExit(f"REFUSING: frozen amendment missing: {MECHANISM_AMENDMENT}")
        bind_mechanism_amendment_outputs()
        print(f"{LABEL}  [SPEC_CONFORMANCE_RERUN]  sealed outputs are read-only")

    if intervention_mode:
        for path in (INTERVENTION_SPEC, INTERVENTION_AMENDMENT):
            if not path.is_file():
                raise SystemExit(f"REFUSING: frozen record missing: {path}")
        bind_intervention_outputs(repair_enabled=not args.repair_off)
        EXPERIMENT_ID = INTERVENTION_ID
        print(
            f"{LABEL}  [MANIPULATION_CHECK]  repair="
            f"{'ON' if R_ENABLED else 'OFF'}  R(theta={R_THETA_DEG}, k={R_K_TICKS})"
        )

    sealed_before = {path.name: (_sha256(path) if path.is_file() else None) for path in SEALED_OUTPUTS}

    contract = run_contracts()
    _write_json(CONTRACT_RESULT, contract)
    print(f"{LABEL} contracts: {'PASS' if contract['overall_pass'] else 'FAIL'}")
    print(f"  -> {CONTRACT_RESULT}")
    if not contract["overall_pass"]:
        return 2
    if args.contracts_only:
        return 0

    for path in (RESULT, TRACE_MANIFEST, TICK_ROWS, EPISODE_ROWS):
        if path.exists():
            raise SystemExit(f"REFUSING: one-shot output already exists: {path}")

    seeds, allocation = _allocate_seeds()
    print(f"  seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)}) [{allocation['status']}]")
    print("  CPU only; outcome fields prohibited; production macros read-only")
    print(f"  source traces: {len(SCALES) * len(STYLES) * len(seeds)}\n", flush=True)

    episodes: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    jobs = [(scale, style, seed) for scale in SCALES for style in STYLES for seed in seeds]
    from experiments.tqdm_loop import set_postfix, tqdm_iter

    with RunLock(LOCK, run_id=LABEL):
        bar = tqdm_iter(jobs, desc=LABEL, unit="trace")
        for scale, style, seed in bar:
            set_postfix(bar, f"{scale}v{scale} {style} seed={seed}")
            rows, episode_rows, manifest = run_source_trace(scale, style, seed)
            _append_csv(TICK_ROWS, TICK_FIELDS, rows)
            _append_csv(EPISODE_ROWS, EPISODE_FIELDS, episode_rows)
            episodes.extend(episode_rows)
            manifests.append(manifest)

        manifest_doc = {
            "record_id": "ACTION_INTERFACE_SCALE_DIAGNOSTIC_TRACE_MANIFEST",
            "status": "COMPLETE",
            "utc": _now(),
            "seed_allocation": allocation,
            "traces": manifests,
            "tick_rows_sha256": _sha256(TICK_ROWS),
            "episode_rows_sha256": _sha256(EPISODE_ROWS),
            "runner_sha256": _sha256(Path(__file__)),
            "spec_sha256": _sha256(SPEC),
            "amendment_sha256": _sha256(AMENDMENT),
        }
        _write_json(TRACE_MANIFEST, manifest_doc)
        result = (
            _analyze_intervention(episodes, manifests) if intervention_mode
            else _analyze(episodes, manifests)
        )
        if intervention_mode:
            result["CONTRACTS"] = {
                "G_L1a_legacy_arms_unchanged": _legacy_equivalence(episodes),
                "G_L1b_repair_off_equals_legacy": (
                    _repair_off_equivalence(episodes) if not R_ENABLED else
                    {"status": "NOT_APPLICABLE_REPAIR_ON", "pass": None}
                ),
            }
            failed = [
                name for name, item in result["CONTRACTS"].items()
                if item.get("pass") is False  # None = informational/superseded, not gating
            ]
            if failed:
                result["VERDICT"] = "CONTRACT_FAILURE"
                result["NEXT"] = f"Stop. Contract(s) failed: {failed}. No verdict is issued."
        result["seed_allocation"] = allocation
        result["artifacts"] = {
            "trace_manifest": TRACE_MANIFEST.name,
            "agent_tick_rows": TICK_ROWS.name,
            "episode_rows": EPISODE_ROWS.name,
        }
        if amendment_mode:
            sealed_after = {
                path.name: (_sha256(path) if path.is_file() else None)
                for path in SEALED_OUTPUTS
            }
            result["record_id"] = MECHANISM_AMENDMENT_ID + "_RESULT"
            result["implements"] = [SPEC.name, AMENDMENT.name, MECHANISM_AMENDMENT.name, CONTRACT_RESULT.name]
            result["classification"] = "SPEC_CONFORMANCE_RERUN"
            result["SANITY_CHECKS"] = {
                "S2_OBSERVATIONAL_EQUIVALENCE": _observational_equivalence(episodes),
                "S1_MACRO_RESIDUAL_SIGN": _macro_residual_sign_probe(TICK_ROWS),
            }
            result["sealed_outputs_unchanged"] = bool(sealed_after == sealed_before)
            result["sealed_outputs_sha256"] = sealed_after
            if not result["sealed_outputs_unchanged"]:
                result["status"] = "INTEGRITY_REQUIRED"
                result["VERDICT"] = "INTEGRITY_REQUIRED"
                result["NEXT"] = "Sealed artifact digest changed during the rerun. Stop and audit."
        _write_json(RESULT, result)

        import experiments.seed_registry as registry
        registry.set_status(
            EXPERIMENT_ID,
            "SPENT",
            note=f"{LABEL} completed with verdict {result['VERDICT']}",
        )

    if intervention_mode:
        print(json.dumps({
            "verdict": result["VERDICT"],
            "repair_enabled": result["repair_enabled"],
            "primary_gate": {
                name: item["pass"] for name, item in result["PRIMARY_GATE"].items()
            },
            "persistence_guard": {
                name: item["pass"] for name, item in result["PERSISTENCE_GUARD"].items()
            },
            "contracts": {
                name: item.get("pass") for name, item in result["CONTRACTS"].items()
            },
            "result": str(RESULT),
        }, indent=2))
        return 0 if result["VERDICT"] != "CONTRACT_FAILURE" else 3

    print(json.dumps({
        "status": result["status"],
        "verdict": result["VERDICT"],
        "localized_candidates": result["localized_candidates"],
        "general_scale_costs_without_B_localization": result["general_scale_costs_without_B_localization"],
        "result": str(RESULT),
    }, indent=2))
    return 0 if result["status"] == "FROZEN_RESULT" else 3


if __name__ == "__main__":
    raise SystemExit(main())
