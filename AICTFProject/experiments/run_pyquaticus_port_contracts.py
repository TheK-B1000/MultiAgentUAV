"""Run CPU-only G1-G6 contracts for the Pyquaticus behavioral-role port.

This script never runs a team episode, opponent, PPO policy, or GPU kernel. It
uses controlled states and the local CPU dynamics helper to decide whether the
pinned TRUE semantics are correctly implemented and whether each semantic
branch is representable through the existing macro action interface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_env import BatchedCTFCore, GPUFieldConfig  # noqa: E402
from gpu_env.pyquaticus_port import (  # noqa: E402
    DEFENDER_RADIUS_CELLS,
    EASY_SPEED_FRACTION,
    UPSTREAM_COMMIT,
    UPSTREAM_LICENSE,
    UPSTREAM_MODE,
    UPSTREAM_REPOSITORY,
    UPSTREAM_SHA256,
    PortState,
    ProjectionCandidate,
    Role,
    SemanticBranch,
    direction_cosine,
    effective_projected_target,
    expected_away_direction,
    projection_candidates,
    roles_tensor,
    true_motion,
)
from macro_actions import MacroAction  # noqa: E402


SPEC = ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json"
DEFAULT_OUT = ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_PORT_CONTRACT_RESULT.json"
HORIZON = 16
DIRECTION_COSINE_MIN = 0.99
INTERACTION_RADIUS_CELLS = 2.5
FLOAT_TOL = 1e-5


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_upstream(upstream_root: Path) -> dict[str, Any]:
    root = upstream_root.resolve()
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-C",
        str(root),
        "rev-parse",
        "HEAD",
    ]
    observed_commit = subprocess.check_output(command, text=True).strip()
    files: dict[str, Any] = {}
    all_match = observed_commit == UPSTREAM_COMMIT
    for relative, expected in UPSTREAM_SHA256.items():
        path = root / relative
        observed = _sha256(path) if path.is_file() else None
        matches = observed == expected
        files[relative] = {
            "expected_sha256": expected,
            "observed_sha256": observed,
            "matches": matches,
        }
        all_match = all_match and matches
    return {
        "root": str(root),
        "expected_commit": UPSTREAM_COMMIT,
        "observed_commit": observed_commit,
        "commit_matches": observed_commit == UPSTREAM_COMMIT,
        "files": files,
        "all_match": all_match,
    }


def _state(
    *,
    positions: tuple[tuple[float, float], ...],
    headings: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    own_flag_pos: tuple[float, float] = (2.0, 10.0),
    own_flag_home: tuple[float, float] = (2.0, 10.0),
    enemy_flag_pos: tuple[float, float] = (17.0, 10.0),
    carrying: tuple[bool, ...] = (False, False, False, False),
    tagged: tuple[bool, ...] = (False, False, False, False),
    alive: tuple[bool, ...] = (True, True, True, True),
) -> PortState:
    return PortState(
        positions=torch.tensor(positions, dtype=torch.float32),
        headings=torch.tensor(headings, dtype=torch.float32),
        own_flag_pos=torch.tensor(own_flag_pos, dtype=torch.float32),
        own_flag_home=torch.tensor(own_flag_home, dtype=torch.float32),
        enemy_flag_pos=torch.tensor(enemy_flag_pos, dtype=torch.float32),
        carrying=torch.tensor(carrying, dtype=torch.bool),
        tagged=torch.tensor(tagged, dtype=torch.bool),
        alive=torch.tensor(alive, dtype=torch.bool),
    )


def _fixtures() -> dict[str, tuple[PortState, torch.Tensor, int]]:
    attack_positions = ((5.0, 10.0), (6.0, 5.0), (6.0, 15.0), (8.0, 10.0))
    return {
        "G1_ATTACK_NO_CARRIER": (
            _state(positions=attack_positions),
            roles_tensor([Role.ATTACK] * 4),
            0,
        ),
        "G2_ATTACK_SELF_CARRIER": (
            _state(positions=attack_positions, carrying=(True, False, False, False)),
            roles_tensor([Role.ATTACK] * 4),
            0,
        ),
        "G2_ATTACK_TEAMMATE_CARRIER_NONCARRIER": (
            _state(positions=attack_positions, carrying=(True, False, False, False)),
            roles_tensor([Role.ATTACK] * 4),
            1,
        ),
        "G3_DEFEND_OUTSIDE": (
            _state(positions=((9.0, 10.0), (9.0, 5.0), (9.0, 15.0), (8.0, 10.0))),
            roles_tensor([Role.DEFEND] * 4),
            0,
        ),
        "G4_DEFEND_INSIDE": (
            _state(positions=((4.0, 10.0), (2.0, 12.0), (2.0, 8.0), (5.5, 10.0))),
            roles_tensor([Role.DEFEND] * 4),
            0,
        ),
        "G5_TAGGED": (
            _state(positions=attack_positions, tagged=(True, False, False, False)),
            roles_tensor([Role.ATTACK, Role.DEFEND, Role.ATTACK, Role.DEFEND]),
            0,
        ),
        "G5_DEAD_LOCAL_ADAPTATION": (
            _state(positions=attack_positions, alive=(True, False, True, True)),
            roles_tensor([Role.ATTACK, Role.DEFEND, Role.ATTACK, Role.DEFEND]),
            1,
        ),
    }


def _v2_extra_fixtures() -> dict[str, Any]:
    """DEFEND_SEMANTIC_COMMITMENT_V2's anchor fixtures. Additions only -- the
    original _fixtures() (including G3/G4) are never modified.

    A 16-tick "stable outward" anchor is not constructed: DEFEND_OUTWARD's own
    target is a ray toward the arena boundary, so distance from the flag is
    monotonically non-decreasing under it. Starting anywhere inside the
    3.5-cell radius, the agent crosses back outside within roughly
    (R - d0) / (0.5 * max_speed_cps * dt) <= 3.5 / 0.545 ~= 6.4 ticks -- well
    inside a 16-tick horizon at every valid starting distance, including
    d0 = 0. No fixture can hold this branch stable for the full horizon; that
    is a property of the reference controller, not a fixture-construction
    gap. The decisive G3/G4 fixtures already exercise real outward segments,
    and outward-branch resolution correctness is established at the single
    tick level by tests/test_pyquaticus_port_contracts.py.
    """
    # 8.712 = 0.5 * 2.2 cps * 0.495 dt * 16 ticks: max possible closing
    # distance under DEFEND_INWARD across the full horizon. 14 cells clears
    # 3.5 (radius) + 8.712 with a >1.7-cell margin for every agent below.
    return {
        "V2_ANCHOR_STABLE_INWARD": (
            _state(positions=((16.0, 10.0), (16.0, 5.0), (16.0, 15.0), (15.0, 10.0))),
            roles_tensor([Role.DEFEND] * 4),
            0,
        ),
    }


def _core() -> BatchedCTFCore:
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=4,
        max_red_agents=4,
        map_set="train",
        map_layout="map_a",
        max_decision_steps=32,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=20260918,
        obstacle_obs_channel=True,
    )
    return BatchedCTFCore(cfg)


def _native_gates(core: BatchedCTFCore) -> dict[str, Any]:
    fixtures = _fixtures()
    g1_state, g1_roles, _ = fixtures["G1_ATTACK_NO_CARRIER"]
    g1 = true_motion(g1_state, g1_roles)
    g1_pass = bool(torch.allclose(g1.targets, g1_state.enemy_flag_pos.expand(4, 2)))

    g2_state, g2_roles, _ = fixtures["G2_ATTACK_SELF_CARRIER"]
    g2 = true_motion(g2_state, g2_roles)
    g2_pass = (
        bool(torch.allclose(g2.targets, g2_state.own_flag_home.expand(4, 2)))
        and g2.branches[0] == SemanticBranch.ATTACK_HOME_SELF_CARRIER
        and set(g2.branches[1:]) == {SemanticBranch.ATTACK_HOME_TEAMMATE_CARRIER}
    )

    g3_state, g3_roles, _ = fixtures["G3_DEFEND_OUTSIDE"]
    g3 = true_motion(g3_state, g3_roles)
    g3_pass = (
        set(g3.branches) == {SemanticBranch.DEFEND_INWARD}
        and bool(torch.allclose(g3.targets, g3_state.own_flag_pos.expand(4, 2)))
    )

    g4_state, g4_roles, _ = fixtures["G4_DEFEND_INSIDE"]
    g4 = true_motion(g4_state, g4_roles)
    g4_cosines = [
        direction_cosine(
            g4_state.positions[i],
            g4.targets[i],
            g4_state.positions[i] + expected_away_direction(g4_state, i),
        )
        for i in range(4)
    ]
    g4_pass = (
        set(g4.branches) == {SemanticBranch.DEFEND_OUTWARD}
        and min(g4_cosines) >= 0.999999
    )

    g5_tag_state, g5_tag_roles, _ = fixtures["G5_TAGGED"]
    g5_tag = true_motion(g5_tag_state, g5_tag_roles)
    g5_dead_state, g5_dead_roles, _ = fixtures["G5_DEAD_LOCAL_ADAPTATION"]
    g5_dead = true_motion(g5_dead_state, g5_dead_roles)
    core.blue_tagged.fill_(True)
    core.blue_carrying.fill_(True)
    core.blue_alive.fill_(False)
    core.reset_all()
    reset_pass = (
        not bool(core.blue_tagged.any())
        and not bool(core.blue_carrying.any())
        and bool(core.blue_alive.all())
    )
    g5_pass = (
        g5_tag.branches[0] == SemanticBranch.TAGGED_UPSTREAM_OVERRIDE
        and bool(torch.allclose(g5_tag.targets[0], g5_tag_state.own_flag_home))
        and abs(float(g5_tag.speed_fraction[0]) - 1.0) <= FLOAT_TOL
        and g5_dead.branches[1] == SemanticBranch.DEAD_LOCAL_ADAPTATION
        and bool(torch.allclose(g5_dead.targets[1], g5_dead_state.positions[1]))
        and abs(float(g5_dead.speed_fraction[1])) <= FLOAT_TOL
        and reset_pass
    )
    return {
        "G1_ATTACK_NO_CARRIER": {
            "pass": g1_pass,
            "targets": g1.targets.tolist(),
            "expected": g1_state.enemy_flag_pos.tolist(),
        },
        "G2_ATTACK_FRIENDLY_CARRIER": {
            "pass": g2_pass,
            "targets": g2.targets.tolist(),
            "branches": list(g2.branches),
            "expected": g2_state.own_flag_home.tolist(),
        },
        "G3_DEFEND_OUTSIDE": {
            "pass": g3_pass,
            "targets": g3.targets.tolist(),
            "expected": g3_state.own_flag_pos.tolist(),
            "radius_cells": DEFENDER_RADIUS_CELLS,
        },
        "G4_DEFEND_INSIDE": {
            "pass": g4_pass,
            "targets": g4.targets.tolist(),
            "direction_cosines": g4_cosines,
            "boundary_case_distance_cells": 3.5,
        },
        "G5_RESET_TAG": {
            "pass": g5_pass,
            "tagged_target": g5_tag.targets[0].tolist(),
            "tagged_speed_fraction": float(g5_tag.speed_fraction[0]),
            "dead_local_target": g5_dead.targets[1].tolist(),
            "dead_local_speed_fraction": float(g5_dead.speed_fraction[1]),
            "dead_semantics_source": "LOCAL_ADAPTATION_NO_UPSTREAM_ALIVE_DEAD_AXIS",
            "reset_clears_tagged_carrying_and_revives": reset_pass,
        },
    }


def _set_core_state(core: BatchedCTFCore, state: PortState) -> None:
    core.blue_x[0] = state.positions[:, 0]
    core.blue_y[0] = state.positions[:, 1]
    core.blue_heading[0] = state.headings
    core.blue_speed[0] = 0.0
    core.blue_flag_pos[0] = state.own_flag_pos
    core.blue_flag_home[0] = state.own_flag_home
    core.red_flag_pos[0] = state.enemy_flag_pos
    core.blue_carrying[0] = state.carrying
    core.blue_tagged[0] = state.tagged
    core.blue_alive[0] = state.alive


def _live_effective_target(
    core: BatchedCTFCore,
    state: PortState,
    candidate: ProjectionCandidate,
    agent_index: int,
) -> torch.Tensor:
    _set_core_state(core, state)
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    target_indices = torch.zeros((1, 4), dtype=torch.int64)
    macros[0, agent_index] = int(candidate.macro)
    target_indices[0, agent_index] = int(candidate.target_index)
    tx, ty = core._build_targets_from_action(macros, target_indices, side="blue")
    tx, ty, _, _ = core._redirect_tagged_to_home(tx, ty, tx.clone(), ty.clone())
    if not bool(state.alive[agent_index].item()):
        return state.positions[agent_index].clone()
    return torch.stack([tx[0, agent_index], ty[0, agent_index]])


def _integrate(
    core: BatchedCTFCore,
    x: torch.Tensor,
    y: torch.Tensor,
    heading: torch.Tensor,
    speed: torch.Tensor,
    alive: torch.Tensor,
    targets: torch.Tensor,
    speed_cap: torch.Tensor,
):
    return core._integrate_side(
        x,
        y,
        heading,
        speed,
        alive,
        targets[..., 0],
        targets[..., 1],
        speed_cap=speed_cap,
    )[:4]


def _candidate_by_label(
    state: PortState,
    roles: torch.Tensor,
    agent_index: int,
    waypoints: torch.Tensor,
    label: str,
    *,
    repair_defend: bool = False,
    repair_home_legality: bool = False,
    unified_defend: bool = False,
) -> ProjectionCandidate:
    motion = true_motion(state, roles)
    candidates = projection_candidates(
        state, motion, agent_index, waypoints,
        repair_defend=repair_defend, repair_home_legality=repair_home_legality,
        unified_defend=unified_defend,
    )
    exact = [candidate for candidate in candidates if candidate.label == label]
    if exact:
        return exact[0]
    # A DEFEND branch can flip INWARD<->OUTWARD mid-trajectory as the agent
    # crosses the radius. GO_TO_DEFENDER_SEMANTIC_TARGET is offered unchanged
    # by both branches (exact match above always catches it); the two new
    # semantic labels are branch-specific, so staying committed to "the
    # semantic option" across a flip means picking whichever semantic label
    # the NEW branch actually offers.
    if label in ("DEFEND_FLAG_SEMANTIC_TARGET", "DEFEND_OUTWARD_SEMANTIC_TARGET"):
        semantic = [
            c for c in candidates
            if c.label.endswith("_SEMANTIC_TARGET") and c.label != "GO_TO_DEFENDER_SEMANTIC_TARGET"
        ]
        if semantic:
            return semantic[0]
    raise RuntimeError(f"candidate {label!r} unavailable for {motion.branches[agent_index]}")


def _probe_candidate(
    core: BatchedCTFCore,
    initial_state: PortState,
    roles: torch.Tensor,
    agent_index: int,
    initial_candidate: ProjectionCandidate,
    *,
    repair_defend: bool = False,
    repair_home_legality: bool = False,
    unified_defend: bool = False,
) -> dict[str, Any]:
    i = int(agent_index)
    waypoints = core._macro_targets
    n_agents = initial_state.n_agents
    true_x = initial_state.positions[:, 0].reshape(1, n_agents).clone()
    true_y = initial_state.positions[:, 1].reshape(1, n_agents).clone()
    true_h = initial_state.headings.reshape(1, n_agents).clone()
    true_s = torch.zeros_like(true_x)
    proj_x, proj_y = true_x.clone(), true_y.clone()
    proj_h, proj_s = true_h.clone(), true_s.clone()
    alive_mask = torch.zeros((1, n_agents), dtype=torch.bool)
    alive_mask[0, i] = bool(initial_state.alive[i].item())

    # Align the tested agent with the initial TRUE desired bearing. This isolates
    # target/interface error from an arbitrary initial turn transient.
    initial_motion = true_motion(initial_state, roles)
    initial_vector = initial_motion.targets[i] - initial_state.positions[i]
    if float(torch.linalg.vector_norm(initial_vector)) > 1e-8:
        initial_heading = math.atan2(float(initial_vector[1]), float(initial_vector[0]))
        true_h[0, i] = initial_heading
        proj_h[0, i] = initial_heading

    commit_left = 0
    committed: ProjectionCandidate | None = None
    trajectory_errors: list[float] = []
    target_errors: list[float] = []
    direction_cosines: list[float] = []
    radial_sign_matches: list[bool] = []
    branch_while_committed: list[str] = []
    interface_targets: list[list[float]] = []
    true_targets: list[list[float]] = []
    true_agent_positions: list[list[float]] = []
    proj_agent_positions: list[list[float]] = []

    for _ in range(HORIZON):
        true_positions = torch.stack([true_x[0], true_y[0]], dim=1)
        true_state = initial_state.with_motion_state(true_positions, true_h[0])
        true_step = true_motion(true_state, roles)

        proj_positions = torch.stack([proj_x[0], proj_y[0]], dim=1)
        proj_state = initial_state.with_motion_state(proj_positions, proj_h[0])
        proj_oracle = true_motion(proj_state, roles)
        if commit_left <= 0:
            committed = _candidate_by_label(
                proj_state,
                roles,
                i,
                waypoints,
                initial_candidate.label,
                repair_defend=repair_defend,
                repair_home_legality=repair_home_legality,
                unified_defend=unified_defend,
            )
            tick_tensor = core._macro_commit_ticks(
                torch.tensor([[committed.macro]], dtype=torch.int64)
            )
            commit_left = int(tick_tensor[0, 0].item())
        assert committed is not None
        projected_target = effective_projected_target(
            proj_state,
            committed,
            i,
            waypoints,
        )
        live_target = _live_effective_target(core, proj_state, committed, i)
        live_target_error = float(torch.linalg.vector_norm(live_target - projected_target))
        if live_target_error > FLOAT_TOL:
            raise AssertionError(
                f"pure/live target mismatch {live_target_error} for {committed.label}"
            )

        target_errors.append(
            float(torch.linalg.vector_norm(projected_target - proj_oracle.targets[i]))
        )
        direction_cosines.append(
            direction_cosine(
                proj_state.positions[i],
                projected_target,
                proj_oracle.targets[i],
            )
        )
        branch_while_committed.append(proj_oracle.branches[i])
        interface_targets.append(projected_target.tolist())
        true_targets.append(proj_oracle.targets[i].tolist())
        true_agent_positions.append(true_positions[i].tolist())
        proj_agent_positions.append(proj_positions[i].tolist())

        true_target_batch = true_positions.clone().reshape(1, n_agents, 2)
        true_target_batch[0, i] = true_step.targets[i]
        proj_target_batch = proj_positions.clone().reshape(1, n_agents, 2)
        proj_target_batch[0, i] = projected_target
        true_caps = torch.zeros_like(true_s)
        true_caps[0, i] = float(core.cfg.max_speed_cps) * float(true_step.speed_fraction[i])
        proj_caps = torch.zeros_like(proj_s)
        proj_caps[0, i] = float(core.cfg.max_speed_cps)

        true_before = torch.stack([true_x[0, i], true_y[0, i]])
        proj_before = torch.stack([proj_x[0, i], proj_y[0, i]])
        true_x, true_y, true_h, true_s = _integrate(
            core, true_x, true_y, true_h, true_s, alive_mask, true_target_batch, true_caps
        )
        proj_x, proj_y, proj_h, proj_s = _integrate(
            core, proj_x, proj_y, proj_h, proj_s, alive_mask, proj_target_batch, proj_caps
        )
        true_after = torch.stack([true_x[0, i], true_y[0, i]])
        proj_after = torch.stack([proj_x[0, i], proj_y[0, i]])
        true_progress = float(torch.dot(true_after - true_before, true_step.targets[i] - true_before))
        proj_progress = float(torch.dot(proj_after - proj_before, proj_oracle.targets[i] - proj_before))
        radial_sign_matches.append((true_progress >= -FLOAT_TOL) == (proj_progress >= -FLOAT_TOL))
        trajectory_errors.append(float(torch.linalg.vector_norm(true_after - proj_after)))
        commit_left -= 1

    trajectory_rmse = math.sqrt(sum(error * error for error in trajectory_errors) / len(trajectory_errors))
    hard_checks = {
        "structurally_satisfiable": bool(initial_candidate.structurally_satisfiable),
        "first_direction_cosine_at_least_0_99": direction_cosines[0] >= DIRECTION_COSINE_MIN,
        "radial_sign_matches_every_tick": all(radial_sign_matches),
        "max_target_error_at_most_2_5_cells": max(target_errors) <= INTERACTION_RADIUS_CELLS,
        "trajectory_rmse_at_most_2_5_cells": trajectory_rmse <= INTERACTION_RADIUS_CELLS,
    }
    return {
        "label": initial_candidate.label,
        "macro": MacroAction(initial_candidate.macro).name,
        "target_index_initial": int(initial_candidate.target_index),
        "structural_note": initial_candidate.structural_note,
        "metrics": {
            "first_direction_cosine": direction_cosines[0],
            "minimum_direction_cosine": min(direction_cosines),
            "radial_sign_agreement_fraction": sum(radial_sign_matches) / len(radial_sign_matches),
            "maximum_target_error_cells": max(target_errors),
            "mean_target_error_cells": sum(target_errors) / len(target_errors),
            "trajectory_rmse_cells": trajectory_rmse,
            "maximum_position_error_cells": max(trajectory_errors),
            "semantic_branches_seen": sorted(set(branch_while_committed)),
        },
        "hard_checks": hard_checks,
        "pass": all(hard_checks.values()),
        "trace": {
            "true_targets": true_targets,
            "projected_effective_targets": interface_targets,
            "position_error_cells": trajectory_errors,
            "radial_sign_match": radial_sign_matches,
            "true_agent_position": true_agent_positions,
            "proj_agent_position": proj_agent_positions,
        },
    }


def _candidate_rank(record: dict[str, Any]) -> tuple[float, float, float, int, int]:
    metrics = record["metrics"]
    macro_id = int(MacroAction[record["macro"]])
    return (
        float(metrics["maximum_target_error_cells"]),
        float(metrics["trajectory_rmse_cells"]),
        float(1.0 - metrics["radial_sign_agreement_fraction"]),
        macro_id,
        int(record["target_index_initial"]),
    )


def _projection_gate(
    core: BatchedCTFCore,
    *,
    repair_defend: bool = False,
    repair_home_legality: bool = False,
    unified_defend: bool = False,
    extra_fixtures: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    all_cases_pass = True
    fixtures = dict(_fixtures())
    if extra_fixtures:
        fixtures.update(extra_fixtures)
    for name, (state, roles, agent_index) in fixtures.items():
        motion = true_motion(state, roles)
        candidates = projection_candidates(
            state, motion, agent_index, core._macro_targets,
            repair_defend=repair_defend, repair_home_legality=repair_home_legality,
            unified_defend=unified_defend,
        )
        records = [
            _probe_candidate(
                core, state, roles, agent_index, candidate,
                repair_defend=repair_defend, repair_home_legality=repair_home_legality,
                unified_defend=unified_defend,
            )
            for candidate in candidates
        ]
        passing = sorted(
            (record for record in records if record["pass"]),
            key=_candidate_rank,
        )
        case_pass = bool(passing)
        all_cases_pass = all_cases_pass and case_pass
        cases[name] = {
            "semantic_branch": motion.branches[agent_index],
            "agent_index": agent_index,
            "true_target_initial": motion.targets[agent_index].tolist(),
            "own_flag_pos": state.own_flag_pos.tolist(),
            "candidates": records,
            "pass": case_pass,
            "selected_candidate_if_global_gate_passes": passing[0]["label"] if passing else None,
        }
    return {
        "gate_id": "G6_PROJECTED_REPRESENTABILITY",
        "pass": all_cases_pass,
        "horizon_decision_ticks": HORIZON,
        "thresholds": {
            "first_direction_cosine_min": DIRECTION_COSINE_MIN,
            "radial_sign_agreement_required": 1.0,
            "maximum_target_error_cells": INTERACTION_RADIUS_CELLS,
            "trajectory_rmse_cells": INTERACTION_RADIUS_CELLS,
        },
        "cases": cases,
        "adapter_freeze_status": "ELIGIBLE_TO_FREEZE" if all_cases_pass else "UNREPRESENTABLE_TEAM_EVALUATION_BLOCKED",
    }


def _subgate_summary(projection: dict[str, Any]) -> dict[str, Any]:
    """PYQUATICUS_G6_SUBGATE_DECOMPOSITION's four questions, read off the
    per-candidate hard_checks already computed by _probe_candidate."""
    per_candidate: list[dict[str, Any]] = []
    for case_name, case in projection["cases"].items():
        for record in case["candidates"]:
            checks = record["hard_checks"]
            per_candidate.append({
                "case": case_name,
                "label": record["label"],
                "G6_TARGET": bool(checks["max_target_error_at_most_2_5_cells"]),
                "G6_LEGALITY": bool(checks["structurally_satisfiable"]),
                "G6_DIRECTION": bool(
                    checks["first_direction_cosine_at_least_0_99"]
                    and checks["radial_sign_matches_every_tick"]
                ),
                "G6_TRAJECTORY": bool(checks["trajectory_rmse_at_most_2_5_cells"]),
            })

    def _fails(key: str) -> list[str]:
        return [f"{item['case']}/{item['label']}" for item in per_candidate if not item[key]]

    return {
        "per_candidate": per_candidate,
        "G6_TARGET": {"fails": _fails("G6_TARGET")},
        "G6_LEGALITY": {"fails": _fails("G6_LEGALITY")},
        "G6_DIRECTION": {"fails": _fails("G6_DIRECTION")},
        "G6_TRAJECTORY": {
            "fails": _fails("G6_TRAJECTORY"),
            "note": "reported, never gating -- a cross-simulator dynamics disagreement, not a vocabulary defect",
        },
    }


def _v2_mechanical_attribution(projection: dict[str, Any]) -> dict[str, Any]:
    """GATES.G6_DIRECTION_radial_sign.MECHANICAL_ATTRIBUTION_REQUIRED, computed
    directly from the recorded per-tick trace -- never narrated.

    For each case/candidate: identify the ticks where the TRUE oracle's own
    branch changed (own_flag_pos as reference, radius DEFENDER_RADIUS_CELLS),
    the target error at those ticks, and for every tick with a radial-sign
    disagreement, whether the true and projected arms sat on opposite sides
    of the radius (EXPLAINED) or the same side (UNEXPLAINED).
    """
    report: dict[str, Any] = {}
    for case_name, case in projection["cases"].items():
        case_report = {}
        flag_x, flag_y = case["own_flag_pos"]
        for record in case["candidates"]:
            trace = record["trace"]
            true_pos = trace.get("true_agent_position")
            proj_pos = trace.get("proj_agent_position")
            if not true_pos or not proj_pos:
                continue
            true_targets = trace["true_targets"]
            radial_match = trace["radial_sign_match"]
            true_dist = [math.hypot(p[0] - flag_x, p[1] - flag_y) for p in true_pos]
            proj_dist = [math.hypot(p[0] - flag_x, p[1] - flag_y) for p in proj_pos]
            true_branch = ["INWARD" if d > DEFENDER_RADIUS_CELLS else "OUTWARD" for d in true_dist]
            flips = [t for t in range(1, len(true_branch)) if true_branch[t] != true_branch[t - 1]]
            target_error_at_flips = {
                t: math.hypot(
                    trace["projected_effective_targets"][t][0] - true_targets[t][0],
                    trace["projected_effective_targets"][t][1] - true_targets[t][1],
                )
                for t in flips
            }
            explained, unexplained = [], []
            for t, matched in enumerate(radial_match):
                if matched:
                    continue
                true_inside = true_dist[t] <= DEFENDER_RADIUS_CELLS
                proj_inside = proj_dist[t] <= DEFENDER_RADIUS_CELLS
                (explained if true_inside != proj_inside else unexplained).append(t)
            case_report[record["label"]] = {
                "branch_flip_ticks": flips,
                "target_error_at_flip_ticks": target_error_at_flips,
                "radial_disagreement_ticks_total": int(sum(1 for m in radial_match if not m)),
                "radial_disagreement_explained_opposite_side": explained,
                "radial_disagreement_UNEXPLAINED_same_side": unexplained,
            }
        report[case_name] = case_report
    return report


def build_result(
    upstream_root: Path,
    *,
    repair_defend: bool = False,
    repair_home_legality: bool = False,
    unified_defend: bool = False,
    extra_fixtures: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized; this milestone is CPU-only")
    provenance = _verify_upstream(upstream_root)
    core = _core()
    if str(core.device) != "cpu":
        raise RuntimeError(f"contract core must be CPU, got {core.device}")
    native = _native_gates(core)
    projection = _projection_gate(
        core, repair_defend=repair_defend, repair_home_legality=repair_home_legality,
        unified_defend=unified_defend, extra_fixtures=extra_fixtures,
    )
    native_pass = all(bool(value["pass"]) for value in native.values())
    clean_seal = bool(provenance["all_match"] and native_pass and projection["pass"])
    if not provenance["all_match"]:
        failure_localization = "PROVENANCE_INVALID"
    elif not native_pass:
        failure_localization = "PORT_INVALID"
    elif not projection["pass"]:
        failure_localization = "ACTION_INTERFACE_DISTORTION"
    else:
        failure_localization = "NONE_CONTRACTS_PASS"
    implementation = ROOT / "gpu_env/pyquaticus_port.py"
    runner = Path(__file__).resolve()
    subgates = _subgate_summary(projection)
    return {
        "record_id": "PYQUATICUS_PORT_CONTRACT_RESULT",
        "status": "PASS_CLEAN_CONTRACT_SEAL" if clean_seal else "FAIL_TEAM_EVALUATION_BLOCKED",
        "repair_defend": bool(repair_defend),
        "repair_home_legality": bool(repair_home_legality),
        "unified_defend": bool(unified_defend),
        "utc": _now(),
        "device": "cpu",
        "gpu_used": False,
        "team_episodes_run": 0,
        "ppo": False,
        "question": "Do the provenance-pinned TRUE semantics pass G1-G5, and is every required semantic branch representable through the current action interface under G6?",
        "spec": {
            "path": str(SPEC.relative_to(ROOT)),
            "sha256": _sha256(SPEC),
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "mode": UPSTREAM_MODE,
            "license": UPSTREAM_LICENSE,
            "verification": provenance,
        },
        "implementation": {
            "semantic_module": str(implementation.relative_to(ROOT)),
            "semantic_module_sha256": _sha256(implementation),
            "runner": str(runner.relative_to(ROOT)),
            "runner_sha256": _sha256(runner),
        },
        "G1_G5_NATIVE": native,
        "G6_PROJECTED_REPRESENTABILITY": projection,
        "G6_SUBGATES": subgates,
        "overall": {
            "provenance_pass": bool(provenance["all_match"]),
            "native_G1_G5_pass": native_pass,
            "projected_G6_pass": bool(projection["pass"]),
            "clean_contract_seal": clean_seal,
            "team_evaluation_unlocked": clean_seal,
        },
        "failure_localization": failure_localization,
        "decision": (
            "CONTRACTS_PASS_WRITE_SEPARATE_EVAL_SPEC_DO_NOT_LAUNCH_AUTOMATICALLY"
            if clean_seal
            else "STOP_NO_TEAM_EVALUATION"
        ),
        "claim_boundary": "No outcome here concerns PPO, learned roles, team payoff, or assignment effects.",
    }


DEFEND_SPEC = ROOT / "artifacts/strategic_demand/sppo/DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1_SPEC.json"
DEFEND_RESULT_DIR = ROOT / "artifacts/strategic_demand/sppo"
COMBOS: tuple[tuple[bool, bool], ...] = (
    (False, False),  # C2: must reproduce the sealed frozen result exactly
    (True, False),   # Repair A only
    (False, True),   # Repair B only
    (True, True),    # both
)


def _combo_suffix(repair_defend: bool, repair_home_legality: bool) -> str:
    return f"DEFEND_{'ON' if repair_defend else 'OFF'}_HOME_{'ON' if repair_home_legality else 'OFF'}"


def _c2_repair_off_equals_frozen(off_off: dict[str, Any]) -> dict[str, Any]:
    if not DEFAULT_OUT.is_file():
        return {"pass": False, "reason": f"sealed reference missing: {DEFAULT_OUT}"}
    sealed = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for case_name, case in sealed["G6_PROJECTED_REPRESENTABILITY"]["cases"].items():
        new_case = off_off["G6_PROJECTED_REPRESENTABILITY"]["cases"][case_name]
        if len(case["candidates"]) != len(new_case["candidates"]):
            mismatches.append(f"{case_name}: candidate count {len(case['candidates'])} != {len(new_case['candidates'])}")
            continue
        for old_c, new_c in zip(case["candidates"], new_case["candidates"]):
            if old_c["label"] != new_c["label"]:
                mismatches.append(f"{case_name}: label {old_c['label']} != {new_c['label']}")
                continue
            for key, old_v in old_c["metrics"].items():
                new_v = new_c["metrics"][key]
                if isinstance(old_v, float):
                    if abs(old_v - float(new_v)) > FLOAT_TOL:
                        mismatches.append(f"{case_name}/{old_c['label']}/{key}: {old_v} != {new_v}")
                elif old_v != new_v:
                    mismatches.append(f"{case_name}/{old_c['label']}/{key}: {old_v} != {new_v}")
    return {
        "pass": not mismatches,
        "n_mismatches": len(mismatches),
        "mismatches_sample": mismatches[:10],
        "reference": str(DEFAULT_OUT.relative_to(ROOT)),
    }


def _c4_orthogonality(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    off_off = results["DEFEND_OFF_HOME_OFF"]["G6_SUBGATES"]
    a_only = results["DEFEND_ON_HOME_OFF"]["G6_SUBGATES"]
    b_only = results["DEFEND_OFF_HOME_ON"]["G6_SUBGATES"]
    both = results["DEFEND_ON_HOME_ON"]["G6_SUBGATES"]

    checks = {
        "A_fixes_G3_G4_target": (
            any("G3_DEFEND_OUTSIDE" in f for f in off_off["G6_TARGET"]["fails"])
            and any("G4_DEFEND_INSIDE" in f for f in off_off["G6_TARGET"]["fails"])
            and not any("G3_DEFEND_OUTSIDE" in f for f in a_only["G6_TARGET"]["fails"] if "DEFEND_FLAG" in f or "DEFEND_OUTWARD" in f)
        ),
        "A_does_not_touch_legality": (
            a_only["G6_LEGALITY"]["fails"] == off_off["G6_LEGALITY"]["fails"]
        ),
        "B_fixes_home_legality": (
            any("GO_HOME_NONCARRIER" in f for f in off_off["G6_LEGALITY"]["fails"])
            and not any("GO_HOME_NONCARRIER" in f for f in b_only["G6_LEGALITY"]["fails"])
        ),
        "B_does_not_touch_target": (
            b_only["G6_TARGET"]["fails"] == off_off["G6_TARGET"]["fails"]
        ),
        "both_fix_both": (
            not any("GO_HOME_NONCARRIER" in f for f in both["G6_LEGALITY"]["fails"])
        ),
        "attack_to_flag_unaffected_everywhere": all(
            not any("G1_ATTACK_NO_CARRIER/GET_FLAG" in f for f in r["G6_TARGET"]["fails"])
            for r in (off_off, a_only, b_only, both)
        ),
    }
    return {"pass": all(checks.values()), "checks": checks}


V2_SPEC = ROOT / "artifacts/strategic_demand/sppo/DEFEND_SEMANTIC_COMMITMENT_V2_SPEC.json"
V2_RESULT_DIR = ROOT / "artifacts/strategic_demand/sppo"


def _main_v2(upstream_root: Path) -> int:
    protected = [
        DEFAULT_OUT,
        ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_G6_SUBGATE_DECOMPOSITION.json",
        ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PROJECTION_UNREPRESENTABLE_RESULT.json",
        ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json",
        ROOT / "artifacts/strategic_demand/sppo/DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1_CONTRACT_RESULT.json",
        ROOT / "artifacts/strategic_demand/sppo/HOME_LEGALITY_CONFIRMED_INTERFACE_REPAIR.json",
    ]
    protected_before = {p.name: (_sha256(p) if p.is_file() else None) for p in protected}

    extra = _v2_extra_fixtures()
    off_result = build_result(upstream_root, unified_defend=False, extra_fixtures=extra)
    on_result = build_result(upstream_root, unified_defend=True, extra_fixtures=extra)

    off_path = V2_RESULT_DIR / "DEFEND_SEMANTIC_COMMITMENT_V2_UNIFIED_OFF_RESULT.json"
    on_path = V2_RESULT_DIR / "DEFEND_SEMANTIC_COMMITMENT_V2_UNIFIED_ON_RESULT.json"
    off_path.write_text(json.dumps(off_result, indent=2) + "\n", encoding="utf-8")
    on_path.write_text(json.dumps(on_result, indent=2) + "\n", encoding="utf-8")
    print(f"UNIFIED_OFF: G6={off_result['overall']['projected_G6_pass']} -> {off_path.name}")
    print(f"UNIFIED_ON:  G6={on_result['overall']['projected_G6_pass']} -> {on_path.name}")

    protected_after = {p.name: (_sha256(p) if p.is_file() else None) for p in protected}
    c0_pass = protected_before == protected_after
    if not c0_pass:
        raise SystemExit(f"REFUSING: a protected artifact changed during the run: {protected_before} -> {protected_after}")

    # C2: off must reproduce the sealed frozen result exactly on the ORIGINAL
    # seven fixtures (the anchor is additional and has no sealed counterpart).
    sealed = json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))
    c2_mismatches: list[str] = []
    for case_name, case in sealed["G6_PROJECTED_REPRESENTABILITY"]["cases"].items():
        new_case = off_result["G6_PROJECTED_REPRESENTABILITY"]["cases"][case_name]
        for old_c, new_c in zip(case["candidates"], new_case["candidates"]):
            if old_c["label"] != new_c["label"]:
                c2_mismatches.append(f"{case_name}: label {old_c['label']} != {new_c['label']}")
                continue
            for key, old_v in old_c["metrics"].items():
                new_v = new_c["metrics"][key]
                if isinstance(old_v, float):
                    if abs(old_v - float(new_v)) > FLOAT_TOL:
                        c2_mismatches.append(f"{case_name}/{old_c['label']}/{key}: {old_v} != {new_v}")
                elif old_v != new_v:
                    c2_mismatches.append(f"{case_name}/{old_c['label']}/{key}: {old_v} != {new_v}")
    c2 = {"pass": not c2_mismatches, "n_mismatches": len(c2_mismatches), "mismatches_sample": c2_mismatches[:10]}

    # C4 orthogonality: unified_defend must not touch any non-DEFEND branch.
    off_sub, on_sub = off_result["G6_SUBGATES"], on_result["G6_SUBGATES"]
    non_defend_fails_match = all(
        {f for f in off_sub[k]["fails"] if "DEFEND" not in f and "G3_" not in f and "G4_" not in f}
        == {f for f in on_sub[k]["fails"] if "DEFEND" not in f and "G3_" not in f and "G4_" not in f}
        for k in ("G6_TARGET", "G6_LEGALITY", "G6_DIRECTION")
    )
    c4 = {"pass": non_defend_fails_match, "non_defend_fails_unchanged": non_defend_fails_match}

    attribution = _v2_mechanical_attribution(on_result["G6_PROJECTED_REPRESENTABILITY"])

    decisive_fixtures = ["G3_DEFEND_OUTSIDE", "G4_DEFEND_INSIDE", "V2_ANCHOR_STABLE_INWARD"]
    decisive_target_pass = all(
        any(
            c["label"] == "DEFEND_UNIFIED_SEMANTIC_TARGET" and c["hard_checks"]["max_target_error_at_most_2_5_cells"]
            for c in on_result["G6_PROJECTED_REPRESENTABILITY"]["cases"][fx]["candidates"]
        )
        for fx in decisive_fixtures
    )
    decisive_direction_pass = all(
        any(
            c["label"] == "DEFEND_UNIFIED_SEMANTIC_TARGET" and c["hard_checks"]["first_direction_cosine_at_least_0_99"]
            for c in on_result["G6_PROJECTED_REPRESENTABILITY"]["cases"][fx]["candidates"]
        )
        for fx in decisive_fixtures
    )
    representable = bool(decisive_target_pass and decisive_direction_pass and c2["pass"] and c4["pass"])

    contract_result = {
        "record_id": "DEFEND_SEMANTIC_COMMITMENT_V2_CONTRACT_RESULT",
        "status": "REPRESENTABLE_ACROSS_TRANSITIONS" if representable else "FALSIFIED_OR_PARTIAL",
        "utc": _now(),
        "implements": [str(V2_SPEC.relative_to(ROOT))],
        "spec_sha256": _sha256(V2_SPEC) if V2_SPEC.is_file() else None,
        "C0_prior_artifact_protection": {"pass": c0_pass, "sha256": protected_after},
        "C2_repair_off_equals_sealed_on_original_fixtures": c2,
        "C4_orthogonality_non_defend_branches_unaffected": c4,
        "DECISIVE_G6_TARGET_pass_on": decisive_fixtures,
        "DECISIVE_G6_TARGET_result": decisive_target_pass,
        "DECISIVE_G6_DIRECTION_first_cosine_result": decisive_direction_pass,
        "MECHANICAL_RADIAL_SIGN_ATTRIBUTION": attribution,
        "results": {"unified_off": off_path.name, "unified_on": on_path.name},
        "REPRESENTABLE_ACROSS_TRANSITIONS": representable,
        "team_evaluation_unlocked": False,
        "claim_boundary": (
            "This establishes only that a single committed DEFEND macro can express "
            "the frozen external defender semantics across its own branch transitions "
            "on the tested fixtures. It says nothing about learned specialization, PPO, "
            "or team payoff, and does not by itself unblock the Pyquaticus team evaluation."
        ),
    }
    contract_out = V2_RESULT_DIR / "DEFEND_SEMANTIC_COMMITMENT_V2_CONTRACT_RESULT.json"
    contract_out.write_text(json.dumps(contract_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": contract_result["status"],
        "C2_pass": c2["pass"], "C4_pass": c4["pass"],
        "decisive_target_pass": decisive_target_pass,
        "decisive_direction_pass": decisive_direction_pass,
        "out": str(contract_out),
    }, indent=2))
    return 0 if representable else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--repair-defend", action="store_true")
    parser.add_argument("--repair-home-legality", action="store_true")
    parser.add_argument(
        "--all-combinations", action="store_true",
        help="Run all four repair-flag combinations and freeze the orthogonality contract result.",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help="Run DEFEND_SEMANTIC_COMMITMENT_V2: unified_defend off vs on, plus the mechanical attribution report.",
    )
    args = parser.parse_args()

    if args.v2:
        return _main_v2(args.upstream_root)

    if not args.all_combinations:
        result = build_result(
            args.upstream_root,
            repair_defend=args.repair_defend,
            repair_home_legality=args.repair_home_legality,
        )
        out = args.out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "status": result["status"], "out": str(out), "overall": result["overall"],
        }, indent=2))
        return 0 if result["overall"]["clean_contract_seal"] else 2

    protected = [
        DEFAULT_OUT,
        ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_G6_SUBGATE_DECOMPOSITION.json",
        ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PROJECTION_UNREPRESENTABLE_RESULT.json",
        ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json",
    ]
    protected_before = {p.name: (_sha256(p) if p.is_file() else None) for p in protected}

    results: dict[str, dict[str, Any]] = {}
    for repair_defend, repair_home_legality in COMBOS:
        suffix = _combo_suffix(repair_defend, repair_home_legality)
        result = build_result(
            args.upstream_root,
            repair_defend=repair_defend, repair_home_legality=repair_home_legality,
        )
        results[suffix] = result
        out_path = DEFEND_RESULT_DIR / f"DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1_{suffix}_RESULT.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"{suffix}: G6={result['overall']['projected_G6_pass']} -> {out_path.name}")

    protected_after = {p.name: (_sha256(p) if p.is_file() else None) for p in protected}
    c0_pass = protected_before == protected_after
    if not c0_pass:
        raise SystemExit(f"REFUSING: a protected artifact changed during the run: {protected_before} -> {protected_after}")

    c2 = _c2_repair_off_equals_frozen(results["DEFEND_OFF_HOME_OFF"])
    c4 = _c4_orthogonality(results)

    success = {
        "G6_TARGET_G3_DEFEND_OUTSIDE": not any(
            "G3_DEFEND_OUTSIDE" in f and ("DEFEND_FLAG" in f or "DEFEND_OUTWARD" in f)
            for f in results["DEFEND_ON_HOME_ON"]["G6_SUBGATES"]["G6_TARGET"]["fails"]
        ),
        "G6_TARGET_G4_DEFEND_INSIDE": not any(
            "G4_DEFEND_INSIDE" in f and ("DEFEND_FLAG" in f or "DEFEND_OUTWARD" in f)
            for f in results["DEFEND_ON_HOME_ON"]["G6_SUBGATES"]["G6_TARGET"]["fails"]
        ),
        "G6_LEGALITY_G2_NONCARRIER_HOME": not any(
            "GO_HOME_NONCARRIER" in f
            for f in results["DEFEND_ON_HOME_ON"]["G6_SUBGATES"]["G6_LEGALITY"]["fails"]
        ),
    }
    representability_restored = all(success.values()) and bool(c2["pass"]) and bool(c4["pass"])

    contract_result = {
        "record_id": "DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1_CONTRACT_RESULT",
        "status": "REPRESENTABILITY_RESTORED" if representability_restored else "PARTIAL_OR_REGRESSION",
        "utc": _now(),
        "implements": [str(DEFEND_SPEC.relative_to(ROOT))],
        "spec_sha256": _sha256(DEFEND_SPEC) if DEFEND_SPEC.is_file() else None,
        "C0_prior_artifact_protection": {"pass": c0_pass, "sha256": protected_after},
        "combinations_run": list(results.keys()),
        "per_combination_results": {
            suffix: f"DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1_{suffix}_RESULT.json" for suffix in results
        },
        "C2_repair_off_equals_frozen": c2,
        "C4_orthogonality": c4,
        "SUCCESS_CONDITIONS": success,
        "REPRESENTABILITY_RESTORED": representability_restored,
        "team_evaluation_unlocked": False,
        "claim_boundary": (
            "This establishes only that the repaired interface can express the frozen "
            "external DEFEND semantics and the non-carrier home legality. It does not "
            "establish learned specialization, PPO outcomes, or team payoff."
        ),
    }
    contract_out = DEFEND_RESULT_DIR / "DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1_CONTRACT_RESULT.json"
    contract_out.write_text(json.dumps(contract_result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": contract_result["status"],
        "C2_pass": c2["pass"],
        "C4_pass": c4["pass"],
        "SUCCESS_CONDITIONS": success,
        "out": str(contract_out),
    }, indent=2))
    return 0 if representability_restored else 2


if __name__ == "__main__":
    raise SystemExit(main())
