"""Provenance-pinned Pyquaticus easy-mode behavioral semantics.

This module adapts only the target-selection behavior of MIT Lincoln
Laboratory's Pyquaticus ``BaseAttacker`` and ``BaseDefender`` at commit
``72b50e067ab311929390ecd4e59131452be15c6d``. It does not import the
upstream dispatcher, run Pyquaticus physics, or implement a learned policy.

Upstream copyright: Copyright 2023 Massachusetts Institute of Technology.
Upstream license: BSD-3-Clause. See ``AICTFProject/THIRD_PARTY_NOTICES.md``.
The scientific and provenance contract is frozen in
``artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

import torch

from macro_actions import MacroAction


UPSTREAM_REPOSITORY = "https://github.com/mit-ll-trusted-autonomy/pyquaticus"
UPSTREAM_COMMIT = "72b50e067ab311929390ecd4e59131452be15c6d"
UPSTREAM_MODE = "easy"
UPSTREAM_LICENSE = "BSD-3-Clause"

UPSTREAM_SHA256 = {
    "LICENSE": "0fdad9afd329ec1eb589fedf9e1a00be1ed641e2ebf31bb837eb9223f168d755",
    "pyquaticus/base_policies/base_attack.py": "e3a7f6e19dc4231124d927206efbee8fa1248e8cc081598efbc7b2cefcd92568",
    "pyquaticus/base_policies/base_defend.py": "8bf5947a1028dccf0b35c0a134926d24a33f832b4d331a99a90b8b114396a75b",
    "pyquaticus/base_policies/base_policy.py": "d2db01ac77aaa3ed10739c1ed738b3201e731b3672b2b3a19cc38478d3b4a5d9",
    "pyquaticus/envs/pyquaticus.py": "ab87caaf460762d341b4121dd385eaa19bfa315fff56df0029c9aa1c365b57bc",
    "pyquaticus/config.py": "5c0cb391fb33622df8cff18ee9298f8c76b897c3acc6a8d75f178faf7c7205a4",
    "pyquaticus/base_policies/README.md": "d26fd9ab42b19ed6ce4cb480b2530edb353b461f05f8c08c9d7babae5f5cde6d",
    "pyproject.toml": "9f54cbb440ee4f357bb01c8e3b82bec91016ecfeea8c4533c39cba72ab1e51a9",
    "pyquaticus/base_policies/base_combined.py": "8259679c0bf69a80647a027e9cc104d88f7776a7031ccb19283b2b8ea1f1e828",
}

UPSTREAM_CATCH_RADIUS_M = 10.0
UPSTREAM_FLAG_KEEPOUT_M = 3.0
UPSTREAM_DEFENDER_BUFFER_M = 1.0
LOCAL_TAG_RANGE_CELLS = 2.5
METERS_TO_CELLS = LOCAL_TAG_RANGE_CELLS / UPSTREAM_CATCH_RADIUS_M
DEFENDER_RADIUS_CELLS = (
    UPSTREAM_FLAG_KEEPOUT_M
    + UPSTREAM_CATCH_RADIUS_M
    + UPSTREAM_DEFENDER_BUFFER_M
) * METERS_TO_CELLS
EASY_SPEED_FRACTION = 0.5
TAGGED_SPEED_FRACTION = 1.0


class Role(IntEnum):
    ATTACK = 0
    DEFEND = 1


class SemanticBranch(str):
    ATTACK_ENEMY_FLAG = "ATTACK_ENEMY_FLAG"
    ATTACK_HOME_SELF_CARRIER = "ATTACK_HOME_SELF_CARRIER"
    ATTACK_HOME_TEAMMATE_CARRIER = "ATTACK_HOME_TEAMMATE_CARRIER"
    DEFEND_INWARD = "DEFEND_INWARD"
    DEFEND_OUTWARD = "DEFEND_OUTWARD"
    TAGGED_UPSTREAM_OVERRIDE = "TAGGED_UPSTREAM_OVERRIDE"
    DEAD_LOCAL_ADAPTATION = "DEAD_LOCAL_ADAPTATION"


@dataclass(frozen=True)
class PortState:
    """Global state required by the pinned scripted-controller semantics."""

    positions: torch.Tensor
    headings: torch.Tensor
    own_flag_pos: torch.Tensor
    own_flag_home: torch.Tensor
    enemy_flag_pos: torch.Tensor
    carrying: torch.Tensor
    tagged: torch.Tensor
    alive: torch.Tensor
    max_x: float = 19.0
    max_y: float = 19.0

    def validate(self) -> None:
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape [N, 2]")
        n_agents = int(self.positions.shape[0])
        for name, value in (
            ("headings", self.headings),
            ("carrying", self.carrying),
            ("tagged", self.tagged),
            ("alive", self.alive),
        ):
            if tuple(value.shape) != (n_agents,):
                raise ValueError(f"{name} must have shape [{n_agents}]")
        for name, value in (
            ("own_flag_pos", self.own_flag_pos),
            ("own_flag_home", self.own_flag_home),
            ("enemy_flag_pos", self.enemy_flag_pos),
        ):
            if tuple(value.shape) != (2,):
                raise ValueError(f"{name} must have shape [2]")
        if self.max_x <= 0.0 or self.max_y <= 0.0:
            raise ValueError("arena bounds must be positive")

    @property
    def n_agents(self) -> int:
        return int(self.positions.shape[0])

    def with_motion_state(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
    ) -> "PortState":
        return PortState(
            positions=positions,
            headings=headings,
            own_flag_pos=self.own_flag_pos,
            own_flag_home=self.own_flag_home,
            enemy_flag_pos=self.enemy_flag_pos,
            carrying=self.carrying,
            tagged=self.tagged,
            alive=self.alive,
            max_x=self.max_x,
            max_y=self.max_y,
        )


@dataclass(frozen=True)
class TrueMotion:
    targets: torch.Tensor
    speed_fraction: torch.Tensor
    branches: tuple[str, ...]


@dataclass(frozen=True)
class ProjectionCandidate:
    label: str
    macro: int
    target_index: int
    structurally_satisfiable: bool
    structural_note: str


def _unit(vector: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector)
    if float(norm) <= eps:
        return torch.zeros_like(vector)
    return vector / norm


def _ray_to_boundary(
    position: torch.Tensor,
    direction: torch.Tensor,
    *,
    max_x: float,
    max_y: float,
) -> torch.Tensor:
    """Return the arena-boundary endpoint on a ray without changing bearing."""
    eps = 1e-8
    ux = float(direction[0])
    uy = float(direction[1])
    px = float(position[0])
    py = float(position[1])
    candidates: list[float] = []
    if ux > eps:
        candidates.append((max_x - px) / ux)
    elif ux < -eps:
        candidates.append((0.0 - px) / ux)
    if uy > eps:
        candidates.append((max_y - py) / uy)
    elif uy < -eps:
        candidates.append((0.0 - py) / uy)
    positive = [value for value in candidates if value >= 0.0]
    if not positive:
        return position.clone()
    distance = min(positive)
    return position + direction * distance


def true_motion(
    state: PortState,
    roles: torch.Tensor,
    *,
    defender_radius_cells: float = DEFENDER_RADIUS_CELLS,
) -> TrueMotion:
    """Compute the pinned global-state easy-mode target oracle.

    Tagged execution is an upstream environment override. Dead execution is a
    named local adaptation because pinned Pyquaticus has no equivalent alive /
    respawn state.
    """
    state.validate()
    if tuple(roles.shape) != (state.n_agents,):
        raise ValueError(f"roles must have shape [{state.n_agents}]")

    targets = torch.empty_like(state.positions)
    speed_fraction = torch.full(
        (state.n_agents,),
        EASY_SPEED_FRACTION,
        dtype=state.positions.dtype,
        device=state.positions.device,
    )
    branches: list[str] = []
    friendly_carrying = bool(state.carrying.any().item())

    for i in range(state.n_agents):
        role = Role(int(roles[i].item()))
        if role == Role.ATTACK:
            if friendly_carrying:
                targets[i] = state.own_flag_home
                branch = (
                    SemanticBranch.ATTACK_HOME_SELF_CARRIER
                    if bool(state.carrying[i].item())
                    else SemanticBranch.ATTACK_HOME_TEAMMATE_CARRIER
                )
            else:
                targets[i] = state.enemy_flag_pos
                branch = SemanticBranch.ATTACK_ENEMY_FLAG
        else:
            to_flag = state.own_flag_pos - state.positions[i]
            distance = float(torch.linalg.vector_norm(to_flag))
            if distance > float(defender_radius_cells):
                targets[i] = state.own_flag_pos
                branch = SemanticBranch.DEFEND_INWARD
            else:
                away = -to_flag
                away_unit = _unit(away)
                if not bool(torch.any(away_unit).item()):
                    away_unit = torch.stack(
                        [torch.cos(state.headings[i]), torch.sin(state.headings[i])]
                    ).to(dtype=state.positions.dtype, device=state.positions.device)
                targets[i] = _ray_to_boundary(
                    state.positions[i],
                    away_unit,
                    max_x=state.max_x,
                    max_y=state.max_y,
                )
                branch = SemanticBranch.DEFEND_OUTWARD

        if not bool(state.alive[i].item()):
            targets[i] = state.positions[i]
            speed_fraction[i] = 0.0
            branch = SemanticBranch.DEAD_LOCAL_ADAPTATION
        elif bool(state.tagged[i].item()):
            targets[i] = state.own_flag_home
            speed_fraction[i] = TAGGED_SPEED_FRACTION
            branch = SemanticBranch.TAGGED_UPSTREAM_OVERRIDE
        branches.append(str(branch))

    return TrueMotion(
        targets=targets,
        speed_fraction=speed_fraction,
        branches=tuple(branches),
    )


def nearest_waypoint_index(target: torch.Tensor, waypoints: torch.Tensor) -> int:
    if tuple(target.shape) != (2,):
        raise ValueError("target must have shape [2]")
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("waypoints must have shape [K, 2]")
    distances = torch.linalg.vector_norm(waypoints - target[None, :], dim=1)
    return int(torch.argmin(distances).item())


def projection_candidates(
    state: PortState,
    motion: TrueMotion,
    agent_index: int,
    waypoints: torch.Tensor,
    *,
    repair_defend: bool = False,
    repair_home_legality: bool = False,
) -> tuple[ProjectionCandidate, ...]:
    """Return every pre-frozen adapter candidate for one semantic branch.

    ``repair_defend`` and ``repair_home_legality`` implement
    DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1_SPEC.json. Both default OFF, and OFF
    reproduces the sealed PYQUATICUS_PORT_CONTRACT_RESULT.json candidate set
    exactly -- this function's DEFEND_INWARD/DEFEND_OUTWARD/
    ATTACK_HOME_TEAMMATE_CARRIER branches are the only place either flag is
    read; nothing else in this module or its callers changes behavior.
    """
    i = int(agent_index)
    if i < 0 or i >= state.n_agents:
        raise IndexError(i)
    branch = motion.branches[i]
    target_index = nearest_waypoint_index(motion.targets[i], waypoints)

    if branch == SemanticBranch.ATTACK_ENEMY_FLAG:
        return (
            ProjectionCandidate(
                label="GET_FLAG",
                macro=int(MacroAction.GET_FLAG),
                target_index=0,
                structurally_satisfiable=True,
                structural_note="GET_FLAG success is reachable by flag acquisition",
            ),
            ProjectionCandidate(
                label="GO_TO_ENEMY_FLAG_WAYPOINT",
                macro=int(MacroAction.GO_TO),
                target_index=target_index,
                structurally_satisfiable=True,
                structural_note="GO_TO success is reachable at the selected waypoint",
            ),
        )

    if branch == SemanticBranch.ATTACK_HOME_SELF_CARRIER:
        return (
            ProjectionCandidate(
                label="GO_HOME_SELF_CARRIER",
                macro=int(MacroAction.GO_HOME),
                target_index=0,
                structurally_satisfiable=True,
                structural_note="GO_HOME success is reachable by this carrier's capture",
            ),
            ProjectionCandidate(
                label="GO_TO_HOME_WAYPOINT_SELF_CARRIER",
                macro=int(MacroAction.GO_TO),
                target_index=target_index,
                structurally_satisfiable=True,
                structural_note="movement is overridden home while carrying; GO_TO completion remains interface-dependent",
            ),
        )

    if branch == SemanticBranch.ATTACK_HOME_TEAMMATE_CARRIER:
        return (
            ProjectionCandidate(
                label="GO_HOME_NONCARRIER",
                macro=int(MacroAction.GO_HOME),
                target_index=0,
                structurally_satisfiable=bool(repair_home_legality),
                structural_note=(
                    "REPAIR_B: the engine never blocks selecting GO_HOME for a "
                    "non-carrier -- _advance_blue_macros accepts any macro id "
                    "unconditionally. The prior False declaration reflected only "
                    "that GO_HOME's own capture-success flag (blue_cap_agents, "
                    "gpu_env/_core/_step.py) cannot fire for a non-carrier. This "
                    "repair recognizes that capture-eligibility is the wrong "
                    "satisfiability criterion for an escort assignment, whose "
                    "correctness is target/direction/trajectory fidelity -- "
                    "already exact -- not a capture event that was never this "
                    "agent's to trigger. GO_HOME's target resolution is untouched."
                    if repair_home_legality else
                    "GO_HOME success requires this agent to capture, impossible while it remains the non-carrier"
                ),
            ),
            ProjectionCandidate(
                label="GO_TO_HOME_WAYPOINT_NONCARRIER",
                macro=int(MacroAction.GO_TO),
                target_index=target_index,
                structurally_satisfiable=True,
                structural_note="GO_TO success is reachable at the selected home waypoint",
            ),
        )

    if branch in (SemanticBranch.DEFEND_INWARD, SemanticBranch.DEFEND_OUTWARD):
        waypoint_candidate = ProjectionCandidate(
            label="GO_TO_DEFENDER_SEMANTIC_TARGET",
            macro=int(MacroAction.GO_TO),
            target_index=target_index,
            structurally_satisfiable=True,
            structural_note="GO_TO success is reachable at the selected waypoint",
        )
        if not repair_defend:
            return (waypoint_candidate,)
        if branch == SemanticBranch.DEFEND_INWARD:
            semantic_candidate = ProjectionCandidate(
                label="DEFEND_FLAG_SEMANTIC_TARGET",
                macro=int(MacroAction.DEFEND_FLAG),
                target_index=0,
                structurally_satisfiable=True,
                structural_note="REPAIR_A: state-computed target = current own_flag_pos, resolved in _build_targets_from_action exactly as GET_FLAG/GO_HOME are",
            )
        else:
            semantic_candidate = ProjectionCandidate(
                label="DEFEND_OUTWARD_SEMANTIC_TARGET",
                macro=int(MacroAction.DEFEND_OUTWARD),
                target_index=0,
                structurally_satisfiable=True,
                structural_note="REPAIR_A: agent-relative outward ray to the arena boundary, recomputed every tick from own position; not nameable by any fixed coordinate",
            )
        return (semantic_candidate, waypoint_candidate)

    if branch == SemanticBranch.TAGGED_UPSTREAM_OVERRIDE:
        return (
            ProjectionCandidate(
                label="TAGGED_INTERFACE_IGNORED",
                macro=int(MacroAction.GO_TO),
                target_index=nearest_waypoint_index(state.own_flag_home, waypoints),
                structurally_satisfiable=True,
                structural_note="environment tagged override controls motion, independent of macro",
            ),
        )

    if branch == SemanticBranch.DEAD_LOCAL_ADAPTATION:
        return (
            ProjectionCandidate(
                label="DEAD_INTERFACE_IGNORED",
                macro=int(MacroAction.GO_TO),
                target_index=nearest_waypoint_index(state.positions[i], waypoints),
                structurally_satisfiable=True,
                structural_note="dead local adaptation suppresses motion, independent of macro",
            ),
        )

    raise ValueError(f"unsupported semantic branch {branch!r}")


def effective_projected_target(
    state: PortState,
    candidate: ProjectionCandidate,
    agent_index: int,
    waypoints: torch.Tensor,
) -> torch.Tensor:
    """Mirror the project's target resolution and tagged/dead overrides."""
    i = int(agent_index)
    macro = MacroAction(int(candidate.macro))
    if macro == MacroAction.GET_FLAG:
        target = state.enemy_flag_pos
    elif macro == MacroAction.GO_HOME:
        target = state.own_flag_home
    elif macro == MacroAction.DEFEND_FLAG:
        target = state.own_flag_pos
    elif macro == MacroAction.DEFEND_OUTWARD:
        away_unit = expected_away_direction(state, i)
        target = _ray_to_boundary(
            state.positions[i], away_unit, max_x=state.max_x, max_y=state.max_y,
        )
    else:
        target = waypoints[int(candidate.target_index)]

    if bool(state.carrying[i].item()):
        target = state.own_flag_home
    if bool(state.tagged[i].item()):
        target = state.own_flag_home
    if not bool(state.alive[i].item()):
        target = state.positions[i]
    return target.clone()


def direction_cosine(origin: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    lhs_direction = lhs - origin
    rhs_direction = rhs - origin
    lhs_norm = float(torch.linalg.vector_norm(lhs_direction))
    rhs_norm = float(torch.linalg.vector_norm(rhs_direction))
    if lhs_norm <= 1e-8 and rhs_norm <= 1e-8:
        return 1.0
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return -1.0
    value = torch.dot(lhs_direction / lhs_norm, rhs_direction / rhs_norm)
    return float(torch.clamp(value, -1.0, 1.0).item())


def roles_tensor(roles: Sequence[Role], *, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor([int(role) for role in roles], dtype=torch.int64, device=device)


def defender_radius_from_tag_range(tag_range_cells: float) -> float:
    return float(tag_range_cells) * (
        (UPSTREAM_FLAG_KEEPOUT_M + UPSTREAM_CATCH_RADIUS_M + UPSTREAM_DEFENDER_BUFFER_M)
        / UPSTREAM_CATCH_RADIUS_M
    )


def expected_away_direction(state: PortState, agent_index: int) -> torch.Tensor:
    """Analytical G4 direction, including the upstream zero-vector fallback."""
    i = int(agent_index)
    away = state.positions[i] - state.own_flag_pos
    away_unit = _unit(away)
    if not bool(torch.any(away_unit).item()):
        away_unit = torch.tensor(
            [math.cos(float(state.headings[i])), math.sin(float(state.headings[i]))],
            dtype=state.positions.dtype,
            device=state.positions.device,
        )
    return away_unit
