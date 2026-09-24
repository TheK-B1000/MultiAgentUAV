"""CPU-only executable contracts for the Pyquaticus behavioral-role port."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from gpu_env import BatchedCTFCore, GPUFieldConfig  # noqa: E402
from gpu_env._core._rules import _ray_to_boundary_batched  # noqa: E402
from gpu_env.pyquaticus_port import (  # noqa: E402
    DEFENDER_RADIUS_CELLS,
    UPSTREAM_COMMIT,
    UPSTREAM_SHA256,
    PortState,
    Role,
    SemanticBranch,
    _ray_to_boundary,
    _unit,
    defender_radius_from_tag_range,
    direction_cosine,
    effective_projected_target,
    expected_away_direction,
    projection_candidates,
    roles_tensor,
    true_motion,
)
from macro_actions import MacroAction  # noqa: E402


def _live_core() -> BatchedCTFCore:
    return BatchedCTFCore(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=4,
            max_red_agents=4,
            map_set="train",
            map_layout="map_a",
            device="cpu",
            seed=20260918,
        )
    )


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts/strategic_demand/sppo/PYQUATICUS_BEHAVIORAL_ROLE_PORT_SPEC.json"


def _state(
    *,
    positions=None,
    headings=None,
    own_flag_pos=(2.0, 10.0),
    own_flag_home=(2.0, 10.0),
    enemy_flag_pos=(17.0, 10.0),
    carrying=(False, False, False, False),
    tagged=(False, False, False, False),
    alive=(True, True, True, True),
) -> PortState:
    if positions is None:
        positions = ((5.0, 10.0), (6.0, 5.0), (6.0, 15.0), (7.0, 10.0))
    if headings is None:
        headings = (0.0, 0.0, 0.0, 0.0)
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


def _waypoints() -> torch.Tensor:
    core = BatchedCTFCore(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=4,
            max_red_agents=4,
            map_set="train",
            map_layout="map_a",
            device="cpu",
            seed=20260918,
        )
    )
    return core._macro_targets.detach().clone()


def test_provenance_constants_match_frozen_spec():
    doc = json.loads(SPEC.read_text(encoding="utf-8"))
    gate = doc["UPSTREAM_PROVENANCE_GATE"]
    assert gate["pinned_commit"] == UPSTREAM_COMMIT
    for item in gate["source_files"]:
        assert UPSTREAM_SHA256[item["path"]] == item["sha256"]
    excluded = gate["explicitly_excluded"]
    assert UPSTREAM_SHA256[excluded["path"]] == excluded["sha256"]


def test_g1_attack_no_carrier_targets_enemy_flag_for_every_attacker():
    state = _state()
    motion = true_motion(state, roles_tensor([Role.ATTACK] * 4))
    expected = state.enemy_flag_pos.expand(4, 2)
    torch.testing.assert_close(motion.targets, expected)
    assert set(motion.branches) == {SemanticBranch.ATTACK_ENEMY_FLAG}
    torch.testing.assert_close(motion.speed_fraction, torch.full((4,), 0.5))


def test_g2_any_friendly_carrier_targets_home_for_self_and_noncarriers():
    state = _state(carrying=(True, False, False, False))
    motion = true_motion(state, roles_tensor([Role.ATTACK] * 4))
    torch.testing.assert_close(motion.targets, state.own_flag_home.expand(4, 2))
    assert motion.branches[0] == SemanticBranch.ATTACK_HOME_SELF_CARRIER
    assert set(motion.branches[1:]) == {SemanticBranch.ATTACK_HOME_TEAMMATE_CARRIER}


def test_g2_global_carry_bit_changes_target_without_a_local_agent_change():
    baseline = _state()
    carried = _state(carrying=(False, False, True, False))
    roles = roles_tensor([Role.ATTACK] * 4)
    before = true_motion(baseline, roles)
    after = true_motion(carried, roles)
    # Agent 0's own local tuple is unchanged. Only teammate 2's global carry bit differs.
    torch.testing.assert_close(baseline.positions[0], carried.positions[0])
    assert bool(baseline.carrying[0]) is False and bool(carried.carrying[0]) is False
    torch.testing.assert_close(before.targets[0], baseline.enemy_flag_pos)
    torch.testing.assert_close(after.targets[0], carried.own_flag_home)


def test_g3_defend_outside_targets_current_flag_not_home():
    state = _state(
        positions=((10.0, 10.0), (10.0, 5.0), (10.0, 15.0), (9.0, 10.0)),
        own_flag_pos=(4.0, 10.0),
        own_flag_home=(2.0, 10.0),
    )
    motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
    torch.testing.assert_close(motion.targets, state.own_flag_pos.expand(4, 2))
    assert set(motion.branches) == {SemanticBranch.DEFEND_INWARD}


def test_g4_defend_inside_and_boundary_point_radially_away():
    state = _state(
        positions=((4.0, 10.0), (2.0, 12.0), (2.0, 8.0), (5.5, 10.0)),
    )
    motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
    assert set(motion.branches) == {SemanticBranch.DEFEND_OUTWARD}
    for i in range(4):
        actual = motion.targets[i] - state.positions[i]
        expected = expected_away_direction(state, i)
        assert float(torch.dot(actual, expected)) > 0.0
        assert direction_cosine(state.positions[i], motion.targets[i], state.positions[i] + expected) >= 0.999999


def test_g4_exact_flag_center_uses_current_heading_fallback():
    state = _state(
        positions=((2.0, 10.0), (6.0, 5.0), (6.0, 15.0), (7.0, 10.0)),
        headings=(torch.pi / 2, 0.0, 0.0, 0.0),
    )
    roles = roles_tensor([Role.DEFEND, Role.ATTACK, Role.ATTACK, Role.ATTACK])
    motion = true_motion(state, roles)
    direction = motion.targets[0] - state.positions[0]
    assert abs(float(direction[0])) < 1e-5
    assert float(direction[1]) > 0.0


def test_g5_tagged_override_and_dead_local_adaptation_are_distinct():
    state = _state(tagged=(True, False, False, False), alive=(True, False, True, True))
    roles = roles_tensor([Role.ATTACK, Role.DEFEND, Role.ATTACK, Role.DEFEND])
    motion = true_motion(state, roles)
    assert motion.branches[0] == SemanticBranch.TAGGED_UPSTREAM_OVERRIDE
    torch.testing.assert_close(motion.targets[0], state.own_flag_home)
    assert float(motion.speed_fraction[0]) == pytest.approx(1.0)
    assert motion.branches[1] == SemanticBranch.DEAD_LOCAL_ADAPTATION
    torch.testing.assert_close(motion.targets[1], state.positions[1])
    assert float(motion.speed_fraction[1]) == pytest.approx(0.0)


def test_g5_local_reset_clears_tagged_and_carrying_and_revives_roster():
    core = BatchedCTFCore(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=4,
            max_red_agents=4,
            map_set="train",
            map_layout="map_a",
            device="cpu",
            seed=20260918,
        )
    )
    core.blue_tagged.fill_(True)
    core.blue_carrying.fill_(True)
    core.blue_alive.fill_(False)
    core.reset_all()
    assert not bool(core.blue_tagged.any())
    assert not bool(core.blue_carrying.any())
    assert bool(core.blue_alive.all())


def test_g6_noncarrier_go_home_is_rejected_before_outcomes():
    state = _state(carrying=(True, False, False, False))
    motion = true_motion(state, roles_tensor([Role.ATTACK] * 4))
    candidates = projection_candidates(state, motion, 1, _waypoints())
    by_label = {candidate.label: candidate for candidate in candidates}
    assert by_label["GO_HOME_NONCARRIER"].macro == int(MacroAction.GO_HOME)
    assert by_label["GO_HOME_NONCARRIER"].structurally_satisfiable is False
    assert by_label["GO_TO_HOME_WAYPOINT_NONCARRIER"].structurally_satisfiable is True


@pytest.mark.parametrize(
    ("carrying", "agent_index", "candidate_label"),
    [
        ((False, False, False, False), 0, "GET_FLAG"),
        ((True, False, False, False), 0, "GO_HOME_SELF_CARRIER"),
        ((True, False, False, False), 1, "GO_TO_HOME_WAYPOINT_NONCARRIER"),
    ],
)
def test_g6_effective_target_matches_live_core_resolution(carrying, agent_index, candidate_label):
    core = BatchedCTFCore(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=4,
            max_red_agents=4,
            map_set="train",
            map_layout="map_a",
            device="cpu",
            seed=20260918,
        )
    )
    state = _state(carrying=carrying)
    core.blue_x[0] = state.positions[:, 0]
    core.blue_y[0] = state.positions[:, 1]
    core.blue_flag_home[0] = state.own_flag_home
    core.blue_flag_pos[0] = state.own_flag_pos
    core.red_flag_pos[0] = state.enemy_flag_pos
    core.blue_carrying[0] = state.carrying
    core.blue_tagged[0] = state.tagged
    core.blue_alive[0] = state.alive

    motion = true_motion(state, roles_tensor([Role.ATTACK] * 4))
    candidates = projection_candidates(state, motion, agent_index, core._macro_targets)
    candidate = next(item for item in candidates if item.label == candidate_label)
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    macros[0, agent_index] = candidate.macro
    targets[0, agent_index] = candidate.target_index
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    tx, ty, _, _ = core._redirect_tagged_to_home(tx, ty, tx.clone(), ty.clone())
    actual = torch.stack([tx[0, agent_index], ty[0, agent_index]])
    expected = effective_projected_target(state, candidate, agent_index, core._macro_targets)
    torch.testing.assert_close(actual, expected)


def test_defender_radius_is_pinned_to_upstream_catch_radius_ratio():
    assert DEFENDER_RADIUS_CELLS == pytest.approx(3.5)
    assert defender_radius_from_tag_range(2.5) == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1: known-answer contracts (Rule 12).
# repair_defend / repair_home_legality both default False everywhere below
# unless explicitly enabled, so every test above this line is unaffected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("px", "py", "ux", "uy", "expect"),
    [
        (5.0, 10.0, 1.0, 0.0, (18.0, 10.0)),
        (5.0, 10.0, -1.0, 0.0, (0.0, 10.0)),
        (5.0, 10.0, 0.0, 1.0, (5.0, 18.0)),
        (5.0, 10.0, 0.0, -1.0, (5.0, 0.0)),
        (5.0, 5.0, 0.7071067811865476, 0.7071067811865476, None),  # diagonal, cross-checked below
    ],
)
def test_ray_to_boundary_batched_matches_scalar_reference(px, py, ux, uy, expect):
    scalar = _ray_to_boundary(
        torch.tensor([px, py]), torch.tensor([ux, uy]), max_x=18.0, max_y=18.0,
    )
    batched_x, batched_y = _ray_to_boundary_batched(
        torch.tensor([[px]]), torch.tensor([[py]]),
        torch.tensor([[ux]]), torch.tensor([[uy]]),
        max_x=18.0, max_y=18.0,
    )
    torch.testing.assert_close(torch.tensor([float(batched_x[0, 0]), float(batched_y[0, 0])]), scalar)
    if expect is not None:
        torch.testing.assert_close(scalar, torch.tensor(expect))


def test_ray_to_boundary_batched_zero_direction_returns_origin():
    x, y = _ray_to_boundary_batched(
        torch.tensor([[5.0]]), torch.tensor([[10.0]]),
        torch.tensor([[0.0]]), torch.tensor([[0.0]]),
        max_x=18.0, max_y=18.0,
    )
    assert float(x[0, 0]) == pytest.approx(5.0)
    assert float(y[0, 0]) == pytest.approx(10.0)


def test_ray_to_boundary_batched_vectorizes_over_multiple_agents_independently():
    px = torch.tensor([[5.0, 2.0, 9.0]])
    py = torch.tensor([[10.0, 2.0, 16.0]])
    ux = torch.tensor([[1.0, -1.0, 0.0]])
    uy = torch.tensor([[0.0, 0.0, 1.0]])
    bx, by = _ray_to_boundary_batched(px, py, ux, uy, max_x=18.0, max_y=18.0)
    for i in range(3):
        sx, sy = _ray_to_boundary(
            torch.tensor([float(px[0, i]), float(py[0, i])]),
            torch.tensor([float(ux[0, i]), float(uy[0, i])]),
            max_x=18.0, max_y=18.0,
        )
        assert float(bx[0, i]) == pytest.approx(float(sx))
        assert float(by[0, i]) == pytest.approx(float(sy))


def test_defend_flag_macro_targets_current_own_flag_pos_not_home_live_engine():
    """The G3 candidate-set artifact: own_flag_pos and own_flag_home coincide in
    the frozen fixture, which is why GO_HOME would have looked exact there too.
    This test uses a DISPLACED flag specifically so the two cannot be confused."""
    core = _live_core()
    core.blue_x[0] = torch.tensor([9.0, 0.0, 0.0, 0.0])
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    core.blue_flag_pos[0] = torch.tensor([4.0, 10.0])  # stolen: away from home
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(MacroAction.DEFEND_FLAG)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    torch.testing.assert_close(torch.stack([tx[0, 0], ty[0, 0]]), torch.tensor([4.0, 10.0]))


def test_defend_outward_macro_matches_scalar_ray_to_boundary_live_engine():
    core = _live_core()
    core.blue_x[0] = torch.tensor([4.0, 0.0, 0.0, 0.0])
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_flag_pos[0] = torch.tensor([2.0, 10.0])
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(MacroAction.DEFEND_OUTWARD)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    away_unit = _unit(torch.tensor([4.0, 10.0]) - torch.tensor([2.0, 10.0]))
    max_x = float(max(0, core.cols - 1))
    max_y = float(max(0, core.rows - 1))
    expected = _ray_to_boundary(torch.tensor([4.0, 10.0]), away_unit, max_x=max_x, max_y=max_y)
    torch.testing.assert_close(torch.stack([tx[0, 0], ty[0, 0]]), expected)


def test_defend_outward_zero_vector_uses_heading_fallback_live_engine():
    core = _live_core()
    core.blue_x[0] = torch.tensor([2.0, 0.0, 0.0, 0.0])
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_heading[0] = torch.tensor([math.pi / 2, 0.0, 0.0, 0.0])
    core.blue_flag_pos[0] = torch.tensor([2.0, 10.0])  # agent exactly on the flag
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(MacroAction.DEFEND_OUTWARD)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    assert abs(float(tx[0, 0]) - 2.0) < 1e-4
    assert float(ty[0, 0]) > 10.0


@pytest.mark.parametrize("macro", [MacroAction.DEFEND_FLAG, MacroAction.DEFEND_OUTWARD])
def test_defend_macros_still_obey_the_carrying_override_live_engine(macro):
    """Carrying forces home regardless of assigned macro -- a pre-existing
    invariant. The new branches must sit before that override, not replace it."""
    core = _live_core()
    core.blue_x[0] = torch.tensor([9.0, 0.0, 0.0, 0.0])
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_flag_pos[0] = torch.tensor([4.0, 10.0])
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    core.blue_carrying[0] = torch.tensor([True, False, False, False])
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(macro)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    torch.testing.assert_close(torch.stack([tx[0, 0], ty[0, 0]]), torch.tensor([2.0, 10.0]))


def test_repair_off_reproduces_the_single_legacy_defend_candidate():
    state = _state(
        positions=((10.0, 10.0), (10.0, 5.0), (10.0, 15.0), (9.0, 10.0)),
        own_flag_pos=(4.0, 10.0),
    )
    motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
    candidates = projection_candidates(state, motion, 0, _waypoints())
    assert [c.label for c in candidates] == ["GO_TO_DEFENDER_SEMANTIC_TARGET"]


@pytest.mark.parametrize(
    ("branch_positions", "expected_semantic_label"),
    [
        (((10.0, 10.0), (10.0, 5.0), (10.0, 15.0), (9.0, 10.0)), "DEFEND_FLAG_SEMANTIC_TARGET"),  # DEFEND_INWARD
        (((4.0, 10.0), (2.0, 12.0), (2.0, 8.0), (5.5, 10.0)), "DEFEND_OUTWARD_SEMANTIC_TARGET"),  # DEFEND_OUTWARD
    ],
)
def test_repair_on_adds_semantic_candidate_and_keeps_legacy_waypoint(branch_positions, expected_semantic_label):
    state = _state(positions=branch_positions, own_flag_pos=(4.0, 10.0))
    motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
    candidates = projection_candidates(state, motion, 0, _waypoints(), repair_defend=True)
    labels = [c.label for c in candidates]
    assert expected_semantic_label in labels
    assert "GO_TO_DEFENDER_SEMANTIC_TARGET" in labels
    semantic = next(c for c in candidates if c.label == expected_semantic_label)
    assert semantic.structurally_satisfiable is True
    assert semantic.macro in (int(MacroAction.DEFEND_FLAG), int(MacroAction.DEFEND_OUTWARD))


def test_repair_home_legality_flips_only_the_satisfiability_flag():
    state = _state(carrying=(True, False, False, False))
    motion = true_motion(state, roles_tensor([Role.ATTACK] * 4))
    off = {c.label: c for c in projection_candidates(state, motion, 1, _waypoints())}
    on = {c.label: c for c in projection_candidates(state, motion, 1, _waypoints(), repair_home_legality=True)}
    assert off["GO_HOME_NONCARRIER"].structurally_satisfiable is False
    assert on["GO_HOME_NONCARRIER"].structurally_satisfiable is True
    assert on["GO_HOME_NONCARRIER"].macro == off["GO_HOME_NONCARRIER"].macro == int(MacroAction.GO_HOME)
    assert on["GO_HOME_NONCARRIER"].target_index == off["GO_HOME_NONCARRIER"].target_index == 0
    # legality repair must not touch the sibling waypoint candidate at all
    assert on["GO_TO_HOME_WAYPOINT_NONCARRIER"] == off["GO_TO_HOME_WAYPOINT_NONCARRIER"]


@pytest.mark.parametrize(
    ("branch_positions", "candidate_label"),
    [
        (((10.0, 10.0), (10.0, 5.0), (10.0, 15.0), (9.0, 10.0)), "DEFEND_FLAG_SEMANTIC_TARGET"),
        (((4.0, 10.0), (2.0, 12.0), (2.0, 8.0), (5.5, 10.0)), "DEFEND_OUTWARD_SEMANTIC_TARGET"),
    ],
)
def test_effective_projected_target_matches_live_engine_for_repaired_defend(branch_positions, candidate_label):
    """The exact assertion _probe_candidate relies on (AssertionError on mismatch)."""
    core = _live_core()
    state = _state(positions=branch_positions, own_flag_pos=(4.0, 10.0))
    core.blue_x[0] = state.positions[:, 0]
    core.blue_y[0] = state.positions[:, 1]
    core.blue_flag_pos[0] = state.own_flag_pos
    core.blue_flag_home[0] = state.own_flag_home
    core.red_flag_pos[0] = state.enemy_flag_pos

    motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
    candidates = projection_candidates(state, motion, 0, core._macro_targets, repair_defend=True)
    candidate = next(c for c in candidates if c.label == candidate_label)
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    macros[0, 0] = candidate.macro
    targets[0, 0] = candidate.target_index
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    tx, ty, _, _ = core._redirect_tagged_to_home(tx, ty, tx.clone(), ty.clone())
    actual = torch.stack([tx[0, 0], ty[0, 0]])
    expected = effective_projected_target(state, candidate, 0, core._macro_targets)
    torch.testing.assert_close(actual, expected)


def test_repair_flags_do_not_alter_unrelated_branches():
    """Orthogonality: ATTACK-to-flag, self-carrier, tagged and dead must be
    byte-identical whether or not either repair flag is set."""
    waypoints = _waypoints()

    attack_state = _state()
    attack_motion = true_motion(attack_state, roles_tensor([Role.ATTACK] * 4))
    off = projection_candidates(attack_state, attack_motion, 0, waypoints)
    on = projection_candidates(
        attack_state, attack_motion, 0, waypoints,
        repair_defend=True, repair_home_legality=True,
    )
    assert off == on

    self_carrier_state = _state(carrying=(True, False, False, False))
    self_carrier_motion = true_motion(self_carrier_state, roles_tensor([Role.ATTACK] * 4))
    off = projection_candidates(self_carrier_state, self_carrier_motion, 0, waypoints)
    on = projection_candidates(
        self_carrier_state, self_carrier_motion, 0, waypoints,
        repair_defend=True, repair_home_legality=True,
    )
    assert off == on

    tagged_state = _state(tagged=(True, False, False, False))
    tagged_motion = true_motion(tagged_state, roles_tensor([Role.ATTACK, Role.DEFEND, Role.ATTACK, Role.DEFEND]))
    off = projection_candidates(tagged_state, tagged_motion, 0, waypoints)
    on = projection_candidates(
        tagged_state, tagged_motion, 0, waypoints,
        repair_defend=True, repair_home_legality=True,
    )
    assert off == on


# ---------------------------------------------------------------------------
# DEFEND_SEMANTIC_COMMITMENT_V2: known-answer contracts (Rule 12).
# unified_defend defaults False everywhere above this line, so nothing above
# is affected. MacroAction.DEFEND is reachable only via this explicit flag.
# ---------------------------------------------------------------------------

def test_defend_radius_helper_matches_port_module():
    from gpu_env._core._rules import _pyquaticus_defender_radius_cells
    for tag_range in (1.0, 2.5, 4.0, 10.0):
        assert _pyquaticus_defender_radius_cells(tag_range) == pytest.approx(
            defender_radius_from_tag_range(tag_range)
        )
    # the live core's default tag_range_cells must reproduce the analytical
    # module's DEFENDER_RADIUS_CELLS constant, since effective_projected_target
    # (analytical) and the live engine must threshold at the same radius
    core = _live_core()
    assert _pyquaticus_defender_radius_cells(float(core.cfg.tag_range_cells)) == pytest.approx(
        DEFENDER_RADIUS_CELLS
    )


def test_unified_defend_matches_inward_target_when_outside_radius_live_engine():
    core = _live_core()
    core.blue_x[0] = torch.tensor([9.0, 0.0, 0.0, 0.0])  # 7 cells out, radius 3.5
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_flag_pos[0] = torch.tensor([2.0, 10.0])
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(MacroAction.DEFEND)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    torch.testing.assert_close(torch.stack([tx[0, 0], ty[0, 0]]), torch.tensor([2.0, 10.0]))


def test_unified_defend_matches_outward_target_when_inside_radius_live_engine():
    core = _live_core()
    core.blue_x[0] = torch.tensor([4.0, 0.0, 0.0, 0.0])  # 2 cells in, radius 3.5
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_flag_pos[0] = torch.tensor([2.0, 10.0])
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(MacroAction.DEFEND)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    away_unit = _unit(torch.tensor([4.0, 10.0]) - torch.tensor([2.0, 10.0]))
    max_x = float(max(0, core.cols - 1))
    max_y = float(max(0, core.rows - 1))
    expected = _ray_to_boundary(torch.tensor([4.0, 10.0]), away_unit, max_x=max_x, max_y=max_y)
    torch.testing.assert_close(torch.stack([tx[0, 0], ty[0, 0]]), expected)


def test_unified_defend_boundary_convention_exactly_at_radius_is_outward():
    """true_motion: distance > R is INWARD; distance <= R (inclusive) is
    OUTWARD. The unified macro must match this exactly at the boundary."""
    core = _live_core()
    radius = DEFENDER_RADIUS_CELLS
    core.blue_x[0] = torch.tensor([2.0 + radius, 0.0, 0.0, 0.0])  # exactly R away
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_flag_pos[0] = torch.tensor([2.0, 10.0])
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(MacroAction.DEFEND)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    # OUTWARD at exactly R means the target must NOT equal own_flag_pos
    assert not (abs(float(tx[0, 0]) - 2.0) < 1e-6 and abs(float(ty[0, 0]) - 10.0) < 1e-6)


def test_unified_defend_still_obeys_carrying_override_live_engine():
    core = _live_core()
    core.blue_x[0] = torch.tensor([9.0, 0.0, 0.0, 0.0])
    core.blue_y[0] = torch.tensor([10.0, 0.0, 0.0, 0.0])
    core.blue_flag_pos[0] = torch.tensor([4.0, 10.0])
    core.blue_flag_home[0] = torch.tensor([2.0, 10.0])
    core.blue_carrying[0] = torch.tensor([True, False, False, False])
    macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
    macros[0, 0] = int(MacroAction.DEFEND)
    targets = torch.zeros((1, 4), dtype=torch.int64)
    tx, ty = core._build_targets_from_action(macros, targets, side="blue")
    torch.testing.assert_close(torch.stack([tx[0, 0], ty[0, 0]]), torch.tensor([2.0, 10.0]))


@pytest.mark.parametrize(
    "branch_positions",
    [
        ((10.0, 10.0), (10.0, 5.0), (10.0, 15.0), (9.0, 10.0)),  # DEFEND_INWARD
        ((4.0, 10.0), (2.0, 12.0), (2.0, 8.0), (5.5, 10.0)),      # DEFEND_OUTWARD
    ],
)
def test_unified_defend_candidate_offered_for_both_branches_same_label(branch_positions):
    """The whole point of V2: ONE label/macro spans both native branches, so a
    committed candidate never needs the branch-flip fallback V1 required."""
    state = _state(positions=branch_positions, own_flag_pos=(4.0, 10.0))
    motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
    candidates = projection_candidates(state, motion, 0, _waypoints(), unified_defend=True)
    labels = [c.label for c in candidates]
    assert "DEFEND_UNIFIED_SEMANTIC_TARGET" in labels
    unified = next(c for c in candidates if c.label == "DEFEND_UNIFIED_SEMANTIC_TARGET")
    assert unified.macro == int(MacroAction.DEFEND)


def test_unified_defend_composes_with_repair_defend_and_home_legality():
    """Orthogonality: unified_defend must not remove or alter the V1 or home
    candidates when both flags are combined."""
    state = _state(
        positions=((10.0, 10.0), (10.0, 5.0), (10.0, 15.0), (9.0, 10.0)),
        own_flag_pos=(4.0, 10.0),
    )
    motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
    solo = {c.label for c in projection_candidates(state, motion, 0, _waypoints(), repair_defend=True)}
    combined = {
        c.label for c in projection_candidates(
            state, motion, 0, _waypoints(), repair_defend=True, unified_defend=True,
        )
    }
    assert combined == solo | {"DEFEND_UNIFIED_SEMANTIC_TARGET"}


def test_effective_projected_target_matches_live_engine_for_unified_defend_across_branches():
    """The decisive C3 check: analytical mirror vs live engine, exact match,
    at fixtures stable-inward, stable-outward, and (this is the point) the
    original boundary-crossing G3/G4 fixtures where the branch actually flips
    mid-probe. Uses the raw resolver directly per tick, not the 16-tick probe,
    to isolate resolution-parity from the commit-horizon trajectory question."""
    core = _live_core()
    fixtures = {
        "stable_inward": ((12.0, 10.0), (12.0, 5.0), (12.0, 15.0), (11.0, 10.0)),
        "stable_outward": ((3.0, 10.0), (2.0, 11.0), (2.0, 9.0), (3.5, 10.0)),
        "G3_boundary": ((10.0, 10.0), (10.0, 5.0), (10.0, 15.0), (9.0, 10.0)),
        "G4_boundary": ((4.0, 10.0), (2.0, 12.0), (2.0, 8.0), (5.5, 10.0)),
    }
    for name, positions in fixtures.items():
        state = _state(positions=positions, own_flag_pos=(4.0, 10.0))
        core.blue_x[0] = state.positions[:, 0]
        core.blue_y[0] = state.positions[:, 1]
        core.blue_flag_pos[0] = state.own_flag_pos
        core.blue_flag_home[0] = state.own_flag_home
        core.red_flag_pos[0] = state.enemy_flag_pos
        motion = true_motion(state, roles_tensor([Role.DEFEND] * 4))
        candidates = projection_candidates(state, motion, 0, core._macro_targets, unified_defend=True)
        candidate = next(c for c in candidates if c.label == "DEFEND_UNIFIED_SEMANTIC_TARGET")
        macros = torch.full((1, 4), int(MacroAction.GO_TO), dtype=torch.int64)
        targets = torch.zeros((1, 4), dtype=torch.int64)
        macros[0, 0] = candidate.macro
        targets[0, 0] = candidate.target_index
        tx, ty = core._build_targets_from_action(macros, targets, side="blue")
        tx, ty, _, _ = core._redirect_tagged_to_home(tx, ty, tx.clone(), ty.clone())
        actual = torch.stack([tx[0, 0], ty[0, 0]])
        expected = effective_projected_target(state, candidate, 0, core._macro_targets)
        torch.testing.assert_close(actual, expected, msg=f"mismatch at fixture {name!r}")
        # and it must equal the TRUE oracle's own target for this same state --
        # the entire predicted-0.000 hypothesis in one assertion
        torch.testing.assert_close(actual, motion.targets[0], msg=f"diverges from oracle at fixture {name!r}")
