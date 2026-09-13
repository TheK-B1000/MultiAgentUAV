"""Known-answer anchors for gpu_env._core._entity_obs.build_entity_tensors.

Constructs tiny deterministic states (hand-set positions/alive/carrying/tagged)
and asserts the emitted entity tensors are EXACTLY the numbers the schema
promises -- per Rule 11 ("trace -> prove semantics -> golden test -> negative
control"), before this touches a real rollout.
"""

from __future__ import annotations

import math

import torch

from gpu_env._core._entity_obs import ENTITY_FEATURES, build_entity_tensors


def _make_core(n=4, device="cpu"):
    import experiments.strategic_demand_searcher as S
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    S.AGENTS = n
    cfg = GPUFieldConfig(n_envs=1, max_blue_agents=n, max_red_agents=n,
                         map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
                         aquaticus_profile=True, rules_profile="OURS", device=device,
                         seed=17600001, obstacle_obs_channel=True, tag_telemetry_enabled=True,
                         own_flag_home_required_to_score=True, **S.RULESET)
    env = GPUCTFVecEnv(cfg)
    env.reset()
    return env, env.core


def test_known_answer_relative_geometry():
    """agent A=(5,5), teammate B=(7,4), enemy C=(2,8): dx/dy/dist must be exact."""
    env, core = _make_core(n=4)
    try:
        core.blue_x[0, 0] = 5.0; core.blue_y[0, 0] = 5.0    # agent A, index 0
        core.blue_x[0, 1] = 7.0; core.blue_y[0, 1] = 4.0    # teammate B, index 1
        core.red_x[0, 2] = 2.0; core.red_y[0, 2] = 8.0      # enemy C, index 2

        d = build_entity_tensors(core, "blue")
        idx_b_slot = 0                    # within agent 0's teammate list, B is the first (index 1 -> slot 0)
        tm = d["teammates"][0, 0, idx_b_slot]
        assert math.isclose(float(tm[0]), 2.0, abs_tol=1e-5)    # dx = 7-5
        assert math.isclose(float(tm[1]), -1.0, abs_tol=1e-5)   # dy = 4-5
        assert math.isclose(float(tm[2]), math.hypot(2.0, -1.0), abs_tol=1e-5)

        en = d["enemies"][0, 0, 2]
        assert math.isclose(float(en[0]), -3.0, abs_tol=1e-5)   # dx = 2-5
        assert math.isclose(float(en[1]), 3.0, abs_tol=1e-5)    # dy = 8-5
        assert math.isclose(float(en[2]), math.hypot(-3.0, 3.0), abs_tol=1e-5)
    finally:
        env.close()


def test_self_exclusion_teammates_never_contains_self():
    """No teammate entry may correspond to the querying agent's own index --
    verified both by shape (N-1 slots) and by checking a distinct-position
    scenario never yields a spurious dx=0,dy=0 self-entry."""
    env, core = _make_core(n=4)
    try:
        xs = [1.0, 9.0, 3.0, 15.0]; ys = [2.0, 8.0, 14.0, 1.0]
        for i, (x, y) in enumerate(zip(xs, ys)):
            core.blue_x[0, i] = x; core.blue_y[0, i] = y
        d = build_entity_tensors(core, "blue")
        assert d["teammates"].shape[2] == 3          # N-1 = 3 slots at N=4
        for i in range(4):
            for slot in range(3):
                dx, dy = float(d["teammates"][0, i, slot, 0]), float(d["teammates"][0, i, slot, 1])
                assert not (abs(dx) < 1e-9 and abs(dy) < 1e-9), \
                    f"agent {i} slot {slot} looks like a self-entry (dx=dy=0)"
    finally:
        env.close()


def test_dead_carrying_tagged_pass_through_as_features_not_filters():
    """A dead/tagged/carrying teammate must still APPEAR as an entity (with
    those exact feature values) -- the smoke test never excluded such agents,
    only encoded their status as features."""
    env, core = _make_core(n=4)
    try:
        core.blue_alive[0, 1] = False
        core.blue_carrying[0, 1] = True
        core.blue_tagged[0, 1] = True
        core.red_alive[0, 2] = False
        core.red_carrying[0, 2] = True
        core.red_tagged[0, 2] = True

        d = build_entity_tensors(core, "blue")
        assert d["teammates_valid"][0, 0].all(), "dead teammate slot must stay VALID (present)"
        tm = d["teammates"][0, 0]                      # agent 0's teammate list
        # teammate index 1 sits at slot 0 (ascending, self excluded)
        assert float(tm[0, 3]) == 0.0                   # alive = False -> 0.0
        assert float(tm[0, 4]) == 1.0                   # carrying = True -> 1.0
        assert float(tm[0, 5]) == 1.0                   # tagged = True -> 1.0

        en = d["enemies"][0, 0, 2]
        assert float(en[3]) == 0.0 and float(en[4]) == 1.0 and float(en[5]) == 1.0
        assert d["enemies_valid"][0, 0].all(), "dead enemy slot must stay VALID (present)"
    finally:
        env.close()


def test_shapes_and_feature_width():
    env, core = _make_core(n=4)
    try:
        d = build_entity_tensors(core, "blue")
        B, N = 1, 4
        assert d["teammates"].shape == (B, N, N - 1, ENTITY_FEATURES)
        assert d["enemies"].shape == (B, N, N, ENTITY_FEATURES)
        assert d["teammates_valid"].shape == (B, N, N - 1)
        assert d["enemies_valid"].shape == (B, N, N)
        assert d["teammates_valid"].dtype == torch.bool
        assert d["enemies_valid"].dtype == torch.bool
    finally:
        env.close()


def test_grid_vec_agent_mask_untouched_by_entity_extraction():
    """The existing observation must remain bit-identical -- entity extraction
    is a pure addition, called separately, never mutating the core it reads."""
    env, core = _make_core(n=4)
    try:
        before = core.get_obs_tensors("blue")
        build_entity_tensors(core, "blue")
        after = core.get_obs_tensors("blue")
        assert torch.equal(before["grid"], after["grid"])
        assert torch.equal(before["vec"], after["vec"])
        assert torch.equal(before["agent_mask"], after["agent_mask"])
        assert torch.equal(before["mask"], after["mask"])
    finally:
        env.close()


def test_batched_matches_looped_reference_computation():
    """Cross-check the batched gather implementation against a naive
    Python-loop reference over several random states, at N=6."""
    import numpy as np
    env, core = _make_core(n=6)
    try:
        rng = np.random.default_rng(0)
        core.blue_x[0] = torch.tensor(rng.uniform(0, 20, 6), dtype=torch.float32)
        core.blue_y[0] = torch.tensor(rng.uniform(0, 20, 6), dtype=torch.float32)
        core.red_x[0] = torch.tensor(rng.uniform(0, 20, 6), dtype=torch.float32)
        core.red_y[0] = torch.tensor(rng.uniform(0, 20, 6), dtype=torch.float32)

        d = build_entity_tensors(core, "blue")
        bx = core.blue_x[0].tolist(); by = core.blue_y[0].tolist()
        rx = core.red_x[0].tolist(); ry = core.red_y[0].tolist()
        for i in range(6):
            expect_tm = [(bx[j] - bx[i], by[j] - by[i]) for j in range(6) if j != i]
            got_tm = [(float(d["teammates"][0, i, s, 0]), float(d["teammates"][0, i, s, 1]))
                     for s in range(5)]
            for (edx, edy), (gdx, gdy) in zip(expect_tm, got_tm):
                assert math.isclose(edx, gdx, abs_tol=1e-4)
                assert math.isclose(edy, gdy, abs_tol=1e-4)
            for k in range(6):
                gdx, gdy = float(d["enemies"][0, i, k, 0]), float(d["enemies"][0, i, k, 1])
                assert math.isclose(rx[k] - bx[i], gdx, abs_tol=1e-4)
                assert math.isclose(ry[k] - by[i], gdy, abs_tol=1e-4)
    finally:
        env.close()


def test_unsupported_side_raises_rather_than_silently_mirroring():
    env, core = _make_core(n=4)
    try:
        import pytest
        with pytest.raises(NotImplementedError):
            build_entity_tensors(core, "red")
    finally:
        env.close()
