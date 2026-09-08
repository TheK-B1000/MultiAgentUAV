"""The C1 injector must actually produce C1, and the batched predicate must
agree with the one C1 was confirmed under.

Two failure modes are worth a test rather than a code review.

1. ``c1_active_mask`` is a hand-rolled batched copy of the ``home_threatened``
   arithmetic inside ``run_g0_v2_evaluation.legal_context``. A copy can drift.
   If it does, training would select on a predicate that is not the confirmed
   one, and every downstream O1 gate would silently be measuring something
   else.

2. ``apply_c1_scenario`` writes raw core tensors. Nothing in the engine
   enforces that the result is a state C1 describes, and the neighbouring
   ``v6i26_phase_pods`` injector demonstrates exactly how this goes wrong: its
   clock line guards on ``core.decision_step``, which does not exist, so it
   silently never ran.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = pytest.importorskip("torch")

from experiments.c1_context import (  # noqa: E402
    C1_BLUE_SCORE,
    C1_RED_SCORE,
    apply_c1_scenario,
    c1_active_from_context,
    c1_active_mask,
)

CANONICAL_MAP = "map_a"
EPISODE_HORIZON = 240
AGENTS = 2
V2_RULES = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)


def _env(n_envs: int = 4, seed: int = 4_242):
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=n_envs,
        max_blue_agents=AGENTS,
        max_red_agents=AGENTS,
        map_set="train",
        map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=seed,
        obstacle_obs_channel=True,
        **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    env.reset()
    return env


# --- the predicate ----------------------------------------------------------


def test_batched_predicate_matches_the_confirmed_one():
    """c1_active_mask[0] must equal c1_active_from_context(legal_context(core)).

    ``legal_context`` reads env 0 only, so env 0 is the comparison point. States
    are randomised over the terms that matter -- score, red possession, flag
    displacement and both sides' distance to the blue flag -- so the check
    covers every branch of ``home_threatened``, not just the easy one where red
    is carrying.
    """
    from experiments.run_g0_v2_evaluation import legal_context

    env = _env(n_envs=1)
    core = env.core
    g = torch.Generator().manual_seed(20260804)

    agreements = 0
    seen_true = seen_false = 0
    try:
        for _ in range(120):
            cols, rows = float(core.cols), float(core.rows)
            core.blue_score[0] = int(torch.randint(0, 3, (1,), generator=g))
            core.red_score[0] = int(torch.randint(0, 3, (1,), generator=g))
            core.red_carrying[0] = torch.rand(AGENTS, generator=g) < 0.25
            core.blue_carrying[0] = torch.rand(AGENTS, generator=g) < 0.25
            core.blue_x[0] = torch.rand(AGENTS, generator=g) * cols
            core.blue_y[0] = torch.rand(AGENTS, generator=g) * rows
            core.red_x[0] = torch.rand(AGENTS, generator=g) * cols
            core.red_y[0] = torch.rand(AGENTS, generator=g) * rows
            # Flag either at home or displaced, so flag_away exercises both ways.
            if bool(torch.rand(1, generator=g) < 0.5):
                core.blue_flag_pos[0] = core.blue_flag_home[0]
            else:
                core.blue_flag_pos[0, 0] = float(torch.rand(1, generator=g)) * cols
                core.blue_flag_pos[0, 1] = float(torch.rand(1, generator=g)) * rows

            scalar = c1_active_from_context(legal_context(core))
            batched = bool(c1_active_mask(core)[0].item())
            assert scalar == batched, (
                "batched C1 predicate drifted from the confirmed definition: "
                f"legal_context={scalar} mask={batched}"
            )
            agreements += 1
            seen_true += int(scalar)
            seen_false += int(not scalar)
    finally:
        env.close()

    assert agreements == 120
    # A test that only ever saw one verdict would pass with a constant.
    assert seen_true >= 10 and seen_false >= 10, (
        f"randomisation was degenerate: {seen_true} true / {seen_false} false"
    )


# --- the injector -----------------------------------------------------------


def test_injected_state_satisfies_c1():
    env = _env(n_envs=8)
    core = env.core
    try:
        apply_c1_scenario(core)
        assert bool(c1_active_mask(core).all()), "injected state is not C1"
    finally:
        env.close()


def test_injection_matches_the_frozen_construction():
    """Each clause of C1_PROPOSAL's "recreatable from valid states" line."""
    env = _env(n_envs=16)
    core = env.core
    try:
        apply_c1_scenario(core)
        mid_x = 0.5 * float(core.cols)

        # BLUE ahead on score, and not close enough to score_limit to end.
        assert torch.all(core.blue_score == C1_BLUE_SCORE)
        assert torch.all(core.red_score == C1_RED_SCORE)
        assert C1_BLUE_SCORE < int(core.score_limit)

        # RED carrying the blue flag, with the flag actually on the carrier.
        assert torch.all(core.red_carrying.any(dim=1))
        carrier = torch.argmax(core.red_carrying.to(torch.int64), dim=1)
        rows = torch.arange(core.B)
        assert torch.allclose(core.blue_flag_pos[:, 0], core.red_x[rows, carrier])
        assert torch.allclose(core.blue_flag_pos[:, 1], core.red_y[rows, carrier])

        # Both BLUE agents past the midline -- the clause that distinguishes C1
        # from the v6i26 defend_lead pod, which parks blue at its own flag.
        assert torch.all(core.blue_x > mid_x), (
            "C1 requires both blue agents past the midline; this is the clause "
            "POD_DEFEND_LEAD violates"
        )

        # RED defender off cooldown.
        assert torch.all(core.red_tag_cooldown <= 0.0)

        # Nobody starts tagged or dead.
        assert torch.all(core.blue_alive) and torch.all(core.red_alive)
        assert not torch.any(core.blue_tagged) and not torch.any(core.red_tagged)
    finally:
        env.close()


def test_injection_is_randomised_not_a_single_memorised_state():
    env = _env(n_envs=32)
    core = env.core
    try:
        apply_c1_scenario(core)
        for name, t in (
            ("red carrier x", core.red_x[:, 0]),
            ("blue x", core.blue_x[:, 0]),
            ("blue y", core.blue_y[:, 0]),
        ):
            assert float(t.std()) > 1e-3, f"{name} is constant across envs"
    finally:
        env.close()


def test_clock_is_left_alone():
    """step_count stays 0 -- a declared artifact, not an oversight.

    Asserted so that a later "improvement" that sets a late clock has to change
    this test, and therefore the preregistration, rather than slipping in.
    """
    env = _env(n_envs=4)
    core = env.core
    try:
        apply_c1_scenario(core)
        assert torch.all(core.step_count == 0)
    finally:
        env.close()


def test_injected_state_survives_a_step():
    """The engine must not immediately undo the injection.

    If a flag-return or grace-period rule reset the carry on the first step, the
    scenario would silently decay to ordinary play and O1 would train on the
    wrong distribution.
    """
    import numpy as np

    env = _env(n_envs=4)
    core = env.core
    try:
        apply_c1_scenario(core)
        n_actions = AGENTS * 2 * core.B
        env.step_async(np.zeros((n_actions,), dtype=np.int64))
        env.step_wait()
        assert bool(c1_active_mask(core).all()), (
            "C1 did not survive one step: the engine reverted the injected state"
        )
    finally:
        env.close()
