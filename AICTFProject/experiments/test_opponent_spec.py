"""R0 regression tests — the opponent-genome training seam.

These must pass before a single R1 training step is consumed. They exist
because both failure modes are SILENT:

    "A" requested -> plain OP6 delivered        (undertrained wrong opponent)
    A override left set -> contaminates OP7     (wrong B pole)

Neither shows up in a loss curve. Both would invalidate the whole repertoire
ladder while producing healthy-looking runs.

Every assertion reads the LIVE resolved profile tensors that the behaviour tree
actually consumes, not the override attribute, because the attribute can be set
and still not reach the tree.

Run:  python experiments/test_opponent_spec.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.opponent_spec import (  # noqa: E402
    OpponentSpecError, assert_opponent_resolved, manifest_entry,
    pole_A_genome, pole_B_genome, resolved_profile_scalars, set_opponent_spec,
)
from experiments.sds_genome import canonical_parent  # noqa: E402
from experiments.strategic_demand_searcher import MAP, MAX_STEPS, RULESET  # noqa: E402

DEV = "cpu"


def build_env(n_envs: int = 2):
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    cfg = GPUFieldConfig(
        n_envs=n_envs, max_blue_agents=2, max_red_agents=2, map_set="train",
        map_layout=MAP, max_decision_steps=MAX_STEPS, aquaticus_profile=True,
        rules_profile="OURS", device=DEV, seed=4242, obstacle_obs_channel=True,
        tag_telemetry_enabled=True, own_flag_home_required_to_score=True,
        **RULESET)
    env = GPUCTFVecEnv(cfg)
    return env, env.core


def test_1_pole_A_resolves_overlay():
    """A training env must resolve min_alive_for_defender=2, not OP6's 3."""
    env, core = build_env()
    try:
        set_opponent_spec(env, core, "OP6", pole_A_genome(), context="pole A")
        got = resolved_profile_scalars(core)
        assert int(got["min_alive_for_defender"]) == 2, got["min_alive_for_defender"]
        print("  PASS  1. pole A resolves min_alive_for_defender=2")
    finally:
        env.close()


def test_2_plain_OP6_resolves_canonical():
    """Plain OP6 must resolve 3. If this reads 2, the overlay leaked."""
    env, core = build_env()
    try:
        set_opponent_spec(env, core, "OP6", None, context="plain OP6")
        got = resolved_profile_scalars(core)
        assert int(got["min_alive_for_defender"]) == 3, got["min_alive_for_defender"]
        print("  PASS  2. plain OP6 resolves min_alive_for_defender=3")
    finally:
        env.close()


def test_3_A_then_B_leaves_no_contamination():
    """The nasty inverse: a stale A override must not survive onto OP7.

    Checked on defender_zone_frac (OP6 0.35 vs OP7 0.05) because
    min_alive_for_defender is 2 for BOTH pole A and OP7 and would not detect it.
    """
    env, core = build_env()
    try:
        set_opponent_spec(env, core, "OP6", pole_A_genome(), context="A first")
        assert abs(resolved_profile_scalars(core)["defender_zone_frac"] - 0.35) < 1e-6
        set_opponent_spec(env, core, "OP7", pole_B_genome(), context="then B")
        got = resolved_profile_scalars(core)
        assert abs(got["defender_zone_frac"] - 0.05) < 1e-6, got["defender_zone_frac"]
        assert abs(got["threat_radius"] - 12.0) < 1e-6, got["threat_radius"]
        print("  PASS  3. A -> B leaves no A override on B")
    finally:
        env.close()


def test_4_B_then_A_restores_overlay():
    env, core = build_env()
    try:
        set_opponent_spec(env, core, "OP7", pole_B_genome(), context="B first")
        set_opponent_spec(env, core, "OP6", pole_A_genome(), context="then A")
        got = resolved_profile_scalars(core)
        assert int(got["min_alive_for_defender"]) == 2
        assert abs(got["defender_zone_frac"] - 0.35) < 1e-6
        print("  PASS  4. B -> A correctly restores the A overlay")
    finally:
        env.close()


def test_5_reset_preserves_intended_opponent():
    """Override must survive env.reset() and auto-reset across termination."""
    env, core = build_env()
    try:
        set_opponent_spec(env, core, "OP6", pole_A_genome(), context="pre-reset")
        env.reset()
        assert int(resolved_profile_scalars(core)["min_alive_for_defender"]) == 2, \
            "override lost on env.reset()"
        zero = np.zeros(2 * 2 * 2, dtype=np.float32)
        n_term = 0
        for _ in range(MAX_STEPS + 5):
            env.step_async(zero)
            _o, _r, d, _i = env.step_wait()
            if bool(np.asarray(d).any()):
                n_term += 1
                assert int(resolved_profile_scalars(core)["min_alive_for_defender"]) == 2, \
                    "override lost on auto-reset"
                if n_term >= 2:
                    break
        assert n_term >= 1, "no terminal reached; test did not exercise auto-reset"
        print(f"  PASS  5. reset + {n_term} auto-resets preserve the A overlay")
    finally:
        env.close()


def test_6_assertion_actually_fires():
    """A wrong resolution must raise, not warn.

    Without this, tests 1-5 only prove the happy path and a broken assertion
    would look identical to a working one.
    """
    env, core = build_env()
    try:
        set_opponent_spec(env, core, "OP6", None, context="plain OP6")
        raised = False
        try:
            # Claim this env is pole A when it is plain OP6.
            assert_opponent_resolved(core, "OP6", pole_A_genome(), context="negative control")
        except OpponentSpecError:
            raised = True
        assert raised, "assertion did NOT fire on a genuine mismatch"

        raised2 = False
        try:
            # Claim OP6-with-A-overlay is OP7.
            set_opponent_spec(env, core, "OP6", pole_A_genome())
            assert_opponent_resolved(core, "OP7", pole_B_genome(), context="negative control 2")
        except OpponentSpecError:
            raised2 = True
        assert raised2, "assertion did NOT fire on a cross-pole mismatch"
        print("  PASS  6. assertion fires on both mismatch directions")
    finally:
        env.close()


def test_7_manifest_records_resolved_profile():
    env, core = build_env()
    try:
        set_opponent_spec(env, core, "OP6", pole_A_genome())
        m = manifest_entry(core, "OP6", pole_A_genome())
        assert m["requested_overlay"] == {"min_alive_for_defender": 2}
        assert int(m["resolved_watch_fields"]["min_alive_for_defender"]) == 2
        assert m["live_opponent_key"] == ["OP6"]
        print("  PASS  7. manifest records the resolved profile")
    finally:
        env.close()


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    print("R0 opponent-seam regression tests")
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
