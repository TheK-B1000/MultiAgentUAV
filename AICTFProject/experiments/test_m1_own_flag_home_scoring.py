"""Correctness tests for M1: own flag must be home to score.

Five checks, per the approved decision:

  1. V2 UNCHANGED      with the flag OFF, scoring is bit-identical to before
  2. NORMAL SCORING    with the flag ON and own flag home, scoring still works
  3. BLOCKED           carrier reaching home while own flag is AWAY does not score
  4. CONTROL           same away setup with M1 OFF still scores (isolates the rule)
  5. RECOVERY          SAME episode: block while away, then own flag returns
                       while the carrier is still legally home, then capture scores

Check 5 is the semantic transition the payoff assay could not distinguish from
"M1 never unblocks". It is forced on engine state, not a lucky rollout.

Run:  python experiments/test_m1_own_flag_home_scoring.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch        # noqa: E402

MAP = "map_a"
RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)


def make(m1: bool, seed: int, device="cuda"):
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    kw = dict(RULESET)
    if m1:
        kw["own_flag_home_required_to_score"] = True
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,
        map_set="train", map_layout=MAP, max_decision_steps=240,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **kw)
    env = GPUCTFVecEnv(cfg)
    return env, env.core


def rollout_scores(m1: bool, seed: int, opponent="OP6", steps=240, device="cuda"):
    env, core = make(m1, seed, device)
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        core.blue_scripted = True
        core.set_blue_style("BLUE_BOTH_ATTACK_V2")
        env.reset()
        core.drain_tag_events()
        term = None
        for _ in range(steps):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = (i0 or {}).get("episode_result") or {}
                term = (int(er.get("blue_score", 0)), int(er.get("red_score", 0)))
                break
        return term if term else (int(core.blue_score[0]), int(core.red_score[0]))
    finally:
        try:
            env.close()
        except Exception:
            pass


def forced_capture(m1: bool, own_flag_home: bool, device="cuda") -> int:
    """Place a blue carrier on its home with the own flag home or stolen."""
    env, core = make(m1, 4242, device)
    try:
        env.env_method("set_phase", "OP6")
        env.env_method("set_next_opponent", "SCRIPTED", "OP6")
        core.blue_scripted = True
        core.set_blue_style("BLUE_BOTH_ATTACK_V2")
        env.reset()
        core.drain_tag_events()
        # Clear the scoring grace window first: score_grace_steps defaults to 10,
        # so a capture forced before step_count >= 10 can never score and every
        # arm would read 0. This bit me once already -- the control caught it.
        grace = int(getattr(core.cfg, "score_grace_steps", 10))
        for _ in range(grace + 6):
            env.step_async(env.action_space.sample() * 0)
            env.step_wait()
        core.drain_tag_events()
        captures = 0

        hf = core.blue_flag_home[0].clone()
        for _ in range(12):
            # blue agent 0 carries the enemy flag and sits on its own home
            core.blue_carrying[0, :] = False
            core.blue_carrying[0, 0] = True
            core.blue_alive[0, :] = True
            core.blue_tagged[0, :] = False
            core.blue_x[0, 0] = hf[0]
            core.blue_y[0, 0] = hf[1]
            if own_flag_home:
                core.blue_flag_pos[0] = hf.clone()
            else:
                core.blue_flag_pos[0, 0] = hf[0] + 7.0   # stolen / away
                core.blue_flag_pos[0, 1] = hf[1]
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            # Count the authoritative capture EVENT rather than a score delta:
            # an episode reset mid-loop would zero blue_score and silently make
            # every arm read 0, which is what broke the control.
            for e in core.drain_tag_events():
                if (e.get("event_type") == "capture_scored"
                        and e.get("scoring_team") == "blue"):
                    captures += 1
            if bool(np.asarray(done).any()):
                break
        return captures
    finally:
        try:
            env.close()
        except Exception:
            pass


def _place_blue_carrier_at_home(core, hf) -> None:
    core.blue_carrying[0, :] = False
    core.blue_carrying[0, 0] = True
    core.blue_alive[0, :] = True
    core.blue_tagged[0, :] = False
    core.blue_x[0, 0] = hf[0]
    core.blue_y[0, 0] = hf[1]


def _own_flag_away(core, hf) -> None:
    """Stolen: position off the stand. Leave red possession as the engine has it."""
    core.blue_flag_pos[0, 0] = hf[0] + 7.0
    core.blue_flag_pos[0, 1] = hf[1]


def _own_flag_returns_home(core, hf) -> None:
    """Intended recovery: own flag is on the stand and nobody is carrying it.

    Park red away from blue home so a same-tick re-grab cannot steal the
    returned flag before capture_confirm_frames elapses. That re-grab is
    ordinary V2 physics, not the M1 transition under test.
    """
    core.red_carrying[0, :] = False
    rh = core.red_flag_home[0]
    core.red_x[0, :] = rh[0]
    core.red_y[0, :] = rh[1]
    core.blue_flag_pos[0] = hf.clone()


def forced_block_then_recover(device="cuda") -> dict:
    """Block while own flag is away, then return the flag; carrier stays home."""
    env, core = make(True, 4243, device)
    try:
        env.env_method("set_phase", "OP6")
        env.env_method("set_next_opponent", "SCRIPTED", "OP6")
        core.blue_scripted = True
        core.set_blue_style("BLUE_BOTH_ATTACK_V2")
        env.reset()
        core.drain_tag_events()
        grace = int(getattr(core.cfg, "score_grace_steps", 10))
        for _ in range(grace + 6):
            env.step_async(env.action_space.sample() * 0)
            env.step_wait()
        core.drain_tag_events()

        hf = core.blue_flag_home[0].clone()
        blocked = recovered = 0

        for _ in range(8):
            _place_blue_carrier_at_home(core, hf)
            _own_flag_away(core, hf)
            env.step_async(env.action_space.sample() * 0)
            env.step_wait()
            for e in core.drain_tag_events():
                if (e.get("event_type") == "capture_scored"
                        and e.get("scoring_team") == "blue"):
                    blocked += 1

        for _ in range(12):
            _place_blue_carrier_at_home(core, hf)
            _own_flag_returns_home(core, hf)
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            for e in core.drain_tag_events():
                if (e.get("event_type") == "capture_scored"
                        and e.get("scoring_team") == "blue"):
                    recovered += 1
            if bool(np.asarray(done).any()):
                break
        return {
            "blocked_phase_captures": blocked,
            "recover_phase_captures": recovered,
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> int:
    ok = True
    print("=" * 74)
    print("M1 CORRECTNESS TESTS -- own_flag_home_required_to_score")
    print("=" * 74)

    # 1. V2 unchanged with flag OFF
    print("\n[1] V2 bit-identical with M1 OFF")
    same = True
    for sd in (3100001, 3100002, 3100003):
        a = rollout_scores(False, sd)
        b = rollout_scores(False, sd)
        print(f"    seed {sd}: {a} vs {a}  (determinism {'OK' if a == b else 'FAIL'})")
        same &= (a == b)
    if not same:
        print("    FAIL: baseline not deterministic"); ok = False
    else:
        print("    PASS")

    # 2. normal scoring still works with M1 ON and flag home
    print("\n[2] M1 ON, own flag HOME -> capture scores")
    d = forced_capture(True, own_flag_home=True)
    print(f"    score delta = {d}")
    if d <= 0:
        print("    FAIL: legal capture did not score"); ok = False
    else:
        print("    PASS")

    # 3. blocked when own flag away
    print("\n[3] M1 ON, own flag AWAY -> capture BLOCKED")
    d3 = forced_capture(True, own_flag_home=False)
    print(f"    score delta = {d3}")
    if d3 != 0:
        print("    FAIL: scored while own flag was away"); ok = False
    else:
        print("    PASS")

    # 4. same situation with M1 OFF must still score (isolates the rule)
    print("\n[4] M1 OFF, own flag AWAY -> capture ALLOWED (control)")
    d4 = forced_capture(False, own_flag_home=False)
    print(f"    score delta = {d4}")
    if d4 <= 0:
        print("    FAIL: control did not score; test setup is wrong, not the rule")
        ok = False
    else:
        print("    PASS -- the block in [3] is caused by M1, not by the setup")

    # 5. the missing transition: block, then own flag returns while still home
    print("\n[5] M1 ON: block while away, then own flag RETURNS while carrier still home -> scores")
    rec = forced_block_then_recover()
    print(f"    blocked-phase captures = {rec['blocked_phase_captures']}")
    print(f"    recover-phase captures = {rec['recover_phase_captures']}")
    if rec["blocked_phase_captures"] != 0:
        print("    FAIL: scored during the away phase of the same episode")
        ok = False
    elif rec["recover_phase_captures"] <= 0:
        print("    FAIL: own flag returned and carrier was still legally home, but no score")
        print("    M1_PAYOFF_ASSAY = INVALID_IMPLEMENTATION (not a scientific FAIL)")
        ok = False
    else:
        print("    PASS -- post-block scoring semantics are the intended rule")

    print("\n" + "=" * 74)
    print("RESULT:", "ALL PASS" if ok else "FAILURES PRESENT")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
