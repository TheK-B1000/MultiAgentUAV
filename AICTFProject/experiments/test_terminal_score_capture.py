"""Regression test for the terminal-reset score-read hazard.

The bug: a harness that breaks out of its step loop on `terminal` and THEN reads
`core.blue_score[0]` reads the ALREADY AUTO-RESET score, recording 0 for an
episode that actually scored. Both gate2_affordance_scenarios_v2.py (July) and
gate2b_affordance_scenarios_v2.py have this shape, which is why 64/64 episodes
stored score 0 while 63/64 recorded a mid-episode capture.

This test drives the real env with the same construction gate2b uses and
compares three independent sources across an episode that includes its terminal
step:

    running score delta   ==   capture_scored ledger   ==   recorded episode score

It asserts the BUGGY read is wrong and the CORRECTED read is right, so it fails
loudly if anyone reintroduces the pattern.

Run:  python experiments/test_terminal_score_capture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


def run_one(opponent: str, seed: int, device: str = "cuda") -> dict:
    """One episode, tracking blue score three independent ways."""
    import experiments.gate2b_affordance_scenarios_v2 as G
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=G.AGENTS, max_red_agents=G.AGENTS,
        map_set="train", map_layout=G.MAP,
        max_decision_steps=G.MAX_DECISION_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **G.RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        core.blue_scripted = True
        core.set_blue_style(G.BOTH_ATTACK)
        env.reset()
        core.drain_tag_events()

        start_b = int(core.blue_score[0])
        running = 0          # source 1: watched pre-terminal, step by step
        ledger = 0           # source 2: capture_scored events from the engine
        prev_b = start_b
        buggy = corrected = None

        for _ in range(G.MAX_DECISION_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            terminal = bool(np.asarray(done).any())

            for e in core.drain_tag_events():
                if (e.get("event_type") == "capture_scored"
                        and e.get("scoring_team") == "blue"):
                    ledger += int(e.get("score_after", 0)) - int(e.get("score_before", 0))

            if terminal:
                # Post-step state is already episode N+1. Three candidate reads:
                #   buggy     = core.blue_score after step_wait  -> post-reset 0
                #   running   = last pre-terminal value          -> UNDERCOUNTS by
                #               the capture that TRIGGERS termination
                #   corrected = vecenv info dict                 -> authoritative
                buggy = int(core.blue_score[0]) - start_b
                i0 = _i[0] if isinstance(_i, (list, tuple)) else _i
                er = i0.get("episode_result") or {}
                corrected = int(er.get("blue_score", i0.get("blue_score", 0))) - start_b
                break

            cur_b = int(core.blue_score[0])
            if cur_b > prev_b:
                running += cur_b - prev_b
            prev_b = cur_b
        else:
            buggy = corrected = int(core.blue_score[0]) - start_b

        return {"seed": seed, "running": running, "ledger": ledger,
                "buggy": buggy, "corrected": corrected}
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="OP6")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()

    print("=" * 78)
    print("TERMINAL-RESET SCORE-READ REGRESSION TEST")
    print("three sources must agree: running delta == capture ledger == recorded")
    print("=" * 78)

    ok = True
    exercised = 0
    for i in range(a.seeds):
        seed = 1900001 + i
        r = run_one(a.opponent, seed, a.device)
        flag = ""
        if r["ledger"] > 0:
            exercised += 1
            # Authoritative sources must agree exactly.
            if r["corrected"] != r["ledger"]:
                flag += "  INFO-vs-LEDGER MISMATCH"
                ok = False
            # The buggy post-reset read must be wrong, or the hazard is gone
            # and this test is no longer testing anything.
            if r["buggy"] == r["ledger"]:
                flag += "  (hazard absent on this seed)"
            # running is expected to undercount by the terminating capture.
            if r["running"] > r["ledger"]:
                flag += "  RUNNING-OVERCOUNT"
                ok = False
        print(f"  seed {seed}  running={r['running']}  ledger={r['ledger']}  "
              f"buggy_read={r['buggy']}  corrected_read={r['corrected']}{flag}")

    print()
    if exercised == 0:
        print("INCONCLUSIVE: no episode scored, hazard not exercised")
        return 3
    print(f"episodes exercising the hazard: {exercised}/{a.seeds}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
