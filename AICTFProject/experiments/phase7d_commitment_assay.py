"""Phase 7D -- can a committed offensive allocation undo itself in time?

Pre-authorized mechanism diagnostic under stock RULESET_V2. Explains a Gate A
result; cannot upgrade one.

THE QUESTION. If both blue vehicles commit forward (DOUBLE_BREACH) and the
opponent's raiding intent then becomes LEGALLY observable, can one attacker
turn around and materially prevent the threat? If yes, the allocation was a
reversible tactical choice, not a strategic commitment.

    STRATEGIC PRESSURE = OPPORTUNITY COST  AND  COMMITMENT COST

DESIGN. Two arms on identical seeds, differing only in whether recovery is
ordered:

    NO_RECOVERY   BLUE_BOTH_ATTACK_V2 throughout          (committed, never recalled)
    RECOVERY      BOTH_ATTACK until intent is observable,
                  then switch to BLUE_ONE_DEFENDER_V2     (emergency recall)

The switch uses the existing set_blue_style mechanism -- no new controller.
BLUE_ONE_DEFENDER_V2 sends agent 1 home and holds it there, which IS the fastest
reasonable recovery available in existing code.

LEGAL OBSERVABILITY. Intent is defined as a live untagged red agent crossing
into blue's half, using the engine's own predicate `_is_on_home_side("blue", x)`
(mid = (cols-1)*0.5). Opponent identity is never an input. Nothing recomputes
the midline -- reimplementing it as cols*0.5 is a known half-cell error.

CLASSIFICATION is behavioural, not a timing threshold. Timing is reported but
does not classify:

    COMMITMENT_PRESENT       recovery does not reliably prevent the threat
    COMMITMENT_ABSENT        recovery routinely converts to effective defence
    COMMITMENT_INDETERMINATE evidence genuinely ambiguous

No new percentage threshold is invented; the paired difference and its CI are
reported and the call is made on whether prevention is *practically reliable*.

Run:  python experiments/phase7d_commitment_assay.py --seeds 24
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAP = "map_a"
MAX_STEPS = 240
AGENTS = 2
RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
SEED_BASE = 1_980_001          # fresh, unused by any prior experiment


def run_episode(*, seed: int, opponent: str, recovery: bool,
                device: str = "cuda") -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=MAP, max_decision_steps=MAX_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        core.blue_scripted = True
        core.set_blue_style("BLUE_BOTH_ATTACK_V2")
        env.reset()
        core.drain_tag_events()

        t_commit = None          # both blue across midline
        t_intent = None          # live untagged red in blue half
        t_recover = None         # style switched
        t_red_pickup = None
        t_red_capture = None
        recovered = False
        term_scores = None
        n = 0

        for step_i in range(MAX_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, info = env.step_wait()
            n += 1
            terminal = bool(np.asarray(done).any())
            if terminal:
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = i0.get("episode_result") or {}
                term_scores = (int(er.get("blue_score", i0.get("blue_score", 0))),
                               int(er.get("red_score", i0.get("red_score", 0))))

            for e in core.drain_tag_events():
                if (e.get("event_type") == "capture_scored"
                        and e.get("scoring_team") == "red" and t_red_capture is None):
                    t_red_capture = step_i

            bx = core.blue_x
            rx = core.red_x
            blue_fwd = (~core._is_on_home_side("blue", bx)[0]).detach().cpu().numpy()
            alive_b = core.blue_alive[0].detach().cpu().numpy().astype(bool)
            if t_commit is None and bool((blue_fwd & alive_b).all()) and alive_b.all():
                t_commit = step_i

            # LEGAL intent signal: live untagged red inside blue's half.
            red_in_blue = core._is_on_home_side("blue", rx)[0].detach().cpu().numpy()
            alive_r = core.red_alive[0].detach().cpu().numpy().astype(bool)
            tagged_r = core.red_tagged[0].detach().cpu().numpy().astype(bool)
            intruder = red_in_blue & alive_r & (~tagged_r)
            if t_intent is None and bool(intruder.any()):
                t_intent = step_i
                if recovery and not recovered:
                    core.set_blue_style("BLUE_ONE_DEFENDER_V2")
                    recovered = True
                    t_recover = step_i

            carry_r = core.red_carrying[0].detach().cpu().numpy().astype(bool)
            if t_red_pickup is None and bool(carry_r.any()):
                t_red_pickup = step_i

            if terminal:
                break

        bs, rs_ = term_scores if term_scores else (
            int(core.blue_score[0]), int(core.red_score[0]))
        return {
            "seed": seed, "opponent": opponent, "recovery": recovery,
            "t_commit": t_commit, "t_intent": t_intent, "t_recover": t_recover,
            "t_red_pickup": t_red_pickup, "t_red_capture": t_red_capture,
            "red_picked_up": int(t_red_pickup is not None),
            "red_captured": int(t_red_capture is not None),
            "blue_score": bs, "red_score": rs_,
            "episode_steps": n,
            "recovery_margin": (None if (t_intent is None or t_red_pickup is None)
                                else t_red_pickup - t_intent),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def paired_ci(d: np.ndarray, rng, n_boot=20000, alpha=0.05):
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return float(d.mean()), float(np.percentile(bs, 100 * alpha / 2)), \
        float(np.percentile(bs, 100 * (1 - alpha / 2)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--opponent", default="OP6")   # raider: intent to observe
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/phase7/commitment_assay.json")
    a = ap.parse_args()

    rows = {"NO_RECOVERY": [], "RECOVERY": []}
    for i in range(a.seeds):
        for arm, rec in (("NO_RECOVERY", False), ("RECOVERY", True)):
            try:
                rows[arm].append(run_episode(seed=SEED_BASE + i,
                                             opponent=a.opponent,
                                             recovery=rec, device=a.device))
            except Exception as exc:
                print(f"  {arm} seed {SEED_BASE+i}: ERROR {type(exc).__name__}: {exc}")
                return 2
        print(f"  paired seed {i+1}/{a.seeds}", flush=True)

    rng = np.random.default_rng(7)
    seeds = [r["seed"] for r in rows["NO_RECOVERY"]]
    idx = {arm: {r["seed"]: r for r in rows[arm]} for arm in rows}

    def arr(arm, k):
        return np.array([float(idx[arm][s][k]) for s in seeds])

    d_pickup = arr("NO_RECOVERY", "red_picked_up") - arr("RECOVERY", "red_picked_up")
    d_capture = arr("NO_RECOVERY", "red_captured") - arr("RECOVERY", "red_captured")
    d_redscore = arr("NO_RECOVERY", "red_score") - arr("RECOVERY", "red_score")

    m_p, lo_p, hi_p = paired_ci(d_pickup, rng)
    m_c, lo_c, hi_c = paired_ci(d_capture, rng)
    m_s, lo_s, hi_s = paired_ci(d_redscore, rng)

    margins = [r["recovery_margin"] for r in rows["RECOVERY"]
               if r["recovery_margin"] is not None]

    # Behavioural classification. Recovery "works" when recalling one vehicle
    # measurably prevents the threat; it is absent-of-commitment when that
    # prevention is routine.
    prevents_pickup = m_p
    prevents_capture = m_c
    reliable = (lo_p > 0 and m_p >= 0.5) or (lo_c > 0 and m_c >= 0.5)
    negligible = (hi_p <= 0.1 and hi_c <= 0.1)
    if reliable:
        verdict = "COMMITMENT_ABSENT"
    elif negligible:
        verdict = "COMMITMENT_PRESENT"
    else:
        verdict = "COMMITMENT_INDETERMINATE"

    out = {
        "record": "Phase 7D commitment / emergency-recovery assay, stock RULESET_V2",
        "opponent": a.opponent, "seeds": a.seeds, "seed_base": SEED_BASE,
        "arms": {arm: {
            "red_pickup_rate": float(np.mean([r["red_picked_up"] for r in rows[arm]])),
            "red_capture_rate": float(np.mean([r["red_captured"] for r in rows[arm]])),
            "mean_red_score": float(np.mean([r["red_score"] for r in rows[arm]])),
            "mean_blue_score": float(np.mean([r["blue_score"] for r in rows[arm]])),
            "mean_episode_steps": float(np.mean([r["episode_steps"] for r in rows[arm]])),
        } for arm in rows},
        "paired_effects_no_recovery_minus_recovery": {
            "red_pickup_prevented": {"mean": m_p, "CI95": [lo_p, hi_p]},
            "red_capture_prevented": {"mean": m_c, "CI95": [lo_c, hi_c]},
            "red_score_reduced": {"mean": m_s, "CI95": [lo_s, hi_s]},
        },
        "timing_reported_not_classifying": {
            "mean_t_commit": float(np.mean([r["t_commit"] for r in rows["RECOVERY"]
                                            if r["t_commit"] is not None] or [np.nan])),
            "mean_t_intent": float(np.mean([r["t_intent"] for r in rows["RECOVERY"]
                                            if r["t_intent"] is not None] or [np.nan])),
            "mean_recovery_margin_steps": float(np.mean(margins)) if margins else None,
            "note": "T_intent to T_red_pickup. Reported; does not classify.",
        },
        "verdict": verdict,
        "classification_rule": ("behavioural: COMMITMENT_ABSENT when recalling one "
                                "vehicle reliably prevents the threat; "
                                "COMMITMENT_PRESENT when prevention is negligible; "
                                "otherwise INDETERMINATE. No new percentage "
                                "threshold was invented as a scientific gate."),
    }
    p = PROJECT_ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("arms", "paired_effects_no_recovery_minus_recovery", "verdict")}, indent=2))
    print(f"-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
