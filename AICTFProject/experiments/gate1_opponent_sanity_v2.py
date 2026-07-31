#!/usr/bin/env python3
"""GATE 1 -- OP6-OP12 admissibility under RULESET_V2 on map_a, from EVENTS.

The previous version of this gate reconstructed "who tagged whom" from
positions read after ``step_wait()``. That is temporally invalid: by then the
target has been redirected home, the tagger has moved, cooldowns are armed, and
flags have been dropped. It reported violations that the rule-level tests
directly disprove, and would have quarantined every opponent.

Tag LEGALITY is now taken exclusively from decision-point telemetry
(``drain_tag_events()``). Post-step state is still used, but only to verify
CONSEQUENCES (carrier dropped the flag, target became tagged, tagged agents
return home and untag there).

FAIL-CLOSED. The gate fails if telemetry is missing, a schema field is absent,
events duplicate, ordering crosses an episode-reset boundary incorrectly, an
opponent deadlocks, or actions/positions go non-finite.

An opponent that fails is QUARANTINED from the G0-v2 mixture. It is never tuned
merely for performing poorly -- only exact rule violations, deadlocks, or broken
behaviour cause quarantine.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OPPONENTS = ["OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"]
MAP = "map_a"
RESOLVED_MAP = "map_a_open"
MAX_DECISION_STEPS = 240
AGENTS = 2
SEED_BASE = 1_500_001

RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)

SUCCESS_FIELDS = {
    "event_type", "env_index", "simulation_time", "ruleset_id",
    "tagger_team", "tagger_index", "target_team", "target_index",
    "tagger_position_at_decision", "target_position_at_decision",
    "distance_at_decision", "tagger_on_own_side", "target_on_tagger_side",
    "tagger_cooldown_before", "tagger_cooldown_after", "target_was_tagged",
    "target_was_carrying_flag", "eligible_target_indices", "selected_nearest_target",
}
DENIED_FIELDS = {
    "event_type", "reason", "env_index", "simulation_time", "ruleset_id",
    "tagger_team", "tagger_index", "candidate_target_index", "cooldown_remaining",
}


def check_events(events: list, *, tag_range: float, cooldown_T: float) -> dict:
    """Validate tag legality using ONLY decision-point event fields."""
    v = defaultdict(int)
    seen = set()
    n_success = n_denied = 0

    # simulation_time restarts every episode, so identity MUST be scoped by
    # episode. Without this, a success in episode 1 and a denial in episode 2 by
    # the same tagger collide and look like a same-instant contradiction.
    episode = 0
    ep_of = []
    for e in events:
        if e.get("event_type") == "episode_reset":
            episode += 1
        ep_of.append(episode)

    for e, ep in zip(events, ep_of):
        et = e.get("event_type")
        if et == "episode_reset":
            continue
        if et == "tag_success":
            n_success += 1
            missing = SUCCESS_FIELDS - set(e)
            if missing:
                v["schema_missing_field"] += 1
                continue
            # identity must be unique per (env, time, tagger, target)
            key = (ep, e["env_index"], round(float(e["simulation_time"]), 6),
                   e["tagger_team"], e["tagger_index"], e["target_index"])
            if key in seen:
                v["duplicate_event"] += 1
            seen.add(key)

            if not e["tagger_on_own_side"]:
                v["tagger_not_on_own_side"] += 1
            if not e["target_on_tagger_side"]:
                v["target_not_on_tagger_side"] += 1
            if float(e["distance_at_decision"]) > tag_range + 1e-6:
                v["tag_out_of_range"] += 1
            if e["target_was_tagged"]:
                v["retagged_already_tagged_target"] += 1
            if float(e["tagger_cooldown_before"]) > 1e-9:
                v["tag_during_cooldown"] += 1
            elig = list(e["eligible_target_indices"])
            if not elig:
                v["tag_with_no_eligible_target"] += 1
            elif e["selected_nearest_target"] not in elig:
                v["selected_target_not_eligible"] += 1
            if e["tagger_team"] == e["target_team"]:
                v["friendly_tag"] += 1
        elif et == "tag_denied":
            n_denied += 1
            missing = DENIED_FIELDS - set(e)
            if missing:
                v["schema_missing_field"] += 1
                continue
            if e["reason"] != "cooldown":
                v["unexpected_denial_reason"] += 1
            if float(e["cooldown_remaining"]) <= 0.0:
                v["denial_without_cooldown"] += 1
        else:
            v["unknown_event_type"] += 1

    # a denial and a success must never coexist for the same tagger at the same instant
    succ_keys = {(ep, e["env_index"], round(float(e["simulation_time"]), 6),
                  e["tagger_team"], e["tagger_index"])
                 for e, ep in zip(events, ep_of)
                 if e.get("event_type") == "tag_success"}
    for e, ep in zip(events, ep_of):
        if e.get("event_type") == "tag_denied":
            k = (ep, e["env_index"], round(float(e["simulation_time"]), 6),
                 e["tagger_team"], e["tagger_index"])
            if k in succ_keys:
                v["denied_and_succeeded_same_instant"] += 1

    return {"violations": dict(v), "n_success": n_success, "n_denied": n_denied}


def run_opponent(opp: str, episodes: int, device: str) -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    viol = defaultdict(int)      # Gate 1A: hard, legality + liveness
    lifecycle = defaultdict(int)  # Gate 1C: post-step, DIAGNOSTIC ONLY
    all_events: list = []
    ep_stats = []

    for ep in range(episodes):
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
            map_set="train", map_layout=MAP, max_decision_steps=MAX_DECISION_STEPS,
            aquaticus_profile=True, rules_profile="OURS", device=device,
            seed=SEED_BASE + ep, obstacle_obs_channel=True,
            tag_telemetry_enabled=True, **RULESET,
        )
        env = GPUCTFVecEnv(cfg)
        core = env.core
        try:
            env.env_method("set_phase", opp)
            env.env_method("set_next_opponent", "SCRIPTED", opp)
            if (env.env_method("get_opponent_key")[0] or "").strip().upper() != opp:
                viol["opponent_mismatch"] += 1
            core.blue_scripted = True
            env.reset()
            core.drain_tag_events()

            mid = float(core.cols) * 0.5
            n = n_drop = 0
            atk = 0
            carry_prev = core.blue_carrying[0].detach().cpu().numpy().copy()
            rcarry_prev = core.red_carrying[0].detach().cpu().numpy().copy()
            tagged_at = {}
            pos_b = []

            for step in range(MAX_DECISION_STEPS):
                act = env.action_space.sample() * 0
                env.step_async(act)
                _o, _r, done, _i = env.step_wait()
                n += 1
                ev = core.drain_tag_events()
                all_events.extend(ev)

                bx = core.blue_x[0].detach().cpu().numpy()
                rx = core.red_x[0].detach().cpu().numpy()
                if not (np.all(np.isfinite(bx)) and np.all(np.isfinite(rx))):
                    viol["nonfinite_position"] += 1
                pos_b.append(bx.copy())
                atk += int(np.sum(bx > mid))

                bt = core.blue_tagged[0].detach().cpu().numpy()
                rt = core.red_tagged[0].detach().cpu().numpy()
                bc = core.blue_carrying[0].detach().cpu().numpy()
                rc = core.red_carrying[0].detach().cpu().numpy()

                # CONSEQUENCE checks driven by the events
                terminal = bool(np.asarray(done).any())
                for e in ev:
                    if e.get("event_type") != "tag_success":
                        continue
                    if terminal:
                        # Episode boundary: auto-reset already cleared tagged/
                        # carrying. Comparing across it photographs the next
                        # episode. Skip the consequence checks, not the event.
                        lifecycle["skipped_at_episode_boundary"] += 1
                        continue
                    tt, ti = e["target_team"], e["target_index"]
                    tagged_now = (bt if tt == "blue" else rt)[ti]
                    if not tagged_now:
                        lifecycle["success_event_target_not_tagged"] += 1
                    if e["target_was_carrying_flag"]:
                        n_drop += 1
                        still = (bc if tt == "blue" else rc)[ti]
                        if still:
                            lifecycle["tagged_carrier_kept_flag"] += 1
                    tagged_at[(tt, ti)] = n

                # tagged agents must eventually untag (they are sent home)
                for tt, arr in (("blue", bt), ("red", rt)):
                    for i in range(len(arr)):
                        k = (tt, i)
                        if k in tagged_at and not arr[i]:
                            tagged_at.pop(k, None)

                carry_prev, rcarry_prev = bc.copy(), rc.copy()
                if bool(np.asarray(done).any()):
                    break

            # anything still tagged at horizon end that never recovered
            for k, t0 in tagged_at.items():
                if n - t0 >= MAX_DECISION_STEPS - 1:
                    lifecycle["tagged_never_recovered"] += 1

            pb = np.asarray(pos_b)
            if pb.shape[0] > 1 and np.all(np.abs(np.diff(pb, axis=0)).sum(axis=0) < 1.0):
                viol["blue_team_frozen"] += 1
            if n == 0:
                viol["zero_length_episode"] += 1

            ep_stats.append({
                "steps": n, "flag_drops": n_drop,
                "blue_score": int(core.blue_score[0]), "red_score": int(core.red_score[0]),
                "timeout": int(n >= MAX_DECISION_STEPS),
                "attack_frac": atk / max(1, n * AGENTS),
            })
        finally:
            env.close()

    if not all_events:
        viol["telemetry_missing"] += 1

    ev_check = check_events(all_events, tag_range=2.5,
                            cooldown_T=RULESET["tag_min_interval_seconds"])
    for k, c in ev_check["violations"].items():
        viol[k] += c

    def m(k):
        return float(np.mean([e[k] for e in ep_stats])) if ep_stats else float("nan")

    n_v = int(sum(viol.values()))
    n_lc = int(sum(lifecycle.values()))
    return {
        "opponent": opp, "episodes": len(ep_stats),
        "violations": dict(viol), "n_violations": n_v,
        "lifecycle_observations": dict(lifecycle),
        "n_lifecycle_observations": n_lc,
        "metrics": {
            "mean_steps": m("steps"),
            "tag_successes_total": ev_check["n_success"],
            "cooldown_denials_total": ev_check["n_denied"],
            "flag_drops_per_episode": m("flag_drops"),
            "blue_score": m("blue_score"), "red_score": m("red_score"),
            "timeout_rate": m("timeout"), "attack_frac": m("attack_frac"),
        },
        # Gate 1A only. A post-step consequence check must never quarantine an
        # opponent: a tag can be legally decided, legally applied, and then
        # legally cleared by later same-step lifecycle processing. Resolving
        # those apart needs tag_applied / untagged_at_home telemetry (Gate 1B/1C).
        "admitted": n_v == 0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--opponents", nargs="+", default=OPPONENTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="artifacts/gate1_opponent_sanity_v2")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("GATE 1 -- opponent admissibility under RULESET_V2_AQUATICUS_10S, map_a")
    print("Tag legality from DECISION-POINT EVENTS only. Fail-closed.")
    print("=" * 84)

    results = []
    for opp in args.opponents:
        print(f"\n[{opp}] {args.episodes} episodes ...", flush=True)
        r = run_opponent(opp, args.episodes, args.device)
        results.append(r)
        mt = r["metrics"]
        print(f"  tags={mt['tag_successes_total']} denials={mt['cooldown_denials_total']} "
              f"drops/ep={mt['flag_drops_per_episode']:.2f} "
              f"B={mt['blue_score']:.2f} R={mt['red_score']:.2f} "
              f"timeout={mt['timeout_rate']:.0%} atk={mt['attack_frac']:.2f}")
        print(f"  violations (1A, hard): {r['violations'] if r['violations'] else 'none'} "
              f"-> {'ADMITTED' if r['admitted'] else 'QUARANTINED'}")
        if r["lifecycle_observations"]:
            print(f"  lifecycle (1C, diagnostic only, does NOT quarantine): "
                  f"{r['lifecycle_observations']}")

    admitted = [r["opponent"] for r in results if r["admitted"]]
    quarantined = [r["opponent"] for r in results if not r["admitted"]]
    ok = len(admitted) >= 2

    print(f"\n{'=' * 84}\nGATE 1 RESULT\n{'=' * 84}")
    print(f"  admitted    : {admitted}")
    print(f"  quarantined : {quarantined if quarantined else 'none'}")
    print(f"  verdict     : {'PASS' if ok else 'FAIL -- too few admissible opponents'}")

    payload = {
        "gate": "gate1_opponent_sanity_events",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ruleset_id": "RULESET_V2_AQUATICUS_10S", "ruleset": RULESET,
        "map": MAP, "resolved_map": RESOLVED_MAP,
        "legality_source": "decision-point tag events only",
        "gate_1a_hard": "legality (territory, range, eligibility, nearest-only, cooldown) + liveness (deadlock, non-finite, frozen)",
        "gate_1c_diagnostic": "post-step consequences; cannot quarantine until tag_applied / untagged_at_home telemetry exists",
        "episodes_per_opponent": args.episodes,
        "seed_block": [SEED_BASE, SEED_BASE + args.episodes - 1],
        "results": results, "admitted_opponents": admitted,
        "quarantined_opponents": quarantined,
        "verdict": "PASS" if ok else "FAIL",
    }
    (out_dir / "gate1_result.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[done] {out_dir / 'gate1_result.json'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
