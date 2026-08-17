"""Tag-cooldown saturation diagnostic (2v2, stock RULESET_V2).

Tests the candidate mechanism left standing after suppression was falsified:
suppression occurred ZERO times in 256 episodes across all four range values, so
the second attacker's advantage against FORTRESS cannot come from removing the
defender. The surviving hypothesis is that it comes from OFFENSIVE THROUGHPUT
against a RATE-LIMITED defence.

Rules that make this plausible:
    taggers_required        1      one tagger suffices
    tag_nearest_only        True   only the nearest eligible target is tagged
    tag_min_interval_seconds 10.0  a tagger cannot re-tag for 10 s

The tag_success event already carries everything needed, so no engine change:
    simulation_time, tagger_index, tagger_cooldown_before,
    eligible_target_indices, selected_nearest_target, target_index

Computes the five diagnostics:
    1. per-RED-agent inter-tag intervals      -- do they pile up near 10 s?
    2. cooldown utilisation                   -- share of tags issued at/near floor
    3. post-tag opportunity windows           -- pickups by the OTHER attacker
                                                 between consecutive red tags
    4. nearest-target diversion               -- tags where >1 target was eligible
    5. ONE_DEFENDER vs BOTH_ATTACK contrast   -- does a tag end the whole attempt?

MECHANISM DIAGNOSTIC ONLY. Gates nothing, changes no ruleset, 2v2 throughout.

Run:  python experiments/phase7_tag_saturation.py --seeds 16
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
RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
SEED_BASE = 2_010_001          # fresh, unused
TAG_MIN_INTERVAL = 10.0


def run_episode(*, opponent: str, blue_style: str, seed: int,
                device: str = "cuda") -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,   # 2v2 ONLY
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
        core.set_blue_style(blue_style)
        env.reset()
        core.drain_tag_events()

        red_tags = []          # tags issued BY red ON blue
        pickup_events = []     # (step, blue_agent_index)
        prev_carry = core.blue_carrying[0].detach().cpu().numpy().astype(bool).copy()
        term = None
        n = 0

        for step_i in range(MAX_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, info = env.step_wait()
            n += 1
            terminal = bool(np.asarray(done).any())
            if terminal:
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = i0.get("episode_result") or {}
                term = (int(er.get("blue_score", i0.get("blue_score", 0))),
                        int(er.get("red_score", i0.get("red_score", 0))))

            for e in core.drain_tag_events():
                if e.get("event_type") == "tag_success" and e.get("tagger_team") == "red":
                    red_tags.append({
                        "step": step_i,
                        "sim_time": float(e.get("simulation_time", float("nan"))),
                        "tagger": int(e.get("tagger_index", -1)),
                        "target": int(e.get("target_index", -1)),
                        "cooldown_before": float(e.get("tagger_cooldown_before", float("nan"))),
                        "n_eligible": len(e.get("eligible_target_indices") or []),
                        "target_carrying": bool(e.get("target_was_carrying_flag", False)),
                    })

            carry = core.blue_carrying[0].detach().cpu().numpy().astype(bool)
            for idx in np.where((~prev_carry) & carry)[0]:
                pickup_events.append({"step": step_i, "agent": int(idx)})
            prev_carry = carry.copy()
            if terminal:
                break

        bs, rs_ = term if term else (int(core.blue_score[0]), int(core.red_score[0]))
        return {"seed": seed, "opponent": opponent, "blue_style": blue_style,
                "red_tags": red_tags, "pickups": pickup_events,
                "blue_score": bs, "red_score": rs_, "episode_steps": n}
    finally:
        try:
            env.close()
        except Exception:
            pass


def analyse(eps: list[dict]) -> dict:
    intervals: list[float] = []          # per-tagger consecutive sim_time gaps
    at_floor = 0
    total_tags = 0
    multi_eligible = 0
    windows_with_other_pickup = 0
    windows_total = 0
    tag_ended_attempt = 0
    attempts = 0

    for ep in eps:
        tags = sorted(ep["red_tags"], key=lambda t: t["step"])
        total_tags += len(tags)
        # (1) per-tagger inter-tag intervals
        by_tagger: dict = {}
        for t in tags:
            by_tagger.setdefault(t["tagger"], []).append(t)
        for _g, lst in by_tagger.items():
            times = [t["sim_time"] for t in lst if np.isfinite(t["sim_time"])]
            for a, b in zip(times, times[1:]):
                intervals.append(b - a)
        # (2) cooldown utilisation: tag issued while cooldown just expired
        for t in tags:
            cb = t["cooldown_before"]
            if np.isfinite(cb) and cb <= 1e-6:
                at_floor += 1
            # (4) nearest-target diversion
            if t["n_eligible"] > 1:
                multi_eligible += 1
        # (3) post-tag opportunity windows: after a tag on agent A, does the
        # OTHER blue agent pick up before the next red tag?
        for i, t in enumerate(tags):
            nxt = tags[i + 1]["step"] if i + 1 < len(tags) else ep["episode_steps"]
            windows_total += 1
            other = [p for p in ep["pickups"]
                     if t["step"] < p["step"] <= nxt and p["agent"] != t["target"]]
            if other:
                windows_with_other_pickup += 1
        # (5) did a tag terminate the whole offensive attempt?
        for i, t in enumerate(tags):
            nxt = tags[i + 1]["step"] if i + 1 < len(tags) else ep["episode_steps"]
            any_pickup = [p for p in ep["pickups"] if t["step"] < p["step"] <= nxt]
            attempts += 1
            if not any_pickup:
                tag_ended_attempt += 1

    iv = np.array(intervals, dtype=float) if intervals else np.array([np.nan])
    near_floor = float(np.mean(iv <= TAG_MIN_INTERVAL * 1.25)) if intervals else float("nan")
    return {
        "episodes": len(eps),
        "total_red_tags": total_tags,
        "mean_tags_per_episode": total_tags / max(1, len(eps)),
        "inter_tag_interval": {
            "n": len(intervals),
            "mean": float(np.nanmean(iv)),
            "median": float(np.nanmedian(iv)),
            "min": float(np.nanmin(iv)),
            "frac_within_125pct_of_floor": near_floor,
            "floor_seconds": TAG_MIN_INTERVAL,
        },
        "cooldown_utilisation": {
            "tags_issued_at_zero_cooldown": at_floor,
            "frac": at_floor / max(1, total_tags),
        },
        "nearest_target_diversion": {
            "tags_with_multiple_eligible": multi_eligible,
            "frac": multi_eligible / max(1, total_tags),
        },
        "post_tag_window": {
            "windows": windows_total,
            "with_other_agent_pickup": windows_with_other_pickup,
            "frac": windows_with_other_pickup / max(1, windows_total),
        },
        "tag_ended_attempt": {
            "attempts": attempts,
            "ended": tag_ended_attempt,
            "frac": tag_ended_attempt / max(1, attempts),
        },
        "mean_blue_score": float(np.mean([e["blue_score"] for e in eps])),
        "mean_red_score": float(np.mean([e["red_score"] for e in eps])),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/phase7/tag_saturation.json")
    a = ap.parse_args()

    arms = [("ONE_DEFENDER_vs_OP7", "BLUE_ONE_DEFENDER_V2", "OP7"),
            ("BOTH_ATTACK_vs_OP7", "BLUE_BOTH_ATTACK_V2", "OP7")]
    out = {"record": "tag-cooldown saturation diagnostic, 2v2 stock RULESET_V2",
           "ruleset": RULESET, "seed_base": SEED_BASE, "arms": {}}

    for name, style, opp in arms:
        eps = []
        for i in range(a.seeds):
            eps.append(run_episode(opponent=opp, blue_style=style,
                                   seed=SEED_BASE + i, device=a.device))
        s = analyse(eps)
        out["arms"][name] = s
        print(f"{name:24s} tags/ep={s['mean_tags_per_episode']:5.2f} "
              f"median_gap={s['inter_tag_interval']['median']:6.2f}s "
              f"at_floor={s['cooldown_utilisation']['frac']:.3f} "
              f"multi_elig={s['nearest_target_diversion']['frac']:.3f} "
              f"other_pickup_after_tag={s['post_tag_window']['frac']:.3f} "
              f"tag_ended={s['tag_ended_attempt']['frac']:.3f}", flush=True)

    p = PROJECT_ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
