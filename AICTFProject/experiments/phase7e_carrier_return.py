"""Phase 7E -- what does the strong generalist PPO actually do after a pickup?

Asks whether successful scoring under stock RULESET_V2 relies on a largely
SELF-SUFFICIENT carrier-return loop, or on teammate support. That distinguishes
two accounts of why one PPO handles many opponents:

    self-sufficient carrier   -> the second agent adds little to conversion,
                                 so a repertoire has little to encode
    escorted carrier          -> conversion is a genuinely joint act

Checkpoint is PRE-COMMITTED to vgc_d1_seed3600001 (frozen best-fixed policy in
the held-out repertoire-value evaluation), chosen before any carrier-return
behaviour was inspected. No retraining. 2v2 throughout.

Reuses the project's existing loaders and rollout helpers rather than
reimplementing them: rl.evaluation.checkpoint.load_policy,
rl.custom_ppo.checkpoints.loader.read_checkpoint_payload, and the
_adapt_obs_for_policy / _predict / _reset_obs / _unpack_step / _done helpers
from eval_v6i9_map_awareness.

Metrics, per carrier possession episode-segment:
    grab -> capture time, path length, straight-line distance, efficiency
    heading reversals during return
    teammate distance during return (mean / min)
    escort incidence (teammate within ESCORT_RADIUS for a share of the return)
    opponent encounters after pickup
    tags/drops after pickup, conversion outcome

Run:  python experiments/phase7e_carrier_return.py --seeds 24
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT = "artifacts/vgc_diversity/vgc_d1_seed3600001/ckpts/final_vgc_d1_seed3600001.zip"
MAP = "map_a"
MAX_STEPS = 240
RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
SEED_BASE = 2_020_001          # fresh, unused
ESCORT_RADIUS = 6.0            # teammate "nearby" threshold, cells


def run_episode(*, policy, opponent: str, seed: int, device: str) -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy, _done, _predict, _reset_obs, _unpack_step)

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,     # 2v2 ONLY
        map_set="train", map_layout=MAP, max_decision_steps=MAX_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        obs = _reset_obs(env.reset())
        core.drain_tag_events()

        segments = []          # one per carrier possession
        active = None
        prev_carry = core.blue_carrying[0].detach().cpu().numpy().astype(bool).copy()
        prev_heading = None
        term = None
        n = 0

        for step_i in range(MAX_STEPS):
            act = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs2, _r, done, info = _unpack_step(env.step(act))
            n += 1
            terminal = _done(done)
            if terminal:
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = (i0 or {}).get("episode_result") or {}
                term = (int(er.get("blue_score", (i0 or {}).get("blue_score", 0))),
                        int(er.get("red_score", (i0 or {}).get("red_score", 0))))

            tags_now = 0
            for e in core.drain_tag_events():
                if e.get("event_type") == "tag_success" and e.get("target_team") == "blue":
                    tags_now += 1

            bx = core.blue_x[0].detach().cpu().numpy()
            by = core.blue_y[0].detach().cpu().numpy()
            rx = core.red_x[0].detach().cpu().numpy()
            ry = core.red_y[0].detach().cpu().numpy()
            hf = core.blue_flag_home[0].detach().cpu().numpy()
            carry = core.blue_carrying[0].detach().cpu().numpy().astype(bool)

            # new possession -> open a segment
            for idx in np.where((~prev_carry) & carry)[0]:
                active = {
                    "agent": int(idx), "start_step": step_i,
                    "start_xy": (float(bx[idx]), float(by[idx])),
                    "path": 0.0, "prev_xy": (float(bx[idx]), float(by[idx])),
                    "reversals": 0, "prev_heading": None,
                    "mate_d": [], "opp_encounters": 0, "tags": 0,
                    "converted": False,
                }
            if active is not None:
                a = active["agent"]
                if a < len(bx):
                    cx, cy = float(bx[a]), float(by[a])
                    px, py = active["prev_xy"]
                    d = math.hypot(cx - px, cy - py)
                    active["path"] += d
                    if d > 1e-6:
                        h = math.atan2(cy - py, cx - px)
                        ph = active["prev_heading"]
                        if ph is not None:
                            dh = abs((h - ph + math.pi) % (2 * math.pi) - math.pi)
                            if dh > math.pi / 2:
                                active["reversals"] += 1
                        active["prev_heading"] = h
                    active["prev_xy"] = (cx, cy)
                    mate = [j for j in range(len(bx)) if j != a]
                    if mate:
                        m = mate[0]
                        active["mate_d"].append(
                            float(math.hypot(bx[m] - cx, by[m] - cy)))
                    if len(rx):
                        if float(np.min(np.hypot(rx - cx, ry - cy))) <= 3.0:
                            active["opp_encounters"] += 1
                    active["tags"] += tags_now

            # possession ended
            for idx in np.where(prev_carry & (~carry))[0]:
                if active is not None and active["agent"] == int(idx):
                    sx, sy = active["start_xy"]
                    straight = math.hypot(sx - float(hf[0]), sy - float(hf[1]))
                    md = active["mate_d"]
                    segments.append({
                        "duration": step_i - active["start_step"],
                        "path_len": active["path"],
                        "straight_line": straight,
                        "efficiency": (straight / active["path"]) if active["path"] > 1e-6 else float("nan"),
                        "reversals": active["reversals"],
                        "mate_dist_mean": float(np.mean(md)) if md else float("nan"),
                        "mate_dist_min": float(np.min(md)) if md else float("nan"),
                        "escort_frac": float(np.mean([x <= ESCORT_RADIUS for x in md])) if md else float("nan"),
                        "opp_encounters": active["opp_encounters"],
                        "tags_during": active["tags"],
                    })
                    active = None

            prev_carry = carry.copy()
            obs = obs2
            if terminal:
                break

        bs, rs_ = term if term else (int(core.blue_score[0]), int(core.red_score[0]))
        return {"seed": seed, "opponent": opponent, "segments": segments,
                "blue_score": bs, "red_score": rs_, "episode_steps": n}
    finally:
        try:
            env.close()
        except Exception:
            pass


def summarize(eps: list[dict]) -> dict:
    segs = [s for e in eps for s in e["segments"]]
    if not segs:
        return {"possessions": 0}
    def m(k):
        v = [s[k] for s in segs if s[k] == s[k]]     # drop NaN
        return float(np.mean(v)) if v else float("nan")
    caps = sum(e["blue_score"] for e in eps)
    return {
        "episodes": len(eps),
        "possessions": len(segs),
        "possessions_per_episode": len(segs) / len(eps),
        "total_captures": caps,
        "conversion_per_possession": caps / max(1, len(segs)),
        "mean_duration_steps": m("duration"),
        "mean_path_len": m("path_len"),
        "mean_straight_line": m("straight_line"),
        "mean_path_efficiency": m("efficiency"),
        "mean_heading_reversals": m("reversals"),
        "mean_teammate_dist": m("mate_dist_mean"),
        "mean_teammate_min_dist": m("mate_dist_min"),
        "mean_escort_fraction": m("escort_frac"),
        "mean_opponent_encounters": m("opp_encounters"),
        "mean_tags_during_possession": m("tags_during"),
        "escort_radius_cells": ESCORT_RADIUS,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--opponents", default="OP6,OP7")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/phase7/carrier_return.json")
    a = ap.parse_args()

    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy
    import experiments.run_g0_v2_evaluation as E

    ck = PROJECT_ROOT / CHECKPOINT
    if not ck.is_file():
        print(f"BLOCKED: missing checkpoint {ck}", file=sys.stderr)
        return 2
    payload = read_checkpoint_payload(str(ck), map_location="cpu")
    policy = load_policy(str(ck), device=a.device,
                         num_cnn_channels=E.resolve_cnn_channels(payload, context=str(ck)))

    out = {"record": "Phase 7E carrier-return analysis, 2v2 stock RULESET_V2",
           "checkpoint": CHECKPOINT,
           "checkpoint_precommitted_reason":
               "frozen best-fixed policy in the held-out repertoire-value "
               "evaluation; chosen before any carrier-return behaviour was seen",
           "seed_base": SEED_BASE, "arms": {}}

    for opp in [o.strip() for o in a.opponents.split(",") if o.strip()]:
        eps = [run_episode(policy=policy, opponent=opp, seed=SEED_BASE + i,
                           device=a.device) for i in range(a.seeds)]
        s = summarize(eps)
        out["arms"][opp] = s
        print(f"{opp:6s} poss/ep={s.get('possessions_per_episode',0):5.2f} "
              f"conv={s.get('conversion_per_possession',0):.3f} "
              f"eff={s.get('mean_path_efficiency',float('nan')):.3f} "
              f"mate_d={s.get('mean_teammate_dist',float('nan')):6.2f} "
              f"escort={s.get('mean_escort_fraction',float('nan')):.3f} "
              f"revs={s.get('mean_heading_reversals',float('nan')):.2f}", flush=True)

    p = PROJECT_ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n-> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
