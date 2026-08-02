#!/usr/bin/env python3
"""Read-only Reward V3 economy scan — static + scripted trajectories.

Does NOT touch live V3 training processes, their configs, or their artifact
directories. Writes only to artifacts/reward_v3_scan/.

Order (matches the requested priority):
  1. Discounted terminal value vs dense shaping
  2. Full reward-source enumeration
  3. Scripted whole-trajectory reward comparison (Gate-2D reward equivalent)
  4–9. offense / team / PBRS / OOB / failed-commit / symmetry notes

V3 reward knobs are COPIED here (tags=0, failed_commit=-0.004). Everything
else inherits GPUFieldConfig defaults matching RULESET_V2_AQUATICUS_10S.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_manager import (  # noqa: E402
    ACTION_FAILED_PUNISHMENT,
    DRAW_TEAM_PENALTY,
    ENEMY_MAV_KILL_REWARD,
    FLAG_CARRY_HOME_REWARD,
    FLAG_PICKUP_REWARD,
    LOSE_TEAM_PUNISH,
    SPARSE_FLAG_CAPTURE_POINTS,
    SPARSE_MINE_TAG_POINTS,
    SPARSE_OOB_POINTS,
    SPARSE_TAG_NO_FLAG_POINTS,
    SPARSE_TAG_WITH_FLAG_POINTS,
    WIN_TEAM_REWARD,
)

# Copied from experiments/run_reward_v3_probe.py — do not import that module's
# training entrypoints (keeps this scan isolated from the live probe).
V3_TAG_NOFLAG = 0.0
V3_TAG_CARRIER = 0.0
V3_FAILED_COMMIT = -0.004
BASELINE_FAILED_COMMIT = ACTION_FAILED_PUNISHMENT  # -0.2

MAP = "map_a"
RESOLVED_MAP = "map_a_open"
OPPONENT = "OP6"
AGENTS = 2
HORIZON = 240
SEED_BASE = 3_100_001  # fresh block; unused by V3 training
N_EPISODES = 16

RULESET = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)

# Scripted blue treatments. Names map to existing style IDs where possible.
STYLES = (
    "BLUE_BOTH_ATTACK_V2",
    "BLUE_ONE_DEFENDER_V2",
    "BLUE_TURTLE",   # proxy for ALWAYS_HOME / passive camp
    "BLUE_RUSH",     # aggressive dual pressure
    "BLUE_ESCORT",
)

REWARD_KEYS = (
    "reward_terminal",
    "reward_offense",
    "reward_pbrs",
    "reward_team",
    "reward_sparse",
    "reward_failure",
    "reward_total",
)


def static_enumeration() -> dict:
    """Build the reward-source table from code constants + V3 overrides."""
    dense_w = 0.25
    sparse_w = 1.0
    pbrs_g = 0.995
    ppo_g = 0.995
    # Measured baseline event rates from REWARD_V3_DESIGN.json (pre-V3 policy).
    tags_per_ep = 17.0
    fails_per_ep = 184.0
    captures_per_ep = 2.2  # design estimate of sparse capture contribution / 1.0

    rows = [
        {
            "source": "terminal_win",
            "raw_value": WIN_TEAM_REWARD,
            "path": "rterm -> _reward_total (unsquashed component)",
            "frequency": "once per episode (terminal)",
            "farmable": False,
            "requires_objective": True,
            "expected_ep_contrib_v3": WIN_TEAM_REWARD,
            "notes": "win +1 / lose -1 / draw -0.5",
        },
        {
            "source": "sparse_capture",
            "raw_value": SPARSE_FLAG_CAPTURE_POINTS,
            "path": "sparse_points/100 * sparse_weight",
            "frequency": "once per capture",
            "farmable": "Hard",
            "requires_objective": True,
            "expected_ep_contrib_v3": sparse_w * (SPARSE_FLAG_CAPTURE_POINTS / 100.0) * captures_per_ep,
            "notes": f"composed contrib = {sparse_w}*(points/100) per event",
        },
        {
            "source": "sparse_tag_no_flag",
            "raw_value_baseline": SPARSE_TAG_NO_FLAG_POINTS,
            "raw_value_v3": V3_TAG_NOFLAG,
            "path": "sparse_points/100 * sparse_weight",
            "frequency": "once per tag",
            "farmable": "Was yes; V3 zeros it",
            "requires_objective": False,
            "expected_ep_contrib_v3": 0.0,
        },
        {
            "source": "sparse_tag_carrier",
            "raw_value_baseline": SPARSE_TAG_WITH_FLAG_POINTS,
            "raw_value_v3": V3_TAG_CARRIER,
            "path": "sparse_points/100 * sparse_weight",
            "frequency": "once per carrier tag",
            "farmable": "Was yes; V3 zeros it",
            "requires_objective": "Partly",
            "expected_ep_contrib_v3": 0.0,
        },
        {
            "source": "offense_enemy_mav_kill",
            "raw_value": ENEMY_MAV_KILL_REWARD,
            "path": "roff DIRECT (NOT sparse, NOT /100)",
            "frequency": "once per tag (and per mine-tag)",
            "farmable": "YES — still live under V3",
            "requires_objective": False,
            "expected_ep_contrib_v3": ENEMY_MAV_KILL_REWARD * tags_per_ep,
            "notes": (
                "CRITICAL: V3 closed sparse tag points but left enemy_mav_kill_reward=0.5 "
                "in the offense channel. At 17 tags/ep this is ~+8.5 raw if BLUE tags, "
                "or ~-8.5 if RED tags BLUE — dwarfs terminal +/-1."
            ),
        },
        {
            "source": "offense_flag_pickup",
            "raw_value": FLAG_PICKUP_REWARD,
            "path": "roff DIRECT",
            "frequency": "once per grab",
            "farmable": "Hard",
            "requires_objective": "Partly",
            "expected_ep_contrib_v3": FLAG_PICKUP_REWARD * captures_per_ep,  # order-of-magnitude
        },
        {
            "source": "offense_flag_carry_home",
            "raw_value": FLAG_CARRY_HOME_REWARD,
            "path": "roff DIRECT",
            "frequency": "once per capture (also sparse capture pays)",
            "farmable": "Hard",
            "requires_objective": True,
            "expected_ep_contrib_v3": FLAG_CARRY_HOME_REWARD * captures_per_ep,
            "notes": "Double-pays with sparse capture on the same event",
        },
        {
            "source": "offense_mine_place",
            "raw_value": 0.2,
            "path": "roff DIRECT (enabled_mine_reward)",
            "frequency": "once per placement",
            "farmable": "Maybe",
            "requires_objective": False,
        },
        {
            "source": "sparse_mine_tag",
            "raw_value": SPARSE_MINE_TAG_POINTS,
            "path": "sparse_points/100 * sparse_weight (HARDCODED constant)",
            "frequency": "once per mine tag",
            "farmable": "Maybe",
            "requires_objective": False,
            "notes": "Not exposed as a V3 knob; still +100 points if mines fire",
        },
        {
            "source": "sparse_oob",
            "raw_value": SPARSE_OOB_POINTS,
            "path": "_sparse_reward_points uses SPARSE_OOB_POINTS constant (cfg.sparse_oob_points is instrumentation-only today)",
            "frequency": "once per OOB agent-step",
            "farmable": "avoidance exploit if aggressive play OOBs",
            "requires_objective": False,
            "expected_ep_contrib_v3": "UNMEASURED rate",
        },
        {
            "source": "failed_commit",
            "raw_value_baseline": BASELINE_FAILED_COMMIT,
            "raw_value_v3": V3_FAILED_COMMIT,
            "path": "rfail DIRECT",
            "frequency": "once per ended unsuccessful macro (per agent)",
            "farmable": "avoidance / inactivity",
            "requires_objective": False,
            "expected_ep_contrib_v3": V3_FAILED_COMMIT * fails_per_ep,
            "notes": (
                "ended_commit = success | ticks_left<=0 | ~alive | tagged; "
                "failed = ended & ~success & was_alive. Tagged useful attacks count as failures."
            ),
        },
        {
            "source": "team_defense_presence",
            "raw_value": 0.03,
            "path": "rteam * dense_weight",
            "frequency": "every step while red has flag and blue near home",
            "farmable": "camping while losing can still collect this",
            "requires_objective": False,
            "composed_per_step": dense_w * 0.03,
        },
        {
            "source": "team_escort",
            "raw_value": 0.02,
            "path": "rteam * dense_weight",
            "frequency": "every step while blue has flag and teammate within 5",
            "farmable": False,
            "requires_objective": "Partly",
            "composed_per_step": dense_w * 0.02,
        },
        {
            "source": "team_intercept",
            "raw_value": 0.02,
            "path": "rteam * dense_weight",
            "frequency": "every step while red has flag and blue within 5 of carrier",
            "farmable": "Maybe",
            "requires_objective": "Partly",
            "composed_per_step": dense_w * 0.02,
        },
        {
            "source": "spin_penalty",
            "raw_value": 0.05,
            "path": "subtracted inside rteam * dense_weight",
            "frequency": "every step (yaw without motion)",
            "farmable": "avoidance",
            "requires_objective": False,
        },
        {
            "source": "idle_penalty",
            "raw_value": 0.03,
            "path": "subtracted inside rteam * dense_weight",
            "frequency": "every step if mean blue speed < 0.15",
            "farmable": "ALWAYS_HOME pays this continuously",
            "requires_objective": False,
            "composed_per_step": -dense_w * 0.03,
        },
        {
            "source": "pbrs_attack_return_defend",
            "raw_value": "coef=0.5 each",
            "path": "F=coef*(pbrs_gamma*Phi'-Phi); then * dense_weight",
            "frequency": "every step (phase-masked)",
            "farmable": "ideally no (potential-based); bugs can create camping attractors",
            "requires_objective": "ideally no",
            "pbrs_gamma": pbrs_g,
            "ppo_gamma": ppo_g,
            "gamma_match": abs(pbrs_g - ppo_g) < 1e-12,
        },
        {
            "source": "stalemate_penalty",
            "raw_value": -0.08,
            "path": "_reward_total when stalemate_steps >= 120",
            "frequency": "once when triggered (also truncates)",
            "farmable": "avoidance",
            "requires_objective": False,
        },
    ]

    disc = {
        "ppo_gamma": ppo_g,
        "gae_lambda": 0.99,
        "horizon": HORIZON,
        "discount_at_step": {
            str(t): round(pbrs_g ** t, 4)
            for t in (50, 100, 150, 200, 220, 240)
        },
        "terminal_win_face_value": WIN_TEAM_REWARD,
        "terminal_win_discounted_if_at_step_240": round(WIN_TEAM_REWARD * (pbrs_g ** 240), 4),
        "terminal_win_discounted_if_at_step_200": round(WIN_TEAM_REWARD * (pbrs_g ** 200), 4),
        "enemy_mav_kill_immediate": ENEMY_MAV_KILL_REWARD,
        "enemy_mav_vs_discounted_terminal_at_240": round(
            ENEMY_MAV_KILL_REWARD / max(1e-9, WIN_TEAM_REWARD * (pbrs_g ** 240)), 2
        ),
        "note": (
            "At gamma=0.995, a win at step 240 is worth ~30% of face value. "
            "One immediate offense kill reward (0.5) is already ~1.66x that "
            "discounted terminal. Seventeen such events dominate completely."
        ),
    }

    return {
        "v3_overrides": {
            "sparse_tag_no_flag_points": V3_TAG_NOFLAG,
            "sparse_tag_with_flag_points": V3_TAG_CARRIER,
            "action_failed_punishment": V3_FAILED_COMMIT,
        },
        "composition": {
            "raw": "rterm + roff + rfail + dense_weight*(rpbrs+rteam) + sparse_weight*(sparse_points/100) [+ stalemate]",
            "then": "tanh(raw / reward_scale=4.0), clip +/- reward_clip=1.0",
            "dense_weight": dense_w,
            "sparse_weight": sparse_w,
        },
        "sources": rows,
        "discounting": disc,
        "hidden_channel_alert": (
            "V3 zeros sparse tag points but enemy_mav_kill_reward (+/-0.5 per tag) "
            "remains in offense. Mine-tag sparse (+100) is also still hardcoded."
        ),
    }


def _info0(infos) -> dict:
    if isinstance(infos, (list, tuple)):
        return infos[0] if infos else {}
    if isinstance(infos, dict):
        # SB3 VecEnv sometimes returns dict-of-lists
        if "reward_total" in infos and not isinstance(infos["reward_total"], (list, tuple, np.ndarray)):
            return infos
        out = {}
        for k, v in infos.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                out[k] = v[0]
            else:
                out[k] = v
        return out
    return {}


def run_style_episode(style: str, seed: int, device: str) -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=AGENTS,
        max_red_agents=AGENTS,
        map_set="train",
        map_layout=MAP,
        max_decision_steps=HORIZON,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
        obstacle_obs_channel=True,
        tag_telemetry_enabled=True,
        # V3 knobs (copied; do not mutate shared module globals)
        sparse_tag_no_flag_points=V3_TAG_NOFLAG,
        sparse_tag_with_flag_points=V3_TAG_CARRIER,
        action_failed_punishment=V3_FAILED_COMMIT,
        **RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", OPPONENT)
        env.env_method("set_next_opponent", "SCRIPTED", OPPONENT)
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()
        core.drain_tag_events()

        sums = {k: 0.0 for k in REWARD_KEYS}
        disc_return = 0.0
        gamma = 0.995
        gpow = 1.0
        steps = 0
        oob_blue = oob_red = 0
        tags_blue = tags_red = 0
        denials = 0
        # Scores must come from the TERMINAL info payload. After done, the vec
        # env may auto-reset and core.blue_score/red_score belong to episode N+1.
        bs = rs = 0
        terminal_info = {}

        for _ in range(HORIZON):
            env.step_async(env.action_space.sample() * 0)
            _o, rewards, dones, infos = env.step_wait()
            steps += 1
            info = _info0(infos)
            r = float(np.asarray(rewards).reshape(-1)[0])
            disc_return += gpow * r
            gpow *= gamma
            for k in REWARD_KEYS:
                if k in info:
                    sums[k] += float(info[k])
            if "reward_total" not in info:
                sums["reward_total"] += r

            for e in core.drain_tag_events():
                et = e.get("event_type")
                if et == "tag_success":
                    if e.get("tagger_team") == "blue":
                        tags_blue += 1
                    else:
                        tags_red += 1
                elif et == "tag_denied":
                    denials += 1
                elif et == "out_of_bounds":
                    if e.get("team") == "blue":
                        oob_blue += 1
                    else:
                        oob_red += 1

            if "blue_score" in info:
                bs = int(info["blue_score"])
            if "red_score" in info:
                rs = int(info["red_score"])

            if bool(np.asarray(dones).any()):
                terminal_info = info
                if "blue_score" in info:
                    bs = int(info["blue_score"])
                if "red_score" in info:
                    rs = int(info["red_score"])
                break

        return {
            "blue_style": style,
            "episode_seed": seed,
            "steps": steps,
            "blue_score": bs,
            "red_score": rs,
            "win": int(bs > rs),
            "loss": int(bs < rs),
            "draw": int(bs == rs),
            "terminal_reward_from_info": float(terminal_info.get("reward_terminal", float("nan"))),
            "discounted_return": disc_return,
            "undiscounted_reward_total": sums["reward_total"],
            "sum_terminal": sums["reward_terminal"],
            "sum_offense": sums["reward_offense"],
            "sum_pbrs": sums["reward_pbrs"],
            "sum_team": sums["reward_team"],
            "sum_sparse": sums["reward_sparse"],
            "sum_failure": sums["reward_failure"],
            "tags_blue": tags_blue,
            "tags_red": tags_red,
            "tag_denials": denials,
            "oob_blue": oob_blue,
            "oob_red": oob_red,
            "ruleset_id": str(cfg.ruleset_id),
        }
    finally:
        env.close()


def aggregate(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["blue_style"]].append(r)
    out = {}
    for style, eps in by.items():
        def mean(key):
            return float(np.mean([e[key] for e in eps]))

        out[style] = {
            "n": len(eps),
            "win_rate": mean("win"),
            "loss_rate": mean("loss"),
            "draw_rate": mean("draw"),
            "mean_blue_score": mean("blue_score"),
            "mean_red_score": mean("red_score"),
            "mean_discounted_return": mean("discounted_return"),
            "mean_undiscounted_total": mean("undiscounted_reward_total"),
            "mean_terminal": mean("sum_terminal"),
            "mean_offense": mean("sum_offense"),
            "mean_pbrs": mean("sum_pbrs"),
            "mean_team": mean("sum_team"),
            "mean_sparse": mean("sum_sparse"),
            "mean_failure": mean("sum_failure"),
            "mean_tags_blue": mean("tags_blue"),
            "mean_tags_red": mean("tags_red"),
            "mean_oob_blue": mean("oob_blue"),
            "mean_oob_red": mean("oob_red"),
        }
    return out


def alignment_verdict(agg: dict) -> dict:
    """Critical Gate-2D-style reward test: does winning beat camping?"""
    rush = agg.get("BLUE_BOTH_ATTACK_V2") or agg.get("BLUE_RUSH")
    camp = agg.get("BLUE_TURTLE")
    if not rush or not camp:
        return {"status": "INCOMPLETE", "reason": "missing styles"}

    d_rush = rush["mean_discounted_return"]
    d_camp = camp["mean_discounted_return"]
    u_rush = rush["mean_undiscounted_total"]
    u_camp = camp["mean_undiscounted_total"]
    win_better_disc = d_rush > d_camp
    win_better_undisc = u_rush > u_camp
    scored = (rush["mean_blue_score"] + rush["mean_red_score"]
              + camp["mean_blue_score"] + camp["mean_red_score"]) > 0.0
    # If nobody ever scores, discounted-return ordering is not evidence about
    # win-vs-camp alignment — only about shaping under 0-0 draws.
    if not scored:
        return {
            "status": "INCONCLUSIVE_NO_SCORE",
            "test": "discounted_return(attack) > discounted_return(camp) UNDER REAL WINS",
            "attack_style": "BLUE_BOTH_ATTACK_V2" if "BLUE_BOTH_ATTACK_V2" in agg else "BLUE_RUSH",
            "camp_style": "BLUE_TURTLE",
            "attack_discounted": d_rush,
            "camp_discounted": d_camp,
            "attack_undiscounted": u_rush,
            "camp_undiscounted": u_camp,
            "attack_win_rate": rush["win_rate"],
            "camp_win_rate": camp["win_rate"],
            "attack_mean_blue_score": rush["mean_blue_score"],
            "camp_mean_blue_score": camp["mean_blue_score"],
            "margin_discounted": d_rush - d_camp,
            "interpretation": (
                "All scripted styles finished 0-0 on this opponent/seed block. "
                "Return ordering under pure draws cannot certify that winning "
                "beats camping. Re-run vs a softer opponent or longer horizon, "
                "or inject a forced-capture trajectory. Hidden offense tag "
                "payoffs can still be compared from the per-style offense sums."
            ),
        }

    outcome_ok = rush["win_rate"] > camp["win_rate"] or rush["mean_blue_score"] > camp["mean_blue_score"]
    status = "PASS" if (win_better_disc and outcome_ok) else "FAIL"
    return {
        "status": status,
        "test": "discounted_return(BOTH_ATTACK/RUSH) > discounted_return(TURTLE/camp)",
        "attack_style": "BLUE_BOTH_ATTACK_V2" if "BLUE_BOTH_ATTACK_V2" in agg else "BLUE_RUSH",
        "camp_style": "BLUE_TURTLE",
        "attack_discounted": d_rush,
        "camp_discounted": d_camp,
        "attack_undiscounted": u_rush,
        "camp_undiscounted": u_camp,
        "attack_win_rate": rush["win_rate"],
        "camp_win_rate": camp["win_rate"],
        "margin_discounted": d_rush - d_camp,
        "win_better_discounted": win_better_disc,
        "win_better_undiscounted": win_better_undisc,
        "outcome_ok": outcome_ok,
        "interpretation": (
            "V3 reward economy favors attacking over camping on these seeds."
            if status == "PASS"
            else "V3 still misaligned: camping/losing earns >= discounted return of attacking."
        ),
    }


def pbrs_audit() -> dict:
    return {
        "form": "F = coef * (pbrs_gamma * Phi(s') - Phi(s))",
        "pbrs_gamma": 0.995,
        "ppo_gamma": 0.995,
        "gamma_match": True,
        "phase_masks": {
            "attack": "active only when NOT carrying before AND after (pickup transition excluded)",
            "return": "active only when carrying before AND after (capture transition excluded)",
            "defend": "active only when red has flag before AND after",
        },
        "terminal_potential": (
            "No explicit Phi(terminal)=0 reset found in _pbrs_reward; "
            "episode boundary handled by env reset of positions/carrying. "
            "Stale prev_* snapshot on the terminal->reset step is a known class of bugs — "
            "verify via unit tests that the first step after reset does not compare "
            "against the previous episode's carrier state."
        ),
        "signs": "BLUE potentials only (attack/return/defend); no mirrored RED Phi in PBRS",
    }


def failed_commit_semantics() -> dict:
    return {
        "definition": (
            "ended_commit = blue_commit_success | (ticks_left<=0) | (~alive) | tagged; "
            "failed_commit = ended_commit & (~success) & prev_alive"
        ),
        "per_agent": True,
        "both_agents_same_step": "Yes — sum(dim=1), so 2 agents can each add one penalty",
        "tagged_converts_to_failure": True,
        "episode_end_unfinished": (
            "ticks expire or tag/death ends commit; truncation itself does not specially "
            "mark remaining commits beyond ticks_left countdown"
        ),
        "helpful_but_failed": (
            "A commit that helped a teammate but did not hit its own success predicate "
            "still pays the failure penalty"
        ),
        "v3_cost_at_184_events": V3_FAILED_COMMIT * 184.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=N_EPISODES)
    p.add_argument("--styles", nargs="+", default=list(STYLES))
    p.add_argument("--static-only", action="store_true")
    p.add_argument("--out-dir", default="artifacts/reward_v3_scan")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    static = static_enumeration()
    static["pbrs_audit"] = pbrs_audit()
    static["failed_commit_semantics"] = failed_commit_semantics()
    static["authored_utc"] = datetime.now(timezone.utc).isoformat()
    static["isolation"] = {
        "touches_live_v3_training": False,
        "out_dir": str(out_dir),
        "live_v3_dir": "artifacts/reward_v3/ (read-only if referenced)",
    }
    (out_dir / "static_enumeration.json").write_text(json.dumps(static, indent=2) + "\n")

    print("=" * 78)
    print("REWARD V3 ECONOMY SCAN (read-only; live V3 training untouched)")
    print("=" * 78)
    print(f"HIDDEN CHANNEL: {static['hidden_channel_alert']}")
    d = static["discounting"]
    print(f"gamma={d['ppo_gamma']}  disc@240={d['discount_at_step']['240']}  "
          f"kill/disc_terminal={d['enemy_mav_vs_discounted_terminal_at_240']}x")
    print(f"wrote {out_dir / 'static_enumeration.json'}")

    if args.static_only:
        return 0

    rows = []
    cells = [(s, SEED_BASE + i) for s in args.styles for i in range(args.episodes)]
    for idx, (style, seed) in enumerate(cells, 1):
        print(f"[{idx}/{len(cells)}] {style} seed={seed}", flush=True)
        rows.append(run_style_episode(style, seed, args.device))

    rows_path = out_dir / "trajectory_rows.csv"
    with open(rows_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    agg = aggregate(rows)
    verdict = alignment_verdict(agg)
    report = {
        "map": MAP,
        "resolved_map": RESOLVED_MAP,
        "opponent": OPPONENT,
        "n_episodes_per_style": args.episodes,
        "seed_base": SEED_BASE,
        "v3_knobs": static["v3_overrides"],
        "aggregate": agg,
        "alignment_verdict": verdict,
        "hidden_channel_alert": static["hidden_channel_alert"],
        "discounting": static["discounting"],
    }
    (out_dir / "trajectory_summary.json").write_text(json.dumps(report, indent=2) + "\n")

    print("\nSTYLE SUMMARY (mean discounted return / win rate / offense / team / failure / oob_blue)")
    for style, a in agg.items():
        print(
            f"  {style:<22s}  Rdisc={a['mean_discounted_return']:+.3f}  "
            f"WR={a['win_rate']:.2f}  off={a['mean_offense']:+.3f}  "
            f"team={a['mean_team']:+.3f}  fail={a['mean_failure']:+.3f}  "
            f"oobB={a['mean_oob_blue']:.2f}  tagsB/R={a['mean_tags_blue']:.1f}/{a['mean_tags_red']:.1f}"
        )
    print(f"\nALIGNMENT VERDICT: {verdict['status']}")
    print(f"  {verdict['interpretation']}")
    print(f"  margin_discounted={verdict.get('margin_discounted', float('nan')):+.3f}")
    print(f"wrote {rows_path}")
    print(f"wrote {out_dir / 'trajectory_summary.json'}")
    # INCONCLUSIVE is exit 3 (measurement gap); FAIL is exit 2; PASS is 0.
    if verdict["status"] == "PASS":
        return 0
    if verdict["status"] == "INCONCLUSIVE_NO_SCORE":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
