"""M1 2v2 payoff assay -- frozen Gate B, no searcher, no PPO.

M1 (own_flag_home_required_to_score=True) is already implemented and frozen.
This script does not edit it. It asks whether the single boolean creates a
two-way game-level reversal:

    OP6  GUARD_RAID     > DOUBLE_BREACH     (defence-demanding)
    OP7  DOUBLE_BREACH  > GUARD_RAID        (offence-demanding)

Gate B (frozen in experiments/M1_PAYOFF_ASSAY_FROZEN.json):
    paired win-rate advantage >= 0.15 AND LCB95 > 0 in BOTH directions.

Scores come from the terminal episode_result / capture_scored ledger, never
from post-reset core.blue_score. M1 binding telemetry is reported, not gated.

Run:  python experiments/m1_payoff_assay.py --seeds 32
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GUARD = "BLUE_ONE_DEFENDER_V2"
BREACH = "BLUE_BOTH_ATTACK_V2"
STYLE_LABEL = {GUARD: "GUARD_RAID", BREACH: "DOUBLE_BREACH"}
STYLES = (GUARD, BREACH)
OPPONENTS = ("OP6", "OP7")

MAP = "map_a"
MAX_STEPS = 240
AGENTS = 2
SEED_BASE = 2_300_001
RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
FLOOR = 0.15
FLAG_HOME_EPS = 1e-3


def paired_ci(d: np.ndarray, rng, n_boot=20000, alpha=0.05):
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def _flag_home(core, team: str) -> bool:
    if team == "blue":
        d = ((core.blue_flag_pos[0] - core.blue_flag_home[0]) ** 2).sum()
    else:
        d = ((core.red_flag_pos[0] - core.red_flag_home[0]) ** 2).sum()
    return bool((d ** 0.5).item() <= FLAG_HOME_EPS)


def run_episode(*, style: str, opponent: str, seed: int, device: str) -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=MAP, max_decision_steps=MAX_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True, **RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
        actual = (env.env_method("get_opponent_key")[0] or "").strip().upper()
        if actual != opponent:
            raise RuntimeError(f"opponent mismatch: {actual!r} != {opponent!r}")
        core.blue_scripted = True
        core.set_blue_style(style)
        env.reset()
        core.drain_tag_events()

        n = 0
        term_scores = None
        blue_caps = red_caps = 0
        blue_stole = red_stole = False
        m1_block_steps = 0
        saw_block = False
        scored_after_block = False
        binding = False  # stole + blocked at home with own flag away

        for _ in range(MAX_STEPS):
            env.step_async(env.action_space.sample() * 0)
            _o, _r, done, _i = env.step_wait()
            n += 1
            terminal = bool(np.asarray(done).any())
            if terminal:
                i0 = _i[0] if isinstance(_i, (list, tuple)) else _i
                er = (i0 or {}).get("episode_result") or {}
                term_scores = (
                    int(er.get("blue_score", i0.get("blue_score", 0) if i0 else 0)),
                    int(er.get("red_score", i0.get("red_score", 0) if i0 else 0)),
                )

            blocked = False
            # Post-terminal core is episode N+1; do not read M1 tensors there.
            if not terminal:
                blk = getattr(core, "_m1_blue_blocked", None)
                if blk is not None and bool(blk[0].any().item()):
                    blocked = True
                    m1_block_steps += 1
                    saw_block = True

            for e in core.drain_tag_events():
                et = e.get("event_type")
                if et == "capture_scored":
                    if e.get("scoring_team") == "blue":
                        blue_caps += 1
                        if saw_block:
                            scored_after_block = True
                    else:
                        red_caps += 1

            if terminal:
                break

            if not _flag_home(core, "red"):
                blue_stole = True
            if not _flag_home(core, "blue"):
                red_stole = True
            if (blocked and blue_stole and red_stole
                    and bool(core.blue_carrying[0].any().item())):
                binding = True

        if term_scores is None:
            # Horizon truncation never auto-resets; live counters are valid.
            bs, rs = int(core.blue_score[0]), int(core.red_score[0])
        else:
            bs, rs = term_scores
        return {
            "episode_seed": seed, "opponent": opponent,
            "blue_style": style, "allocation": STYLE_LABEL[style],
            "map": MAP, "m1": True, "episode_steps": n,
            "blue_score": bs, "red_score": rs,
            "win": int(bs > rs), "blue_captures": blue_caps,
            "red_captures": red_caps,
            "blue_stole_red_flag": int(blue_stole),
            "red_stole_blue_flag": int(red_stole),
            "m1_block_steps": m1_block_steps,
            "m1_block_occurred": int(saw_block),
            "m1_scored_after_block": int(scored_after_block),
            "m1_binding_sequence": int(binding),
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="artifacts/strategic_demand/m1_payoff")
    a = ap.parse_args()

    out = PROJECT_ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("M1 2v2 PAYOFF ASSAY  own_flag_home_required_to_score=True")
    print("Gate B frozen: paired WR adv >= 0.15 AND LCB95>0 in BOTH directions")
    print("M1 will not be edited after these numbers.")
    print("=" * 78)

    rows = []
    for opp in OPPONENTS:
        for i in range(a.seeds):
            seed = SEED_BASE + i
            for style in STYLES:
                rows.append(run_episode(style=style, opponent=opp,
                                        seed=seed, device=a.device))
            if (i + 1) % 8 == 0 or i == 0:
                print(f"  {opp} paired seed {i + 1}/{a.seeds}", flush=True)

    fields = list(rows[0].keys())
    with (out / "episode_rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(7)

    def col(opp, alloc, k):
        return np.array([r[k] for r in rows
                         if r["opponent"] == opp and r["allocation"] == alloc],
                        dtype=float)

    cells = {}
    for opp in OPPONENTS:
        cells[opp] = {}
        for alloc in ("GUARD_RAID", "DOUBLE_BREACH"):
            wr = float(col(opp, alloc, "win").mean())
            cells[opp][alloc] = {
                "n": int(col(opp, alloc, "win").size),
                "win_rate": wr,
                "mean_blue_score": float(col(opp, alloc, "blue_score").mean()),
                "mean_red_score": float(col(opp, alloc, "red_score").mean()),
                "frac_m1_block": float(col(opp, alloc, "m1_block_occurred").mean()),
                "frac_binding_sequence": float(col(opp, alloc, "m1_binding_sequence").mean()),
                "frac_scored_after_block": float(col(opp, alloc, "m1_scored_after_block").mean()),
            }

    # OP6: GUARD should beat BREACH. OP7: BREACH should beat GUARD.
    d6 = col("OP6", "GUARD_RAID", "win") - col("OP6", "DOUBLE_BREACH", "win")
    d7 = col("OP7", "DOUBLE_BREACH", "win") - col("OP7", "GUARD_RAID", "win")
    m6, lo6, hi6 = paired_ci(d6, rng)
    m7, lo7, hi7 = paired_ci(d7, rng)
    op6_pass = m6 >= FLOOR and lo6 > 0
    op7_pass = m7 >= FLOOR and lo7 > 0
    verdict = "PASS" if (op6_pass and op7_pass) else "FAIL"

    gate = {
        "OP6_GUARD_minus_BREACH": {
            "delta_win_rate": m6, "LCB95": lo6, "UCB95": hi6,
            "floor": FLOOR, "verdict": "PASS" if op6_pass else "FAIL",
        },
        "OP7_BREACH_minus_GUARD": {
            "delta_win_rate": m7, "LCB95": lo7, "UCB95": hi7,
            "floor": FLOOR, "verdict": "PASS" if op7_pass else "FAIL",
        },
        "GATE_B": verdict,
    }

    summary = {
        "record": "M1 2v2 payoff assay",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "experiments/M1_PAYOFF_ASSAY_FROZEN.json",
        "m1": True, "seed_base": SEED_BASE, "seeds": a.seeds,
        "cells": cells, "gate_B": gate,
        "not_launched": ["searcher", "PPO", "specialists", "latent"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCELLS")
    for opp in OPPONENTS:
        for alloc in ("GUARD_RAID", "DOUBLE_BREACH"):
            c = cells[opp][alloc]
            print(f"  {opp:4s} {alloc:14s} WR={c['win_rate']:.3f}  "
                  f"B={c['mean_blue_score']:.2f} R={c['mean_red_score']:.2f}  "
                  f"M1block={c['frac_m1_block']:.2f} bind={c['frac_binding_sequence']:.2f}")
    print("\nGATE B")
    print(f"  OP6 GUARD-BREACH  {m6:+.3f}  LCB95={lo6:+.3f}  [{gate['OP6_GUARD_minus_BREACH']['verdict']}]")
    print(f"  OP7 BREACH-GUARD  {m7:+.3f}  LCB95={lo7:+.3f}  [{gate['OP7_BREACH_minus_GUARD']['verdict']}]")
    print(f"  GATE_B = {verdict}")
    print(f"\n-> {out / 'summary.json'}")

    md = PROJECT_ROOT / "artifacts/strategic_demand/HUMAN_DECISION_REQUIRED_M1.md"
    md.write_text(
        f"# HUMAN_DECISION_REQUIRED — M1 payoff assay\n\n"
        f"Generated {summary['utc']}. M1 was not edited after these numbers.\n\n"
        f"## Gate B: **{verdict}**\n\n"
        f"- OP6 GUARD − BREACH: ΔWR={m6:+.3f}, LCB95={lo6:+.3f} → "
        f"{gate['OP6_GUARD_minus_BREACH']['verdict']}\n"
        f"- OP7 BREACH − GUARD: ΔWR={m7:+.3f}, LCB95={lo7:+.3f} → "
        f"{gate['OP7_BREACH_minus_GUARD']['verdict']}\n\n"
        f"Floor 0.15 AND LCB95>0 required in both directions.\n\n"
        f"Not started: searcher, PPO, specialists, latent.\n"
        f"Next step is a human decision. Do not retune M1.\n",
        encoding="utf-8")
    print(f"-> {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
