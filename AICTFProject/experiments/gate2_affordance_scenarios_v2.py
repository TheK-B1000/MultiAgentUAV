#!/usr/bin/env python3
"""GATE 2 -- episode-level strategic affordances under RULESET_V2 on map_a.

The rule-level probe showed a lone defender CAN now tag (0% -> 100%). That is
necessary but not sufficient. This gate asks whether defensive allocation
actually changes OUTCOMES over full episodes, and -- critically -- whether it
COSTS something.

Both sides of the trade-off are required:

    one defender significantly improves home defense
    AND
    keeping one defender significantly reduces offensive output

Without the second half there is no opportunity cost, and therefore still no
reason for a policy pool to contain different allocations. That absence is
exactly what collapsed the strategy space under RULESET_V1.

Scenarios are driven by the scripted blue styles (no learned policy), so this
measures the ENVIRONMENT's affordances rather than any policy's skill.

Additional diagnostics (reported, not gated): whether escorting or decoy timing
against the tag cooldown changes outcomes. These are affordances, not evidence
of latent strategies.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAP = "map_a"
MAX_DECISION_STEPS = 240
AGENTS = 2
SEED_BASE = 1_600_001
OPPONENT = "OP6"  # a straightforward attacker; the gate is about OUR allocation

RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)

# Blue styles that differ in how many agents stay home.
STYLE_BOTH_ATTACK = "BLUE_RUSH"     # both forward
STYLE_ONE_DEFENDS = "BLUE_SPLIT"    # split posture
STYLE_BOTH_DEFEND = "BLUE_TURTLE"   # both home
STYLE_ESCORT = "BLUE_ESCORT"


def run_style(style: str, episodes: int, device: str, opponent: str) -> dict:
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    rows = []
    for ep in range(episodes):
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
            map_set="train", map_layout=MAP, max_decision_steps=MAX_DECISION_STEPS,
            aquaticus_profile=True, rules_profile="OURS", device=device,
            seed=SEED_BASE + ep, obstacle_obs_channel=True, **RULESET,
        )
        env = GPUCTFVecEnv(cfg)
        core = env.core
        try:
            env.env_method("set_phase", opponent)
            env.env_method("set_next_opponent", "SCRIPTED", opponent)
            core.blue_scripted = True
            core.set_blue_style(style)
            env.reset()
            mid = float(core.cols) * 0.5
            home_steps = 0
            red_in_our_half = 0
            n = 0
            for _ in range(MAX_DECISION_STEPS):
                env.step_async(env.action_space.sample() * 0)
                _o, _r, done, _i = env.step_wait()
                n += 1
                bx = core.blue_x[0].detach().cpu().numpy()
                rx = core.red_x[0].detach().cpu().numpy()
                home_steps += int(np.sum(bx <= mid))
                red_in_our_half += int(np.sum(rx <= mid))
                if bool(done.any()):
                    break
            rows.append({
                "blue_score": int(core.blue_score[0]),
                "red_score": int(core.red_score[0]),
                "margin": int(core.blue_score[0]) - int(core.red_score[0]),
                "defend_frac": home_steps / max(1, n * AGENTS),
                "red_incursion": red_in_our_half / max(1, n * AGENTS),
                "steps": n,
            })
        finally:
            env.close()

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    return {"style": style, "episodes": len(rows),
            "blue_score": m("blue_score"), "red_score": m("red_score"),
            "margin": m("margin"), "defend_frac": m("defend_frac"),
            "red_incursion": m("red_incursion"), "mean_steps": m("steps")}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponent", default=OPPONENT)
    p.add_argument("--out-dir", default="artifacts/gate2_affordance_v2")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print(f"GATE 2 -- strategic affordances under RULESET_V2, map_a, vs {args.opponent}")
    print("Requires BOTH: a defender improves defense AND costs offense.")
    print("=" * 84)

    styles = [STYLE_BOTH_ATTACK, STYLE_ONE_DEFENDS, STYLE_BOTH_DEFEND, STYLE_ESCORT]
    res = {}
    print(f"\n{'style':<16}{'blue':>8}{'red':>8}{'margin':>9}{'defend':>9}{'red_incur':>11}")
    for s in styles:
        r = run_style(s, args.episodes, args.device, args.opponent)
        res[s] = r
        print(f"{s:<16}{r['blue_score']:>8.2f}{r['red_score']:>8.2f}{r['margin']:>9.2f}"
              f"{r['defend_frac']:>9.2f}{r['red_incursion']:>11.3f}")

    atk, one = res[STYLE_BOTH_ATTACK], res[STYLE_ONE_DEFENDS]

    # (a) defense benefit: fewer red points conceded when one agent stays home
    defense_gain = atk["red_score"] - one["red_score"]
    # (b) offensive cost: fewer blue points scored when one agent stays home
    offense_cost = atk["blue_score"] - one["blue_score"]

    print(f"\n{'=' * 84}\nTRADE-OFF\n{'=' * 84}")
    print(f"  defense benefit (red_score both-attack minus one-defends) = {defense_gain:+.3f}")
    print(f"  offensive cost  (blue_score both-attack minus one-defends) = {offense_cost:+.3f}")
    print(f"  defend_frac: both-attack={atk['defend_frac']:.2f} one-defends={one['defend_frac']:.2f}")

    a_ok = defense_gain > 0.0
    b_ok = offense_cost > 0.0
    ok = a_ok and b_ok
    print(f"\n  (a) defender improves defense : {'PASS' if a_ok else 'FAIL'}")
    print(f"  (b) defender costs offense    : {'PASS' if b_ok else 'FAIL'}")
    print(f"  GATE 2: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("\n  Without BOTH halves there is no opportunity cost, so allocation is")
        print("  still a free choice and a policy pool has no reason to differ. Do NOT")
        print("  proceed to G0-v2 training; revisit the rules or the scenario scaling.")

    payload = {
        "gate": "gate2_affordance", "created_utc": datetime.now(timezone.utc).isoformat(),
        "ruleset_id": "RULESET_V2_AQUATICUS_10S", "ruleset": RULESET,
        "map": MAP, "resolved_map": "map_a_open", "opponent": args.opponent,
        "episodes_per_style": args.episodes,
        "seed_block": [SEED_BASE, SEED_BASE + args.episodes - 1],
        "by_style": res,
        "defense_benefit": defense_gain, "offense_cost": offense_cost,
        "defense_ok": bool(a_ok), "offense_cost_ok": bool(b_ok),
        "verdict": "PASS" if ok else "FAIL",
        "note": "Scripted blue styles only -- measures ENVIRONMENT affordances, not policy skill.",
    }
    (out_dir / "gate2_result.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[done] {out_dir / 'gate2_result.json'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
