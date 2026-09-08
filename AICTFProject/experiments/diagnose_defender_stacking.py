"""Diagnostic: how often does the normalized GUARD probe's defender stacking cost coverage?

Explains the 4v4 Pole-A reversal recorded in STRATEGIC_DEMAND_4v4_CERTIFICATION.json. This is
a MECHANISM diagnostic, not a gate: it changes no threshold, licenses no verdict, and reuses
the already-spent certification seeds deliberately (re-deriving mechanism from a published
verdict, as the decision-leverage diagnostic did).

A CORRECTION TO THE OBVIOUS METRIC
----------------------------------
The natural quantity is "P(two or more defenders select the same threat while another threat
is available)". But _scripted_blue_styles.py assigns

    target_x[:, lo:] = def_x.unsqueeze(1)

i.e. EVERY defender receives the identical target. Collision is therefore 1.0 BY
CONSTRUCTION, not stochastic -- measuring it would only re-derive the source code.

The quantity that actually varies, and that determines whether stacking COSTS anything, is
how often more than one distinct legal threat exists at the same time. With k simultaneous
legal intruders and ceil(N/2) stacked defenders, exactly one intruder is contested and k-1
are unopposed regardless of how many defenders GUARD commits.

So this measures, at real decision steps, the distribution of simultaneous legal intruders,
and reports the fraction of contested steps where stacking wastes defensive capacity.

"Legal intruder" uses the SAME predicate the behaviour tree uses -- alive, untagged, and on
our side -- read from the live core rather than reimplemented.

Run:  python experiments/diagnose_defender_stacking.py --sizes 2,4 --n-seeds 16 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "DEFENDER_STACKING_DIAGNOSTIC.json"

# The already-spent certification blocks. Reused deliberately: this is a diagnostic.
SEED_BASE = {2: 12_400_001, 4: 12_400_001, 6: 12_600_001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def probe_size(n: int, n_seeds: int, device: str) -> dict:
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome

    S.AGENTS = n
    genome = pole_A_genome(n)
    n_def = (n + 1) // 2

    counts: Counter = Counter()
    steps_total = 0
    episodes = 0

    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from experiments.sds_genome import apply_genome_to_core

    for i in range(n_seeds):
        seed = SEED_BASE[n] + i
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=n, max_red_agents=n,
            map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
            aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
            obstacle_obs_channel=True, tag_telemetry_enabled=True,
            own_flag_home_required_to_score=True, **S.RULESET,
        )
        env = GPUCTFVecEnv(cfg)
        core = env.core
        try:
            env.env_method("set_phase", genome.base_opponent)
            env.env_method("set_next_opponent", "SCRIPTED", genome.base_opponent)
            apply_genome_to_core(core, genome)
            core.blue_scripted = True
            core.set_blue_style(S.GUARD)
            env.reset()
            apply_genome_to_core(core, genome)

            for _ in range(S.MAX_STEPS):
                env.step_async(env.action_space.sample() * 0)
                _o, _r, done, _info = env.step_wait()

                # The behaviour tree's own legal-intruder predicate, read from the live core.
                red_alive = core.red_alive[0]
                red_tagged = core.red_tagged[0]
                on_our_side = core._is_on_home_side("blue", core.red_x)[0]
                intruders = int(
                    (red_alive & (~red_tagged) & on_our_side).sum().item())
                counts[intruders] += 1
                steps_total += 1

                if bool(np.asarray(done).any()):
                    break
            episodes += 1
        finally:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass

    contested = sum(v for k, v in counts.items() if k >= 1)
    multi = sum(v for k, v in counts.items() if k >= 2)
    return {
        "team_size": n,
        "defenders_committed": n_def,
        "episodes": episodes,
        "decision_steps": steps_total,
        "intruder_count_histogram": {str(k): counts[k] for k in sorted(counts)},
        "steps_with_any_intruder": contested,
        "steps_with_multiple_intruders": multi,
        "frac_of_contested_steps_that_are_multi_threat": (
            float(multi / contested) if contested else 0.0),
        "mean_intruders_when_contested": (
            float(sum(k * v for k, v in counts.items() if k >= 1) / contested)
            if contested else 0.0),
        "wasted_defenders_when_multi": max(0, n_def - 1),
        "unopposed_intruders_when_multi": (
            float(sum((k - 1) * v for k, v in counts.items() if k >= 2) / multi)
            if multi else 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="2,4")
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print("=" * 78)
    print(f"DEFENDER STACKING DIAGNOSTIC   sizes={sizes}  n_seeds={args.n_seeds}")
    print("  collision rate is 1.0 BY CONSTRUCTION (all defenders share one target);")
    print("  what varies is how often >=2 distinct threats exist to be covered.")
    print("=" * 78, flush=True)

    results = []
    for n in sizes:
        r = probe_size(n, int(args.n_seeds), args.device)
        results.append(r)
        print(f"\n  --- {n}v{n}  ({r['defenders_committed']} defenders committed) ---")
        print(f"    episodes {r['episodes']}  decision steps {r['decision_steps']}")
        print(f"    intruder histogram        {r['intruder_count_histogram']}")
        print(f"    steps with any intruder   {r['steps_with_any_intruder']}")
        print(f"    steps with >=2 intruders  {r['steps_with_multiple_intruders']}")
        print(f"    frac of contested steps that are multi-threat "
              f"{r['frac_of_contested_steps_that_are_multi_threat']:.4f}")
        print(f"    mean intruders when contested "
              f"{r['mean_intruders_when_contested']:.3f}")
        print(f"    unopposed intruders when multi-threat "
              f"{r['unopposed_intruders_when_multi']:.3f}")

    OUT.write_text(json.dumps({
        "record": "Defender stacking mechanism diagnostic",
        "status": "DIAGNOSTIC_NOT_A_GATE", "utc": _now(), "device": args.device,
        "explains": "STRATEGIC_DEMAND_4v4_CERTIFICATION.json (Pole A reversal)",
        "collision_rate_note": (
            "P(2+ defenders on the same threat) is 1.0 by construction -- "
            "_scripted_blue_styles.py assigns every defender the identical target. The "
            "measured quantity is instead how often >=2 distinct legal threats exist, i.e. "
            "how often that guaranteed stacking actually costs coverage."),
        "predicate": "alive & ~tagged & on_our_side, read from the live core",
        "seeds_reused_deliberately": "the already-spent certification blocks; this is a "
                                     "mechanism diagnostic and licenses no verdict",
        "results": results,
        "changes_no_threshold": True,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
