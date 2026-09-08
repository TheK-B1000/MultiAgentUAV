"""C7 Stage 1 — did increasing team size actually expand allocation capacity?

Asked BEFORE any reversal question, because an inert manipulation would produce
a NO_REVERSAL that says nothing about team size. If capacity did not expand,
C7 stops here and Stage 2's block 9920000 stays unspent.

The frozen criterion (C7_TEAM_SIZE_DEMAND_FROZEN.json):

    some opponent must achieve an offence+contest sum EXCEEDING the maximum
    observed at 2v2, which was 0.885 (OP7)

Measures are the C6 Stage-1 measures unchanged -- offensive_pressure,
home_retention, red_contest_fraction from bt_red_role -- reused by import rather
than reimplemented, so the two arms cannot drift apart through a transcription
error. The only difference is AGENTS, on both the training and evaluation sides.

The offence-vs-contest correlation is also reported, DESCRIPTIVELY, against the
2v2 value of -0.898. It is not a C7 criterion: the r >= -0.70 bar belongs to
SRCTF's affordance gate, which is a different experiment with a different
admission question. Conflating them would import a bar C7 never froze.

Run:  python experiments/run_c7_stage1_capacity.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FROZEN = ROOT / "artifacts/c7_preregistration/C7_TEAM_SIZE_DEMAND_FROZEN.json"
STAGE0 = ROOT / "artifacts/c7_stage0/C7_STAGE0_RESULT.json"
OUT = ROOT / "artifacts/c7_stage1/C7_STAGE1_RESULT.json"

C7_SEEDS = (3_300_001, 3_300_002, 3_300_003)
C7_AGENTS = 4
CEILING_2V2 = 0.885          # max offence+contest at 2v2 (OP7), from C6 Stage 1
HISTORICAL_R = -0.898        # descriptive reference only


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.exists():
        print(f"REFUSED: {OUT.name} already exists.", file=sys.stderr)
        return 2
    if not STAGE0.exists() or json.loads(STAGE0.read_text(encoding="utf-8"))["verdict"] != "STAGE0_PASS":
        print("REFUSED: Stage 0 has not passed. An incompetent baseline cannot be scanned.",
              file=sys.stderr)
        return 2

    # AGENTS must be rebound on BOTH sides before anything is measured. Patching
    # only one produced a 4-agent policy evaluated in a 2v2 env during Stage 0.
    import experiments.run_g0_v2_evaluation as E
    E.AGENTS = C7_AGENTS
    import experiments.run_c6_stage1_mechanical as C6   # measures, reused unchanged
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy
    from srctf.opponent_sets import get as opponent_set

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    base = int(frozen["stage_1_allocation_capacity"]["seed_block"][0])
    opponents = list(opponent_set("historical"))

    print("=" * 74)
    print("C7 STAGE 1 — allocation capacity at 4v4")
    print(f"  block {base}..{base + args.episodes - 1}   agents {C7_AGENTS}")
    print(f"  criterion: some opponent offence+contest > {CEILING_2V2} (2v2 max, OP7)")
    print("=" * 74, flush=True)

    per_policy = {}
    for pseed in C7_SEEDS:
        tag = f"c7_4v4_seed{pseed}"
        ck = ROOT / "artifacts" / "c7_stage0" / tag / "ckpts" / f"final_{tag}.zip"
        payload = read_checkpoint_payload(str(ck), map_location="cpu")
        pol = load_policy(str(ck), device=args.device,
                          num_cnn_channels=resolve_cnn_channels(payload, context=str(ck)))
        res = {}
        for opp in opponents:
            rows = []
            for i in range(args.episodes):
                rows += C6.measure(pol, opponent=opp, seed=base + i, device=args.device)
            s = C6.summarize(rows)
            s["offence_plus_contest"] = round(
                s["offensive_pressure"] + s["red_contest_fraction"], 4)
            res[opp] = s
            print(f"  {pseed} {opp:5s} offence={s['offensive_pressure']:.3f} "
                  f"contest={s['red_contest_fraction']:.3f} "
                  f"sum={s['offence_plus_contest']:.3f}", flush=True)
        per_policy[str(pseed)] = res

    sums = [(p, o, r["offence_plus_contest"])
            for p, res in per_policy.items() for o, r in res.items()]
    best_p, best_o, best_sum = max(sums, key=lambda t: t[2])
    ceiling_broken = best_sum > CEILING_2V2

    # Descriptive correlation, pooled across policies.
    off = np.array([r["offensive_pressure"] for res in per_policy.values() for r in res.values()])
    con = np.array([r["red_contest_fraction"] for res in per_policy.values() for r in res.values()])
    r_hat = (float(np.corrcoef(off, con)[0, 1])
             if off.std() > 0 and con.std() > 0 else None)

    verdict = "C7_STAGE1_CAPACITY_PASS" if ceiling_broken else "C7_CAPACITY_DID_NOT_EXPAND"

    out = {
        "record": "C7 Stage 1 — allocation capacity at 4v4",
        "verdict": verdict,
        "criterion": f"some opponent offence+contest > {CEILING_2V2} (the 2v2 maximum, OP7)",
        "max_offence_plus_contest": best_sum,
        "max_achieved_by": {"policy": best_p, "opponent": best_o},
        "ceiling_2v2": CEILING_2V2,
        "ceiling_broken": bool(ceiling_broken),
        "descriptive_correlation": {
            "r_offence_contest_4v4": None if r_hat is None else round(r_hat, 4),
            "r_offence_contest_2v2": HISTORICAL_R,
            "status": "DESCRIPTIVE ONLY. Not a C7 criterion. The r >= -0.70 bar belongs to "
                      "SRCTF's affordance gate, a different experiment with a different "
                      "admission question.",
        },
        "per_policy": per_policy,
        "seed_block": [base, base + args.episodes - 1],
        "agents": C7_AGENTS,
        "meaning": (
            "capacity expanded: the 4v4 manipulation is not inert, so a Stage 2 reversal "
            "result would be informative either way"
            if ceiling_broken else
            "capacity did NOT expand: the manipulation was inert and nothing is learned "
            "about demand. Stage 2 must NOT run and block 9920000 stays unspent."),
        "stage_2_authorized": bool(ceiling_broken),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}")
    print(f"  max offence+contest {best_sum:.4f} ({best_p}/{best_o}) vs 2v2 ceiling {CEILING_2V2}")
    print(f"  descriptive r: 4v4 {r_hat if r_hat is None else round(r_hat, 4)} "
          f"vs 2v2 {HISTORICAL_R}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
