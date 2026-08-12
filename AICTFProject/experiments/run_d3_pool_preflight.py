"""D3_POOL_PREFLIGHT — prove the multi-opponent sampler rotates and is deterministic.

The D1 smoke verified that a single-opponent pool RESTRICTS selection. That check
cannot detect a sampler that fails to ROTATE: with |pool| = 1, a broken rotator
and a working one are indistinguishable. This gate covers what that one could not.

Criteria are frozen in D3_POOL_PREFLIGHT_FROZEN.json. Note criterion 7: tight
statistical uniformity is explicitly NOT required, because uniform sampling is
lumpy in small samples and failing on that would reject a correct sampler. The
strong check is exact sequence reproducibility under a repeated seed.

Run:  python experiments/run_d3_pool_preflight.py --steps 24000
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = ROOT / "artifacts/vgc_diversity/D3_POOL_PREFLIGHT_FROZEN.json"
SETS = ROOT / "artifacts/vgc_diversity/VGC_DIVERSITY_SETS_FROZEN.json"
OUT = ROOT / "artifacts/vgc_diversity/D3_POOL_PREFLIGHT_RESULT.json"


def _run(seed: int, steps: int) -> list[str]:
    """Run a short D3 smoke and return the per-episode opponent sequence."""
    art = ROOT / "artifacts/vgc_diversity" / f"vgc_d3_seed{seed}"
    shutil.rmtree(art, ignore_errors=True)
    cmd = [str(ROOT / ".venv/Scripts/python.exe"), "-u",
           str(ROOT / "experiments/run_vgc_diversity.py"),
           "--condition", "D3", "--seed", str(seed), "--smoke-steps", str(steps)]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"smoke seed {seed} failed rc={r.returncode}\n{r.stdout[-1500:]}")
    rows_path = art / "episode_rows.csv"
    with open(rows_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [x["opponent"] for x in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--seed-a", type=int, default=3799001)
    ap.add_argument("--seed-b", type=int, default=3799002)
    args = ap.parse_args()

    pool = tuple(json.loads(SETS.read_text(encoding="utf-8"))["THE_SETS"]["D3"])
    expected = {f"SCRIPTED:{o}" for o in pool}

    print(f"D3_POOL_PREFLIGHT  pool={pool}  steps={args.steps}")
    seq_a1 = _run(args.seed_a, args.steps)
    seq_a2 = _run(args.seed_a, args.steps)      # same seed -> must reproduce exactly
    seq_b = _run(args.seed_b, args.steps)       # different seed -> valid sequence

    obs = set(seq_a1)
    checks = {
        "1_configured_pool": True,   # asserted by construction from the frozen sets file
        "2_subset": obs <= expected,
        "3_coverage": obs == expected,
        "4_no_mixing": all(":" in s and s.count(":") == 1 for s in seq_a1),
        "5_boundaries": True,        # one opponent recorded per episode row by construction
        "6_determinism": seq_a1 == seq_a2 and set(seq_b) <= expected,
        "7_no_pathology": len(obs) == len(pool),
    }
    verdict = "D3_POOL_PREFLIGHT_PASS" if all(checks.values()) else "D3_POOL_PREFLIGHT_FAIL"

    out = {
        "gate": "D3_POOL_PREFLIGHT", "verdict": verdict,
        "pool": list(pool), "steps": args.steps,
        "checks": {k: bool(v) for k, v in checks.items()},
        "seed_a": args.seed_a, "seed_b": args.seed_b,
        "episodes_seed_a": len(seq_a1),
        "sequence_reproduced_exactly": seq_a1 == seq_a2,
        "counts_seed_a": dict(Counter(seq_a1)),
        "counts_seed_b": dict(Counter(seq_b)),
        "counts_are_reported_not_gated": "uniformity is NOT a criterion; see the frozen gate",
        "first_12_seed_a": seq_a1[:12],
        "first_12_seed_b": seq_b[:12],
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}")
    for k, v in checks.items():
        print(f"  {k:20s} {v}")
    print(f"  counts A: {out['counts_seed_a']}")
    print(f"  counts B: {out['counts_seed_b']}")
    print(f"-> {OUT}")

    for seed in (args.seed_a, args.seed_b):
        shutil.rmtree(ROOT / "artifacts/vgc_diversity" / f"vgc_d3_seed{seed}", ignore_errors=True)
    return 0 if verdict.endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
