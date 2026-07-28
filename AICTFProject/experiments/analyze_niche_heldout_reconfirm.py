#!/usr/bin/env python3
"""Paired-bootstrap niche reconfirm for a finished scripted payoff matrix.

Use the *project* interpreter so NumPy is available::

    AICTFProject\\.venv\\Scripts\\python.exe \\
      experiments/analyze_niche_heldout_reconfirm.py \\
      --episode-csv artifacts/op9_split_heldout16_blue_probes_v2_seed521001/episode_results.csv \\
      --intended-best BLUE_SPLIT \\
      --out-json artifacts/op9_split_heldout16_blue_probes_v2_seed521001/reconfirm_verdict.json

Do **not** call bare ``python`` from PATH — system Python often lacks NumPy
and will exit 1 after a successful matrix collection.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_margins(path: Path) -> dict[str, dict[int, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by: dict[str, dict[int, float]] = {}
    for row in rows:
        by.setdefault(str(row["blue_style"]), {})[int(row["episode_index"])] = float(
            row["win_margin"]
        )
    if not by:
        raise ValueError(f"no rows in {path}")
    n = {s: len(eps) for s, eps in by.items()}
    if len(set(n.values())) != 1:
        raise ValueError(f"unequal episode counts across blues: {n}")
    return by


def _paired_ci(
    by: dict[str, dict[int, float]],
    a: str,
    b: str,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, object]:
    n = len(by[a])
    d = np.array([by[a][i] - by[b][i] for i in range(n)], dtype=np.float64)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for k in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[k] = d[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "mean": float(d.mean()),
        "ci95": [float(lo), float(hi)],
        "clear": bool(float(lo) > 0.0),
        "pos": int((d > 0).sum()),
        "neg": int((d < 0).sum()),
        "zero": int((d == 0).sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode-csv", type=Path, required=True)
    p.add_argument("--intended-best", required=True)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--n-boot", type=int, default=4000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--protocol", default="BLUE_PROBES_V3")
    p.add_argument("--base-seed", type=int, default=None)
    args = p.parse_args()

    by = _load_margins(Path(args.episode_csv))
    styles = sorted(by)
    n = len(by[styles[0]])
    intended = str(args.intended_best)
    if intended not in by:
        raise SystemExit(f"intended-best {intended!r} missing; have {styles}")

    rng = np.random.default_rng(int(args.seed))
    means = {s: float(np.mean(list(by[s].values()))) for s in styles}
    wrs = {
        s: float(np.mean([1.0 if by[s][i] > 0 else 0.0 for i in range(n)]))
        for s in styles
    }
    uniquely_best = max(styles, key=lambda s: means[s])

    pair: dict[str, object] = {}
    ok = uniquely_best == intended
    print(f"means: {{{', '.join(f'{k}: {v:.4f}' for k, v in means.items())}}}")
    print(f"wr: {{{', '.join(f'{k}: {v:.3f}' for k, v in wrs.items())}}}")
    print(f"uniquely_best: {uniquely_best}  intended: {intended}")
    print(f"paired {intended} advantages (bootstrap 95% CI):")
    for other in [s for s in styles if s != intended]:
        stats = _paired_ci(by, intended, other, rng=rng, n_boot=int(args.n_boot))
        pair[other] = stats
        ok = ok and bool(stats["clear"])
        print(
            f"  {intended} - {other}: mean={stats['mean']:+.4f} "
            f"CI95=[{stats['ci95'][0]:+.4f},{stats['ci95'][1]:+.4f}] "
            f"clear={stats['clear']} "
            f"(+{stats['pos']}/-{stats['neg']}/={stats['zero']})"
        )

    others = [s for s in styles if s != intended]
    pooled = np.array(
        [by[intended][i] - max(by[s][i] for s in others) for i in range(n)],
        dtype=np.float64,
    )
    boots = np.empty(int(args.n_boot), dtype=np.float64)
    for k in range(int(args.n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[k] = pooled[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pooled_clear = bool(float(lo) > 0.0)
    ok = ok and pooled_clear
    print(
        f"pooled vs best-other: mean={pooled.mean():+.4f} "
        f"CI95=[{lo:+.4f},{hi:+.4f}] clear={pooled_clear}"
    )

    verdict = "RECONFIRM_PASS" if ok else "RECONFIRM_FAIL"
    print(f"VERDICT: {verdict}")
    payload = {
        "verdict": verdict,
        "protocol": str(args.protocol),
        "base_seed": args.base_seed,
        "n_episodes": int(n),
        "intended_best": intended,
        "means": means,
        "win_rates": wrs,
        "uniquely_best": uniquely_best,
        "paired_advantages": pair,
        "pooled_vs_best_other": {
            "mean": float(pooled.mean()),
            "ci95": [float(lo), float(hi)],
            "clear": pooled_clear,
        },
        "episode_csv": str(Path(args.episode_csv).resolve()),
    }
    out_json = args.out_json
    if out_json is None:
        out_json = Path(args.episode_csv).resolve().parent / "reconfirm_verdict.json"
    Path(out_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
