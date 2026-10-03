r"""Read-only diagnostic of dual-branch training runs: how a side's own-pole performance moves over training.

    python experiments/diagnose_dual_branch_training.py --out <dir>

Reads only the existing episode_rows.csv of the 2v2 and 4v4 dual-branch A/B runs (no episodes, no training).
Per 20k-step bin: win rate, blue (own) score, red (conceded) score, margin, attack-route crossings, time to
first score, and the offense / terminal / failure reward parts -- next to the frozen DEFEND-teacher weight at
that step (lambda 0.1, linear decay 50k -> 150k, 0 after; from the run config). Separates "the side stops
scoring" from "the side concedes more", and lines the change up with the teacher schedule. Not available in
the logs and therefore NOT reported: per-update teacher loss, per-update DEFEND share (roles are fixed per
episode at k = ceil(N/3) by construction).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "2v2 A": "artifacts/scale_2v2_specialists/pi_A_specialist_2v2_dual_branch_v1",
    "2v2 B": "artifacts/scale_2v2_specialists/pi_B_specialist_2v2_dual_branch_v1",
    "4v4 A": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_dual_branch_v1",
    "4v4 B": "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_dual_branch_v1",
}
BIN = 20_000
LAM_PEAK, LAM_END, D0, D1 = 0.1, 0.0, 50_000, 150_000


def lam(step: float) -> float:
    if step <= D0:
        return LAM_PEAK
    if step >= D1:
        return LAM_END
    return LAM_PEAK + (LAM_END - LAM_PEAK) * (step - D0) / (D1 - D0)


def f(r: dict, k: str) -> float:
    v = r.get(k, "")
    return float(v) if v not in ("", None) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    out = ROOT / ap.parse_args().out
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    res, md = {}, ["# Dual-branch training diagnostic (read-only; episode_rows.csv)", "",
                   "DEFEND-teacher weight lambda: 0.1 to 50k, linear to 0 at 150k, 0 after (frozen schedule).", ""]
    for name, d in RUNS.items():
        p = ROOT / d / "episode_rows.csv"
        if not p.is_file():
            raise SystemExit(f"FAIL-CLOSED: missing {p}")
        cfg = next((ROOT / d).glob("*_run_config.json"), None)
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        steps = np.array([f(r, "timesteps") for r in rows])
        cols = {
            "win": [f(r, "success") for r in rows],
            "own_score": [f(r, "blue_score") for r in rows],
            "conceded": [f(r, "red_score") for r in rows],
            "margin": [f(r, "win_margin") for r in rows],
            "attack_crossings": [f(r, "blue_attack_upper_crossings") + f(r, "blue_attack_lower_crossings") for r in rows],
            "time_to_first_score": [f(r, "time_to_first_score") for r in rows],
            "reward_offense": [f(r, "reward_offense") for r in rows],
            "reward_terminal": [f(r, "reward_terminal") for r in rows],
            "reward_failure": [f(r, "reward_failure") for r in rows],
        }
        cols = {k: np.array(v) for k, v in cols.items()}
        bins = []
        for lo in range(0, int(np.nanmax(steps)) + 1, BIN):
            m = (steps >= lo) & (steps < lo + BIN)
            if not m.any():
                continue
            b = {"steps": f"{lo // 1000}-{(lo + BIN) // 1000}k", "episodes": int(m.sum()),
                 "teacher_lambda_mid": round(lam(lo + BIN / 2), 4)}
            for k, v in cols.items():
                vv = v[m]
                b[k] = float(np.nanmean(vv)) if np.isfinite(vv).any() else None
            bins.append(b)
        res[name] = {"episode_rows": str(p.relative_to(ROOT)).replace("\\", "/"),
                     "run_config": str(cfg.relative_to(ROOT)).replace("\\", "/") if cfg else None,
                     "opponent": sorted({r["opponent"] for r in rows}), "bins": bins}
        md += [f"## {name} ({res[name]['opponent'][0]})", "",
               "| steps | eps | λ | win | own score | conceded | margin | attack crossings | t first score | r_offense |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        g = lambda x, fmt: "—" if x is None else format(x, fmt)  # noqa: E731
        for b in bins:
            md.append(f"| {b['steps']} | {b['episodes']} | {b['teacher_lambda_mid']:.3f} | {g(b['win'], '.2f')} | "
                      f"{g(b['own_score'], '.2f')} | {g(b['conceded'], '.2f')} | {g(b['margin'], '+.2f')} | "
                      f"{g(b['attack_crossings'], '.2f')} | {g(b['time_to_first_score'], '.0f')} | {g(b['reward_offense'], '+.3f')} |")
        md.append("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "TRAINING_DIAGNOSTIC.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / "TRAINING_DIAGNOSTIC.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
