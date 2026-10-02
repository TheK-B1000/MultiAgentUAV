r"""Apply the HISTORICAL top-50 rule to a system's own 128-seed crossover rows, then read the crossover on them.

    python experiments/select_own_top50.py --verify-historical --scale 2
    python experiments/select_own_top50.py --verify-historical --scale 6
    python experiments/select_own_top50.py --rows <rows.csv> --label DUAL_BRANCH_6V6_OWN_TOP50 --out <dir> --scale 6

Rule (verbatim from paper/aamas2027/true_top50_provenance/<scale>_Ours_heuristic_roles_seeds.json):
  score(seed) = outcome(strategy A on Pole A) + outcome(strategy B on Pole B), outcome = win (0/1);
  rank descending; tie-break ascending seed ID; take top 50.
No other rule is used. --verify-historical proves this code reproduces the frozen list for that scale.

Reading: own-top50 = BEST-CASE strategic capability of the system on its own best scenarios, NOT an unbiased
estimate of general performance (A@A and B@B are pushed toward 1 by construction; the informative cells are
B@A and A@B). The full-128 readout is the general-performance number.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
PROV_DIR = ROOT / "paper" / "aamas2027" / "true_top50_provenance"
CELLS = (("pi_A", "A"), ("pi_B", "A"), ("pi_A", "B"), ("pi_B", "B"))
N_TOP = 50
PROV_FOR = {
    2: PROV_DIR / "2v2_Ours_heuristic_roles_seeds.json",
    4: PROV_DIR / "4v4_Ours_heuristic_roles_seeds.json",
    6: PROV_DIR / "6v6_Ours_heuristic_roles_seeds.json",
}


def load(path: Path) -> dict:
    by: dict = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"], r["pole"]), {})[int(r["seed"])] = (float(r["win"]), float(r["margin"]))
    seeds = sorted(by[("pi_A", "A")])
    for c in CELLS:
        if sorted(by.get(c, {})) != seeds:
            raise SystemExit(f"FAIL-CLOSED: {path.name} cell {c} does not cover the same seeds as (pi_A, A)")
    return by


def select(by: dict) -> tuple[list[int], dict]:
    """The historical rule, exactly. Returns (top-50 sorted ascending, scores of every seed)."""
    seeds = sorted(by[("pi_A", "A")])
    score = {s: by[("pi_A", "A")][s][0] + by[("pi_B", "B")][s][0] for s in seeds}
    ranked = sorted(seeds, key=lambda s: (-score[s], s))
    return sorted(ranked[:N_TOP]), score


def readout(by: dict, seeds: list[int]) -> dict:
    from experiments.eval_hog_psp_v3 import _mean_ci
    out = {}
    for i, field in enumerate(("win", "margin")):
        v = {c: np.array([by[c][s][i] for s in seeds]) for c in CELLS}
        d = {"Delta_A": v[("pi_A", "A")] - v[("pi_B", "A")], "Delta_B": v[("pi_B", "B")] - v[("pi_A", "B")]}
        out[field] = {**{f"V_{p[-1]}_on_{q}": float(v[(p, q)].mean()) for p, q in CELLS},
                      **{k: {"mean": float(x.mean()), "std": float(x.std(ddof=1)), **{kk: float(_mean_ci(x)[kk])
                         for kk in ("lcb95", "ucb95")}} for k, x in d.items()}}
    return out


def verify_historical(scale: int) -> list[int]:
    prov_path = PROV_FOR[scale]
    if not prov_path.is_file():
        raise SystemExit(f"FAIL: missing provenance {prov_path}")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    by = load(SD / prov["source_csv"])
    got, score = select(by)
    want = sorted(int(s) for s in prov["selected_seed_ids"])
    if got != want:
        raise SystemExit(
            f"FAIL: rule does not reproduce the frozen {scale}v{scale} list "
            f"({len(set(got) ^ set(want))} seeds differ)"
        )
    if any(score[s] != prov["scores"][str(s)] for s in want):
        raise SystemExit(f"FAIL: per-seed scores differ from {prov_path.name}")
    return got


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify-historical", action="store_true")
    ap.add_argument("--scale", type=int, choices=(2, 4, 6), default=2,
                    help="which frozen Ours top-50 provenance to verify / overlap against")
    ap.add_argument("--rows")
    ap.add_argument("--label")
    ap.add_argument("--out")
    a = ap.parse_args()
    hist = verify_historical(int(a.scale))
    print(f"historical rule reproduced exactly for {a.scale}v{a.scale}: "
          f"{len(hist)} seeds match {PROV_FOR[int(a.scale)].name}")
    if a.verify_historical:
        return 0
    if not (a.rows and a.label and a.out):
        raise SystemExit("--rows, --label and --out are required")
    rows = ROOT / a.rows
    by = load(rows)
    top, score = select(by)
    n_all = len(score)
    cut = score[sorted(score, key=lambda s: (-score[s], s))[N_TOP - 1]]
    tied = sum(1 for s in score if score[s] == cut)
    above = sum(1 for s in score if score[s] > cut)
    prov_path = PROV_FOR[int(a.scale)]
    res = {"record": a.label, "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "classification": "POST-HOC DESCRIPTIVE SUBSET: best-case strategic capability on the system's own best 50 "
                             "scenarios. Not an unbiased estimate of general performance; the full-128 readout is.",
           "scale": f"{a.scale}v{a.scale}",
           "rule": json.loads(prov_path.read_text(encoding="utf-8"))["rule"],
           "rule_verified_against": prov_path.name,
           "rows": str(rows.relative_to(ROOT)).replace("\\", "/"),
           "rows_sha256": hashlib.sha256(rows.read_bytes()).hexdigest(),
           "n_candidates": n_all, "selected_seed_ids": top,
           "cutoff": {"score": cut, "seeds_above": above, "seeds_tied_at_cutoff": tied,
                      "tie_break": "ascending seed ID (historical rule)"},
           "score_distribution": {str(k): sum(1 for s in score if score[s] == k) for k in (2.0, 1.0, 0.0)},
           "top50": readout(by, top), "all_128": readout(by, sorted(score)),
           "overlap_with_old_ours_top50": len(set(top) & set(hist))}
    f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f} [{s['lcb95']:+.3f}, {s['ucb95']:+.3f}]"  # noqa: E731
    md = [f"# {a.label}", "", res["classification"], "",
          f"Rule (historical, verified to reproduce the old {a.scale}v{a.scale} list): {res['rule']}", "",
          f"Scores over {n_all} seeds: both won (2) {res['score_distribution']['2.0']}, one (1) "
          f"{res['score_distribution']['1.0']}, neither (0) {res['score_distribution']['0.0']}. Cutoff score {cut:g}: "
          f"{above} seeds above, {tied} tied (tie-break ascending seed ID). Overlap with the old Ours top-50: "
          f"{res['overlap_with_old_ours_top50']}/50.", ""]
    for field, title in (("win", "Win rate"), ("margin", "Score margin (blue − red)")):
        md += [f"## {title}", "", "| Seeds | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |",
               "|---|---|---|---|---|---|---|"]
        for name in ("top50", "all_128"):
            t = res[name][field]
            md.append(f"| {'own top-50' if name == 'top50' else 'all 128'} | {t['V_A_on_A']:.3f} | {t['V_B_on_A']:.3f} | "
                      f"{t['V_A_on_B']:.3f} | {t['V_B_on_B']:.3f} | {f(t['Delta_A'])} | {f(t['Delta_B'])} |")
        md.append("")
    res["table_markdown"] = "\n".join(md)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.label}.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    (out / f"{a.label}.md").write_text(res["table_markdown"] + "\n", encoding="utf-8")
    (out / f"{a.label}_seed_ids.json").write_text(json.dumps(top) + "\n", encoding="utf-8")
    print(res["table_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
