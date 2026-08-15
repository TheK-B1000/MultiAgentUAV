"""Classify the completed D1/D3/D7 cross-play matrix into Path A vs Path B.

Criteria are frozen in artifacts/summer_2026/PAPER_PATH_READOUT_FROZEN.json.
Do not edit that file after d1_d3_d7_summary.json exists.

Run: python experiments/analyze_summer_2026_paper_path.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FROZEN = ROOT / "artifacts/summer_2026/PAPER_PATH_READOUT_FROZEN.json"
SUMMARY = ROOT / "artifacts/vgc_diversity/crossplay/d1_d3_d7_summary.json"
OUT_JSON = ROOT / "artifacts/summer_2026/gate_results.json"
OUT_CSV = ROOT / "artifacts/summer_2026/figures/payoff_heatmap.csv"
OPPONENTS = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")


def _wald_diff_lcb95(p1: float, n1: int, p2: float, n2: int) -> float:
    """Lower 95% bound on p1-p2 (Wald). Conservative enough to freeze before the full board."""
    if n1 <= 0 or n2 <= 0:
        return float("-inf")
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return (p1 - p2) - 1.96 * se


def _spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return float("nan")
    return num / (denx * deny)


def main() -> int:
    if not SUMMARY.exists():
        print("d1_d3_d7_summary.json not ready", file=sys.stderr)
        return 2
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = [s for s in data["summaries"] if s.get("arm") == "PRIMARY"]
    if len(rows) < 2:
        print("need >=2 PRIMARY policies", file=sys.stderr)
        return 1

    heatmap = []
    for s in rows:
        rec = {
            "policy_id": s["policy_id"],
            "diversity_condition": s["diversity_condition"],
            "seed": s.get("train_seed"),
            "overall": s["overall_avg"],
            "worst_opponent": s["worst_opponent"],
            "worst_wr": s["worst_opponent_wr"],
            "var_across_opponents": s["variance_across_opponents"],
        }
        for o in OPPONENTS:
            rec[o] = s["per_opponent"][o]["win_rate"]
            rec[f"{o}_n"] = s["per_opponent"][o]["n"]
        heatmap.append(rec)

    col_mean = {o: sum(r[o] for r in heatmap) / len(heatmap) for o in OPPONENTS}
    col_rank_vec = [col_mean[o] for o in OPPONENTS]
    for rec in heatmap:
        rec["spearman_vs_column_means"] = _spearman([rec[o] for o in OPPONENTS], col_rank_vec)

    crossovers = []
    for i, a in enumerate(heatmap):
        for b in heatmap[i + 1 :]:
            for x in OPPONENTS:
                for y in OPPONENTS:
                    if x >= y:
                        continue
                    lcb_ax = _wald_diff_lcb95(a[x], a[f"{x}_n"], b[x], b[f"{x}_n"])
                    lcb_by = _wald_diff_lcb95(b[y], b[f"{y}_n"], a[y], a[f"{y}_n"])
                    lcb_ay = _wald_diff_lcb95(a[y], a[f"{y}_n"], b[y], b[f"{y}_n"])
                    lcb_bx = _wald_diff_lcb95(b[x], b[f"{x}_n"], a[x], a[f"{x}_n"])
                    hit = {
                            "A": a["policy_id"], "B": b["policy_id"],
                            "A_condition": a["diversity_condition"],
                            "B_condition": b["diversity_condition"],
                            "same_condition": a["diversity_condition"] == b["diversity_condition"],
                            "X": x, "Y": y,
                            "WR_A_X": a[x], "WR_B_X": b[x],
                            "WR_A_Y": a[y], "WR_B_Y": b[y],
                            "LCB95_A_minus_B_on_X": round(lcb_ax, 4),
                            "LCB95_B_minus_A_on_Y": round(lcb_by, 4),
                        }
                    if lcb_ax > 0 and lcb_by > 0:
                        crossovers.append(hit)
                    elif lcb_ay > 0 and lcb_bx > 0:
                        hit = {
                            "A": a["policy_id"], "B": b["policy_id"],
                            "A_condition": a["diversity_condition"],
                            "B_condition": b["diversity_condition"],
                            "same_condition": a["diversity_condition"] == b["diversity_condition"],
                            "X": y, "Y": x,
                            "WR_A_X": a[y], "WR_B_X": b[y],
                            "WR_A_Y": a[x], "WR_B_Y": b[x],
                            "LCB95_A_minus_B_on_X": round(lcb_ay, 4),
                            "LCB95_B_minus_A_on_Y": round(lcb_bx, 4),
                        }
                        crossovers.append(hit)

    by_cond: dict[str, list] = defaultdict(list)
    for rec in heatmap:
        by_cond[rec["diversity_condition"]].append(rec)

    def _mean(vals: list[float]) -> float | None:
        return None if not vals else sum(vals) / len(vals)

    diversity = {}
    for cond, items in sorted(by_cond.items()):
        diversity[cond] = {
            "n_seeds": len(items),
            "mean_overall_WR": _mean([x["overall"] for x in items]),
            "mean_worst_WR": _mean([x["worst_wr"] for x in items]),
            "mean_var_across_opponents": _mean([x["var_across_opponents"] for x in items]),
            "worst_opponent_identities": [x["worst_opponent"] for x in items],
        }

    seed_noise = [c for c in crossovers if c["same_condition"]]
    strategic = [c for c in crossovers if not c["same_condition"]]
    crossover_found = bool(strategic)
    paper_path = "PATH_B" if crossover_found else "PATH_A"
    out = {
        "gate": "SUMMER_2026_PAPER_PATH",
        "protocol": str(FROZEN.relative_to(ROOT)).replace("\\", "/"),
        "n_policies": len(heatmap),
        "CROSSOVER_FOUND": crossover_found,
        "n_significant_crossover_pairs": len(crossovers),
        "n_seed_noise_same_condition_crossovers": len(seed_noise),
        "n_cross_condition_crossovers": len(strategic),
        "seed_noise_crossovers_sample": seed_noise[:10],
        "crossovers_sample": strategic[:20],
        "paper_path": paper_path,
        "paper_title": frozen["two_papers"][paper_path]["title"],
        "permitted_claim": frozen["two_papers"][paper_path]["permitted_claim"],
        "next": frozen["two_papers"][paper_path]["next"],
        "opponent_column_means": {k: round(v, 4) for k, v in col_mean.items()},
        "hardest_opponent_by_column_mean": min(col_mean, key=col_mean.get),
        "easiest_opponent_by_column_mean": max(col_mean, key=col_mean.get),
        "per_policy_spearman_vs_column_means": {
            r["policy_id"]: r["spearman_vs_column_means"] for r in heatmap
        },
        "diversity_value_OBSERVED": diversity,
        "evidence_label": "EXPLORATORY" if any(
            r["diversity_condition"] == "D3" for r in heatmap
        ) else "COMPLETE_HISTORICAL",
        "status": "OBSERVED",
        "if_PATH_B": "preregistered DISCOVERY only; family-wise error not controlled; confirm via 1-seed specialist pilot",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "policy_id", "diversity_condition", "seed", *OPPONENTS,
            "overall", "worst_opponent", "worst_wr", "var_across_opponents",
            "spearman_vs_column_means",
        ])
        w.writeheader()
        for rec in heatmap:
            w.writerow({k: rec.get(k) for k in w.fieldnames})

    print(json.dumps({
        "paper_path": paper_path,
        "CROSSOVER_FOUND": crossover_found,
        "n_crossovers": len(crossovers),
        "n_cross_condition": len(strategic),
        "n_seed_noise": len(seed_noise),
        "hardest_column": out["hardest_opponent_by_column_mean"],
        "diversity": diversity,
        "->": str(OUT_JSON),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
