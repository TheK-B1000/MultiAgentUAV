r"""Readout of a post-hoc matched crossover: every system on the identical seed block, win rate AND score margin.

    python experiments/readout_posthoc_matched_crossover.py --spec artifacts/strategic_demand/sppo/<SPEC>.json

The spec names its systems' row files (READOUT_SYSTEMS, first = the system under study), its post-hoc block
(POST_HOC_MATCHED_ROLE_ABLATIONS, one entry) and READOUT_OUT. For each system: V(A,A), V(B,A), V(A,B), V(B,B),
Delta_A = V(A,A) - V(B,A), Delta_B = V(B,B) - V(A,B), on win and on margin (blue - red); then each delta paired
within seed against every other system. mean, std (ddof=1), 95% paired percentile bootstrap (eval_hog_psp_v3).
A system missing any (policy, pole, seed) cell of the block refuses -- absence is an error, never a default.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CELLS = (("pi_A", "A"), ("pi_B", "A"), ("pi_A", "B"), ("pi_B", "B"))


def cells(path: Path, seeds: list[int], field: str) -> dict:
    by: dict = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"], r["pole"]), {})[int(r["seed"])] = float(r[field])
    out = {}
    for c in CELLS:
        missing = [s for s in seeds if s not in by.get(c, {})]
        if missing:
            raise SystemExit(f"FAIL-CLOSED: {path.name} lacks {c} on {len(missing)} seed(s), e.g. {missing[:3]}")
        out[c] = np.array([by[c][s] for s in seeds], dtype=np.float64)
    return out


def stat(x: np.ndarray) -> dict:
    from experiments.eval_hog_psp_v3 import _mean_ci
    ci = _mean_ci(x)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "lcb95": float(ci["lcb95"]),
            "ucb95": float(ci["ucb95"]), "n": int(x.size)}


def deltas(v: dict) -> dict:
    return {"Delta_A": v[("pi_A", "A")] - v[("pi_B", "A")], "Delta_B": v[("pi_B", "B")] - v[("pi_A", "B")]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec_p = ROOT / a.spec
    spec = json.loads(spec_p.read_text(encoding="utf-8"))
    (label, entry), = spec["POST_HOC_MATCHED_ROLE_ABLATIONS"].items()
    lo, hi = (int(x) for x in entry["block"].split(".."))
    seeds = list(range(lo, hi + 1))
    systems = {name: ROOT / p for name, p in spec["READOUT_SYSTEMS"].items()}
    for name, p in systems.items():
        if not p.is_file():
            raise SystemExit(f"FAIL-CLOSED: rows for {name} missing: {p}")
    res = {"record": f"{label}_READOUT", "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "spec": spec_p.name, "classification": spec["classification"], "seeds": {"block": entry["block"], "n": len(seeds)},
           "systems": {}, "paired_vs": {}}
    D = {}
    for name, p in systems.items():
        res["systems"][name] = {"rows": str(p.relative_to(ROOT)).replace("\\", "/")}
        for field in ("win", "margin"):
            v = cells(p, seeds, field)
            d = deltas(v)
            D[(name, field)] = d
            res["systems"][name][field] = {
                **{f"V_{pol[-1]}_on_{pole}": float(v[(pol, pole)].mean()) for pol, pole in CELLS},
                **{k: stat(x) for k, x in d.items()}}
    first = next(iter(systems))
    for other in list(systems)[1:]:
        res["paired_vs"][f"{first}_minus_{other}"] = {
            field: {k: stat(D[(first, field)][k] - D[(other, field)][k]) for k in ("Delta_A", "Delta_B")}
            for field in ("win", "margin")}
    f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f} [{s['lcb95']:+.3f}, {s['ucb95']:+.3f}]"  # noqa: E731
    md = [f"# {label} (n = {len(seeds)} seeds, block {entry['block']})", "", spec["classification"], ""]
    for field, title in (("win", "Win rate"), ("margin", "Score margin (blue − red)")):
        md += [f"## {title}", "", "| System | A@A | B@A | A@B | B@B | Δ_A mean ± std [95% CI] | Δ_B mean ± std [95% CI] |",
               "|---|---|---|---|---|---|---|"]
        for name in systems:
            t = res["systems"][name][field]
            md.append(f"| {name} | {t['V_A_on_A']:.3f} | {t['V_B_on_A']:.3f} | {t['V_A_on_B']:.3f} | {t['V_B_on_B']:.3f} | "
                      f"{f(t['Delta_A'])} | {f(t['Delta_B'])} |")
        for pair, t in res["paired_vs"].items():
            md.append(f"\nPaired {pair}: Δ_A {f(t[field]['Delta_A'])}; Δ_B {f(t[field]['Delta_B'])}")
        md.append("")
    res["table_markdown"] = "\n".join(md)
    out = ROOT / spec["READOUT_OUT"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(res["table_markdown"] + "\n", encoding="utf-8")
    print(res["table_markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
