"""Patch state.json after post-hoc Path A/B classification."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # AICTFProject/
p = ROOT / "artifacts/summer_2026/state.json"
g = json.loads((ROOT / "artifacts/summer_2026/gate_results.json").read_text(encoding="utf-8"))
st = json.loads(p.read_text(encoding="utf-8"))
st["gates"]["CROSSOVER_FOUND"] = True
st["gates"]["PAPER_PATH"] = "PATH_B"
st["gates"]["PAPER_PATH_DISCOVERY"] = "cross_condition_Wald_LCB95"
st["gates"]["CLASSIFIER_RAN_AFTER_STALE_SUPERVISOR_STOP"] = True
st["history"].append({
    "from": "STOPPED_SCIENTIFIC_GATE",
    "to": "STOPPED_SCIENTIFIC_GATE",
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "reason": (
        "PATH_B applied post-hoc: stale supervisor skipped "
        "analyze_summer_2026_paper_path.py; ran manually on complete matrix. "
        f"{g['paper_title']}. next={g['next']}"
    ),
    "CROSSOVER_FOUND": True,
    "n_cross_condition": g.get("n_cross_condition"),
})
tmp = p.with_suffix(".tmp")
tmp.write_text(json.dumps(st, indent=2), encoding="utf-8")
os.replace(tmp, p)
print("state updated; PAPER_PATH=PATH_B")
print("crossovers:")
for c in g.get("crossovers_sample", []):
    print(c)
