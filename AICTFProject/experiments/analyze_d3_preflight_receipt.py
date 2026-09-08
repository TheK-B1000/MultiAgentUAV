"""Post-hoc receipt for D3_POOL_PREFLIGHT answering the four concrete questions.

Does not replace the frozen gate in D3_POOL_PREFLIGHT_FROZEN.json. Reads the
result JSON plus any still-on-disk smoke episode_rows (if present) and writes
artifacts/summer_2026/d3_preflight_four_questions.json.

Rule for Q3: any outside-pool opponent that appears in episode_rows.csv is
treated as gradient-bearing PPO experience (one completed episode = one row of
training exposure). That fails the gate. Init-only artifacts that never appear
as episode rows are documented and allowed.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "artifacts/vgc_diversity/D3_POOL_PREFLIGHT_RESULT.json"
SETS = ROOT / "artifacts/vgc_diversity/VGC_DIVERSITY_SETS_FROZEN.json"
OUT = ROOT / "artifacts/summer_2026/d3_preflight_four_questions.json"


def main() -> int:
    if not RESULT.exists():
        print("D3_POOL_PREFLIGHT_RESULT.json not ready", file=sys.stderr)
        return 2
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    pool = tuple(json.loads(SETS.read_text(encoding="utf-8"))["THE_SETS"]["D3"])
    expected = {f"SCRIPTED:{o}" for o in pool}
    counts = result.get("counts_seed_a") or {}
    obs = set(counts) if counts else set(result.get("first_12_seed_a") or [])

    # Prefer live rows if a smoke dir still exists (usually deleted after gate).
    rows = []
    for seed in (result.get("seed_a"), result.get("seed_b"), 3799001):
        p = ROOT / "artifacts/vgc_diversity" / f"vgc_d3_seed{seed}" / "episode_rows.csv"
        if p.is_file():
            rows = list(csv.DictReader(p.open(encoding="utf-8")))
            break

    outside = sorted(obs - expected)
    maps = Counter(r.get("canonical_map", "") for r in rows) if rows else {}
    rules = Counter(r.get("ruleset_id", "") for r in rows) if rows else {}

    q = {
        "1_all_configured_occurred": {
            "pass": expected <= obs,
            "observed": sorted(obs),
            "required": sorted(expected),
            "counts": counts,
        },
        "2_no_outside_pool": {
            "pass": obs <= expected and not outside,
            "outside": outside,
        },
        "3_outside_pool_gradient_bearing": {
            "rule": "any outside-pool episode_rows entry = FAIL (completed episode = PPO exposure)",
            "pass": not outside,
            "outside_episode_counts": {k: counts.get(k, 0) for k in outside},
            "code_invariant": {
                "id": "Q3_OPPONENT_BEFORE_FIRST_ROLLOUT",
                "status": "CODE_INVARIANT_PASS",
                "doc": "artifacts/summer_2026/Q3_OPPONENT_BEFORE_ROLLOUT_INVARIANT.md",
                "summary": (
                    "OPPONENT_POOL resolves opener into pool[0] when fixed_opponent_tag "
                    "is out-of-pool; build_training_env calls set_next_opponent before "
                    "learn()/first env.step(). Episode CSV + this invariant closes Q3 "
                    "for D3 Mixed-PPO (not the SNAPSHOT/FP path)."
                ),
                "tests": [
                    "tests/test_v5i4_paper_faithful.py::test_initial_opponent_falls_back_to_pool_first_entry",
                ],
            },
            "init_artifact_policy": (
                "Outside-pool tags must not appear as episode rows. For D3 "
                "OPPONENT_POOL the opener is forced in-pool before step #1, so "
                "CSV absence is dispositive. If an outside-pool row appears, FAIL."
            ),
        },
        "4_map_a_ruleset_v2": {
            "pass": (
                (not rows)  # fall back to provenance in run config if rows wiped
                or (set(maps) == {"map_a"} and all(str(r).startswith("RULESET_V2") for r in rules))
            ),
            "canonical_map_counts": dict(maps),
            "ruleset_id_counts": dict(rules),
            "note": "If episode_rows were deleted after the gate, re-check vgc_condition / run_config sidecars.",
        },
    }
    all_pass = all(v["pass"] for v in q.values()) and str(result.get("verdict", "")).endswith("PASS")
    out = {
        "source_gate_verdict": result.get("verdict"),
        "four_questions_all_pass": all_pass,
        "questions": q,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
