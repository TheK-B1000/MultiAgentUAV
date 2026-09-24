"""CPU-only oracle regime-conditioned Pyquaticus composition test.

The oracle dispatches Pole A to 2A/2D and Pole B to 4A/0D. Fixed 2A/2D and
fixed 4A/0D arms are retained as paired controls. No PPO or training path is
reachable from this module.
"""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.run_pyquaticus_4v4_role_composition_sweep import (
    COMPOSITIONS as SWEEP_COMPOSITIONS,
    _action_for_roles,
    _make_env,
    _sha256,
    composition_roles,
    run_episode as run_composition_episode,
)
from experiments.tqdm_loop import tqdm_iter

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "PYQUATICUS_4V4_ORACLE_COMPOSITION_SPEC.json"
PREFLIGHT_PATH = SD / "PYQUATICUS_4V4_ORACLE_COMPOSITION_PREFLIGHT.json"
RESULT_PATH = SD / "PYQUATICUS_4V4_ORACLE_COMPOSITION_RESULT.json"
EPISODE_CSV = SD / "PYQUATICUS_4V4_ORACLE_COMPOSITION_EPISODES.csv"
MAPPING_PATH = SD / "PYQUATICUS_4V4_ORACLE_COMPOSITION_MAPPING_AUDIT.json"
ORIGINAL_RESULT_PATH = SD / "PYQUATICUS_4V4_ORACLE_COMPOSITION_RESULT_PREAMENDMENT_BUGGY.json"
POLES = ("A", "B")
ARMS = ("ORACLE_REGIME", "FIXED_2A2D", "FIXED_4A0D")
COMPOSITION_BY_ARM = {
    "FIXED_2A2D": {"A": "2A_2D", "B": "2A_2D"},
    "FIXED_4A0D": {"A": "4A_0D", "B": "4A_0D"},
    "ORACLE_REGIME": {"A": "2A_2D", "B": "4A_0D"},
}
NUM_BOOT = 20_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_SEED_ALLOCATION":
        raise RuntimeError(f"oracle spec is not frozen: {spec.get('status')!r}")
    return spec


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lcb95": None, "ucb95": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(NUM_BOOT, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "lcb95": float(lo), "ucb95": float(hi), "n": int(values.size)}


def run_job(pole: str, arm: str, seed: int) -> tuple[dict, dict]:
    composition = COMPOSITION_BY_ARM[arm][pole]
    row, mapping = run_composition_episode(pole, composition, seed)
    row["arm"] = arm
    row["effective_composition"] = composition
    mapping["arm"] = arm
    mapping["effective_composition"] = composition
    return row, mapping


def _pair(rows_by_key: dict, pole: str, left: str, right: str, seeds: list[int], field: str) -> np.ndarray:
    return np.asarray([
        rows_by_key[(pole, left, seed)][field] - rows_by_key[(pole, right, seed)][field]
        for seed in seeds
    ])


def run_preflight() -> dict:
    spec = _spec()
    from gpu_env.pyquaticus_port import UPSTREAM_COMMIT

    semantic_path = SD / "DEFEND_SEMANTIC_COMMITMENT_V2_CONTRACT_RESULT.json"
    prior_path = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_RESULT.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    checks = {
        "spec_status": spec["status"],
        "upstream_commit": UPSTREAM_COMMIT == spec["prerequisites"]["upstream_commit"],
        "semantic_status": semantic.get("status") == spec["prerequisites"]["validated_semantic_status"],
        "prior_result_complete": prior.get("status") == spec["prerequisites"]["prior_confirmation_status"],
        "prior_result_label": prior.get("decision_label") == spec["prerequisites"]["prior_confirmation_label"],
        "n_macros_eval": all(int(value) < 8 for value in (2, 7)),
        "poles": list(spec["opponent_poles"]) == list(POLES),
        "arms": list(spec["arms"]) == list(ARMS),
        "seed_block_shape": int(spec["execution"]["seed_block"]["n"]) == 64 and int(spec["execution"]["seed_block"]["last"]) - int(spec["execution"]["seed_block"]["base"]) == 63,
        "sweep_compositions_available": all(value in SWEEP_COMPOSITIONS for value in ("2A_2D", "4A_0D")),
        "oracle_mapping": COMPOSITION_BY_ARM == {
            "ORACLE_REGIME": {"A": "2A_2D", "B": "4A_0D"},
            "FIXED_2A2D": {"A": "2A_2D", "B": "2A_2D"},
            "FIXED_4A0D": {"A": "4A_0D", "B": "4A_0D"},
        },
    }
    pole_checks = {}
    for pole in POLES:
        env, core, genome, live = _make_env(pole, 20260918)
        try:
            macros_by_arm = {}
            for arm in ARMS:
                composition = COMPOSITION_BY_ARM[arm][pole]
                roles = composition_roles(composition)
                action = _action_for_roles(core, roles)
                assert action.shape == (1, 4, 2)
                assert int(action[..., 0].max()) < 8
                macros_by_arm[arm] = action[0, :, 0].tolist()
            pole_checks[pole] = {
                "pass": True,
                "genome_id": genome.genome_id,
                "live_config_hash": live.get("live_config_hash"),
                "macros_by_arm": macros_by_arm,
                "blue_scripted": bool(core.blue_scripted),
            }
        finally:
            env.close()
    expected = {
        "A": {"ORACLE_REGIME": [7, 7, 2, 2], "FIXED_2A2D": [7, 7, 2, 2], "FIXED_4A0D": [2, 2, 2, 2]},
        "B": {"ORACLE_REGIME": [2, 2, 2, 2], "FIXED_2A2D": [7, 7, 2, 2], "FIXED_4A0D": [2, 2, 2, 2]},
    }
    checks["opponent_and_oracle_action_preflight"] = all(
        item["pass"] and item["blue_scripted"] is False and item["macros_by_arm"] == expected[pole]
        for pole, item in pole_checks.items()
    )
    passed = all(bool(value) for key, value in checks.items() if key != "spec_status")
    out = {
        "record_id": "PYQUATICUS_4V4_ORACLE_COMPOSITION_PREFLIGHT",
        "status": "PASS" if passed else "FAIL",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "semantic_result_sha256": _sha256(semantic_path),
        "prior_result_sha256": _sha256(prior_path),
        "checks": checks,
        "pole_checks": pole_checks,
        "episodes_run": 0,
        "seed_allocation": "not performed by preflight",
    }
    PREFLIGHT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "blue_win_rate": _bootstrap(np.asarray([row["blue_win"] for row in rows]), 17),
        "score_difference": _bootstrap(np.asarray([row["blue_score"] - row["red_score"] for row in rows]), 19),
        "draw_rate": float(np.mean([row["draw"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_defend_inward_count": float(np.mean([row["defend_inward_count"] for row in rows])),
        "mean_defend_outward_count": float(np.mean([row["defend_outward_count"] for row in rows])),
    }


def _load_existing_rows() -> list[dict]:
    int_fields = {
        "seed", "blue_score", "red_score", "blue_win", "draw", "steps", "blue_alive_end",
        "red_alive_end", "attack_ticks", "defend_ticks", "attack_enemy_flag_branch_count",
        "carrier_home_branch_count", "defend_inward_count", "defend_outward_count",
        "tagged_ticks_by_role", "role_switch_count",
    }
    with EPISODE_CSV.open(newline="", encoding="utf-8") as fh:
        rows = []
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for field in int_fields:
                if row.get(field) in (None, "", "None"):
                    row[field] = None
                else:
                    row[field] = int(row[field])
            rows.append(row)
    return rows


def run_evaluation(workers: int = 4, existing: bool = False) -> dict:
    spec = _spec()
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8")) if PREFLIGHT_PATH.is_file() else {}
    if preflight.get("status") != "PASS":
        raise RuntimeError(f"preflight is not PASS: {preflight.get('status')!r}")
    seeds = list(range(int(spec["execution"]["seed_block"]["base"]), int(spec["execution"]["seed_block"]["last"]) + 1))
    rows: list[dict]
    mappings: list[dict] = []
    if existing:
        if not EPISODE_CSV.is_file():
            raise RuntimeError(f"existing episode CSV not found: {EPISODE_CSV}")
        rows = _load_existing_rows()
        expected_rows = len(POLES) * len(ARMS) * len(seeds)
        if len(rows) != expected_rows:
            raise RuntimeError(f"existing episode CSV has {len(rows)} rows, expected {expected_rows}")
    else:
        jobs = [(pole, arm, seed) for pole in POLES for arm in ARMS for seed in seeds]
        rows = []
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(run_job, *job): job for job in jobs}
            for future in tqdm_iter(
                as_completed(futures),
                desc="4v4 oracle composition",
                total=len(futures),
                unit="ep",
            ):
                job = futures[future]
                row, mapping = future.result()
                rows.append(row)
                mappings.append(mapping)
    rows.sort(key=lambda row: (row["pole"], ARMS.index(row["arm"]), row["seed"]))
    if not existing:
        mappings.sort(key=lambda row: (row["pole"], ARMS.index(row["arm"]), row["seed"]))
        with EPISODE_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        MAPPING_PATH.write_text(json.dumps({"record_id": "PYQUATICUS_4V4_ORACLE_COMPOSITION_MAPPING_AUDIT", "rows": mappings}, indent=2) + "\n", encoding="utf-8")

    summaries = {
        f"{pole}/{arm}": _summary([row for row in rows if row["pole"] == pole and row["arm"] == arm])
        for pole in POLES for arm in ARMS
    }
    rows_by_key = {(row["pole"], row["arm"], row["seed"]): row for row in rows}
    contrasts = {}
    for pole, seed_base in (("A", 101), ("B", 103)):
        wrong_fixed = "FIXED_4A0D" if pole == "A" else "FIXED_2A2D"
        for left, right, name in (
            ("ORACLE_REGIME", wrong_fixed, "oracle_gain_over_wrong_fixed"),
            ("ORACLE_REGIME", "FIXED_2A2D", "oracle_vs_2A2D"),
            ("ORACLE_REGIME", "FIXED_4A0D", "oracle_vs_4A0D"),
            ("FIXED_2A2D", "FIXED_4A0D", "fixed_2A2D_minus_fixed_4A0D"),
        ):
            values = _pair(rows_by_key, pole, left, right, seeds, "blue_win")
            score_values = _pair(rows_by_key, pole, left, right, seeds, "blue_score") - _pair(rows_by_key, pole, left, right, seeds, "red_score")
            contrasts[f"{pole}/{name}"] = {
                "definition": f"{left} - {right}",
                "blue_win_rate": _bootstrap(values, seed_base),
                "score_difference": _bootstrap(score_values, seed_base + 1),
            }

    parity_fields = ("blue_score", "red_score", "blue_win", "draw", "steps", "blue_alive_end", "attack_ticks", "defend_ticks", "attack_enemy_flag_branch_count", "carrier_home_branch_count", "defend_inward_count", "defend_outward_count", "tagged_ticks_by_role", "role_switch_count")
    parity_mismatches = []
    for pole, matched_arm in (("A", "FIXED_2A2D"), ("B", "FIXED_4A0D")):
        for seed in seeds:
            oracle = rows_by_key[(pole, "ORACLE_REGIME", seed)]
            matched = rows_by_key[(pole, matched_arm, seed)]
            for field in parity_fields:
                if oracle.get(field) != matched.get(field):
                    parity_mismatches.append({"pole": pole, "seed": seed, "field": field, "oracle": oracle.get(field), "matched": matched.get(field)})
    oracle_parity = {"pass": not parity_mismatches, "n_mismatches": len(parity_mismatches), "mismatches": parity_mismatches[:20]}
    a_gain = contrasts["A/oracle_gain_over_wrong_fixed"]["blue_win_rate"]
    b_gain = contrasts["B/oracle_gain_over_wrong_fixed"]["blue_win_rate"]
    gate = {
        "oracle_parity": oracle_parity["pass"],
        "A_oracle_gain_over_wrong_fixed_lcb95_gt_zero": float(a_gain["lcb95"]) > 0.0,
        "B_oracle_gain_over_wrong_fixed_lcb95_gt_zero": float(b_gain["lcb95"]) > 0.0,
    }
    label = "ORACLE_COMPOSITION_PASS" if all(gate.values()) else "NO_ORACLE_COMPOSITION_CONFIRMATION"
    out = {
        "record_id": "PYQUATICUS_4V4_ORACLE_COMPOSITION_RESULT",
        "status": "COMPLETE",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "preflight_sha256": _sha256(PREFLIGHT_PATH),
        "n_rows": len(rows),
        "summaries": summaries,
        "contrasts": contrasts,
        "oracle_parity": oracle_parity,
        "decision_gate_details": gate,
        "decision_label": label,
        "claim_boundary": spec["claim_boundary"],
        "six_v_six": "NOT_RUN",
        "ppo": "OFF",
    }
    if existing:
        out["analysis_amendment"] = {
            "status": "CORRECTED",
            "reason": "The original result wired the B-side wrong-composition gate to matched FIXED_4A0D instead of frozen wrong FIXED_2A2D. Raw rows were retained and reanalyzed without rerunning episodes or changing the decision rule.",
            "original_artifact": str(ORIGINAL_RESULT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "original_artifact_sha256": _sha256(ORIGINAL_RESULT_PATH) if ORIGINAL_RESULT_PATH.is_file() else None,
        }
    RESULT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    selected = sum((args.preflight, args.run, args.analyze_existing))
    if selected != 1:
        parser.error("choose exactly one of --preflight, --run, or --analyze-existing")
    if args.preflight:
        out = run_preflight()
    else:
        out = run_evaluation(args.workers, existing=args.analyze_existing)
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
