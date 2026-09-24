"""CPU-only Pyquaticus-derived 4v4 role-composition sweep.

The semantic adapter and opponent construction are reused from the completed
Pyquaticus assignment evaluation.  This runner changes only the number of
DEFEND identities under the frozen prefix-identity ordering in
``PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_SPEC.json``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.run_pyquaticus_4v4_team_evaluation import (
    HORIZON,
    N_MACROS_EVAL,
    _action_for_roles,
    _make_env,
    _sha256,
    _telemetry_for_tick,
)
from experiments.tqdm_loop import tqdm_iter

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_SPEC.json"
PREFLIGHT_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_PREFLIGHT.json"
RESULT_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_RESULT.json"
EPISODE_CSV = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_EPISODES.csv"
MAPPING_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_MAPPING_AUDIT.json"
COMPOSITIONS = ("4A_0D", "3A_1D", "2A_2D", "1A_3D", "0A_4D")
POLES = ("A", "B")
BASELINE = "2A_2D"
NUM_BOOT = 20_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_SEED_ALLOCATION":
        raise RuntimeError(f"composition spec is not frozen: {spec.get('status')!r}")
    return spec


def composition_roles(composition: str) -> tuple[int, ...]:
    if composition not in COMPOSITIONS:
        raise ValueError(f"unknown composition {composition!r}")
    defenders = int(composition.split("_")[1][:-1])
    roles = tuple(1 if i < defenders else 0 for i in range(4))
    if roles.count(1) != defenders or roles.count(0) != 4 - defenders:
        raise AssertionError((composition, roles))
    return roles


def _ids(roles: tuple[int, ...]) -> tuple[list[int], list[int]]:
    defenders = [i for i, role in enumerate(roles) if role == 1]
    attackers = [i for i, role in enumerate(roles) if role == 0]
    return attackers, defenders


def run_episode(pole: str, composition: str, seed: int) -> tuple[dict, dict]:
    env, core, genome, live = _make_env(pole, seed)
    try:
        roles = composition_roles(composition)
        attackers, defenders = _ids(roles)
        counters = {
            "attack_ticks": 0,
            "defend_ticks": 0,
            "attack_enemy_flag_branch_count": 0,
            "carrier_home_branch_count": 0,
            "defend_inward_count": 0,
            "defend_outward_count": 0,
            "tagged_ticks_by_role": 0,
        }
        terminal_info = None
        steps = 0
        for _ in range(HORIZON):
            _telemetry_for_tick(core, roles, counters)
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, infos = env.step_wait()
            steps += 1
            if bool(np.asarray(done).any()):
                terminal_info = dict(infos[0])
                break
        if terminal_info is None:
            terminal_info = {
                "episode_result": {
                    "blue_score": int(core.blue_score[0].item()),
                    "red_score": int(core.red_score[0].item()),
                },
                "terminal_observation": {},
            }
        result = dict(terminal_info.get("episode_result") or {})
        blue_score = int(result.get("blue_score", 0))
        red_score = int(result.get("red_score", 0))
        terminal_obs = terminal_info.get("terminal_observation") or {}
        agent_mask = terminal_obs.get("agent_mask")
        blue_alive_end = int(np.asarray(agent_mask).sum()) if agent_mask is not None else None
        row = {
            "seed": int(seed),
            "pole": pole,
            "composition": composition,
            "defender_ids": ",".join(map(str, defenders)),
            "attacker_ids": ",".join(map(str, attackers)),
            "blue_score": blue_score,
            "red_score": red_score,
            "blue_win": int(blue_score > red_score),
            "draw": int(blue_score == red_score),
            "steps": int(steps),
            "blue_alive_end": blue_alive_end,
            "red_alive_end": None,
            "genome_id": str(genome.genome_id),
            "pole_config_hash": str(live.get("live_config_hash", "")),
            **counters,
            "role_switch_count": 0,
        }
        mapping = {
            "seed": int(seed),
            "pole": pole,
            "composition": composition,
            "defender_ids": defenders,
            "attacker_ids": attackers,
        }
        return row, mapping
    finally:
        env.close()


def _bootstrap(values: np.ndarray, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lcb95": None, "ucb95": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(NUM_BOOT, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "lcb95": float(lo), "ucb95": float(hi), "n": int(values.size)}


def run_preflight() -> dict:
    spec = _spec()
    from gpu_env.pyquaticus_port import UPSTREAM_COMMIT

    checks = {
        "spec_status": spec["status"],
        "upstream_commit": UPSTREAM_COMMIT == spec["prerequisites"]["upstream_commit"],
        "n_macros_eval": int(spec["execution"]["seed_block"]["n"]) == 64 and N_MACROS_EVAL == 8,
        "composition_names": set(spec["compositions"]) == set(COMPOSITIONS),
        "composition_counts": all(composition_roles(name).count(1) + composition_roles(name).count(0) == 4 for name in COMPOSITIONS),
        "baseline_is_2a2d": BASELINE == "2A_2D",
        "v2_status": spec["prerequisites"]["validated_semantic_status"] == "REPRESENTABLE_ACROSS_TRANSITIONS",
    }
    fixture = {name: list(composition_roles(name)) for name in COMPOSITIONS}
    checks["synthetic_compositions"] = fixture == {
        "4A_0D": [0, 0, 0, 0],
        "3A_1D": [1, 0, 0, 0],
        "2A_2D": [1, 1, 0, 0],
        "1A_3D": [1, 1, 1, 0],
        "0A_4D": [1, 1, 1, 1],
    }
    pole_checks = {}
    for pole in POLES:
        env, core, genome, live = _make_env(pole, 20260918)
        try:
            macros_by_composition = {}
            for name in COMPOSITIONS:
                roles = composition_roles(name)
                action = _action_for_roles(core, roles)
                assert action.shape == (1, 4, 2)
                assert int(action[..., 0].max()) < N_MACROS_EVAL
                macros_by_composition[name] = action[0, :, 0].tolist()
            pole_checks[pole] = {
                "pass": True,
                "genome_id": genome.genome_id,
                "live_config_hash": live.get("live_config_hash"),
                "macros_by_composition": macros_by_composition,
                "blue_scripted": bool(core.blue_scripted),
            }
        finally:
            env.close()
    checks["opponent_and_action_preflight"] = all(item["pass"] and item["blue_scripted"] is False for item in pole_checks.values())
    passed = all(bool(value) for key, value in checks.items() if key != "spec_status")
    out = {
        "record_id": "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_PREFLIGHT",
        "status": "PASS" if passed else "FAIL",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "checks": checks,
        "pole_checks": pole_checks,
        "episodes_run": 0,
        "seed_allocation": "not performed by preflight",
    }
    PREFLIGHT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def run_evaluation(workers: int = 8) -> dict:
    spec = _spec()
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8")) if PREFLIGHT_PATH.is_file() else {}
    if preflight.get("status") != "PASS":
        raise RuntimeError(f"preflight is not PASS: {preflight.get('status')!r}")
    seeds = range(int(spec["execution"]["seed_block"]["base"]), int(spec["execution"]["seed_block"]["last"]) + 1)
    jobs = [(pole, composition, seed) for pole in POLES for composition in COMPOSITIONS for seed in seeds]
    rows: list[dict] = []
    mappings: list[dict] = []
    with ProcessPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(run_episode, *job): job for job in jobs}
        for future in tqdm_iter(
            as_completed(futures),
            desc="4v4 role-composition sweep",
            total=len(futures),
            unit="ep",
        ):
            job = futures[future]
            row, mapping = future.result()
            rows.append(row)
            mappings.append(mapping)
    rows.sort(key=lambda row: (row["pole"], COMPOSITIONS.index(row["composition"]), row["seed"]))
    mappings.sort(key=lambda row: (row["pole"], COMPOSITIONS.index(row["composition"]), row["seed"]))
    with EPISODE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    MAPPING_PATH.write_text(json.dumps({"record_id": "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_MAPPING_AUDIT", "rows": mappings}, indent=2) + "\n", encoding="utf-8")

    summaries = {}
    for pole in POLES:
        for composition in COMPOSITIONS:
            subset = [row for row in rows if row["pole"] == pole and row["composition"] == composition]
            summaries[f"{pole}/{composition}"] = {
                "n": len(subset),
                "blue_win_rate": _bootstrap(np.asarray([row["blue_win"] for row in subset]), 17),
                "score_difference": _bootstrap(np.asarray([row["blue_score"] - row["red_score"] for row in subset]), 19),
                "draw_rate": float(np.mean([row["draw"] for row in subset])),
                "mean_steps": float(np.mean([row["steps"] for row in subset])),
                "mean_defend_inward_count": float(np.mean([row["defend_inward_count"] for row in subset])),
                "mean_defend_outward_count": float(np.mean([row["defend_outward_count"] for row in subset])),
            }

    baseline_rows = {(row["pole"], row["seed"]): row for row in rows if row["composition"] == BASELINE}
    composition_contrasts = {}
    for pole in POLES:
        for composition in COMPOSITIONS:
            if composition == BASELINE:
                continue
            current = {(row["pole"], row["seed"]): row for row in rows if row["pole"] == pole and row["composition"] == composition}
            seeds_common = sorted(set(current) & set(baseline_rows))
            win_delta = np.asarray([current[key]["blue_win"] - baseline_rows[key]["blue_win"] for key in seeds_common])
            score_delta = np.asarray([
                (current[key]["blue_score"] - current[key]["red_score"]) -
                (baseline_rows[key]["blue_score"] - baseline_rows[key]["red_score"])
                for key in seeds_common
            ])
            composition_contrasts[f"{pole}/{composition}_minus_{BASELINE}"] = {
                "n": len(seeds_common),
                "blue_win_rate": _bootstrap(win_delta, 23),
                "score_difference": _bootstrap(score_delta, 29),
            }

    regime_contrasts = {}
    for composition in COMPOSITIONS:
        a = {(row["seed"]): row for row in rows if row["pole"] == "A" and row["composition"] == composition}
        b = {(row["seed"]): row for row in rows if row["pole"] == "B" and row["composition"] == composition}
        common = sorted(set(a) & set(b))
        regime_contrasts[composition] = {
            "n": len(common),
            "B_minus_A_blue_win_rate": _bootstrap(np.asarray([b[s]["blue_win"] - a[s]["blue_win"] for s in common]), 31),
            "B_minus_A_score_difference": _bootstrap(np.asarray([
                (b[s]["blue_score"] - b[s]["red_score"]) - (a[s]["blue_score"] - a[s]["red_score"])
                for s in common
            ]), 37),
        }

    effects = []
    for composition in COMPOSITIONS:
        if composition == BASELINE:
            continue
        a = composition_contrasts[f"A/{composition}_minus_{BASELINE}"]["blue_win_rate"]
        b = composition_contrasts[f"B/{composition}_minus_{BASELINE}"]["blue_win_rate"]
        a_excludes = float(a["lcb95"]) > 0.0 or float(a["ucb95"]) < 0.0
        b_excludes = float(b["lcb95"]) > 0.0 or float(b["ucb95"]) < 0.0
        opposite = (float(a["mean"]) > 0.0 and float(b["mean"]) < 0.0) or (float(a["mean"]) < 0.0 and float(b["mean"]) > 0.0)
        effects.append({"composition": composition, "A_excludes_zero": a_excludes, "B_excludes_zero": b_excludes, "opposite_sign": opposite})
    regime_signal = any(item["A_excludes_zero"] and item["B_excludes_zero"] and item["opposite_sign"] for item in effects)
    any_composition_signal = any(item["A_excludes_zero"] or item["B_excludes_zero"] for item in effects)
    if regime_signal:
        label = "REGIME_DEPENDENT_COMPOSITION_SIGNAL"
    elif any_composition_signal:
        label = "COMPOSITION_DESCRIPTIVE_SIGNAL"
    else:
        label = "NO_DEMONSTRATED_COMPOSITION_EFFECT"
    out = {
        "record_id": "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_RESULT",
        "status": "COMPLETE",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "preflight_sha256": _sha256(PREFLIGHT_PATH),
        "n_rows": len(rows),
        "summaries": summaries,
        "composition_contrasts_vs_2A2D": composition_contrasts,
        "regime_B_minus_A_contrasts": regime_contrasts,
        "regime_gate_details": effects,
        "decision_label": label,
        "claim_boundary": spec["claim_boundary"],
        "six_v_six": "NOT_RUN",
    }
    RESULT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    out = run_preflight() if args.preflight else run_evaluation(args.workers)
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
