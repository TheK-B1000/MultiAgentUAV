"""CPU-only targeted confirmation of the Pyquaticus role-composition signal.

This runner keeps the validated Pyquaticus-derived controllers and fixed
identity ordering unchanged.  Its only experimental factor is 4A/0D versus
2A/2D, evaluated on both certified 4v4 poles with paired fresh seeds.
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
    HORIZON,
    N_MACROS_EVAL,
    _action_for_roles,
    _make_env,
    _sha256,
    _telemetry_for_tick,
    composition_roles,
)
from experiments.tqdm_loop import tqdm_iter

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_SPEC.json"
PREFLIGHT_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_PREFLIGHT.json"
RESULT_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_RESULT.json"
EPISODE_CSV = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_EPISODES.csv"
MAPPING_PATH = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_MAPPING_AUDIT.json"
POLES = ("A", "B")
COMPOSITIONS = ("4A_0D", "2A_2D")
BASELINE = "2A_2D"
NUM_BOOT = 20_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_SEED_ALLOCATION":
        raise RuntimeError(f"confirmation spec is not frozen: {spec.get('status')!r}")
    return spec


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
    return {
        "mean": float(values.mean()),
        "lcb95": float(lo),
        "ucb95": float(hi),
        "n": int(values.size),
    }


def run_preflight() -> dict:
    spec = _spec()
    from gpu_env.pyquaticus_port import UPSTREAM_COMMIT

    prior_result_path = SD / "PYQUATICUS_4V4_ROLE_COMPOSITION_SWEEP_RESULT.json"
    semantic_path = SD / "DEFEND_SEMANTIC_COMMITMENT_V2_CONTRACT_RESULT.json"
    prior = json.loads(prior_result_path.read_text(encoding="utf-8"))
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    checks = {
        "spec_status": spec["status"],
        "upstream_commit": UPSTREAM_COMMIT == spec["prerequisites"]["upstream_commit"],
        "semantic_status": semantic.get("status") == spec["prerequisites"]["validated_semantic_status"],
        "prior_result_complete": prior.get("status") == spec["prerequisites"]["prior_composition_result_status"],
        "prior_result_label": prior.get("decision_label") == spec["prerequisites"]["prior_composition_label"],
        "n_macros_eval": N_MACROS_EVAL == 8,
        "poles": list(spec["opponent_poles"]) == list(POLES),
        "compositions": list(spec["compositions"]) == list(COMPOSITIONS),
        "seed_block_shape": int(spec["execution"]["seed_block"]["n"]) == 64 and int(spec["execution"]["seed_block"]["last"]) - int(spec["execution"]["seed_block"]["base"]) == 63,
        "baseline_is_2a2d": BASELINE == "2A_2D",
        "composition_counts": all(len(composition_roles(name)) == 4 for name in COMPOSITIONS),
    }
    fixture = {name: list(composition_roles(name)) for name in COMPOSITIONS}
    checks["frozen_role_fixture"] = fixture == {
        "4A_0D": [0, 0, 0, 0],
        "2A_2D": [1, 1, 0, 0],
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
    expected_macros = {"4A_0D": [2, 2, 2, 2], "2A_2D": [7, 7, 2, 2]}
    checks["opponent_and_action_preflight"] = all(
        item["pass"] and item["blue_scripted"] is False and item["macros_by_composition"] == expected_macros
        for item in pole_checks.values()
    )
    passed = all(bool(value) for key, value in checks.items() if key != "spec_status")
    out = {
        "record_id": "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_PREFLIGHT",
        "status": "PASS" if passed else "FAIL",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "prior_result_sha256": _sha256(prior_result_path),
        "semantic_result_sha256": _sha256(semantic_path),
        "checks": checks,
        "pole_checks": pole_checks,
        "episodes_run": 0,
        "seed_allocation": "not performed by preflight",
    }
    PREFLIGHT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _cell_summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "blue_win_rate": _bootstrap(np.asarray([row["blue_win"] for row in rows]), 17),
        "score_difference": _bootstrap(np.asarray([row["blue_score"] - row["red_score"] for row in rows]), 19),
        "draw_rate": float(np.mean([row["draw"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_defend_inward_count": float(np.mean([row["defend_inward_count"] for row in rows])),
        "mean_defend_outward_count": float(np.mean([row["defend_outward_count"] for row in rows])),
    }


def run_evaluation(workers: int = 4) -> dict:
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
            desc="4v4 role-composition confirmation",
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
    MAPPING_PATH.write_text(json.dumps({
        "record_id": "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_MAPPING_AUDIT",
        "rows": mappings,
    }, indent=2) + "\n", encoding="utf-8")

    summaries = {
        f"{pole}/{composition}": _cell_summary([
            row for row in rows if row["pole"] == pole and row["composition"] == composition
        ])
        for pole in POLES for composition in COMPOSITIONS
    }
    by_key = {(row["pole"], row["composition"], row["seed"]): row for row in rows}
    delta_by_pole = {}
    for pole, seed_base in (("A", 101), ("B", 103)):
        current = np.asarray([
            by_key[(pole, "4A_0D", seed)]["blue_win"] - by_key[(pole, BASELINE, seed)]["blue_win"]
            for seed in seeds
        ])
        score_delta = np.asarray([
            (by_key[(pole, "4A_0D", seed)]["blue_score"] - by_key[(pole, "4A_0D", seed)]["red_score"]) -
            (by_key[(pole, BASELINE, seed)]["blue_score"] - by_key[(pole, BASELINE, seed)]["red_score"])
            for seed in seeds
        ])
        delta_by_pole[pole] = {
            "n": int(current.size),
            "blue_win_rate": _bootstrap(current, seed_base),
            "score_difference": _bootstrap(score_delta, seed_base + 1),
        }
    delta_a = np.asarray([
        by_key[("A", "4A_0D", seed)]["blue_win"] - by_key[("A", BASELINE, seed)]["blue_win"]
        for seed in seeds
    ])
    delta_b = np.asarray([
        by_key[("B", "4A_0D", seed)]["blue_win"] - by_key[("B", BASELINE, seed)]["blue_win"]
        for seed in seeds
    ])
    interaction_values = delta_b - delta_a
    interaction = {
        "definition": "Delta_B - Delta_A",
        "paired_seed_values": interaction_values.tolist(),
        "blue_win_rate": _bootstrap(interaction_values, 107),
    }
    b_gate = float(delta_by_pole["B"]["blue_win_rate"]["lcb95"]) > 0.0
    interaction_gate = float(interaction["blue_win_rate"]["lcb95"]) > 0.0
    a_nonnegative_requirement = float(delta_by_pole["A"]["blue_win_rate"]["ucb95"]) >= 0.0
    if interaction_gate and b_gate and a_nonnegative_requirement:
        label = "REGIME_COMPOSITION_CONFIRMATION_PASS"
    elif b_gate:
        label = "B_COMPOSITION_EFFECT_ONLY"
    else:
        label = "NO_TARGETED_COMPOSITION_CONFIRMATION"
    out = {
        "record_id": "PYQUATICUS_4V4_ROLE_COMPOSITION_CONFIRMATION_RESULT",
        "status": "COMPLETE",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "preflight_sha256": _sha256(PREFLIGHT_PATH),
        "n_rows": len(rows),
        "summaries": summaries,
        "delta_by_pole": delta_by_pole,
        "interaction": interaction,
        "decision_gate_details": {
            "interaction_lcb95_gt_zero": interaction_gate,
            "delta_B_lcb95_gt_zero": b_gate,
            "delta_A_ucb95_ge_zero": a_nonnegative_requirement,
        },
        "decision_label": label,
        "claim_boundary": spec["claim_boundary"],
        "six_v_six": "NOT_RUN",
        "ppo": "OFF",
    }
    RESULT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    out = run_preflight() if args.preflight else run_evaluation(args.workers)
    print(json.dumps(out, indent=2))
    return 0 if out.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
