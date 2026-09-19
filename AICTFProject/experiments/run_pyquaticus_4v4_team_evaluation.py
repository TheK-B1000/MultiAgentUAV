"""Frozen Pyquaticus-derived scripted-role 4v4 team evaluation.

The role semantics and assignment arms are frozen in
``PYQUATICUS_4V4_TEAM_EVALUATION_SPEC.json``.  This runner deliberately keeps
PPO off and uses the evaluation-only eight-entry macro vocabulary so
``MacroAction.DEFEND == 7`` can be sent directly to the scripted reference
controller path.  It must not be imported by PPO training code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from experiments.tqdm_loop import tqdm_iter

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "PYQUATICUS_4V4_TEAM_EVALUATION_SPEC.json"
PREFLIGHT_PATH = SD / "PYQUATICUS_4V4_TEAM_EVALUATION_PREFLIGHT.json"
RESULT_PATH = SD / "PYQUATICUS_4V4_TEAM_EVALUATION_RESULT.json"
EPISODE_CSV = SD / "PYQUATICUS_4V4_TEAM_EVALUATION_EPISODES.csv"
MAPPING_PATH = SD / "PYQUATICUS_4V4_TEAM_EVALUATION_MAPPING_AUDIT.json"
B3_GENOME = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"
B3_CERT = SD / "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json"
HORIZON = 240
N_MACROS_EVAL = 8
N_TARGETS = 50
NUM_BOOT = 20_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_SEED_ALLOCATION":
        raise RuntimeError(f"team-evaluation spec is not frozen: {spec.get('status')!r}")
    return spec


def assignment_roles(
    positions: Iterable[Iterable[float]],
    home: Iterable[float],
    arm: str,
) -> tuple[int, ...]:
    """Return role ids (0 ATTACK, 1 DEFEND) for the stable agent ids."""
    points = np.asarray(list(positions), dtype=np.float64)
    home_point = np.asarray(tuple(home), dtype=np.float64)
    if points.shape != (4, 2):
        raise ValueError(f"expected four [x,y] positions, got {points.shape}")
    if home_point.shape != (2,):
        raise ValueError(f"expected [x,y] home, got {home_point.shape}")
    if arm == "FIXED_IDENTITY":
        defenders = {0, 1}
    else:
        distance = np.linalg.norm(points - home_point[None, :], axis=1)
        if arm == "CLOSEST_DEFENDS":
            order = sorted(range(4), key=lambda i: (float(distance[i]), i))
        elif arm == "FARTHEST_DEFENDS":
            order = sorted(range(4), key=lambda i: (-float(distance[i]), i))
        else:
            raise ValueError(f"unknown assignment arm {arm!r}")
        defenders = set(order[:2])
    roles = tuple(1 if i in defenders else 0 for i in range(4))
    if roles.count(1) != 2 or roles.count(0) != 2:
        raise AssertionError(f"assignment does not produce 2A/2D: {roles}")
    return roles


def _role_ids_from_names(roles: tuple[int, ...]) -> tuple[list[int], list[int]]:
    defenders = [i for i, role in enumerate(roles) if role == 1]
    attackers = [i for i, role in enumerate(roles) if role == 0]
    return attackers, defenders


def _make_env(pole: str, seed: int):
    import experiments.strategic_demand_searcher as searcher
    from experiments.pole_attestation import (
        assert_resolved_matches_certification,
        attest_live_pole,
        resolve_pole_genome,
    )
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    genome_path = str(B3_GENOME) if pole == "B" else None
    genome = resolve_pole_genome(pole, 4, genome_path)
    attestation = assert_resolved_matches_certification("A", 4, B3_CERT, genome) if pole == "A" else assert_resolved_matches_certification("B", 4, B3_CERT, genome)
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=4,
        max_red_agents=4,
        n_macros=N_MACROS_EVAL,
        map_set="train",
        map_layout=searcher.MAP,
        max_decision_steps=HORIZON,
        score_limit=1_000_000,
        aquaticus_profile=True,
        rules_profile="OURS",
        device="cpu",
        seed=int(seed),
        obstacle_obs_channel=True,
        tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True,
        **searcher.RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    opp = genome.base_opponent
    env.env_method("set_phase", opp)
    env.env_method("set_next_opponent", "SCRIPTED", opp)
    apply_genome_to_core(core, genome)
    core.blue_scripted = False
    env.reset()
    apply_genome_to_core(core, genome)
    core.blue_scripted = False
    core.drain_tag_events()
    live = attest_live_pole(core, pole, 4, attestation, context="Pyquaticus 4v4 team-eval preflight")
    return env, core, genome, live


def _action_for_roles(core, roles: tuple[int, ...]) -> np.ndarray:
    from macro_actions import MacroAction

    carrying = bool(core.blue_carrying[0].any().item())
    action = np.zeros((1, 4, 2), dtype=np.int64)
    for i, role in enumerate(roles):
        if role == 1:
            macro = int(MacroAction.DEFEND)
        elif carrying:
            macro = int(MacroAction.GO_HOME)
        else:
            macro = int(MacroAction.GET_FLAG)
        action[0, i, 0] = macro
        action[0, i, 1] = 0
    if int(action[..., 0].max()) >= N_MACROS_EVAL:
        raise AssertionError("evaluation action escaped the frozen n_macros=8 vocabulary")
    return action


def _telemetry_for_tick(core, roles: tuple[int, ...], counters: dict[str, int]) -> None:
    from gpu_env.pyquaticus_port import DEFENDER_RADIUS_CELLS

    carrying = bool(core.blue_carrying[0].any().item())
    for i, role in enumerate(roles):
        prefix = "defend" if role == 1 else "attack"
        counters[f"{prefix}_ticks"] += 1
        if role == 0:
            counters["carrier_home_branch_count"] += int(carrying)
            counters["attack_enemy_flag_branch_count"] += int(not carrying)
        else:
            px = float(core.blue_x[0, i].item())
            py = float(core.blue_y[0, i].item())
            fx = float(core.blue_flag_pos[0, 0].item())
            fy = float(core.blue_flag_pos[0, 1].item())
            distance = float(np.hypot(px - fx, py - fy))
            counters["defend_inward_count"] += int(distance > DEFENDER_RADIUS_CELLS)
            counters["defend_outward_count"] += int(distance <= DEFENDER_RADIUS_CELLS)
        counters["tagged_ticks_by_role"] += int(bool(core.blue_tagged[0, i].item()))


def run_episode(pole: str, arm: str, seed: int) -> tuple[dict, dict]:
    env, core, genome, live = _make_env(pole, seed)
    try:
        positions = np.stack([
            core.blue_x[0].detach().cpu().numpy(),
            core.blue_y[0].detach().cpu().numpy(),
        ], axis=1)
        home = core.blue_flag_home[0].detach().cpu().numpy()
        roles = assignment_roles(positions, home, arm)
        attackers, defenders = _role_ids_from_names(roles)
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
                    "success": int(core.blue_score[0].item() > core.red_score[0].item()),
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
            "assignment_arm": arm,
            "defender_ids": ",".join(map(str, defenders)),
            "attacker_ids": ",".join(map(str, attackers)),
            "blue_score": blue_score,
            "red_score": red_score,
            "blue_win": int(blue_score > red_score),
            "draw": int(blue_score == red_score),
            "steps": int(steps),
            "blue_flag_captures": blue_score,
            "red_flag_captures": red_score,
            "blue_alive_end": blue_alive_end,
            "red_alive_end": None,
            "genome_id": str(genome.genome_id),
            "pole_config_hash": str(live.get("live_config_hash", "")),
            **counters,
            "role_switch_count": 0,
        }
        return row, {
            "seed": int(seed),
            "pole": pole,
            "assignment_arm": arm,
            "defender_ids": defenders,
            "attacker_ids": attackers,
        }
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
    spec = _load_spec()
    from gpu_env.pyquaticus_port import UPSTREAM_COMMIT, UPSTREAM_SHA256

    checks: dict[str, object] = {}
    checks["spec_status"] = spec["status"]
    checks["upstream_commit"] = UPSTREAM_COMMIT == spec["prerequisites"]["upstream_provenance"]["commit"]
    checks["n_macros_eval"] = N_MACROS_EVAL == int(spec["execution"]["environment"]["evaluation_only_n_macros"])
    checks["role_arms"] = set(spec["assignment_arms"]["arms"]) == {"FIXED_IDENTITY", "CLOSEST_DEFENDS", "FARTHEST_DEFENDS"}
    checks["role_count"] = spec["assignment_arms"]["role_count"] == {"attack": 2, "defend": 2}
    checks["v2_status"] = spec["prerequisites"]["semantic_contracts"]["unified_defend_status"] == "REPRESENTABLE_ACROSS_TRANSITIONS"
    checks["upstream_hash_block_present"] = len(UPSTREAM_SHA256) >= 9

    fixture = np.array([[1.0, 0.0], [2.0, 0.0], [8.0, 0.0], [9.0, 0.0]])
    expected = {
        "FIXED_IDENTITY": (1, 1, 0, 0),
        "CLOSEST_DEFENDS": (1, 1, 0, 0),
        "FARTHEST_DEFENDS": (0, 0, 1, 1),
    }
    fixture_results = {
        arm: assignment_roles(fixture, (0.0, 0.0), arm)
        for arm in expected
    }
    checks["synthetic_assignment_fixtures"] = fixture_results == expected

    pole_checks = {}
    for pole in ("A", "B"):
        env, core, genome, live = _make_env(pole, 20260918)
        try:
            positions = np.stack([
                core.blue_x[0].detach().cpu().numpy(),
                core.blue_y[0].detach().cpu().numpy(),
            ], axis=1)
            home = core.blue_flag_home[0].detach().cpu().numpy()
            mappings = {
                arm: assignment_roles(positions, home, arm)
                for arm in ("FIXED_IDENTITY", "CLOSEST_DEFENDS", "FARTHEST_DEFENDS")
            }
            roles = mappings["CLOSEST_DEFENDS"]
            action = _action_for_roles(core, roles)
            assert action.shape == (1, 4, 2)
            assert int(action[..., 0].max()) < N_MACROS_EVAL
            assert int(action[0, roles.index(1), 0]) == 7
            assert bool(core.blue_scripted) is False
            pole_checks[pole] = {
                "pass": True,
                "genome_id": genome.genome_id,
                "live_config_hash": live.get("live_config_hash"),
                "mappings": {name: list(values) for name, values in mappings.items()},
                "direct_action_macros": action[0, :, 0].tolist(),
            }
        finally:
            env.close()
    checks["opponent_and_action_preflight"] = all(item.get("pass") for item in pole_checks.values())
    passed = all(bool(value) for key, value in checks.items() if key != "spec_status")
    result = {
        "record_id": "PYQUATICUS_4V4_TEAM_EVALUATION_PREFLIGHT",
        "status": "PASS" if passed else "FAIL",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "checks": checks,
        "pole_checks": pole_checks,
        "episodes_run": 0,
        "seed_allocation": "not performed by preflight",
    }
    PREFLIGHT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_evaluation(workers: int = 1) -> dict:
    spec = _load_spec()
    if not PREFLIGHT_PATH.is_file():
        raise RuntimeError("preflight artifact missing; run --preflight first")
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError(f"preflight is not PASS: {preflight.get('status')!r}")
    seeds = range(int(spec["execution"]["seed_block"]["base"]), int(spec["execution"]["seed_block"]["last"]) + 1)
    jobs = [
        (pole, arm, seed)
        for pole in ("A", "B")
        for arm in ("FIXED_IDENTITY", "CLOSEST_DEFENDS", "FARTHEST_DEFENDS")
        for seed in seeds
    ]
    rows: list[dict] = []
    mappings: list[dict] = []
    if int(workers) <= 1:
        for job in tqdm_iter(jobs, desc="4v4 team evaluation", total=len(jobs), unit="ep"):
            row, mapping = run_episode(*job)
            rows.append(row)
            mappings.append(mapping)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futures = {pool.submit(run_episode, *job): job for job in jobs}
            for future in tqdm_iter(
                as_completed(futures),
                desc="4v4 team evaluation",
                total=len(futures),
                unit="ep",
            ):
                job = futures[future]
                row, mapping = future.result()
                rows.append(row)
                mappings.append(mapping)
        rows.sort(key=lambda row: (row["pole"], row["assignment_arm"], row["seed"]))
        mappings.sort(key=lambda row: (row["pole"], row["assignment_arm"], row["seed"]))
    fieldnames = list(rows[0].keys()) if rows else []
    with EPISODE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    MAPPING_PATH.write_text(json.dumps({"record_id": "PYQUATICUS_4V4_TEAM_EVALUATION_MAPPING_AUDIT", "rows": mappings}, indent=2) + "\n", encoding="utf-8")

    summaries = {}
    for pole in ("A", "B"):
        for arm in ("FIXED_IDENTITY", "CLOSEST_DEFENDS", "FARTHEST_DEFENDS"):
            subset = [row for row in rows if row["pole"] == pole and row["assignment_arm"] == arm]
            win = np.asarray([row["blue_win"] for row in subset], dtype=np.float64)
            score = np.asarray([row["blue_score"] - row["red_score"] for row in subset], dtype=np.float64)
            summaries[f"{pole}/{arm}"] = {
                "n": len(subset),
                "blue_win_rate": _bootstrap(win, seed=17),
                "score_difference": _bootstrap(score, seed=19),
                "draw_rate": float(np.mean([row["draw"] for row in subset])),
                "mean_steps": float(np.mean([row["steps"] for row in subset])),
                "mean_defend_inward_count": float(np.mean([row["defend_inward_count"] for row in subset])),
                "mean_defend_outward_count": float(np.mean([row["defend_outward_count"] for row in subset])),
            }
    contrasts = {}
    arm_pairs = (("FIXED_IDENTITY", "CLOSEST_DEFENDS"), ("FIXED_IDENTITY", "FARTHEST_DEFENDS"), ("CLOSEST_DEFENDS", "FARTHEST_DEFENDS"))
    for pole in ("A", "B"):
        for left, right in arm_pairs:
            l = {row["seed"]: row for row in rows if row["pole"] == pole and row["assignment_arm"] == left}
            r = {row["seed"]: row for row in rows if row["pole"] == pole and row["assignment_arm"] == right}
            common = sorted(set(l) & set(r))
            contrasts[f"{pole}/{left}_minus_{right}"] = {
                "n": len(common),
                "blue_win_rate": _bootstrap(np.asarray([l[s]["blue_win"] - r[s]["blue_win"] for s in common]), seed=23),
                "score_difference": _bootstrap(np.asarray([(l[s]["blue_score"] - l[s]["red_score"]) - (r[s]["blue_score"] - r[s]["red_score"]) for s in common]), seed=29),
            }
    all_assignment_signals = [
        c["blue_win_rate"] for c in contrasts.values()
    ]
    signal = any(float(c["lcb95"]) > 0.0 or float(c["ucb95"]) < 0.0 for c in all_assignment_signals)
    result = {
        "record_id": "PYQUATICUS_4V4_TEAM_EVALUATION_RESULT",
        "status": "COMPLETE",
        "utc": _now(),
        "implements": str(SPEC_PATH.relative_to(ROOT)).replace("\\", "/"),
        "spec_sha256": _sha256(SPEC_PATH),
        "preflight_sha256": _sha256(PREFLIGHT_PATH),
        "n_rows": len(rows),
        "summaries": summaries,
        "assignment_contrasts": contrasts,
        "decision_label": "ASSIGNMENT_EFFECT_DESCRIPTIVE_SIGNAL" if signal else "NO_DEMONSTRATED_ASSIGNMENT_EFFECT",
        "claim_boundary": spec["claim_boundary"],
        "six_v_six": "NOT_RUN",
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("choose exactly one of --preflight or --run")
    result = run_preflight() if args.preflight else run_evaluation(workers=args.workers)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
