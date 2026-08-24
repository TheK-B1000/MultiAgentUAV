"""Exact Phase 0 branch-environment reuse equivalence diagnostic.

This spends no scorer-training or evaluation data. It reuses the two frozen
first-interval evidence seeds and compares a candidate per-seed environment
pool against the production rebuild-per-branch reference across both source
policies, both poles, and early/mid/late branch points.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import phase0_collect_scorer_data as phase0  # noqa: E402
from experiments import r2_learned_crossover as R2  # noqa: E402
from rl.behavior_telemetry import compute_behavior_telemetry_batch  # noqa: E402


OUT = phase0.SD / "PHASE0_ENV_REUSE_EQUIVALENCE.json"
SEEDS = (phase0.SEED_BASE, phase0.SEED_BASE + 1)


def _array(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().numpy()
    return np.asarray(value)


def _digest(value: Any) -> str | None:
    """Stable exact digest for numeric/core state, including RNG state."""
    h = hashlib.sha256()
    if isinstance(value, torch.Generator):
        value = value.get_state()
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        arr = _array(value)
        h.update(str(arr.dtype).encode())
        h.update(repr(tuple(arr.shape)).encode())
        h.update(arr.tobytes(order="C"))
        return h.hexdigest()
    if value is None or isinstance(value, (bool, int, float, str)):
        h.update(f"{type(value).__name__}:{value!r}".encode())
        return h.hexdigest()
    if isinstance(value, (list, tuple)):
        parts = [_digest(v) for v in value]
        if any(p is None for p in parts):
            return None
        h.update(type(value).__name__.encode())
        h.update(json.dumps(parts, separators=(",", ":")).encode())
        return h.hexdigest()
    if isinstance(value, dict):
        parts = []
        for key in sorted(value, key=lambda x: str(x)):
            item = _digest(value[key])
            if item is None:
                return None
            parts.append((str(key), item))
        h.update(json.dumps(parts, separators=(",", ":")).encode())
        return h.hexdigest()
    return None


def _core_state(core: Any) -> dict[str, str]:
    state: dict[str, str] = {}
    for name, value in sorted(vars(core).items()):
        digest = _digest(value)
        if digest is not None:
            state[name] = digest
    return state


def _obs_state(obs: dict[str, Any]) -> dict[str, str]:
    return {name: _digest(value) for name, value in sorted(obs.items())}


def _mask_state(core: Any, obs: dict[str, Any]) -> dict[str, str]:
    masks = {
        "core_blue": _digest(core._build_action_mask(side="blue")),
    }
    for name, value in sorted(obs.items()):
        if "mask" in name.lower():
            masks[f"obs:{name}"] = _digest(value)
    return masks


def _terminal(core: Any, info: Any) -> tuple[int, int]:
    return phase0._terminal(core, info)


def _run_branch(env: Any, model: Any, pole: str, seed: int,
                prefix: list[np.ndarray], branch_t: int) -> dict[str, Any]:
    core = env.core
    obs = phase0._prep(env, core, pole, seed)
    for i in range(branch_t):
        env.step_async(prefix[i])
        obs, _reward, done, _info = env.step_wait()
        if bool(np.asarray(done).any()):
            raise RuntimeError(f"episode ended before branch t={branch_t}")

    action, _ = model.predict(obs, deterministic=True)
    action = np.asarray(action).reshape(-1).astype(np.int64)
    restored = {
        "observation": _obs_state(obs),
        "action_masks": _mask_state(core, obs),
        "core_numeric_ledger": _core_state(core),
        "behavior_telemetry": _digest(
            compute_behavior_telemetry_batch(
                core, torch.as_tensor(action, dtype=torch.long, device=core.device).reshape(1, -1)
            )
        ),
    }

    total_reward = None
    env.step_async(action)
    obs, reward, done, info = env.step_wait()
    reward_arr = np.asarray(reward, dtype=np.float64)
    total_reward = reward_arr.copy()
    after_first = {
        "observation": _obs_state(obs),
        "action_masks": _mask_state(core, obs),
        "core_numeric_ledger": _core_state(core),
        "reward": _digest(reward_arr),
        "done": _digest(np.asarray(done)),
    }
    steps = 1
    terminal = _terminal(core, info) if bool(np.asarray(done).any()) else None
    while terminal is None and steps < R2.MAX_STEPS - branch_t:
        action_next, _ = model.predict(obs, deterministic=True)
        action_next = np.asarray(action_next).reshape(-1).astype(np.int64)
        env.step_async(action_next)
        obs, reward, done, info = env.step_wait()
        total_reward += np.asarray(reward, dtype=np.float64)
        steps += 1
        if bool(np.asarray(done).any()):
            terminal = _terminal(core, info)
    if terminal is None:
        terminal = (int(core.blue_score[0]), int(core.red_score[0]))
    blue, red = terminal
    return {
        "restored": restored,
        "first_action": action.tolist(),
        "after_first": after_first,
        "continuation": {
            "blue": blue,
            "red": red,
            "win": int(blue > red),
            "margin": blue - red,
            "steps": steps,
            "return": np.asarray(total_reward).tolist(),
            "return_digest": _digest(total_reward),
            "terminal_core_numeric_ledger": _core_state(core),
        },
    }


def _reference(model: Any, pole: str, seed: int, prefix: list[np.ndarray],
               branch_t: int, device: str) -> dict[str, Any]:
    env = R2.build_env(device, seed)
    try:
        return _run_branch(env, model, pole, seed, prefix, branch_t)
    finally:
        env.close()


def _compare(reference: dict[str, Any], reuse: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "restored_observation": reference["restored"]["observation"] == reuse["restored"]["observation"],
        "restored_masks": reference["restored"]["action_masks"] == reuse["restored"]["action_masks"],
        "restored_core_ledger": reference["restored"]["core_numeric_ledger"] == reuse["restored"]["core_numeric_ledger"],
        "restored_behavior_telemetry": reference["restored"]["behavior_telemetry"] == reuse["restored"]["behavior_telemetry"],
        "first_branch_action": reference["first_action"] == reuse["first_action"],
        "after_first_observation": reference["after_first"]["observation"] == reuse["after_first"]["observation"],
        "after_first_masks": reference["after_first"]["action_masks"] == reuse["after_first"]["action_masks"],
        "after_first_core_ledger": reference["after_first"]["core_numeric_ledger"] == reuse["after_first"]["core_numeric_ledger"],
        "first_reward_done": (
            reference["after_first"]["reward"] == reuse["after_first"]["reward"]
            and reference["after_first"]["done"] == reuse["after_first"]["done"]
        ),
        "continuation_outcome": all(
            reference["continuation"][key] == reuse["continuation"][key]
            for key in ("blue", "red", "win", "margin", "steps")
        ),
        "continuation_return": reference["continuation"]["return_digest"] == reuse["continuation"]["return_digest"],
        "terminal_core_ledger": (
            reference["continuation"]["terminal_core_numeric_ledger"]
            == reuse["continuation"]["terminal_core_numeric_ledger"]
        ),
    }
    return {
        "checks": checks,
        "exact": all(checks.values()),
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "core_fields_compared": len(reference["restored"]["core_numeric_ledger"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from rl.custom_ppo import load_custom_ppo_policy

    probe = R2.build_env(args.device, SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    models = {
        name: load_custom_ppo_policy(str(path), obs_space, act_space, device=args.device)
        for name, path in phase0.TEACHERS.items()
    }

    rows: list[dict[str, Any]] = []
    reference_seconds = 0.0
    reuse_seconds = 0.0
    for seed in SEEDS:
        pools = {name: R2.build_env(args.device, seed) for name in ("pi_A", "pi_B")}
        try:
            for pole in ("A", "B"):
                source = phase0.source_policy_for(seed, pole)
                rollout = phase0.rollout(models[source], pole, seed, args.device, record_prefix=True)
                points, notes = phase0.select_tertile_points(
                    rollout["decision_steps"], rollout["steps"]
                )
                if len(points) != 3:
                    raise RuntimeError(
                        f"seed={seed} pole={pole} did not yield three branches: {points} {notes}"
                    )
                for tertile, branch_t in zip(("early", "mid", "late"), points):
                    for policy in ("pi_A", "pi_B"):
                        start = time.perf_counter()
                        ref = _reference(
                            models[policy], pole, seed, rollout["prefix"], branch_t, args.device
                        )
                        reference_seconds += time.perf_counter() - start
                        start = time.perf_counter()
                        candidate = _run_branch(
                            pools[policy], models[policy], pole, seed,
                            rollout["prefix"], branch_t,
                        )
                        reuse_seconds += time.perf_counter() - start
                        comparison = _compare(ref, candidate)
                        rows.append({
                            "seed": seed,
                            "pole": pole,
                            "source_policy": source,
                            "branch_policy": policy,
                            "tertile": tertile,
                            "branch_t": branch_t,
                            **comparison,
                        })
        finally:
            for env in pools.values():
                env.close()

    required_paths = {
        f"{row['pole']}|{row['source_policy']}" for row in rows
    }
    required_branch_coverage = {
        (row["pole"], row["branch_policy"], row["tertile"]) for row in rows
    }
    expected_branch_coverage = {
        (pole, policy, tertile)
        for pole in ("A", "B")
        for policy in ("pi_A", "pi_B")
        for tertile in ("early", "mid", "late")
    }
    passed = (
        all(row["exact"] for row in rows)
        and required_paths == {"A|pi_A", "A|pi_B", "B|pi_A", "B|pi_B"}
        and required_branch_coverage == expected_branch_coverage
    )
    record = {
        "record": "PHASE0 environment-reuse exact equivalence",
        "classification": "IMPLEMENTATION_DIAGNOSTIC_NO_SCIENCE_DATA",
        "reference": "rebuild one environment per branch-policy continuation",
        "candidate": "build one environment per seed and branch policy; reset/replay for all poles and tertiles",
        "seeds": list(SEEDS),
        "four_source_paths": sorted(required_paths),
        "comparisons": len(rows),
        "required_branch_path_count": len(required_branch_coverage),
        "timing": {
            "reference_seconds": reference_seconds,
            "reuse_seconds": reuse_seconds,
            "speedup": reference_seconds / max(reuse_seconds, 1e-12),
        },
        "rows": rows,
        "verdict": "PASS_EXACT" if passed else "FAIL_REJECT_REUSE",
        "reuse_authorized": passed,
        "failure_rule": "Any failed exact check rejects reuse; the original collector remains authoritative.",
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({
        "verdict": record["verdict"],
        "comparisons": len(rows),
        "four_source_paths": sorted(required_paths),
        "timing": record["timing"],
        "failed_rows": sum(not row["exact"] for row in rows),
        "output": str(OUT),
    }, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
