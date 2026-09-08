"""Collect the frozen RASR-PPO DEV scorer-qualification block.

This is a DEV-only adaptation of ``phase0_full_scorer_collection.py``.
Every counterfactual continuation reconstructs a fresh seeded environment;
environment reuse across branch teachers or branch points is forbidden.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import phase0_collect_scorer_data as P0  # noqa: E402
from experiments import r2_learned_crossover as R2  # noqa: E402
from experiments.phase0_scorer_common import sha256_file  # noqa: E402
from experiments.run_rasrppo_ladder import require_dev_collection_gate  # noqa: E402

RASR_DIR = ROOT / "artifacts" / "strategic_demand" / "rasrppo"
OUT = RASR_DIR / "dev_scorer_data"
COMPLETE = OUT / "COLLECTION_COMPLETE.json"
DEV_BASE, N_DEV = 10_500_001, 96
DEV_SEEDS = list(range(DEV_BASE, DEV_BASE + N_DEV))
TRAIN_RANGE = range(10_400_001, 10_400_033)
FINAL_RANGE = range(10_600_001, 10_600_193)
EXPECTED_TEACHER_SHA = {
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_policy_for(seed: int, pole: str) -> str:
    if seed not in DEV_SEEDS or pole not in ("A", "B"):
        raise ValueError("RASR DEV source assignment escaped the frozen block")
    return ("pi_A", "pi_B")[(seed - DEV_BASE + (pole == "B")) % 2]


def _copy_obs(obs):
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def _existing_seed_ids() -> set[int]:
    found: set[int] = set()
    for directory in (OUT / "seed_shards", OUT / "seed_summaries"):
        if not directory.exists():
            continue
        for path in directory.glob("seed_*.*"):
            try:
                found.add(int(path.stem.removeprefix("seed_")))
            except ValueError as exc:
                raise RuntimeError(f"unrecognized seed artifact {path}") from exc
    manifest = OUT / "collection_manifest.json"
    if manifest.is_file():
        found.update(
            int(seed)
            for seed in json.loads(manifest.read_text(encoding="utf-8")).get(
                "completed_seeds", []
            )
        )
    return found


def _preflight() -> dict:
    gate = require_dev_collection_gate()
    if COMPLETE.exists():
        raise RuntimeError(f"REFUSING: DEV output is already COMPLETE: {COMPLETE}")
    if DEV_SEEDS != list(range(10_500_001, 10_500_097)):
        raise RuntimeError("REFUSING: DEV seed block drifted")
    existing = _existing_seed_ids()
    train = sorted(existing.intersection(TRAIN_RANGE))
    final = sorted(existing.intersection(FINAL_RANGE))
    outside = sorted(existing.difference(DEV_SEEDS))
    if train:
        raise RuntimeError(f"REFUSING: TRAIN 104* seeds found in DEV output: {train}")
    if final:
        raise RuntimeError(f"REFUSING: FINAL 106* seeds found in DEV output: {final}")
    if outside:
        raise RuntimeError(f"REFUSING: non-DEV seeds found in DEV output: {outside}")
    for name, path in P0.TEACHERS.items():
        actual = sha256_file(path) if path.is_file() else None
        if actual != EXPECTED_TEACHER_SHA[name]:
            raise RuntimeError(
                f"REFUSING: {name} teacher hash mismatch "
                f"(expected {EXPECTED_TEACHER_SHA[name]}, got {actual})"
            )
    return gate


def _plain_episode(model, policy: str, pole: str, seed: int, device: str):
    env = R2.build_env(device, seed)
    core = env.core
    try:
        obs = P0._prep(env, core, pole, seed)
        prefix, decisions, records, rewards = [], [], [], []
        terminal = None
        for step in range(R2.MAX_STEPS):
            decision = bool((core.blue_commit_ticks_left[0] <= 0).any().item())
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action).reshape(-1).astype(np.int64)
            if decision:
                decisions.append(step)
                records.append(
                    {"step": step, "obs": _copy_obs(obs), "action": action.copy()}
                )
            prefix.append(action.copy())
            env.step_async(action)
            obs, reward, done, info = env.step_wait()
            rewards.append(np.asarray(reward, dtype=np.float64).reshape(-1))
            if bool(np.asarray(done).any()):
                terminal = P0._terminal(core, info)
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        reward_matrix = np.stack(rewards)
        suffix = np.flip(
            np.cumsum(np.flip(reward_matrix, axis=0), axis=0), axis=0
        )
        for record in records:
            record["return"] = suffix[record["step"]].copy()
        blue, red = terminal
        summary = {
            "seed": seed,
            "split": "DEV_QUALIFICATION_ONLY",
            "policy": policy,
            "pole": pole,
            "blue": blue,
            "red": red,
            "win": int(blue > red),
            "margin": blue - red,
            "steps": len(rewards),
            "decision_records": len(records),
        }
        return summary, records, prefix, decisions
    finally:
        env.close()


def _branch_one(model, policy: str, pole: str, seed: int, prefix, step: int, device: str):
    """Build a fresh environment for exactly one teacher continuation."""
    env = R2.build_env(device, seed)
    core = env.core
    try:
        obs = P0._prep(env, core, pole, seed)
        for index in range(step):
            env.step_async(prefix[index])
            obs, _reward, done, _info = env.step_wait()
            if bool(np.asarray(done).any()):
                raise RuntimeError(
                    f"episode ended before branch seed={seed} pole={pole} step={step}"
                )
        restored = _copy_obs(obs)
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action).reshape(-1).astype(np.int64)
        rewards, terminal = [], None
        for _ in range(R2.MAX_STEPS - step):
            env.step_async(action)
            obs, reward, done, info = env.step_wait()
            rewards.append(np.asarray(reward, dtype=np.float64).reshape(-1))
            if bool(np.asarray(done).any()):
                terminal = P0._terminal(core, info)
                break
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action).reshape(-1).astype(np.int64)
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        blue, red = terminal
        branch_action = np.asarray(
            model.predict(restored, deterministic=True)[0]
        ).reshape(-1).astype(np.int64)
        return {
            "policy": policy,
            "obs": restored,
            "action": branch_action,
            "return": np.stack(rewards).sum(axis=0),
            "blue": blue,
            "red": red,
            "win": int(blue > red),
            "margin": blue - red,
        }
    finally:
        env.close()


def _stack_records(records, prefix: str, arrays: dict) -> None:
    if not records:
        raise RuntimeError(f"no {prefix} decision records")
    for key in records[0]["obs"]:
        arrays[f"{prefix}_obs_{key}"] = np.stack(
            [record["obs"][key] for record in records]
        )
    arrays[f"{prefix}_action"] = np.stack([record["action"] for record in records])
    arrays[f"{prefix}_return"] = np.stack([record["return"] for record in records])


def _collect_seed(models, seed: int, device: str):
    summaries, plain_records, sources = [], [], {}
    for policy in ("pi_A", "pi_B"):
        for pole in ("A", "B"):
            summary, records, prefix, decisions = _plain_episode(
                models[policy], policy, pole, seed, device
            )
            summaries.append(summary)
            for record in records:
                record.update({"policy": policy, "pole": pole})
            plain_records.extend(records)
            if policy == source_policy_for(seed, pole):
                sources[pole] = (policy, prefix, decisions, summary["steps"])

    branches = []
    for pole in ("A", "B"):
        source, prefix, decisions, horizon = sources[pole]
        points, notes = P0.select_tertile_points(decisions, horizon)
        if len(points) != P0.BRANCHES_PER_SOURCE:
            raise RuntimeError(
                f"seed={seed} pole={pole} has {len(points)} branch points, expected 3"
            )
        for branch_index, step in enumerate(points):
            pair = [
                _branch_one(models[policy], policy, pole, seed, prefix, step, device)
                for policy in ("pi_A", "pi_B")
            ]
            if not all(
                np.array_equal(pair[0]["obs"][key], pair[1]["obs"][key])
                for key in pair[0]["obs"]
            ):
                raise RuntimeError(
                    f"matched branch state mismatch seed={seed} pole={pole} step={step}"
                )
            branches.append(
                {
                    "seed": seed,
                    "pole": pole,
                    "source_policy": source,
                    "branch_index": branch_index,
                    "step": step,
                    "notes": notes,
                    "pair": pair,
                }
            )

    arrays = {
        "plain_seed": np.asarray([seed] * len(plain_records), dtype=np.int64),
        "plain_step": np.asarray([row["step"] for row in plain_records], dtype=np.int32),
        "plain_policy": np.asarray(
            [0 if row["policy"] == "pi_A" else 1 for row in plain_records],
            dtype=np.int8,
        ),
        "plain_pole": np.asarray(
            [0 if row["pole"] == "A" else 1 for row in plain_records], dtype=np.int8
        ),
        "branch_seed": np.asarray([seed] * len(branches), dtype=np.int64),
        "branch_step": np.asarray([row["step"] for row in branches], dtype=np.int32),
        "branch_index": np.asarray(
            [row["branch_index"] for row in branches], dtype=np.int8
        ),
        "branch_pole": np.asarray(
            [0 if row["pole"] == "A" else 1 for row in branches], dtype=np.int8
        ),
        "branch_source_policy": np.asarray(
            [0 if row["source_policy"] == "pi_A" else 1 for row in branches],
            dtype=np.int8,
        ),
    }
    _stack_records(plain_records, "plain", arrays)
    for key in branches[0]["pair"][0]["obs"]:
        arrays[f"branch_obs_{key}"] = np.stack(
            [row["pair"][0]["obs"][key] for row in branches]
        )
    for policy_index, policy in enumerate(("pi_A", "pi_B")):
        items = [row["pair"][policy_index] for row in branches]
        arrays[f"branch_{policy}_action"] = np.stack(
            [item["action"] for item in items]
        )
        arrays[f"branch_{policy}_return"] = np.stack(
            [item["return"] for item in items]
        )
        arrays[f"branch_{policy}_blue"] = np.asarray(
            [item["blue"] for item in items], dtype=np.int16
        )
        arrays[f"branch_{policy}_red"] = np.asarray(
            [item["red"] for item in items], dtype=np.int16
        )
    return summaries, arrays


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gate = _preflight()
    balance = {
        pole: {
            policy: sum(
                source_policy_for(seed, pole) == policy for seed in DEV_SEEDS
            )
            for policy in ("pi_A", "pi_B")
        }
        for pole in ("A", "B")
    }
    contract = {
        "mode": "DRY_RUN" if args.dry_run else "DEV_COLLECTION",
        "gate_verdict": gate["verdict"],
        "dev_collection_authorized": gate["dev_collection_authorized"],
        "seed_block": [DEV_SEEDS[0], DEV_SEEDS[-1], len(DEV_SEEDS)],
        "split": "DEV qualification only; forbidden for fitting",
        "plain_episodes": len(DEV_SEEDS) * 4,
        "branch_states": len(DEV_SEEDS) * 2 * P0.BRANCHES_PER_SOURCE,
        "branch_continuations": len(DEV_SEEDS) * 2 * P0.BRANCHES_PER_SOURCE * 2,
        "source_balance": balance,
        "four_paths": ["pi_A|A", "pi_A|B", "pi_B|A", "pi_B|B"],
        "tertile_branch_points": ["early", "mid", "late"],
        "environment_semantics": "fresh rebuild per teacher per branch; reuse forbidden",
        "output": str(OUT.relative_to(ROOT)),
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "train_104_seeds_used_for_fitting": False,
        "final_106_seeds_touched": False,
    }
    print(json.dumps(contract, indent=2))
    if args.dry_run:
        print("\nDRY RUN -- gate checked; no environment constructed and no DEV step spent.")
        return 0

    from rl.custom_ppo import load_custom_ppo_policy

    OUT.mkdir(parents=True, exist_ok=True)
    shards = OUT / "seed_shards"
    summaries_dir = OUT / "seed_summaries"
    shards.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)
    probe = R2.build_env(args.device, DEV_SEEDS[0])
    observation_space, action_space = probe.observation_space, probe.action_space
    probe.close()
    models = {
        name: load_custom_ppo_policy(
            str(path), observation_space, action_space, device=args.device
        )
        for name, path in P0.TEACHERS.items()
    }

    manifest_path = OUT / "collection_manifest.json"
    completed: list[int] = []
    if manifest_path.exists():
        completed = [
            int(seed)
            for seed in json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "completed_seeds", []
            )
        ]
    if not set(completed).issubset(DEV_SEEDS):
        raise RuntimeError("REFUSING: manifest contains a non-DEV seed")
    for seed in DEV_SEEDS:
        if seed in completed:
            continue
        summaries, arrays = _collect_seed(models, seed, args.device)
        shard = shards / f"seed_{seed}.npz"
        with shard.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(shard.with_suffix(".npz.tmp"), shard)
        _write_json_atomic(summaries_dir / f"seed_{seed}.json", summaries)
        completed.append(seed)
        _write_json_atomic(
            manifest_path,
            {
                "record": "RASR-PPO DEV scorer collection",
                "updated_utc": _now(),
                "semantics": "fresh rebuild per teacher per branch",
                "use": "DEV qualification only; never scorer fitting",
                "completed_seeds": completed,
                "target_seeds": len(DEV_SEEDS),
                "plain_episodes_per_seed": 4,
                "branch_states_per_seed": 6,
                "teacher_sha256": EXPECTED_TEACHER_SHA,
            },
        )
        print(f"seed {seed} complete ({len(completed)}/{len(DEV_SEEDS)})", flush=True)

    episode_path = OUT / "episode_summaries.jsonl"
    temporary = episode_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for seed in DEV_SEEDS:
            rows = json.loads(
                (summaries_dir / f"seed_{seed}.json").read_text(encoding="utf-8")
            )
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    os.replace(temporary, episode_path)
    _write_json_atomic(
        COMPLETE,
        {
            "verdict": "COLLECTION_COMPLETE",
            "utc": _now(),
            "seed_block": [DEV_SEEDS[0], DEV_SEEDS[-1], len(DEV_SEEDS)],
            "completed_seeds": len(completed),
            "plain_episodes": len(completed) * 4,
            "branch_states": len(completed) * 6,
            "branch_continuations": len(completed) * 12,
            "semantics": "fresh rebuild per teacher per branch",
            "use": "DEV qualification only; never scorer fitting",
            "train_104_seeds_used_for_fitting": False,
            "final_106_seeds_touched": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
