"""Collect the frozen V2 oracle-gated rehearsal bank expansion block.

Implements ORACLE_GATED_K2_V2_COLLECTION_PROTOCOL.json.

320 training-only seeds (11000001..11000320), 12 uniform branch points per seed
drawn from all four (policy, pole) trajectories. Same shard schema and branch
semantics as the stratified collector; selection is NOT 16-cell stratified.

Run:  python experiments/collect_oracle_gated_v2_bank_data.py --device cuda
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

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "sppo" / "ORACLE_GATED_K2_V2_COLLECTION_PROTOCOL.json"
OUT = SD / "sppo" / "oracle_gated_k2_v2_bank_data"
COMPLETE = OUT / "COLLECTION_COMPLETE.json"

BASE, N_SEEDS = 11_000_001, 320
SEEDS = list(range(BASE, BASE + N_SEEDS))
BRANCH_POINTS_PER_SEED = 12
LATE_BOUNDARY = 127
RNG_SEED = 29
POLICIES = ("pi_A", "pi_B")
POLES = ("A", "B")

FORBIDDEN = [
    range(6_500_001, 6_500_257),
    range(10_400_001, 10_400_033),
    range(10_500_001, 10_500_097),
    range(10_600_001, 10_600_193),
    range(10_700_001, 10_700_161),
    range(10_800_001, 10_800_002),
    range(10_900_001, 10_900_002),
    range(11_100_001, 11_100_002),
    range(11_200_001, 11_200_033),
]
EXPECTED_TEACHER_SHA = {
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_name(cell: tuple) -> str:
    pole, regime, horizon = cell
    return f"{pole}_r{regime}_{horizon}"


def _copy_obs(obs):
    return {k: np.asarray(v).copy() for k, v in obs.items()}


def _preflight(seed_lo: int, seed_hi: int, out: Path, worker_id: str) -> dict:
    if not PROTOCOL.is_file():
        raise RuntimeError(f"REFUSING: protocol missing: {PROTOCOL}")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not spec["status"].startswith("FROZEN"):
        raise RuntimeError(f"REFUSING: protocol not frozen: {spec['status']!r}")
    blk = spec["seed_block"]
    if list(blk) != [BASE, BASE + N_SEEDS - 1, N_SEEDS]:
        raise RuntimeError("REFUSING: protocol seed block drifted")
    worker_seeds = list(range(seed_lo, seed_hi + 1))
    if not worker_seeds:
        raise RuntimeError("REFUSING: empty seed range")
    if not set(worker_seeds).issubset(set(SEEDS)):
        raise RuntimeError("REFUSING: seed range escapes the frozen 320-seed block")
    if (out / "COLLECTION_COMPLETE.json").is_file():
        raise RuntimeError(f"REFUSING: already COMPLETE: {out / 'COLLECTION_COMPLETE.json'}")
    worker_done = out / "WORKER_RANGE_COMPLETE.json"
    if worker_done.is_file():
        raise RuntimeError(f"REFUSING: worker range already COMPLETE: {worker_done}")
    for forbidden in FORBIDDEN:
        overlap = set(worker_seeds) & set(forbidden)
        if overlap:
            raise RuntimeError(f"REFUSING: seed range collides: {sorted(overlap)[:5]}")
    for name, path in P0.TEACHERS.items():
        actual = sha256_file(Path(path))
        if actual != EXPECTED_TEACHER_SHA[name]:
            raise RuntimeError(f"REFUSING: {name} teacher hash mismatch")
    return spec


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
                records.append({"step": step, "obs": _copy_obs(obs), "action": action.copy()})
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
        suffix = np.flip(np.cumsum(np.flip(reward_matrix, axis=0), axis=0), axis=0)
        for record in records:
            record["return"] = suffix[record["step"]].copy()
        blue, red = terminal
        summary = {
            "seed": seed, "split": "V2_BANK", "policy": policy, "pole": pole,
            "blue": blue, "red": red, "win": int(blue > red), "margin": blue - red,
            "steps": len(rewards), "decision_records": len(records),
        }
        return summary, records, prefix, decisions
    finally:
        env.close()


def _cells_for(tagger, records, pole: str, policy: str):
    import torch
    if not records:
        return {}
    vec = np.stack([np.asarray(r["obs"]["vec"])[0] for r in records])
    regimes = tagger.regime_from_vec(torch.as_tensor(vec, dtype=torch.float32)).numpy()
    buckets: dict[tuple, list[tuple[str, int]]] = {}
    for record, regime in zip(records, regimes):
        step = int(record["step"])
        cell = (pole, int(regime), "late" if step > LATE_BOUNDARY else "not_late")
        buckets.setdefault(cell, []).append((policy, step))
    return buckets


def select_uniform_points(tagger, sources: dict, seed: int):
    """Uniform draw of branch points from all four trajectories."""
    candidates: list[tuple[str, str, int, tuple]] = []
    for (policy, pole), (_, records, _) in sources.items():
        for cell, items in _cells_for(tagger, records, pole, policy).items():
            for src_policy, step in items:
                candidates.append((pole, src_policy, int(step), cell))
    if not candidates:
        raise RuntimeError(f"seed={seed} has no branch candidates")
    rng = np.random.default_rng([RNG_SEED, seed])
    n_pick = min(BRANCH_POINTS_PER_SEED, len(candidates))
    pick = rng.choice(len(candidates), size=n_pick, replace=False)
    selections = [candidates[int(i)] for i in sorted(pick)]
    selections.sort(key=lambda item: (item[0], item[2], item[1]))
    visited = {}
    for pole, _, step, cell in selections:
        visited[cell_name(cell)] = visited.get(cell_name(cell), 0) + 1
    return selections, visited


def _branch_one(model, policy: str, pole: str, seed: int, prefix, step: int, device: str):
    env = R2.build_env(device, seed)
    core = env.core
    try:
        obs = P0._prep(env, core, pole, seed)
        for index in range(step):
            env.step_async(prefix[index])
            obs, _, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                raise RuntimeError(f"episode ended before branch seed={seed} step={step}")
        restored = _copy_obs(obs)
        rewards = []
        branch_action = np.asarray(model.predict(restored, deterministic=True)[0]).reshape(-1).astype(np.int64)
        action = branch_action
        terminal = None
        for _ in range(step, R2.MAX_STEPS):
            env.step_async(action)
            obs, reward, done, info = env.step_wait()
            rewards.append(np.asarray(reward, dtype=np.float64).reshape(-1))
            if bool(np.asarray(done).any()):
                terminal = P0._terminal(core, info)
                break
            action = np.asarray(model.predict(obs, deterministic=True)[0]).reshape(-1).astype(np.int64)
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        blue, red = terminal
        return {
            "obs": restored, "action": branch_action,
            "return": np.stack(rewards).sum(axis=0), "blue": blue, "red": red,
        }
    finally:
        env.close()


def _stack(arrays, prefix, records):
    for key in records[0]["obs"]:
        arrays[f"{prefix}_obs_{key}"] = np.stack([r["obs"][key] for r in records])
    arrays[f"{prefix}_action"] = np.stack([r["action"] for r in records])
    arrays[f"{prefix}_return"] = np.stack([r["return"] for r in records])


def _collect_seed(models, tagger, seed: int, device: str):
    summaries, plain_records = [], []
    sources: dict[tuple[str, str], tuple] = {}
    for policy in POLICIES:
        for pole in POLES:
            summary, records, prefix, _ = _plain_episode(models[policy], policy, pole, seed, device)
            summaries.append(summary)
            for record in records:
                record.update({"policy": policy, "pole": pole})
            plain_records.extend(records)
            sources[(policy, pole)] = (prefix, records, None)

    selections, visited = select_uniform_points(tagger, sources, seed)
    branches = []
    for branch_index, (pole, source_policy, step, cell) in enumerate(selections):
        prefix, _, _ = sources[(source_policy, pole)]
        pair = [_branch_one(models[p], p, pole, seed, prefix, step, device) for p in POLICIES]
        if not all(np.array_equal(pair[0]["obs"][k], pair[1]["obs"][k]) for k in pair[0]["obs"]):
            raise RuntimeError(f"branch obs mismatch seed={seed} step={step}")
        branches.append({
            "seed": seed, "pole": pole, "source_policy": source_policy, "step": step,
            "branch_index": branch_index, "cell": cell_name(cell),
            "regime": cell[1], "horizon": cell[2], "pair": pair,
        })

    arrays = {
        "plain_seed": np.asarray([seed] * len(plain_records), dtype=np.int64),
        "plain_step": np.asarray([r["step"] for r in plain_records], dtype=np.int32),
        "plain_policy": np.asarray([0 if r["policy"] == "pi_A" else 1 for r in plain_records], dtype=np.int8),
        "plain_pole": np.asarray([0 if r["pole"] == "A" else 1 for r in plain_records], dtype=np.int8),
        "branch_seed": np.asarray([seed] * len(branches), dtype=np.int64),
        "branch_step": np.asarray([b["step"] for b in branches], dtype=np.int32),
        "branch_index": np.asarray([b["branch_index"] for b in branches], dtype=np.int8),
        "branch_pole": np.asarray([0 if b["pole"] == "A" else 1 for b in branches], dtype=np.int8),
        "branch_source_policy": np.asarray(
            [0 if b["source_policy"] == "pi_A" else 1 for b in branches], dtype=np.int8),
        "branch_regime": np.asarray([b["regime"] for b in branches], dtype=np.int8),
        "branch_is_late": np.asarray([b["horizon"] == "late" for b in branches], dtype=np.int8),
        "branch_cell": np.asarray([b["cell"] for b in branches], dtype="U24"),
    }
    _stack(arrays, "plain", plain_records)
    for key in branches[0]["pair"][0]["obs"]:
        arrays[f"branch_obs_{key}"] = np.stack([b["pair"][0]["obs"][key] for b in branches])
    for policy_index, policy in enumerate(POLICIES):
        items = [b["pair"][policy_index] for b in branches]
        arrays[f"branch_{policy}_action"] = np.stack([i["action"] for i in items])
        arrays[f"branch_{policy}_return"] = np.stack([i["return"] for i in items])
        arrays[f"branch_{policy}_blue"] = np.asarray([i["blue"] for i in items], dtype=np.int16)
        arrays[f"branch_{policy}_red"] = np.asarray([i["red"] for i in items], dtype=np.int16)
    return summaries, arrays, [b["cell"] for b in branches], visited


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="smoke only")
    parser.add_argument("--seed-lo", type=int, default=BASE)
    parser.add_argument("--seed-hi", type=int, default=BASE + N_SEEDS - 1)
    parser.add_argument("--out-dir", type=str, default="",
                        help="isolated worker output root (default: canonical OUT)")
    parser.add_argument("--worker-id", type=str, default="serial",
                        help="provenance label for logs/manifests")
    args = parser.parse_args()

    out = Path(args.out_dir) if args.out_dir else OUT
    worker_seeds = list(range(int(args.seed_lo), int(args.seed_hi) + 1))
    spec = _preflight(int(args.seed_lo), int(args.seed_hi), out, args.worker_id)
    print(json.dumps({
        "mode": "DRY_RUN" if args.dry_run else "V2_BANK_COLLECTION",
        "worker_id": args.worker_id,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "seed_range": [worker_seeds[0], worker_seeds[-1], len(worker_seeds)],
        "frozen_block": [SEEDS[0], SEEDS[-1], len(SEEDS)],
        "branch_points_per_seed": BRANCH_POINTS_PER_SEED,
        "selection": spec["selection"],
        "output": str(out.relative_to(ROOT)),
    }, indent=2), flush=True)
    if args.dry_run:
        return 0

    from rl.custom_ppo import load_custom_ppo_policy
    from rl.scorer.qpsi import QPsi, QPsiConfig

    shards = out / "seed_shards"
    summaries_dir = out / "seed_summaries"
    shards.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    probe = R2.build_env(args.device, worker_seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    models = {
        name: load_custom_ppo_policy(str(path), obs_space, act_space, device=args.device)
        for name, path in P0.TEACHERS.items()
    }
    tagger = QPsi(QPsiConfig())

    manifest_path = out / "collection_manifest.json"
    completed: list[int] = []
    if manifest_path.exists():
        completed = [int(s) for s in json.loads(manifest_path.read_text(encoding="utf-8")).get("completed_seeds", [])]
    todo = [s for s in worker_seeds if s not in completed]
    if args.limit:
        todo = todo[: args.limit]

    for seed in todo:
        summaries, arrays, cells, visited = _collect_seed(models, tagger, seed, args.device)
        shard = shards / f"seed_{seed}.npz"
        with shard.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(shard.with_suffix(".npz.tmp"), shard)
        _write_json_atomic(summaries_dir / f"seed_{seed}.json", {
            "episodes": summaries, "branch_cells": cells, "visited_cell_counts": visited,
            "split": "V2_BANK", "seed": seed, "worker_id": args.worker_id,
        })
        completed.append(seed)
        _write_json_atomic(manifest_path, {
            "record": "V2 oracle-gated bank expansion collection",
            "protocol": "ORACLE_GATED_K2_V2_COLLECTION_PROTOCOL.json",
            "worker_id": args.worker_id,
            "seed_range": [int(args.seed_lo), int(args.seed_hi)],
            "updated_utc": _now(),
            "completed_seeds": completed,
            "target_seeds": len(worker_seeds),
            "branch_points_per_seed": BRANCH_POINTS_PER_SEED,
        })
        print(f"seed {seed} [{len(cells)}] branch pts worker={args.worker_id} "
              f"({len(completed)}/{len(worker_seeds)})", flush=True)

    if len(completed) < len(worker_seeds):
        print(f"\nPARTIAL: {len(completed)}/{len(worker_seeds)} worker={args.worker_id}")
        return 0

    _write_json_atomic(out / "WORKER_RANGE_COMPLETE.json", {
        "verdict": "WORKER_RANGE_COMPLETE",
        "utc": _now(),
        "worker_id": args.worker_id,
        "seed_range": [int(args.seed_lo), int(args.seed_hi), len(worker_seeds)],
        "completed_seeds": len(completed),
        "next_step": "experiments/merge_oracle_gated_v2_bank_workers.py after ALL workers + serial prefix",
    })
    print(f"\nWORKER_RANGE_COMPLETE worker={args.worker_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
