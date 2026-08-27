"""Collect the frozen 16-cell STRATIFIED REGIME block.

Implements STRATIFIED_REGIME_COLLECTION_PROTOCOL_DESIGN.json, frozen by PI ruling
at commit c318bcea and launch-authorized 2026-08-27.

This is a selection-only adaptation of ``collect_rasr_dev_scorer_data.py``. The
rollout, branch_at, teacher-consistent continuation and shard schema are unchanged
-- the ONLY difference is which decision points become branch states.

RASR chose branch points by tertile, uniformly over a heavily skewed visitation
distribution, and 5 of 8 pole x regime cells fell below the 32-distinct-seed floor.
The states were present in the trajectories the whole time; uniform selection simply
never landed on them. Here, each seed walks the 16 cells in a FROZEN rarest-first
order and spends one branch slot on each cell it actually visits, up to 12.

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

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "rasrppo" / "STRATIFIED_REGIME_COLLECTION_PROTOCOL_DESIGN.json"
OUT = SD / "stratified_regime_data"
COMPLETE = OUT / "COLLECTION_COMPLETE.json"

# ---- frozen parameters (must match the protocol; verified at preflight) ----
BASE, N_SEEDS = 10_700_001, 160
SEEDS = list(range(BASE, BASE + N_SEEDS))
BRANCH_POINTS_PER_SEED = 12
LATE_BOUNDARY = 127          # D5 late tertile boundary, reused unchanged
RNG_SEED = 13
SUPPORT_FLOOR = 32

FROZEN_RANK = [
    ("B", 3, "not_late"), ("B", 2, "not_late"), ("B", 2, "late"), ("B", 3, "late"),
    ("A", 2, "late"), ("A", 1, "late"), ("A", 2, "not_late"), ("A", 0, "late"),
    ("A", 3, "not_late"), ("B", 1, "late"), ("B", 1, "not_late"), ("B", 0, "late"),
    ("A", 0, "not_late"), ("A", 1, "not_late"), ("A", 3, "late"), ("B", 0, "not_late"),
]

SPLITS = {"FIT": range(10_700_001, 10_700_097),
          "CALIB": range(10_700_097, 10_700_129),
          "EVAL": range(10_700_129, 10_700_161)}

# blocks that are already spent; the new block must not intersect any of them
FORBIDDEN = [range(6_500_001, 6_500_257), range(10_400_001, 10_400_033),
             range(10_500_001, 10_500_097), range(10_600_001, 10_600_193),
             range(10_300_001, 10_300_193)]

EXPECTED_TEACHER_SHA = {
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_name(cell) -> str:
    return f"{cell[0]}_r{cell[1]}_{cell[2]}"


def source_policy_for(seed: int, pole: str) -> str:
    """Unchanged in structure from the RASR collector; only the base moves."""
    if seed not in SEEDS or pole not in ("A", "B"):
        raise ValueError("source assignment escaped the frozen block")
    return ("pi_A", "pi_B")[(seed - BASE + (pole == "B")) % 2]


def _copy_obs(obs):
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def _preflight() -> dict:
    """Refuse loudly rather than silently collecting the wrong thing."""
    if not PROTOCOL.is_file():
        raise RuntimeError(f"REFUSING: frozen protocol not found at {PROTOCOL}")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not spec["status"].startswith("FROZEN -- STRATIFIED_16CELL_COLLECTION_PROTOCOL_FROZEN"):
        raise RuntimeError(f"REFUSING: protocol is not frozen: {spec['status']!r}")
    auth = spec["AUTHORIZATION"]["collection_launch"]
    if not auth.startswith("AUTHORIZED"):
        raise RuntimeError(f"REFUSING: collection launch is not authorized: {auth!r}")

    frozen = spec["FREEZE_RECORD"]["frozen_parameters_restated"]
    drift = []
    if frozen["seeds"] != N_SEEDS:
        drift.append(f"seeds {frozen['seeds']} != {N_SEEDS}")
    if frozen["branch_points_per_seed"] != BRANCH_POINTS_PER_SEED:
        drift.append(f"branch_points {frozen['branch_points_per_seed']} != {BRANCH_POINTS_PER_SEED}")
    if frozen["cells"] != len(FROZEN_RANK):
        drift.append(f"cells {frozen['cells']} != {len(FROZEN_RANK)}")
    if set(spec["ALLOCATION_RULE"]["frozen_rank_order"]) != {cell_name(c) for c in FROZEN_RANK}:
        drift.append("rank order membership differs from the frozen protocol")
    if spec["ALLOCATION_RULE"]["frozen_rank_order"] != [cell_name(c) for c in FROZEN_RANK]:
        drift.append("rank ORDER differs from the frozen protocol")
    if drift:
        raise RuntimeError("REFUSING: parameters drifted from the frozen protocol: " + "; ".join(drift))

    if COMPLETE.exists():
        raise RuntimeError(f"REFUSING: output is already COMPLETE: {COMPLETE}")
    if SEEDS != list(range(10_700_001, 10_700_161)):
        raise RuntimeError("REFUSING: seed block drifted")
    for blk in FORBIDDEN:
        overlap = set(SEEDS) & set(blk)
        if overlap:
            raise RuntimeError(f"REFUSING: seed block collides with a spent block: {sorted(overlap)[:5]}")
    if set().union(*(set(r) for r in SPLITS.values())) != set(SEEDS):
        raise RuntimeError("REFUSING: FIT/CALIB/EVAL splits do not partition the block")

    for name, path in P0.TEACHERS.items():
        from experiments.phase0_scorer_common import sha256_file
        actual = sha256_file(Path(path))
        if actual != EXPECTED_TEACHER_SHA[name]:
            raise RuntimeError(
                f"REFUSING: {name} teacher hash mismatch "
                f"(expected {EXPECTED_TEACHER_SHA[name]}, got {actual})")
    return spec


def _plain_episode(model, policy: str, pole: str, seed: int, device: str):
    """Unchanged from the RASR collector."""
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
            "seed": seed, "split": _split_of(seed), "policy": policy, "pole": pole,
            "blue": blue, "red": red, "win": int(blue > red), "margin": blue - red,
            "steps": len(rewards), "decision_records": len(records),
        }
        return summary, records, prefix, decisions
    finally:
        env.close()


def _split_of(seed: int) -> str:
    for name, rng in SPLITS.items():
        if seed in rng:
            return name
    raise ValueError(f"seed {seed} is outside every split")


def _cells_for(tagger, records, pole: str):
    """Bucket a source trajectory's decision steps by (pole, regime, horizon)."""
    import torch
    if not records:
        return {}
    vec = np.stack([np.asarray(r["obs"]["vec"])[0] for r in records])   # (N,2,20)
    regimes = tagger.regime_from_vec(torch.as_tensor(vec, dtype=torch.float32)).numpy()
    buckets: dict[tuple, list[int]] = {}
    for record, regime in zip(records, regimes):
        step = int(record["step"])
        cell = (pole, int(regime), "late" if step > LATE_BOUNDARY else "not_late")
        buckets.setdefault(cell, []).append(step)
    return buckets


def select_stratified_points(tagger, sources, seed: int):
    """The frozen allocation rule.

    Walk the 16 cells in FROZEN_RANK order; for each cell this seed actually
    visits, spend one branch slot, until BRANCH_POINTS_PER_SEED are gone. No
    backfilling from abundant cells -- backfilling would re-create the uniform
    sampling skew this protocol exists to remove.
    """
    buckets: dict[tuple, list[int]] = {}
    for pole, (_, _, records, _) in sources.items():
        buckets.update(_cells_for(tagger, records, pole))
    rng = np.random.default_rng([RNG_SEED, seed])
    chosen: dict[str, list[tuple[int, tuple]]] = {"A": [], "B": []}
    slots = BRANCH_POINTS_PER_SEED
    for cell in FROZEN_RANK:
        if slots == 0:
            break
        steps = buckets.get(cell)
        if not steps:
            continue
        chosen[cell[0]].append((int(rng.choice(sorted(steps))), cell))
        slots -= 1
    for pole in ("A", "B"):
        chosen[pole].sort()
    return chosen, {cell_name(c): len(v) for c, v in sorted(buckets.items())}


def _branch_one(model, policy: str, pole: str, seed: int, prefix, step: int, device: str):
    """Unchanged from the RASR collector: fresh env, replay prefix, branch."""
    env = R2.build_env(device, seed)
    core = env.core
    try:
        obs = P0._prep(env, core, pole, seed)
        for index in range(step):
            env.step_async(prefix[index])
            obs, _, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                raise RuntimeError(f"episode ended before branch seed={seed} pole={pole} step={step}")
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
        return {"obs": restored, "action": branch_action,
                "return": np.stack(rewards).sum(axis=0), "blue": blue, "red": red}
    finally:
        env.close()


def _stack(arrays, prefix, records):
    for key in records[0]["obs"]:
        arrays[f"{prefix}_obs_{key}"] = np.stack([r["obs"][key] for r in records])
    arrays[f"{prefix}_action"] = np.stack([r["action"] for r in records])
    arrays[f"{prefix}_return"] = np.stack([r["return"] for r in records])


def _collect_seed(models, tagger, seed: int, device: str):
    summaries, plain_records, sources = [], [], {}
    for policy in ("pi_A", "pi_B"):
        for pole in ("A", "B"):
            summary, records, prefix, decisions = _plain_episode(models[policy], policy, pole, seed, device)
            summaries.append(summary)
            for record in records:
                record.update({"policy": policy, "pole": pole})
            plain_records.extend(records)
            if policy == source_policy_for(seed, pole):
                sources[pole] = (policy, prefix, records, decisions)

    chosen, visited = select_stratified_points(tagger, sources, seed)
    n_chosen = sum(len(v) for v in chosen.values())
    if n_chosen == 0:
        raise RuntimeError(f"seed={seed} selected no branch points")

    branches = []
    for pole in ("A", "B"):
        source, prefix, _, _ = sources[pole]
        for branch_index, (step, cell) in enumerate(chosen[pole]):
            pair = [_branch_one(models[policy], policy, pole, seed, prefix, step, device)
                    for policy in ("pi_A", "pi_B")]
            if not all(np.array_equal(pair[0]["obs"][k], pair[1]["obs"][k]) for k in pair[0]["obs"]):
                raise RuntimeError(f"matched branch state mismatch seed={seed} pole={pole} step={step}")
            branches.append({"seed": seed, "pole": pole, "source_policy": source,
                             "step": step, "branch_index": branch_index,
                             "cell": cell_name(cell), "regime": cell[1],
                             "horizon": cell[2], "pair": pair})

    arrays = {
        "plain_seed": np.asarray([r["step"] * 0 + seed for r in plain_records], dtype=np.int64),
        "plain_step": np.asarray([r["step"] for r in plain_records], dtype=np.int32),
        "plain_policy": np.asarray([0 if r["policy"] == "pi_A" else 1 for r in plain_records], dtype=np.int8),
        "plain_pole": np.asarray([0 if r["pole"] == "A" else 1 for r in plain_records], dtype=np.int8),
        "branch_seed": np.asarray([seed] * len(branches), dtype=np.int64),
        "branch_step": np.asarray([b["step"] for b in branches], dtype=np.int32),
        "branch_index": np.asarray([b["branch_index"] for b in branches], dtype=np.int8),
        "branch_pole": np.asarray([0 if b["pole"] == "A" else 1 for b in branches], dtype=np.int8),
        "branch_source_policy": np.asarray([0 if b["source_policy"] == "pi_A" else 1 for b in branches], dtype=np.int8),
        "branch_regime": np.asarray([b["regime"] for b in branches], dtype=np.int8),
        "branch_is_late": np.asarray([b["horizon"] == "late" for b in branches], dtype=np.int8),
        "branch_cell": np.asarray([b["cell"] for b in branches], dtype="U24"),
    }
    _stack(arrays, "plain", plain_records)
    for key in branches[0]["pair"][0]["obs"]:
        arrays[f"branch_obs_{key}"] = np.stack([b["pair"][0]["obs"][key] for b in branches])
    for policy_index, policy in enumerate(("pi_A", "pi_B")):
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
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after N seeds; SMOKE USE ONLY, never for the frozen run")
    args = parser.parse_args()

    spec = _preflight()
    contract = {
        "mode": "DRY_RUN" if args.dry_run else "STRATIFIED_COLLECTION",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_status": spec["status"],
        "seed_block": [SEEDS[0], SEEDS[-1], len(SEEDS)],
        "splits": {k: [min(v), max(v), len(v)] for k, v in SPLITS.items()},
        "cells": len(FROZEN_RANK),
        "frozen_rank_order": [cell_name(c) for c in FROZEN_RANK],
        "branch_points_per_seed": BRANCH_POINTS_PER_SEED,
        "branch_states_max": len(SEEDS) * BRANCH_POINTS_PER_SEED,
        "branch_continuations_max": len(SEEDS) * BRANCH_POINTS_PER_SEED * 2,
        "late_boundary": LATE_BOUNDARY,
        "rng_seed": RNG_SEED,
        "support_floor_per_cell": SUPPORT_FLOOR,
        "support_floor_scope": "checked ONCE over the FULL block after collection; never per split, never during",
        "environment_semantics": "fresh rebuild per teacher per branch; reuse forbidden",
        "output": str(OUT.relative_to(ROOT)),
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "final_106_seeds_touched": False,
    }
    print(json.dumps(contract, indent=2), flush=True)
    if args.dry_run:
        print("\nDRY RUN -- protocol verified; no environment constructed, no seed spent.")
        return 0

    from rl.custom_ppo import load_custom_ppo_policy
    from rl.scorer.qpsi import QPsi, QPsiConfig
    tagger = QPsi(QPsiConfig())

    OUT.mkdir(parents=True, exist_ok=True)
    shards, summaries_dir = OUT / "seed_shards", OUT / "seed_summaries"
    shards.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)
    probe = R2.build_env(args.device, SEEDS[0])
    observation_space, action_space = probe.observation_space, probe.action_space
    probe.close()
    models = {name: load_custom_ppo_policy(str(path), observation_space, action_space, device=args.device)
              for name, path in P0.TEACHERS.items()}

    manifest_path = OUT / "collection_manifest.json"
    completed: list[int] = []
    if manifest_path.exists():
        completed = [int(s) for s in json.loads(manifest_path.read_text(encoding="utf-8")).get("completed_seeds", [])]
    if not set(completed).issubset(SEEDS):
        raise RuntimeError("REFUSING: manifest contains a seed outside the frozen block")

    todo = [s for s in SEEDS if s not in completed]
    if args.limit:
        todo = todo[: args.limit]
    for seed in todo:
        summaries, arrays, cells, visited = _collect_seed(models, tagger, seed, args.device)
        shard = shards / f"seed_{seed}.npz"
        with shard.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(shard.with_suffix(".npz.tmp"), shard)
        _write_json_atomic(summaries_dir / f"seed_{seed}.json",
                           {"episodes": summaries, "branch_cells": cells,
                            "visited_cell_counts": visited, "split": _split_of(seed)})
        completed.append(seed)
        _write_json_atomic(manifest_path, {
            "record": "16-cell stratified regime collection",
            "protocol": "STRATIFIED_REGIME_COLLECTION_PROTOCOL_DESIGN.json, frozen c318bcea",
            "updated_utc": _now(),
            "semantics": "fresh rebuild per teacher per branch",
            "completed_seeds": completed,
            "target_seeds": len(SEEDS),
            "branch_points_per_seed": BRANCH_POINTS_PER_SEED,
            "teacher_sha256": EXPECTED_TEACHER_SHA,
        })
        print(f"seed {seed} [{_split_of(seed)}] {len(cells)} branch pts "
              f"({len(completed)}/{len(SEEDS)})", flush=True)

    if len(completed) < len(SEEDS):
        print(f"\nPARTIAL: {len(completed)}/{len(SEEDS)}. No COMPLETE marker written.")
        return 0

    episode_path = OUT / "episode_summaries.jsonl"
    temporary = episode_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for seed in SEEDS:
            rows = json.loads((summaries_dir / f"seed_{seed}.json").read_text(encoding="utf-8"))
            for row in rows["episodes"]:
                handle.write(json.dumps(row) + "\n")
    os.replace(temporary, episode_path)
    _write_json_atomic(COMPLETE, {
        "verdict": "COLLECTION_COMPLETE",
        "utc": _now(),
        "seed_block": [SEEDS[0], SEEDS[-1], len(SEEDS)],
        "completed_seeds": len(completed),
        "splits": {k: [min(v), max(v), len(v)] for k, v in SPLITS.items()},
        "semantics": "fresh rebuild per teacher per branch",
        "support_floor_check": "NOT PERFORMED HERE -- a separate one-shot audit scores the 32-distinct-seed floor over all 16 cells",
        "final_106_seeds_touched": False,
    })
    print("\nCOLLECTION_COMPLETE written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
