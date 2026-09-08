"""Collect the frozen 16-cell STRATIFIED REGIME block.

Implements STRATIFIED_REGIME_COLLECTION_PROTOCOL_DESIGN.json, frozen by PI ruling
at commit c318bcea, launch-authorized 2026-08-27, and amended by AMENDMENT_2
(all four (policy, pole) trajectories eligible; secondary source-balance tie-break).

This is a selection-only adaptation of ``collect_rasr_dev_scorer_data.py``. The
rollout, branch_at, teacher-consistent continuation and shard schema are unchanged
-- the ONLY scientific deltas vs RASR are (1) which decision points become branch
states and (2) AMENDMENT_2 eligibility / provenance / tie-break.

Attempt 1 (source-only eligibility) is STOPPED BEFORE FULL COLLECTION /
FEASIBILITY DEFECT; its seed is quarantined and recollected under this file.
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

FORBIDDEN = [range(6_500_001, 6_500_257), range(10_400_001, 10_400_033),
             range(10_500_001, 10_500_097), range(10_600_001, 10_600_193),
             range(10_300_001, 10_300_193)]

EXPECTED_TEACHER_SHA = {
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}

POLICIES = ("pi_A", "pi_B")
POLES = ("A", "B")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_name(cell) -> str:
    return f"{cell[0]}_r{cell[1]}_{cell[2]}"


def _copy_obs(obs):
    return {key: np.asarray(value).copy() for key, value in obs.items()}


def _empty_source_counts() -> dict[str, dict[str, int]]:
    return {cell_name(cell): {"pi_A": 0, "pi_B": 0} for cell in FROZEN_RANK}


def rebuild_source_counts_from_shards(
    completed_seeds: list[int], shards_dir: Path
) -> dict[str, dict[str, int]]:
    """Mandatory resume rebuild: reproduce the uninterrupted global counter."""
    counts = _empty_source_counts()
    for seed in sorted(int(s) for s in completed_seeds):
        shard = shards_dir / f"seed_{seed}.npz"
        if not shard.is_file():
            raise RuntimeError(
                f"REFUSING: completed seed {seed} has no shard for counter rebuild"
            )
        with np.load(shard, allow_pickle=False) as data:
            cells = [str(c) for c in data["branch_cell"]]
            sources = np.asarray(data["branch_source_policy"], dtype=np.int64)
            if len(cells) != len(sources):
                raise RuntimeError(f"REFUSING: shard {shard.name} cell/source length mismatch")
            for name, source_idx in zip(cells, sources):
                policy = "pi_A" if int(source_idx) == 0 else "pi_B"
                if name not in counts:
                    raise RuntimeError(
                        f"REFUSING: shard {shard.name} has unknown cell {name!r}"
                    )
                counts[name][policy] += 1
    return counts


def _preflight() -> dict:
    """Refuse loudly rather than silently collecting the wrong thing."""
    if not PROTOCOL.is_file():
        raise RuntimeError(f"REFUSING: frozen protocol not found at {PROTOCOL}")
    spec = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if not spec["status"].startswith("FROZEN -- STRATIFIED_16CELL_COLLECTION_PROTOCOL_FROZEN"):
        raise RuntimeError(f"REFUSING: protocol is not frozen: {spec['status']!r}")
    if "AMENDMENT_2_ALL_FOUR_TRAJECTORIES_ELIGIBLE" not in spec:
        raise RuntimeError("REFUSING: AMENDMENT_2 is absent from the protocol")
    auth = spec["AUTHORIZATION"]["collection_launch"]
    if not auth.startswith("AUTHORIZED"):
        raise RuntimeError(f"REFUSING: collection launch is not authorized: {auth!r}")
    if "AMENDMENT_2" not in auth:
        raise RuntimeError(
            "REFUSING: collection launch authorization does not cite AMENDMENT_2"
        )

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


def _cells_for(tagger, records, pole: str, policy: str):
    """Bucket a trajectory's decision steps by (pole, regime, horizon).

    Candidates carry source-policy provenance; cell identity ignores policy.
    """
    import torch
    if not records:
        return {}
    vec = np.stack([np.asarray(r["obs"]["vec"])[0] for r in records])   # (N,2,20)
    regimes = tagger.regime_from_vec(torch.as_tensor(vec, dtype=torch.float32)).numpy()
    buckets: dict[tuple, list[tuple[str, int]]] = {}
    for record, regime in zip(records, regimes):
        step = int(record["step"])
        cell = (pole, int(regime), "late" if step > LATE_BOUNDARY else "not_late")
        buckets.setdefault(cell, []).append((policy, step))
    return buckets


def select_stratified_points(
    tagger,
    sources: dict[tuple[str, str], tuple],
    seed: int,
    source_counts: dict[str, dict[str, int]],
):
    """Rarest-first allocation over all four (policy, pole) trajectories.

    AMENDMENT_2: every (policy, pole) path is eligible. Cell identity is still
    pole x regime x horizon only. When both policies supply candidates for a
    cell, prefer the policy currently less represented in that cell (secondary
    tie-break); if equal, use the frozen uniform RNG.
    """
    buckets: dict[tuple, list[tuple[str, int]]] = {}
    for (policy, pole), (_, records, _) in sources.items():
        for cell, candidates in _cells_for(tagger, records, pole, policy).items():
            buckets.setdefault(cell, []).extend(candidates)

    visited = {
        cell_name(cell): len(candidates) for cell, candidates in sorted(buckets.items())
    }
    rng = np.random.default_rng([RNG_SEED, seed])
    # selections: list of (pole, policy, step, cell)
    selections: list[tuple[str, str, int, tuple]] = []
    slots = BRANCH_POINTS_PER_SEED
    for cell in FROZEN_RANK:
        if slots == 0:
            break
        candidates = buckets.get(cell)
        if not candidates:
            continue
        name = cell_name(cell)
        count_a = int(source_counts[name]["pi_A"])
        count_b = int(source_counts[name]["pi_B"])
        if count_a < count_b:
            preferred = [c for c in candidates if c[0] == "pi_A"] or list(candidates)
        elif count_b < count_a:
            preferred = [c for c in candidates if c[0] == "pi_B"] or list(candidates)
        else:
            preferred = list(candidates)
        # Stable order then uniform draw among the preferred set.
        preferred = sorted(preferred, key=lambda item: (item[0], item[1]))
        policy, step = preferred[int(rng.integers(0, len(preferred)))]
        selections.append((cell[0], policy, int(step), cell))
        source_counts[name][policy] += 1
        slots -= 1

    selections.sort(key=lambda item: (item[0], item[2], item[1]))
    return selections, visited


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
                raise RuntimeError(
                    f"episode ended before branch seed={seed} pole={pole} step={step}"
                )
        restored = _copy_obs(obs)
        rewards = []
        branch_action = np.asarray(
            model.predict(restored, deterministic=True)[0]
        ).reshape(-1).astype(np.int64)
        action = branch_action
        terminal = None
        for _ in range(step, R2.MAX_STEPS):
            env.step_async(action)
            obs, reward, done, info = env.step_wait()
            rewards.append(np.asarray(reward, dtype=np.float64).reshape(-1))
            if bool(np.asarray(done).any()):
                terminal = P0._terminal(core, info)
                break
            action = np.asarray(
                model.predict(obs, deterministic=True)[0]
            ).reshape(-1).astype(np.int64)
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        blue, red = terminal
        return {
            "obs": restored,
            "action": branch_action,
            "return": np.stack(rewards).sum(axis=0),
            "blue": blue,
            "red": red,
        }
    finally:
        env.close()


def _stack(arrays, prefix, records):
    for key in records[0]["obs"]:
        arrays[f"{prefix}_obs_{key}"] = np.stack([r["obs"][key] for r in records])
    arrays[f"{prefix}_action"] = np.stack([r["action"] for r in records])
    arrays[f"{prefix}_return"] = np.stack([r["return"] for r in records])


def _collect_seed(models, tagger, seed: int, device: str, source_counts):
    summaries, plain_records = [], []
    # AMENDMENT_2: retain all four (policy, pole) trajectories.
    sources: dict[tuple[str, str], tuple] = {}
    for policy in POLICIES:
        for pole in POLES:
            summary, records, prefix, decisions = _plain_episode(
                models[policy], policy, pole, seed, device
            )
            summaries.append(summary)
            for record in records:
                record.update({"policy": policy, "pole": pole})
            plain_records.extend(records)
            sources[(policy, pole)] = (prefix, records, decisions)

    selections, visited = select_stratified_points(
        tagger, sources, seed, source_counts
    )
    if not selections:
        raise RuntimeError(f"seed={seed} selected no branch points")

    branches = []
    for branch_index, (pole, source_policy, step, cell) in enumerate(selections):
        prefix, _, _ = sources[(source_policy, pole)]
        pair = [
            _branch_one(models[policy], policy, pole, seed, prefix, step, device)
            for policy in POLICIES
        ]
        if not all(
            np.array_equal(pair[0]["obs"][k], pair[1]["obs"][k])
            for k in pair[0]["obs"]
        ):
            raise RuntimeError(
                f"matched branch state mismatch seed={seed} pole={pole} step={step}"
            )
        branches.append({
            "seed": seed,
            "pole": pole,
            "source_policy": source_policy,
            "step": step,
            "branch_index": branch_index,
            "cell": cell_name(cell),
            "regime": cell[1],
            "horizon": cell[2],
            "pair": pair,
        })

    arrays = {
        "plain_seed": np.asarray(
            [seed for _ in plain_records], dtype=np.int64
        ),
        "plain_step": np.asarray([r["step"] for r in plain_records], dtype=np.int32),
        "plain_policy": np.asarray(
            [0 if r["policy"] == "pi_A" else 1 for r in plain_records], dtype=np.int8
        ),
        "plain_pole": np.asarray(
            [0 if r["pole"] == "A" else 1 for r in plain_records], dtype=np.int8
        ),
        "branch_seed": np.asarray([seed] * len(branches), dtype=np.int64),
        "branch_step": np.asarray([b["step"] for b in branches], dtype=np.int32),
        "branch_index": np.asarray(
            [b["branch_index"] for b in branches], dtype=np.int8
        ),
        "branch_pole": np.asarray(
            [0 if b["pole"] == "A" else 1 for b in branches], dtype=np.int8
        ),
        "branch_source_policy": np.asarray(
            [0 if b["source_policy"] == "pi_A" else 1 for b in branches], dtype=np.int8
        ),
        "branch_regime": np.asarray([b["regime"] for b in branches], dtype=np.int8),
        "branch_is_late": np.asarray(
            [b["horizon"] == "late" for b in branches], dtype=np.int8
        ),
        "branch_cell": np.asarray([b["cell"] for b in branches], dtype="U24"),
    }
    _stack(arrays, "plain", plain_records)
    for key in branches[0]["pair"][0]["obs"]:
        arrays[f"branch_obs_{key}"] = np.stack(
            [b["pair"][0]["obs"][key] for b in branches]
        )
    for policy_index, policy in enumerate(POLICIES):
        items = [b["pair"][policy_index] for b in branches]
        arrays[f"branch_{policy}_action"] = np.stack([i["action"] for i in items])
        arrays[f"branch_{policy}_return"] = np.stack([i["return"] for i in items])
        arrays[f"branch_{policy}_blue"] = np.asarray(
            [i["blue"] for i in items], dtype=np.int16
        )
        arrays[f"branch_{policy}_red"] = np.asarray(
            [i["red"] for i in items], dtype=np.int16
        )
    return summaries, arrays, [b["cell"] for b in branches], visited, [
        b["source_policy"] for b in branches
    ]


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="stop after N seeds; SMOKE USE ONLY, never for the frozen run",
    )
    args = parser.parse_args()

    spec = _preflight()
    amendment = spec["AMENDMENT_2_ALL_FOUR_TRAJECTORIES_ELIGIBLE"]
    contract = {
        "mode": "DRY_RUN" if args.dry_run else "STRATIFIED_COLLECTION",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_status": spec["status"],
        "amendment_2": True,
        "amendment_2_ruling": amendment["ruling"],
        "eligibility": "all four (policy, pole) trajectories",
        "source_balance_tie_break": True,
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
        "support_floor_scope": (
            "checked ONCE over the FULL block after collection; "
            "never per split, never during"
        ),
        "environment_semantics": "fresh rebuild per teacher per branch; reuse forbidden",
        "output": str(OUT.relative_to(ROOT)),
        "teacher_sha256": EXPECTED_TEACHER_SHA,
        "final_106_seeds_touched": False,
        "attempt_1_disposition": amendment["attempt_1_disposition"]["classification"],
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
            int(s)
            for s in json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "completed_seeds", []
            )
        ]
    if not set(completed).issubset(SEEDS):
        raise RuntimeError("REFUSING: manifest contains a seed outside the frozen block")

    # Mandatory: rebuild the global source-balance counter from completed shards
    # before any new selection so resume matches an uninterrupted run.
    source_counts = rebuild_source_counts_from_shards(completed, shards)
    print(
        json.dumps(
            {
                "source_balance_counter_rebuilt_from_shards": True,
                "completed_seeds_for_rebuild": len(completed),
                "source_counts": source_counts,
            },
            indent=2,
        ),
        flush=True,
    )

    todo = [s for s in SEEDS if s not in completed]
    if args.limit:
        todo = todo[: args.limit]
    for seed in todo:
        summaries, arrays, cells, visited, sources = _collect_seed(
            models, tagger, seed, args.device, source_counts
        )
        shard = shards / f"seed_{seed}.npz"
        with shard.with_suffix(".npz.tmp").open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(shard.with_suffix(".npz.tmp"), shard)
        _write_json_atomic(
            summaries_dir / f"seed_{seed}.json",
            {
                "episodes": summaries,
                "branch_cells": cells,
                "branch_source_policies": sources,
                "visited_cell_counts": visited,
                "split": _split_of(seed),
                "amendment_2": True,
            },
        )
        completed.append(seed)
        _write_json_atomic(
            manifest_path,
            {
                "record": "16-cell stratified regime collection",
                "protocol": (
                    "STRATIFIED_REGIME_COLLECTION_PROTOCOL_DESIGN.json, "
                    "frozen c318bcea + AMENDMENT_2"
                ),
                "amendment_2": True,
                "updated_utc": _now(),
                "semantics": "fresh rebuild per teacher per branch",
                "eligibility": "all four (policy, pole) trajectories",
                "completed_seeds": completed,
                "target_seeds": len(SEEDS),
                "branch_points_per_seed": BRANCH_POINTS_PER_SEED,
                "teacher_sha256": EXPECTED_TEACHER_SHA,
                "source_counts_after_seed": {
                    cell: dict(counts) for cell, counts in source_counts.items()
                },
            },
        )
        print(
            f"seed {seed} [{_split_of(seed)}] {len(cells)} branch pts "
            f"sources={sources} ({len(completed)}/{len(SEEDS)})",
            flush=True,
        )

    if len(completed) < len(SEEDS):
        print(f"\nPARTIAL: {len(completed)}/{len(SEEDS)}. No COMPLETE marker written.")
        return 0

    episode_path = OUT / "episode_summaries.jsonl"
    temporary = episode_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for seed in SEEDS:
            rows = json.loads(
                (summaries_dir / f"seed_{seed}.json").read_text(encoding="utf-8")
            )
            for row in rows["episodes"]:
                handle.write(json.dumps(row) + "\n")
    os.replace(temporary, episode_path)

    # Realized source composition per cell (report, not a gate).
    composition = rebuild_source_counts_from_shards(completed, shards)
    _write_json_atomic(COMPLETE, {
        "verdict": "COLLECTION_COMPLETE",
        "utc": _now(),
        "seed_block": [SEEDS[0], SEEDS[-1], len(SEEDS)],
        "completed_seeds": len(completed),
        "splits": {k: [min(v), max(v), len(v)] for k, v in SPLITS.items()},
        "semantics": "fresh rebuild per teacher per branch",
        "amendment_2": True,
        "eligibility": "all four (policy, pole) trajectories",
        "realized_source_composition_per_cell": composition,
        "support_floor_check": (
            "NOT PERFORMED HERE -- a separate one-shot audit scores the "
            "32-distinct-seed floor over all 16 cells"
        ),
        "final_106_seeds_touched": False,
    })
    print("\nCOLLECTION_COMPLETE written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
