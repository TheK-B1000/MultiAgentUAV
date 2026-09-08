"""Frozen full Phase 0 scorer collection using rebuild-per-branch semantics."""
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

FULL = P0.OUT / "full_collection_rebuild_per_branch"
EVIDENCE = P0.OUT / "first_interval_treatment_evidence.json"
REUSE = P0.SD / "PHASE0_ENV_REUSE_EQUIVALENCE.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _preflight() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if evidence["checks"].get("four_path_coverage_complete") is not True:
        raise RuntimeError("four-path evidence gate is not complete")
    if evidence["checks"].get("replay_to_state_reproduction_exact") is not True:
        raise RuntimeError("replay-to-state evidence is not exact")
    reuse = json.loads(REUSE.read_text(encoding="utf-8"))
    if reuse.get("verdict") != "FAIL_REJECT_REUSE" or reuse.get("reuse_authorized") is not False:
        raise RuntimeError("rebuild-per-branch launch requires frozen reuse rejection")
    if (FULL / "COLLECTION_COMPLETE.json").exists():
        raise RuntimeError("full Phase 0 collection already completed")


def _copy_obs(obs):
    return {k: np.asarray(v).copy() for k, v in obs.items()}


def _plain_episode(model, policy: str, pole: str, seed: int, device: str):
    env = R2.build_env(device, seed)
    core = env.core
    try:
        obs = P0._prep(env, core, pole, seed)
        prefix, decision_steps, records, rewards = [], [], [], []
        term, info = None, None
        for t in range(R2.MAX_STEPS):
            decision = bool((core.blue_commit_ticks_left[0] <= 0).any().item())
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action).reshape(-1).astype(np.int64)
            if decision:
                decision_steps.append(t)
                records.append({"step": t, "obs": _copy_obs(obs), "action": action.copy()})
            prefix.append(action.copy())
            env.step_async(action)
            obs, reward, done, info = env.step_wait()
            rewards.append(np.asarray(reward, dtype=np.float64).reshape(-1))
            if bool(np.asarray(done).any()):
                term = P0._terminal(core, info)
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        reward_matrix = np.stack(rewards)
        suffix = np.flip(np.cumsum(np.flip(reward_matrix, axis=0), axis=0), axis=0)
        for rec in records:
            rec["return"] = suffix[rec["step"]].copy()
        blue, red = term
        summary = {"seed": seed, "split": P0.split_for(seed), "policy": policy,
                   "pole": pole, "blue": blue, "red": red,
                   "win": int(blue > red), "margin": blue - red,
                   "steps": len(rewards), "decision_records": len(records)}
        return summary, records, prefix, decision_steps
    finally:
        env.close()


def _branch_one(model, policy: str, pole: str, seed: int, prefix, t: int, device: str):
    """Authoritative path: construct a fresh seeded env for this one branch."""
    env = R2.build_env(device, seed)
    core = env.core
    try:
        obs = P0._prep(env, core, pole, seed)
        for i in range(t):
            env.step_async(prefix[i])
            obs, _r, done, _info = env.step_wait()
            if bool(np.asarray(done).any()):
                raise RuntimeError(f"episode ended before branch seed={seed} pole={pole} t={t}")
        restored = _copy_obs(obs)
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action).reshape(-1).astype(np.int64)
        rewards, term = [], None
        for _ in range(R2.MAX_STEPS - t):
            env.step_async(action)
            obs, reward, done, info = env.step_wait()
            rewards.append(np.asarray(reward, dtype=np.float64).reshape(-1))
            if bool(np.asarray(done).any()):
                term = P0._terminal(core, info)
                break
            action_next, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action_next).reshape(-1).astype(np.int64)
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        blue, red = term
        return {"policy": policy, "obs": restored, "action": np.asarray(
                    model.predict(restored, deterministic=True)[0]
                ).reshape(-1).astype(np.int64),
                "return": np.stack(rewards).sum(axis=0), "blue": blue, "red": red,
                "win": int(blue > red), "margin": blue - red}
    finally:
        env.close()


def _stack_records(records, prefix: str, arrays: dict):
    if not records:
        return
    for key in records[0]["obs"]:
        arrays[f"{prefix}_obs_{key}"] = np.stack([r["obs"][key] for r in records])
    arrays[f"{prefix}_action"] = np.stack([r["action"] for r in records])
    arrays[f"{prefix}_return"] = np.stack([r["return"] for r in records])


def _collect_seed(models, seed: int, device: str):
    summaries, plain_records, sources = [], [], {}
    for policy in ("pi_A", "pi_B"):
        for pole in ("A", "B"):
            summary, records, prefix, decisions = _plain_episode(
                models[policy], policy, pole, seed, device
            )
            summaries.append(summary)
            for rec in records:
                rec.update({"policy": policy, "pole": pole})
            plain_records.extend(records)
            if policy == P0.source_policy_for(seed, pole):
                sources[pole] = (policy, prefix, decisions, summary["steps"])

    branch_records = []
    for pole in ("A", "B"):
        source, prefix, decisions, horizon = sources[pole]
        points, notes = P0.select_tertile_points(decisions, horizon)
        for branch_index, t in enumerate(points):
            pair = [
                _branch_one(models[policy], policy, pole, seed, prefix, t, device)
                for policy in ("pi_A", "pi_B")
            ]
            if not all(np.array_equal(pair[0]["obs"][k], pair[1]["obs"][k])
                       for k in pair[0]["obs"]):
                raise RuntimeError(f"matched branch state mismatch seed={seed} pole={pole} t={t}")
            branch_records.append({"seed": seed, "split": P0.split_for(seed),
                                   "pole": pole, "source_policy": source,
                                   "branch_index": branch_index, "step": t,
                                   "notes": notes, "pair": pair})

    arrays = {
        "plain_seed": np.asarray([seed] * len(plain_records), dtype=np.int64),
        "plain_step": np.asarray([r["step"] for r in plain_records], dtype=np.int32),
        "plain_policy": np.asarray([0 if r["policy"] == "pi_A" else 1 for r in plain_records], dtype=np.int8),
        "plain_pole": np.asarray([0 if r["pole"] == "A" else 1 for r in plain_records], dtype=np.int8),
        "branch_step": np.asarray([r["step"] for r in branch_records], dtype=np.int32),
        "branch_index": np.asarray([r["branch_index"] for r in branch_records], dtype=np.int8),
        "branch_pole": np.asarray([0 if r["pole"] == "A" else 1 for r in branch_records], dtype=np.int8),
        "branch_source_policy": np.asarray([0 if r["source_policy"] == "pi_A" else 1 for r in branch_records], dtype=np.int8),
    }
    _stack_records(plain_records, "plain", arrays)
    for key in branch_records[0]["pair"][0]["obs"]:
        arrays[f"branch_obs_{key}"] = np.stack([r["pair"][0]["obs"][key] for r in branch_records])
    for policy_index, policy in enumerate(("pi_A", "pi_B")):
        items = [r["pair"][policy_index] for r in branch_records]
        arrays[f"branch_{policy}_action"] = np.stack([x["action"] for x in items])
        arrays[f"branch_{policy}_return"] = np.stack([x["return"] for x in items])
        arrays[f"branch_{policy}_blue"] = np.asarray([x["blue"] for x in items], dtype=np.int16)
        arrays[f"branch_{policy}_red"] = np.asarray([x["red"] for x in items], dtype=np.int16)
    return summaries, branch_records, arrays


def _write_json_atomic(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--launch", action="store_true")
    args = ap.parse_args()
    _preflight()
    print(json.dumps({"mode": "LAUNCH" if args.launch else "CONTRACT_ONLY",
                      "seeds": f"{P0.SEED_BASE}..{P0.SEED_BASE + P0.N_SEEDS - 1}",
                      "plain_episodes": 1024, "branch_points": 1536,
                      "branch_env_semantics": "fresh rebuild per teacher per branch"}, indent=2))
    if not args.launch:
        return 0

    from rl.custom_ppo import load_custom_ppo_policy
    FULL.mkdir(parents=True, exist_ok=True)
    shards = FULL / "seed_shards"
    shards.mkdir(exist_ok=True)
    summaries_dir = FULL / "seed_summaries"
    summaries_dir.mkdir(exist_ok=True)
    probe = R2.build_env(args.device, P0.SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    models = {name: load_custom_ppo_policy(str(path), obs_space, act_space, device=args.device)
              for name, path in P0.TEACHERS.items()}
    manifest_path = FULL / "collection_manifest.json"
    completed = []
    if manifest_path.exists():
        completed = json.loads(manifest_path.read_text(encoding="utf-8")).get("completed_seeds", [])
    for offset in range(P0.N_SEEDS):
        seed = P0.SEED_BASE + offset
        if seed in completed:
            continue
        summaries, branches, arrays = _collect_seed(models, seed, args.device)
        shard = shards / f"seed_{seed}.npz"
        with open(shard.with_suffix(".npz.tmp"), "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(shard.with_suffix(".npz.tmp"), shard)
        _write_json_atomic(summaries_dir / f"seed_{seed}.json", summaries)
        completed.append(seed)
        _write_json_atomic(manifest_path, {
            "record": "PHASE0 full scorer collection", "updated_utc": _now(),
            "semantics": "rebuild-per-branch", "completed_seeds": completed,
            "target_seeds": P0.N_SEEDS, "plain_episodes_per_seed": 4,
            "branch_points_per_seed": 6,
        })
        print(f"seed {seed} complete ({len(completed)}/{P0.N_SEEDS})", flush=True)
    episode_path = FULL / "episode_summaries.jsonl"
    with open(episode_path.with_suffix(".jsonl.tmp"), "w", encoding="utf-8") as handle:
        for seed in sorted(completed):
            for row in json.loads((summaries_dir / f"seed_{seed}.json").read_text(encoding="utf-8")):
                handle.write(json.dumps(row) + "\n")
    os.replace(episode_path.with_suffix(".jsonl.tmp"), episode_path)
    _write_json_atomic(FULL / "COLLECTION_COMPLETE.json", {
        "verdict": "COLLECTION_COMPLETE", "utc": _now(),
        "completed_seeds": len(completed), "plain_episodes": len(completed) * 4,
        "branch_points": len(completed) * 6,
        "semantics": "fresh rebuild per teacher per branch",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
