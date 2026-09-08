"""Frozen terminal evaluation for EXP2 K=2 supervised compression.

This is an execution surface for EXP2_K2_LATENT_COMPRESSION_PROTOCOL.json.
It does not define or tune any scientific parameter. The default invocation is
contract-only; ``--launch`` spends the single 8300001..8300192 block.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.opponent_spec import (  # noqa: E402
    assert_live_opponent_batch,
    install_keyed_opponent_overlays,
    pole_A_genome,
)
from rl.curriculum import phase_from_tag  # noqa: E402
from rl.custom_ppo.exp2_teacher_compression import decision_eligible_agents  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "EXP2_K2_LATENT_COMPRESSION_PROTOCOL.json"
PROTOCOL_ID = "EXP2_K2_LATENT_COMPRESSION_V1"
EXPERIMENT_ID = "EXP2_K2_LATENT_COMPRESSION"
SEED_BLOCK = "8300001..8300192"
DEVELOPMENT_SEED = 8_200_001
TRAIN_DIR = SD / "exp2_k2_latent_compression/exp2_k2_supervised_compression_seed8100001_2m"
STUDENT = TRAIN_DIR / "ckpts/final_exp2_k2_supervised_compression_seed8100001_2m.zip"
TEACHER_A = SD / "sappo_continuation/sappo_pi_A_specialist_1p5M_seed7100001/ckpts/final_sappo_pi_A_specialist_1p5M_seed7100001.zip"
TEACHER_B = SD / "sappo_continuation/sappo_pi_B_specialist_1p5M_seed7200001/ckpts/final_sappo_pi_B_specialist_1p5M_seed7200001.zip"
OUT = SD / "exp2_k2_terminal_evaluation"

EXPECTED_HASHES = {
    "student": "a4a34c8310b50abb779f7fff8f3921a272c3ec9e23e3cec4f4e96076a84cd44f",
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}
SEED_BASE = 8_300_001
N_PAIRED = 192
N_BOOT = 20_000
ALPHA = 0.05
BOOTSTRAP_SEED = 7
MAX_STEPS = 240
MAP = "map_a_open"
POLES = {"A": "OP6", "B": "OP7"}
POLICY_CELLS = ("z0", "z1", "pi_A", "pi_B")
RULESET = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_protocol() -> dict[str, Any]:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"wrong {EXPERIMENT_ID} protocol")
    terminal = payload["terminal_evaluation"]
    bootstrap = terminal["bootstrap"]
    if terminal["episodes_per_cell"] != N_PAIRED:
        raise RuntimeError("terminal episode count drift")
    if bootstrap != {
        "procedure": "paired percentile bootstrap over seed-level episode outcomes",
        "samples": N_BOOT,
        "alpha": ALPHA,
        "rng_seed": BOOTSTRAP_SEED,
    }:
        raise RuntimeError("bootstrap contract drift")
    if payload["seed_blocks"]["evaluation"]["range"] != SEED_BLOCK:
        raise RuntimeError("evaluation seed block drift")
    return payload


def guard_rails(*, launch: bool) -> dict[str, Any]:
    protocol = _load_protocol()
    paths = {"student": STUDENT, "pi_A": TEACHER_A, "pi_B": TEACHER_B}
    hashes = {}
    for name, path in paths.items():
        if not path.is_file() or not path.name.startswith("final_"):
            raise RuntimeError(f"missing/non-terminal checkpoint for {name}: {path}")
        hashes[name] = _sha256(path)
        if hashes[name] != EXPECTED_HASHES[name]:
            raise RuntimeError(f"checkpoint hash mismatch for {name}")
    result_files = (OUT / "summary.json", OUT / "episode_rows.csv", OUT / "action_identity_by_seed.csv")
    if launch and any(path.exists() for path in result_files):
        raise RuntimeError(f"{EXPERIMENT_ID} evaluation output already exists; the frozen block is spend-once")
    return {"protocol": protocol, "checkpoint_hashes": hashes}


def build_env(device: str, seed: int):
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_set="train",
        map_layout=MAP,
        max_decision_steps=MAX_STEPS,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=int(seed),
        tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True,
        **RULESET,
    )
    return GPUCTFVecEnv(cfg)


def _paired_mean_ci(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(values.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _ratio_ci(numer: np.ndarray, denom: np.ndarray) -> dict[str, Any]:
    numer = np.asarray(numer, dtype=np.float64)
    denom = np.asarray(denom, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(numer), size=(N_BOOT, len(numer)))
    bn, bd = numer[idx].mean(axis=1), denom[idx].mean(axis=1)
    if float(denom.mean()) <= 0 or np.any(bd <= 0):
        raise RuntimeError("retention denominator is zero in point or bootstrap sample")
    boot = bn / bd
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {
        "student_matched_value": float(numer.mean()),
        "teacher_matched_value": float(denom.mean()),
        "rho": float(numer.mean() / denom.mean()),
        "lcb95": float(lo),
        "ucb95": float(hi),
    }


def _tensor_obs(policy, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    batched = policy._batched_obs(obs)
    return policy._tensor_obs(batched)


def _dist_probs(policy, obs: dict[str, np.ndarray], z: int | None) -> list[torch.Tensor]:
    obs_t = _tensor_obs(policy, obs)
    z_idx = None if z is None else torch.tensor([z], dtype=torch.long, device=policy.device)
    with torch.no_grad():
        return [p.detach() for p in policy.get_distribution(obs_t, z_idx=z_idx).probabilities()]


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return (p * (torch.log2(p.clamp_min(eps)) - torch.log2(q.clamp_min(eps)))).sum(-1)


def action_identity(policy_map, obs: dict[str, np.ndarray]) -> dict[str, float]:
    probs = {
        "z0": _dist_probs(policy_map["student"], obs, 0),
        "z1": _dist_probs(policy_map["student"], obs, 1),
        "pi_A": _dist_probs(policy_map["pi_A"], obs, None),
        "pi_B": _dist_probs(policy_map["pi_B"], obs, None),
    }
    mask_t = torch.as_tensor(obs["mask"], dtype=torch.float32).reshape(1, -1)
    alive_t = torch.as_tensor(obs["agent_mask"], dtype=torch.float32).reshape(1, -1)
    eligible = decision_eligible_agents(
        mask_t, action_dims=(5, 50, 5, 50), n_agents=2, agent_mask=alive_t,
    )[0]
    jsd, margin_a, margin_b, disagree = [], [], [], []
    for head in range(4):
        agent = head // 2
        if not bool(eligible[agent]):
            continue
        p0, p1 = probs["z0"][head], probs["z1"][head]
        pa, pb = probs["pi_A"][head], probs["pi_B"][head]
        mix = 0.5 * (p0 + p1)
        jsd.append(0.5 * (_kl(p0, mix) + _kl(p1, mix)))
        margin_a.append(_kl(pa, p1) - _kl(pa, p0))
        margin_b.append(_kl(pb, p0) - _kl(pb, p1))
        disagree.append((p0.argmax(-1) != p1.argmax(-1)).float())
    if not jsd:
        return {"count": 0.0, "jsd_bits": 0.0, "margin_A_bits": 0.0, "margin_B_bits": 0.0, "argmax_disagree": 0.0}
    return {
        "count": float(len(jsd)),
        "jsd_bits": float(torch.stack(jsd).sum().item()),
        "margin_A_bits": float(torch.stack(margin_a).sum().item()),
        "margin_B_bits": float(torch.stack(margin_b).sum().item()),
        "argmax_disagree": float(torch.stack(disagree).sum().item()),
    }


def _np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def trajectory_tick(core, state: dict[str, Any], step: int) -> None:
    bx, by = _np(core.blue_x[0]), _np(core.blue_y[0])
    rx, ry = _np(core.red_x[0]), _np(core.red_y[0])
    alive = _np(core.blue_alive[0] & (~core.blue_tagged[0])).astype(bool)
    red_alive = _np(core.red_alive[0] & (~core.red_tagged[0])).astype(bool)
    carrying = _np(core.blue_carrying[0]).astype(bool)
    bhome = _np(core.blue_flag_home[0])
    rhome = _np(core.red_flag_home[0])
    midpoint = 0.5 * (float(bhome[0]) + float(rhome[0]))
    enemy = alive & (bx > midpoint if rhome[0] > bhome[0] else bx < midpoint)
    noncarrier = alive & (~carrying)
    home_dist = np.sqrt((bx - bhome[0]) ** 2 + (by - bhome[1]) ** 2)

    state["ticks"] += 1
    state["home_defense"] += float(np.any(noncarrier & (home_dist <= 8.0)))
    state["simultaneous_attackers"] += float(enemy.sum() >= 2)
    state["enemy_half_occupancy"] += float(enemy.sum() / max(1, alive.sum()))
    if state["commitment_timing"] is None and np.any(enemy):
        state["commitment_timing"] = step
        state["commitment_occurred"] = 1
    state["carrier_ticks"] += float(np.any(carrying))

    red_flag = _np(core.red_flag_pos[0])
    dist_to_flag = np.sqrt((bx - red_flag[0]) ** 2 + (by - red_flag[1]) ** 2)
    if state["first_flag_contact"] is None and np.any(alive & (dist_to_flag <= 1.5)):
        state["first_flag_contact"] = step
    if state["first_pickup"] is None and np.any(carrying):
        state["first_pickup"] = step
    if np.any(carrying):
        ci = int(np.argmax(carrying))
        others = np.flatnonzero(noncarrier)
        if len(others) and red_alive.any():
            carrier = np.array([bx[ci], by[ci]])
            reds = np.stack([rx[red_alive], ry[red_alive]], axis=1)
            nearest = reds[np.argmin(np.linalg.norm(reds - carrier, axis=1))]
            segment = carrier - nearest
            denom = float(np.dot(segment, segment))
            screened = False
            for oi in others:
                point = np.array([bx[oi], by[oi]])
                u = 0.0 if denom == 0 else float(np.clip(np.dot(point - nearest, segment) / denom, 0, 1))
                distance = float(np.linalg.norm(point - (nearest + u * segment)))
                screened |= distance <= 3.0
            state["interception_eligible"] += 1
            state["interception"] += float(screened)
            state["screen_eligible"] += 1
            state["screen"] += float(screened)

    score = int(core.blue_score[0].item())
    if score > state["last_score"]:
        state["captures"] += score - state["last_score"]
        if state["first_pickup"] is not None and state["pickup_to_capture"] is None:
            state["pickup_to_capture"] = step - state["first_pickup"]
    state["last_score"] = score


def trajectory_finalize(state: dict[str, Any], steps: int) -> dict[str, Any]:
    ticks = max(1, state["ticks"])
    return {
        "home_defense_fraction": state["home_defense"] / ticks,
        "simultaneous_attackers": state["simultaneous_attackers"] / ticks,
        "enemy_half_occupancy": state["enemy_half_occupancy"] / ticks,
        "commitment_timing": state["commitment_timing"] if state["commitment_timing"] is not None else steps,
        "commitment_occurred": state["commitment_occurred"],
        "interception_behavior": state["interception"] / max(1, state["interception_eligible"]),
        "interception_eligible_ticks": state["interception_eligible"],
        "first_enemy_flag_contact": state["first_flag_contact"] if state["first_flag_contact"] is not None else steps,
        "first_pickup": state["first_pickup"] if state["first_pickup"] is not None else steps,
        "pickup_occurred": int(state["first_pickup"] is not None),
        "pickup_to_capture": state["pickup_to_capture"] if state["pickup_to_capture"] is not None else steps,
        "carrier_fraction": state["carrier_ticks"] / ticks,
        "capture_count": state["captures"],
        "noncarrier_screen_fraction": state["screen"] / max(1, state["screen_eligible"]),
        "screen_eligible_ticks": state["screen_eligible"],
    }


def run_episode(policy_name: str, pole: str, seed: int, policy_map, device: str):
    env = build_env(device, seed)
    core = env.core
    base_key = POLES[pole]
    genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
    policy = policy_map["student"] if policy_name.startswith("z") else policy_map[policy_name]
    if policy_name.startswith("z"):
        policy.fixed_latent_strategy = True
        policy.fixed_latent_strategy_id = int(policy_name[-1])
    policy.reset_strategy()
    state = defaultdict(float)
    state.update({"commitment_timing": None, "first_flag_contact": None, "first_pickup": None,
                  "pickup_to_capture": None, "last_score": 0, "commitment_occurred": 0})
    action_acc = defaultdict(float)
    try:
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        install_keyed_opponent_overlays(core, genomes)
        env.env_method("set_phase", phase_from_tag(base_key))
        env.env_method("set_next_opponent", "SCRIPTED", base_key)
        obs = env.reset()
        obs["global_state"] = env.state()
        assert_live_opponent_batch(core, genomes, allowed_keys=(base_key,), context=f"EXP2 {policy_name}|{pole} seed {seed}")
        term = None
        steps = 0
        for step in range(MAX_STEPS):
            trajectory_tick(core, state, step)
            ai = action_identity(policy_map, obs)
            for key, value in ai.items():
                action_acc[key] += value
            action, _ = policy.predict(obs, deterministic=True)
            env.step_async(action)
            obs, _reward, done, info = env.step_wait()
            obs["global_state"] = env.state()
            steps += 1
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                result = (i0 or {}).get("episode_result") or {}
                term = (int(result.get("blue_score", 0)), int(result.get("red_score", 0)))
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        blue, red = term
        count = max(1.0, action_acc["count"])
        action_row = {key: action_acc[key] / count for key in ("jsd_bits", "margin_A_bits", "margin_B_bits", "argmax_disagree")}
        action_row["eligible_heads"] = action_acc["count"]
        row = {
            "policy": policy_name, "pole": pole, "opponent": base_key,
            "episode_seed": seed, "blue_score": blue, "red_score": red,
            "win": int(blue > red), "draw": int(blue == red), "steps": steps,
            **trajectory_finalize(state, steps),
        }
        return row, action_row
    finally:
        env.close()


def _wins(rows, policy: str, pole: str) -> np.ndarray:
    selected = sorted((r for r in rows if r["policy"] == policy and r["pole"] == pole), key=lambda r: r["episode_seed"])
    if len(selected) != N_PAIRED:
        raise RuntimeError(f"incomplete cell {policy}|{pole}: {len(selected)}")
    return np.asarray([r["win"] for r in selected], dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--development-smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.launch and args.development_smoke:
        raise RuntimeError("choose either --launch or --development-smoke")
    preflight = guard_rails(launch=args.launch)
    print(json.dumps({
        "mode": "LAUNCH" if args.launch else ("DEVELOPMENT_SMOKE" if args.development_smoke else "CONTRACT_ONLY"),
        "block": SEED_BLOCK,
        "cells": [f"{p}|{pole}" for p in POLICY_CELLS for pole in POLES],
        "episodes": len(POLICY_CELLS) * len(POLES) * N_PAIRED,
        "checkpoint_hashes": preflight["checkpoint_hashes"],
        "bootstrap": {"samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
    }, indent=2))
    if not args.launch and not args.development_smoke:
        print("CONTRACT ONLY. No environment constructed; evaluation block untouched.")
        return 0

    from rl.custom_ppo import load_custom_ppo_policy

    OUT.mkdir(parents=True, exist_ok=True)
    freeze = {
        "record": f"{EXPERIMENT_ID} terminal checkpoint locked before evaluation",
        "utc": _now(), "protocol": str(PROTOCOL.relative_to(ROOT)),
        "seed_block": SEED_BLOCK, "checkpoint_hashes": preflight["checkpoint_hashes"],
        "evaluator": str(Path(__file__).resolve().relative_to(ROOT)),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "terminal_checkpoint_only": True,
    }
    (OUT / "terminal_freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    probe = build_env(args.device, SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy_map = {
        "student": load_custom_ppo_policy(str(STUDENT), obs_space, act_space, device=args.device),
        "pi_A": load_custom_ppo_policy(str(TEACHER_A), obs_space, act_space, device=args.device),
        "pi_B": load_custom_ppo_policy(str(TEACHER_B), obs_space, act_space, device=args.device),
    }
    if not policy_map["student"].model.uses_latent_strategy or policy_map["student"].model.latent_k != 2:
        raise RuntimeError("terminal student is not K=2")
    if policy_map["student"].model.strategy_encoder is not None:
        raise RuntimeError("terminal student unexpectedly contains q_phi/router")

    if args.development_smoke:
        smoke_rows = []
        for policy_name in POLICY_CELLS:
            row, identity = run_episode(policy_name, "A", 8_200_001, policy_map, args.device)
            smoke_rows.append({"episode": row, "action_identity": identity})
        print(json.dumps({
            "verdict": "DEVELOPMENT_SMOKE_PASS",
            "seed": DEVELOPMENT_SEED,
            "cells": smoke_rows,
            "evaluation_block_untouched": True,
        }, indent=2))
        return 0

    rows, action_seed = [], defaultdict(lambda: defaultdict(float))
    for policy_name in POLICY_CELLS:
        for pole in POLES:
            print(f"scoring {policy_name}|{pole} ...", flush=True)
            for offset in range(N_PAIRED):
                seed = SEED_BASE + offset
                row, identity = run_episode(policy_name, pole, seed, policy_map, args.device)
                rows.append(row)
                acc = action_seed[seed]
                weight = identity["eligible_heads"]
                acc["eligible_heads"] += weight
                for key in ("jsd_bits", "margin_A_bits", "margin_B_bits", "argmax_disagree"):
                    acc[key] += identity[key] * weight
                if (offset + 1) % 24 == 0:
                    print(f"  {offset + 1}/{N_PAIRED}", flush=True)

    action_rows = []
    for seed in range(SEED_BASE, SEED_BASE + N_PAIRED):
        acc = action_seed[seed]
        denom = max(1.0, acc["eligible_heads"])
        action_rows.append({"episode_seed": seed, "eligible_heads": acc["eligible_heads"], **{
            key: acc[key] / denom for key in ("jsd_bits", "margin_A_bits", "margin_B_bits", "argmax_disagree")
        }})

    matrix = {f"{p}|{pole}": float(_wins(rows, p, pole).mean()) for p in POLICY_CELLS for pole in POLES}
    delta_a = _paired_mean_ci(_wins(rows, "z0", "A") - _wins(rows, "z1", "A"))
    delta_b = _paired_mean_ci(_wins(rows, "z1", "B") - _wins(rows, "z0", "B"))
    delta_a["passes"] = delta_a["lcb95"] > 0
    delta_b["passes"] = delta_b["lcb95"] > 0
    student_value = 0.5 * (_wins(rows, "z0", "A") + _wins(rows, "z1", "B"))
    teacher_value = 0.5 * (_wins(rows, "pi_A", "A") + _wins(rows, "pi_B", "B"))
    retention = _ratio_ci(student_value, teacher_value)
    retention["passes"] = retention["lcb95"] >= 0.90

    action_gate = {}
    for key in ("jsd_bits", "margin_A_bits", "margin_B_bits"):
        ci = _paired_mean_ci(np.asarray([row[key] for row in action_rows]))
        ci["passes"] = ci["lcb95"] > 0
        action_gate[key] = ci
    action_gate["argmax_disagree"] = _paired_mean_ci(np.asarray([row["argmax_disagree"] for row in action_rows]))
    action_pass = all(action_gate[key]["passes"] for key in ("jsd_bits", "margin_A_bits", "margin_B_bits"))

    trajectory = {}
    metric_names = [
        "home_defense_fraction", "simultaneous_attackers", "enemy_half_occupancy",
        "commitment_timing", "commitment_occurred", "interception_behavior",
        "first_enemy_flag_contact", "first_pickup", "pickup_occurred",
        "pickup_to_capture", "carrier_fraction", "capture_count", "noncarrier_screen_fraction",
    ]
    for policy_name in POLICY_CELLS:
        for pole in POLES:
            cell = [r for r in rows if r["policy"] == policy_name and r["pole"] == pole]
            trajectory[f"{policy_name}|{pole}"] = {
                metric: _paired_mean_ci(np.asarray([r[metric] for r in cell])) for metric in metric_names
            }

    overall_pass = bool(delta_a["passes"] and delta_b["passes"] and retention["passes"] and action_pass)
    verdict = f"{EXPERIMENT_ID}_CONFIRMED" if overall_pass else f"{EXPERIMENT_ID}_NOT_CONFIRMED"
    summary = {
        "record": f"{EXPERIMENT_ID} frozen terminal evaluation", "utc": _now(),
        "protocol": str(PROTOCOL.relative_to(ROOT)), "block": SEED_BLOCK,
        "n_paired": N_PAIRED, "total_episodes": len(rows),
        "checkpoint_hashes": preflight["checkpoint_hashes"], "payoff_matrix": matrix,
        "delta_A_z": delta_a, "delta_B_z": delta_b,
        "retention": retention,
        "behavioral_action_identity": {**action_gate, "passes": action_pass},
        "trajectory_telemetry": trajectory,
        "bootstrap": {"samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED,
                      "unit": "evaluation seed"},
        "verdict": verdict,
    }
    with (OUT / "episode_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with (OUT / "action_identity_by_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(action_rows[0]))
        writer.writeheader(); writer.writerows(action_rows)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"payoff_matrix": matrix, "delta_A_z": delta_a, "delta_B_z": delta_b,
                      "retention": retention, "behavioral_action_identity": action_gate,
                      "verdict": verdict}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
