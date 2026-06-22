"""State-conditioned forced-z branch evaluation for v6i5 repertoires.

This is an evaluation-only diagnostic. It loads one checkpoint, collects
tactical state snapshots during matched driver rollouts, and from each exact
snapshot forces every latent z for a short horizon and optionally to terminal.

The main question is whether any latent beats the z3 generalist in specific
game states, not whether the router is ready.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.inference import (
    apply_deterministic_sampling_generators,
    load_custom_ppo_policy,
    read_custom_ppo_metadata,
)
from rl.training.env_factory import build_training_env
from tools.q_probe_local_counterfactual import (
    _env_opponent_tag,
    _restore_env,
    _restore_policy,
    _set_global_seeds,
    _single_obs,
    _snapshot_env,
    _snapshot_policy,
)


BUCKETS: tuple[str, ...] = (
    "neutral_opening",
    "team_carrying_enemy_flag",
    "enemy_carrying_team_flag",
    "leading_late",
    "trailing_late",
    "carrier_near_capture_zone",
    "high_enemy_pressure",
)

BRANCH_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "checkpoint_steps",
    "opponent",
    "probe_seed",
    "driver_mode",
    "driver_z",
    "bucket",
    "state_id",
    "snapshot_step",
    "start_blue_score",
    "start_red_score",
    "start_blue_carrier_count",
    "start_red_carrier_count",
    "forced_z",
    "short_horizon_steps",
    "short_return",
    "short_blue_score_delta",
    "short_red_score_delta",
    "short_score_delta",
    "short_blue_pickups",
    "short_red_pickups",
    "short_blue_drops",
    "short_red_drops",
    "terminal_return",
    "terminal_step",
    "terminal_blue_score",
    "terminal_red_score",
    "terminal_score_delta",
    "terminal_blue_won",
)

ADVANTAGE_FIELDS: tuple[str, ...] = (
    "opponent",
    "bucket",
    "forced_z",
    "state_count",
    "mean_short_return",
    "mean_short_score_delta",
    "mean_terminal_score_delta",
    "terminal_win_rate",
    "latent_advantage_vs_uniform",
    "latent_advantage_vs_z3",
    "is_bucket_best_z",
)

ORACLE_FIELDS: tuple[str, ...] = (
    "opponent",
    "bucket",
    "state_count",
    "best_z",
    "best_mean_terminal_score_delta",
    "z3_mean_terminal_score_delta",
    "oracle_minus_z3",
)

TARGET_BUCKETS: tuple[tuple[str, str], ...] = (
    ("OP6", "team_carrying_enemy_flag"),
    ("OP6", "enemy_carrying_team_flag"),
    ("OP5", "neutral_opening"),
    ("OP5", "enemy_carrying_team_flag"),
    ("OP7", "enemy_carrying_team_flag"),
)

TERMINAL_BRANCH_FIELDS: tuple[str, ...] = (
    "checkpoint",
    "opponent",
    "state_bucket",
    "snapshot_id",
    "snapshot_source_seed",
    "snapshot_episode_id",
    "snapshot_step",
    "paired_branch_seed",
    "forced_z",
    "initial_team_score",
    "initial_enemy_score",
    "final_team_score",
    "final_enemy_score",
    "final_score_differential",
    "score_change_from_snapshot",
    "branch_end_score_delta",
    "terminal_score_delta",
    "terminal_reward_or_return",
    "win",
    "loss",
    "draw",
    "terminated_naturally",
    "truncated_by_safety_cap",
    "branch_steps",
    "time_to_natural_termination",
    "team_flag_initially_carried",
    "enemy_flag_initially_carried",
    "team_flag_recovered",
    "enemy_flag_capture_completed",
    "own_flag_returned",
    "team_capture_count_after_branch",
    "enemy_capture_count_after_branch",
)

TERMINAL_PAIR_FIELDS: tuple[str, ...] = (
    "opponent",
    "state_bucket",
    "snapshot_id",
    "z0_outcome",
    "z3_outcome",
    "z0_final_score_differential",
    "z3_final_score_differential",
    "paired_terminal_return_advantage_z0_minus_z3",
    "paired_final_score_advantage_z0_minus_z3",
    "z0_wins_pair",
    "z3_wins_pair",
    "pair_tied",
    "z0_terminated_naturally",
    "z3_terminated_naturally",
    "pair_valid_for_terminal_analysis",
)

TERMINAL_SUMMARY_FIELDS: tuple[str, ...] = (
    "scope",
    "opponent",
    "state_bucket",
    "number_of_snapshots",
    "number_of_valid_terminal_pairs",
    "number_of_truncated_pairs",
    "truncated_pair_rate",
    "mean_paired_terminal_return_advantage",
    "median_paired_terminal_return_advantage",
    "mean_paired_final_score_advantage",
    "z0_better_pair_count",
    "z3_better_pair_count",
    "tie_count",
    "paired_win_rate_difference",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "per_seed_results",
)


def _to_np(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _score(core: Any) -> tuple[int, int]:
    blue = int(_to_np(getattr(core, "blue_score", np.zeros((1,), dtype=np.int64))).reshape(-1)[0])
    red = int(_to_np(getattr(core, "red_score", np.zeros((1,), dtype=np.int64))).reshape(-1)[0])
    return blue, red


def _carrier_counts(core: Any) -> tuple[int, int]:
    blue_c = int(_to_np(getattr(core, "blue_carrying", np.zeros((1, 1), dtype=np.int64)))[0].sum())
    red_c = int(_to_np(getattr(core, "red_carrying", np.zeros((1, 1), dtype=np.int64)))[0].sum())
    return blue_c, red_c


def _model_parameter_hash(model: Any) -> str:
    h = hashlib.sha256()
    inner = getattr(model, "model", model)
    state_dict = inner.state_dict()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(str(tensor.dtype).encode("ascii"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_info() -> dict[str, Any]:
    import subprocess

    root = str(_REPO_ROOT.parent)
    out: dict[str, Any] = {"commit": "unavailable", "dirty": "unavailable"}
    try:
        commit = subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "status", "--short"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        out = {"commit": commit, "dirty": bool(status.strip()), "status_short": status.splitlines()}
    except Exception:
        pass
    return out


def _diag(core: Any) -> float:
    val = float(getattr(core, "max_dist", 1.0) or 1.0)
    return val if val > 1e-6 else 1.0


def _nearest_red_to_blue_flag_frac(core: Any) -> float:
    red_alive = _to_np(getattr(core, "red_alive", np.zeros((1, 1), dtype=np.int64)))[0].astype(bool)
    if not red_alive.any():
        return 1.0
    rx = _to_np(getattr(core, "red_x"))[0][red_alive]
    ry = _to_np(getattr(core, "red_y"))[0][red_alive]
    blue_flag = _to_np(getattr(core, "blue_flag_pos"))[0]
    d = np.sqrt((rx - blue_flag[0]) ** 2 + (ry - blue_flag[1]) ** 2)
    return float(d.min() / _diag(core))


def _carrier_home_distance_frac(core: Any) -> float:
    diag = _diag(core)
    best = float("inf")
    blue_carry = _to_np(getattr(core, "blue_carrying", np.zeros((1, 1), dtype=np.int64)))[0].astype(bool)
    if blue_carry.any():
        bx = _to_np(getattr(core, "blue_x"))[0][blue_carry]
        by = _to_np(getattr(core, "blue_y"))[0][blue_carry]
        blue_flag = _to_np(getattr(core, "blue_flag_pos"))[0]
        best = min(best, float(np.sqrt((bx - blue_flag[0]) ** 2 + (by - blue_flag[1]) ** 2).min() / diag))
    red_carry = _to_np(getattr(core, "red_carrying", np.zeros((1, 1), dtype=np.int64)))[0].astype(bool)
    if red_carry.any():
        rx = _to_np(getattr(core, "red_x"))[0][red_carry]
        ry = _to_np(getattr(core, "red_y"))[0][red_carry]
        red_flag = _to_np(getattr(core, "red_flag_pos"))[0]
        best = min(best, float(np.sqrt((rx - red_flag[0]) ** 2 + (ry - red_flag[1]) ** 2).min() / diag))
    return best if math.isfinite(best) else 1.0


def state_bucket_labels(core: Any, *, step: int, max_steps: int) -> list[str]:
    """Return tactical state buckets for the current env core state."""
    blue_score, red_score = _score(core)
    blue_carry, red_carry = _carrier_counts(core)
    labels: list[str] = []
    if step < max(32, int(0.15 * max_steps)) and blue_score == red_score and blue_carry == 0 and red_carry == 0:
        labels.append("neutral_opening")
    if blue_carry > 0:
        labels.append("team_carrying_enemy_flag")
    if red_carry > 0:
        labels.append("enemy_carrying_team_flag")
    if step >= int(0.70 * max_steps) and blue_score > red_score:
        labels.append("leading_late")
    if step >= int(0.70 * max_steps) and blue_score < red_score:
        labels.append("trailing_late")
    if _carrier_home_distance_frac(core) <= 0.20:
        labels.append("carrier_near_capture_zone")
    if red_carry > 0 or _nearest_red_to_blue_flag_frac(core) <= 0.25:
        labels.append("high_enemy_pressure")
    return [b for b in BUCKETS if b in labels]


def _target_bucket_pairs(values: list[str] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return TARGET_BUCKETS
    if len(values) % 2 != 0:
        raise ValueError("--target-buckets expects OP BUCKET pairs")
    allowed = set(TARGET_BUCKETS)
    pairs: list[tuple[str, str]] = []
    for i in range(0, len(values), 2):
        pair = (values[i].upper(), values[i + 1])
        if pair not in allowed:
            raise ValueError(f"unsupported target bucket {pair[0]} {pair[1]}")
        pairs.append(pair)
    return tuple(pairs)


def _copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            out[key] = val.copy()
        elif isinstance(val, torch.Tensor):
            out[key] = val.detach().clone()
        else:
            out[key] = val
    return out


@dataclass
class BranchOutcome:
    short_return: float
    short_blue_score_delta: int
    short_red_score_delta: int
    short_blue_pickups: int
    short_red_pickups: int
    short_blue_drops: int
    short_red_drops: int
    terminal_return: float
    terminal_step: int
    terminal_blue_score: int
    terminal_red_score: int
    branch_steps: int = 0
    terminated_naturally: bool = False
    truncated_by_safety_cap: bool = False
    team_recovered: bool = False
    enemy_capture_completed: bool = False
    own_flag_returned: bool = False


def _roll_branch(
    *,
    env: Any,
    model: Any,
    obs: dict[str, Any],
    forced_z: int,
    starting_step: int,
    short_horizon: int,
    max_steps: int,
    terminal: bool,
) -> BranchOutcome:
    model.fixed_latent_strategy = True
    model.fixed_latent_strategy_id = int(forced_z)
    core = env.core
    start_blue, start_red = _score(core)
    prev_blue_c, prev_red_c = _carrier_counts(core)
    short_return = 0.0
    terminal_return = 0.0
    short_blue_pickups = 0
    short_red_pickups = 0
    short_blue_drops = 0
    short_red_drops = 0
    terminal_blue = start_blue
    terminal_red = start_red
    terminal_step = starting_step
    cur_obs = _copy_obs(obs)
    limit = max_steps if terminal else min(max_steps, starting_step + short_horizon)
    for step in range(starting_step, limit):
        action, _ = model.predict(_single_obs(cur_obs, env), deterministic=False)
        env.step_async(action)
        cur_obs, rewards, dones, infos = env.step_wait()
        reward = float(np.asarray(rewards).reshape(-1)[0])
        terminal_return += reward
        if step < starting_step + short_horizon:
            short_return += reward
        blue_c, red_c = _carrier_counts(env.core)
        if step < starting_step + short_horizon:
            short_blue_pickups += int(blue_c > 0 and prev_blue_c == 0)
            short_red_pickups += int(red_c > 0 and prev_red_c == 0)
            short_blue_drops += int(blue_c == 0 and prev_blue_c > 0)
            short_red_drops += int(red_c == 0 and prev_red_c > 0)
        prev_blue_c, prev_red_c = blue_c, red_c
        terminal_step = step + 1
        terminal_blue, terminal_red = _score(env.core)
        if bool(np.asarray(dones).reshape(-1)[0]):
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info) or {}
            terminal_blue = int(ep_res.get("blue_score", terminal_blue))
            terminal_red = int(ep_res.get("red_score", terminal_red))
            break
    blue_now, red_now = _score(env.core)
    return BranchOutcome(
        short_return=float(short_return),
        short_blue_score_delta=int(blue_now - start_blue),
        short_red_score_delta=int(red_now - start_red),
        short_blue_pickups=int(short_blue_pickups),
        short_red_pickups=int(short_red_pickups),
        short_blue_drops=int(short_blue_drops),
        short_red_drops=int(short_red_drops),
        terminal_return=float(terminal_return),
        terminal_step=int(terminal_step),
        terminal_blue_score=int(terminal_blue),
        terminal_red_score=int(terminal_red),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_branch_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_completed_combo_keys(path: Path) -> set[tuple[str, int, str, int]]:
    keys: set[tuple[str, int, str, int]] = set()
    for row in _read_branch_rows(path):
        try:
            keys.add(
                (
                    str(row["opponent"]).upper(),
                    int(row["probe_seed"]),
                    str(row["driver_mode"]),
                    int(row["driver_z"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return keys


def compute_advantage_rows(rows: list[dict[str, Any]], *, baseline_z: int = 3) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["opponent"]), str(row["bucket"]), int(row["forced_z"]))
        grouped.setdefault(key, []).append(row)
    by_bucket: dict[tuple[str, str], dict[int, dict[str, float]]] = {}
    for (opp, bucket, z), vals in grouped.items():
        n = len(vals)
        if n <= 0:
            continue
        summary = {
            "state_count": float(n),
            "mean_short_return": statistics.fmean(float(v["short_return"]) for v in vals),
            "mean_short_score_delta": statistics.fmean(float(v["short_score_delta"]) for v in vals),
            "mean_terminal_score_delta": statistics.fmean(float(v["terminal_score_delta"]) for v in vals),
            "terminal_win_rate": statistics.fmean(float(v["terminal_blue_won"]) for v in vals),
        }
        by_bucket.setdefault((opp, bucket), {})[z] = summary

    out: list[dict[str, Any]] = []
    for (opp, bucket), per_z in sorted(by_bucket.items()):
        if not per_z:
            continue
        mean_all = statistics.fmean(v["mean_terminal_score_delta"] for v in per_z.values())
        z3_val = per_z.get(int(baseline_z), {}).get("mean_terminal_score_delta", float("nan"))
        best_z = max(per_z, key=lambda z: per_z[z]["mean_terminal_score_delta"])
        for z, vals in sorted(per_z.items()):
            row = {
                "opponent": opp,
                "bucket": bucket,
                "forced_z": int(z),
                "state_count": int(vals["state_count"]),
                "mean_short_return": float(vals["mean_short_return"]),
                "mean_short_score_delta": float(vals["mean_short_score_delta"]),
                "mean_terminal_score_delta": float(vals["mean_terminal_score_delta"]),
                "terminal_win_rate": float(vals["terminal_win_rate"]),
                "latent_advantage_vs_uniform": float(vals["mean_terminal_score_delta"] - mean_all),
                "latent_advantage_vs_z3": float(vals["mean_terminal_score_delta"] - z3_val) if math.isfinite(z3_val) else float("nan"),
                "is_bucket_best_z": int(z == best_z),
            }
            out.append(row)
    return out


def compute_oracle_rows(advantage_rows: list[dict[str, Any]], *, baseline_z: int = 3) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in advantage_rows:
        grouped.setdefault((str(row["opponent"]), str(row["bucket"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (opp, bucket), vals in sorted(grouped.items()):
        best = max(vals, key=lambda r: float(r["mean_terminal_score_delta"]))
        base = next((r for r in vals if int(r["forced_z"]) == int(baseline_z)), None)
        z3_val = float(base["mean_terminal_score_delta"]) if base is not None else float("nan")
        out.append(
            {
                "opponent": opp,
                "bucket": bucket,
                "state_count": int(best["state_count"]),
                "best_z": int(best["forced_z"]),
                "best_mean_terminal_score_delta": float(best["mean_terminal_score_delta"]),
                "z3_mean_terminal_score_delta": z3_val,
                "oracle_minus_z3": float(best["mean_terminal_score_delta"] - z3_val) if math.isfinite(z3_val) else float("nan"),
            }
        )
    return out


def _write_report(path: Path, *, oracle_rows: list[dict[str, Any]], branch_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    useful = [r for r in oracle_rows if math.isfinite(float(r["oracle_minus_z3"])) and float(r["oracle_minus_z3"]) > 0.0 and int(r["best_z"]) != 3]
    lines = [
        "# v6i5 State-Conditioned Branch Evaluation",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- opponents: {', '.join(args.opponents)}",
        f"- seeds: {int(args.n_seeds)} from {int(args.base_seed)}",
        f"- driver mode: {args.driver_mode}",
        f"- branch rows: {len(branch_rows)}",
        f"- non-z3 useful bucket candidates: {len(useful)}",
        "",
        "## Oracle vs Always-z3",
        "",
        "| opponent | bucket | best_z | states | oracle_minus_z3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in oracle_rows:
        lines.append(
            f"| {row['opponent']} | {row['bucket']} | {row['best_z']} | "
            f"{row['state_count']} | {float(row['oracle_minus_z3']):+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_episode_collect_branches(
    *,
    env: Any,
    model: Any,
    checkpoint_path: str,
    checkpoint_steps: int,
    opponent_label: str,
    opponent_env_tag: str,
    probe_seed: int,
    device: str,
    latent_k: int,
    driver_mode: str,
    driver_z: int,
    snapshots_per_bucket: int,
    min_snapshot_gap: int,
    short_horizon: int,
    max_steps: int,
    terminal: bool,
) -> list[dict[str, Any]]:
    _set_global_seeds(probe_seed)
    try:
        env.seed(int(probe_seed))
    except Exception:
        pass
    env.env_method("set_next_opponent", "SCRIPTED", opponent_env_tag)
    model.reset_strategy()
    apply_deterministic_sampling_generators(model.model, int(probe_seed), device=device)
    model.fixed_latent_strategy = driver_mode == "fixed_z"
    model.fixed_latent_strategy_id = int(driver_z)
    obs = env.reset()
    seen: dict[str, int] = {bucket: 0 for bucket in BUCKETS}
    last_step: dict[str, int] = {bucket: -10_000 for bucket in BUCKETS}
    rows: list[dict[str, Any]] = []

    for step in range(max_steps):
        labels = state_bucket_labels(env.core, step=step, max_steps=max_steps)
        for bucket in labels:
            if seen[bucket] >= snapshots_per_bucket:
                continue
            if step - last_step[bucket] < min_snapshot_gap:
                continue
            env_snap = _snapshot_env(env)
            pol_snap = _snapshot_policy(model)
            obs_snap = _copy_obs(obs)
            start_blue, start_red = _score(env.core)
            start_blue_c, start_red_c = _carrier_counts(env.core)
            state_id = f"{opponent_label}|{probe_seed}|{driver_mode}|{driver_z}|{bucket}|{seen[bucket]}"
            for z in range(int(latent_k)):
                _restore_env(env, env_snap)
                _restore_policy(model, pol_snap)
                outcome = _roll_branch(
                    env=env,
                    model=model,
                    obs=obs_snap,
                    forced_z=z,
                    starting_step=step,
                    short_horizon=short_horizon,
                    max_steps=max_steps,
                    terminal=terminal,
                )
                rows.append(
                    {
                        "checkpoint_path": checkpoint_path,
                        "checkpoint_steps": int(checkpoint_steps),
                        "opponent": str(opponent_label).upper(),
                        "probe_seed": int(probe_seed),
                        "driver_mode": driver_mode,
                        "driver_z": int(driver_z),
                        "bucket": bucket,
                        "state_id": state_id,
                        "snapshot_step": int(step),
                        "start_blue_score": int(start_blue),
                        "start_red_score": int(start_red),
                        "start_blue_carrier_count": int(start_blue_c),
                        "start_red_carrier_count": int(start_red_c),
                        "forced_z": int(z),
                        "short_horizon_steps": int(short_horizon),
                        "short_return": float(outcome.short_return),
                        "short_blue_score_delta": int(outcome.short_blue_score_delta),
                        "short_red_score_delta": int(outcome.short_red_score_delta),
                        "short_score_delta": int(outcome.short_blue_score_delta - outcome.short_red_score_delta),
                        "short_blue_pickups": int(outcome.short_blue_pickups),
                        "short_red_pickups": int(outcome.short_red_pickups),
                        "short_blue_drops": int(outcome.short_blue_drops),
                        "short_red_drops": int(outcome.short_red_drops),
                        "terminal_return": float(outcome.terminal_return),
                        "terminal_step": int(outcome.terminal_step),
                        "terminal_blue_score": int(outcome.terminal_blue_score),
                        "terminal_red_score": int(outcome.terminal_red_score),
                        "terminal_score_delta": int((outcome.terminal_blue_score - start_blue) - (outcome.terminal_red_score - start_red)),
                        "terminal_blue_won": int(outcome.terminal_blue_score > outcome.terminal_red_score),
                    }
                )
            _restore_env(env, env_snap)
            _restore_policy(model, pol_snap)
            obs = obs_snap
            seen[bucket] += 1
            last_step[bucket] = step

        if all(v >= snapshots_per_bucket for v in seen.values()):
            break
        action, _ = model.predict(_single_obs(obs, env), deterministic=False)
        env.step_async(action)
        obs, _rewards, dones, _infos = env.step_wait()
        if bool(np.asarray(dones).reshape(-1)[0]):
            break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--opponents", nargs="+", default=["OP5", "OP6", "OP7", "OP4"])
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--driver-mode", choices=["fixed_z", "natural"], default="fixed_z")
    parser.add_argument("--driver-z-values", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--snapshots-per-bucket", type=int, default=4)
    parser.add_argument("--min-snapshot-gap", type=int, default=32)
    parser.add_argument("--short-horizon", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--no-terminal", action="store_true", help="Only roll branches for the short horizon.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent / "state_conditioned_branch_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt.stem
    branch_csv = out_dir / f"{stem}_state_branch_rows.csv"
    advantage_csv = out_dir / f"{stem}_state_branch_advantages.csv"
    oracle_csv = out_dir / f"{stem}_state_branch_oracle_vs_z3.csv"
    manifest_json = out_dir / f"{stem}_state_branch_manifest.json"
    report_md = out_dir / f"{stem}_state_branch_report.md"

    meta = read_custom_ppo_metadata(str(ckpt))
    cfg_meta = meta.get("cfg") or {}
    latent_k = int(meta.get("latent_k", 4))
    ckpt_steps = int(meta.get("global_step", -1) or -1)
    n_blue = int(meta.get("n_blue", args.agents))
    cfg = PPOConfig()
    cfg.use_latent_strategy = True
    cfg.latent_k = latent_k
    cfg.n_envs = 1
    cfg.seed = int(args.base_seed)
    cfg.device = str(args.device)
    cfg.max_blue_agents = n_blue
    cfg.n_agents_per_team = n_blue
    cfg.opponent_randomize = False
    cfg.fixed_opponent_tag = _env_opponent_tag(args.opponents[0])
    cfg.map_layout = str(cfg_meta.get("map_layout", cfg.map_layout))
    cfg.map_set = str(cfg_meta.get("map_set", cfg.map_set))
    cfg.max_decision_steps = int(cfg_meta.get("max_decision_steps", args.max_steps) or args.max_steps)

    env = build_training_env(cfg, initial_phase="PHASE1", initial_opponent_tag=_env_opponent_tag(args.opponents[0]))
    rows: list[dict[str, Any]] = _read_branch_rows(branch_csv)
    completed_keys = _read_completed_combo_keys(branch_csv)
    if rows:
        print(f"[state_branch] resume: loaded {len(rows)} existing branch rows from {branch_csv}", flush=True)
    try:
        model = load_custom_ppo_policy(str(ckpt), env.observation_space, env.action_space, device=str(args.device))
        total = len(args.opponents) * int(args.n_seeds)
        if args.driver_mode == "fixed_z":
            total *= len(args.driver_z_values)
        combo = 0
        for opp in args.opponents:
            opp_tag = _env_opponent_tag(opp)
            for seed_offset in range(int(args.n_seeds)):
                seed = int(args.base_seed) + seed_offset
                driver_zs = list(args.driver_z_values) if args.driver_mode == "fixed_z" else [-1]
                for driver_z in driver_zs:
                    combo += 1
                    combo_key = (str(opp).upper(), int(seed), str(args.driver_mode), int(driver_z))
                    if combo_key in completed_keys:
                        print(f"[state_branch] skip existing {opp} seed={seed} driver={args.driver_mode}:{driver_z} ({combo}/{total})", flush=True)
                        continue
                    start = time.time()
                    ep_rows = _run_episode_collect_branches(
                        env=env,
                        model=model,
                        checkpoint_path=str(ckpt),
                        checkpoint_steps=ckpt_steps,
                        opponent_label=str(opp).upper(),
                        opponent_env_tag=opp_tag,
                        probe_seed=seed,
                        device=str(args.device),
                        latent_k=latent_k,
                        driver_mode=str(args.driver_mode),
                        driver_z=int(driver_z),
                        snapshots_per_bucket=int(args.snapshots_per_bucket),
                        min_snapshot_gap=int(args.min_snapshot_gap),
                        short_horizon=int(args.short_horizon),
                        max_steps=int(args.max_steps),
                        terminal=not bool(args.no_terminal),
                    )
                    rows.extend(ep_rows)
                    _append_csv(branch_csv, ep_rows, BRANCH_FIELDS)
                    completed_keys.add(combo_key)
                    print(f"[state_branch] {opp} seed={seed} driver={args.driver_mode}:{driver_z} rows={len(ep_rows)} total_rows={len(rows)} ({combo}/{total}, {time.time() - start:.1f}s)", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass

    advantage_rows = compute_advantage_rows(rows, baseline_z=3)
    oracle_rows = compute_oracle_rows(advantage_rows, baseline_z=3)
    _write_csv(branch_csv, rows, BRANCH_FIELDS)
    _write_csv(advantage_csv, advantage_rows, ADVANTAGE_FIELDS)
    _write_csv(oracle_csv, oracle_rows, ORACLE_FIELDS)
    manifest = {
        "checkpoint": str(ckpt),
        "checkpoint_steps": ckpt_steps,
        "latent_k": latent_k,
        "opponents": list(args.opponents),
        "n_seeds": int(args.n_seeds),
        "base_seed": int(args.base_seed),
        "driver_mode": str(args.driver_mode),
        "driver_z_values": list(args.driver_z_values),
        "snapshots_per_bucket": int(args.snapshots_per_bucket),
        "short_horizon": int(args.short_horizon),
        "terminal_branches": not bool(args.no_terminal),
        "branch_rows": len(rows),
        "non_z3_oracle_candidates": sum(
            1
            for r in oracle_rows
            if int(r["best_z"]) != 3 and math.isfinite(float(r["oracle_minus_z3"])) and float(r["oracle_minus_z3"]) > 0.0
        ),
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(report_md, oracle_rows=oracle_rows, branch_rows=rows, args=args)
    print(f"[state_branch] wrote {branch_csv}")
    print(f"[state_branch] wrote {advantage_csv}")
    print(f"[state_branch] wrote {oracle_csv}")
    print(f"[state_branch] wrote {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
