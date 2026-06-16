"""Post-hoc qualitative rollout tool for latent PPO checkpoints.

Pure evaluation. No training objectives. No supervised labels. No backward
passes. The tool loads a trained checkpoint, runs eval episodes against the
canonical opponent set, and records per-step trajectory data that lets you
*see* what each learned ``z`` does behaviorally.

Two execution modes per opponent:

* **natural** -- ``q_phi`` chooses ``z`` exactly as it would during training
  (subject to the persistence interval baked into the checkpoint).
* **fixed_z** -- ``z`` is clamped to each value in ``[0, K-1]`` in turn,
  producing K matched sub-rollouts per opponent. This is the only way to
  measure the actor's *true* sensitivity to ``z`` without confounding it
  with q_phi's routing decisions.

Per step we log:

* z timeline -- ``z_active`` per step, plus ``q_phi_entropy``, ``q_phi_prob_k``
* team positions -- ``blue_{i}_x/y``, ``red_{i}_x/y``, alive + carrying bits
* flag events -- ``blue_carrier_count``, ``red_carrier_count`` (transitions
  show pickups / drops)
* captures / returns / score changes -- ``blue_score``, ``red_score`` deltas
* behavior fingerprint -- ``team_spread``, ``num_attackers``, ``num_defenders``,
  ``intercept_pressure``, ``defense_pressure``, ``attack_defense_ratio``,
  plus the rest of ``BEHAVIOR_TELEMETRY_NAMES`` (13 signals total).

Outputs (in ``--out-dir``, default ``<checkpoint_dir>/qualitative/``):

* ``<checkpoint_stem>_qualitative_steps.csv`` -- per-step trace (large)
* ``<checkpoint_stem>_qualitative_rollout_by_z.csv`` -- aggregated by
  ``(opponent, mode, z)``: episode count, win rate, dwell, behavior means
* ``<checkpoint_stem>_qualitative_rollout_summary.md`` -- executive readout

Usage::

    python tools/qualitative_rollout.py \\
        --checkpoint checkpoints/4v4/<run>.zip \\
        --opponents OP3 OP5 OP6 OP4 \\
        --episodes-per-mode 5 \\
        --agents 4

All behaviour numbers are observed, never optimised. The tool changes no
trainer state, writes no weights, and adds no supervised labels.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES, compute_behavior_telemetry_batch
from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.inference import load_custom_ppo_policy, read_custom_ppo_metadata
from rl.training.env_factory import build_training_env


# Map user-friendly opponent labels (matching the training --opponent-pool flag)
# to the env's internal scripted tags. Pass-through for unknown labels so the
# env's own validation can surface mismatches.
_OPPONENT_ENV_TAG: dict[str, str] = {
    "OP5": "OP5_RUSHER",
    "OP6": "OP6_TURTLE",
    "OP7": "OP7_SWITCHER",
}


def _env_opponent_tag(label: str) -> str:
    return _OPPONENT_ENV_TAG.get(str(label).strip().upper(), str(label).strip().upper())


# ---------------------------------------------------------------------------
# Per-step capture
# ---------------------------------------------------------------------------


def _team_positions(core: Any) -> dict[str, list[float]]:
    """Return blue/red x/y/alive/carrying arrays from env.core (n_envs=1 slice)."""

    def _to_np(name: str) -> np.ndarray:
        t = getattr(core, name, None)
        if t is None:
            return np.zeros((1,), dtype=np.float32)
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    return {
        "blue_x": _to_np("blue_x")[0].tolist(),
        "blue_y": _to_np("blue_y")[0].tolist(),
        "blue_alive": _to_np("blue_alive")[0].astype(np.int64).tolist(),
        "blue_carrying": _to_np("blue_carrying")[0].astype(np.int64).tolist(),
        "red_x": _to_np("red_x")[0].tolist(),
        "red_y": _to_np("red_y")[0].tolist(),
        "red_alive": _to_np("red_alive")[0].astype(np.int64).tolist(),
        "red_carrying": _to_np("red_carrying")[0].astype(np.int64).tolist(),
    }


def _summarise_positions(positions: dict[str, list[float]]) -> dict[str, float]:
    """Compact scalars derived from raw positions for the rollout_by_z aggregation."""
    bx, by = np.asarray(positions["blue_x"]), np.asarray(positions["blue_y"])
    rx, ry = np.asarray(positions["red_x"]), np.asarray(positions["red_y"])
    ba = np.asarray(positions["blue_alive"], dtype=np.int64)
    ra = np.asarray(positions["red_alive"], dtype=np.int64)
    out: dict[str, float] = {}
    out["blue_alive_count"] = float(ba.sum())
    out["red_alive_count"] = float(ra.sum())
    if ba.any():
        out["blue_centroid_x"] = float(bx[ba == 1].mean())
        out["blue_centroid_y"] = float(by[ba == 1].mean())
    else:
        out["blue_centroid_x"] = 0.0
        out["blue_centroid_y"] = 0.0
    if ra.any():
        out["red_centroid_x"] = float(rx[ra == 1].mean())
        out["red_centroid_y"] = float(ry[ra == 1].mean())
    else:
        out["red_centroid_x"] = 0.0
        out["red_centroid_y"] = 0.0
    return out


@dataclass
class EpisodeRecord:
    """One eval episode's per-step trace + episode-level outcome."""

    opponent: str
    mode: str  # "natural" or "fixed_z"
    fixed_z_id: int  # -1 in natural mode
    episode_idx: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    # Episode terminator outcome (filled after env reports done).
    outcome_blue_score: int = 0
    outcome_red_score: int = 0
    outcome_blue_won: bool = False
    outcome_decision_steps: int = 0
    n_steps: int = 0


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------


def _run_episode(
    *,
    env: Any,
    model: Any,
    opponent_label: str,
    mode: str,
    fixed_z_id: int,
    episode_idx: int,
    deterministic: bool,
    max_steps: int,
    latent_k: int,
) -> EpisodeRecord:
    rec = EpisodeRecord(
        opponent=opponent_label,
        mode=mode,
        fixed_z_id=fixed_z_id,
        episode_idx=episode_idx,
    )

    if hasattr(model, "reset_strategy"):
        model.reset_strategy()
    # fixed_z is only meaningful for latent checkpoints; for non-latent the
    # attribute does not exist on the policy handle (and predict() ignores it).
    if latent_k > 0 and hasattr(model, "fixed_latent_strategy"):
        if mode == "fixed_z":
            model.fixed_latent_strategy = True
            model.fixed_latent_strategy_id = max(0, min(int(fixed_z_id), latent_k - 1))
        else:
            model.fixed_latent_strategy = False

    obs = env.reset()
    prev_blue_score = 0
    prev_red_score = 0
    prev_blue_carrying_any = False
    prev_red_carrying_any = False

    for step in range(max_steps):
        single = {
            k: v[0] if hasattr(v, "shape") and v.ndim >= 2 and v.shape[0] == 1 else v
            for k, v in obs.items()
        }
        try:
            single["global_state"] = env.state()[0]
        except Exception:
            pass

        act, _ = model.predict(single, deterministic=deterministic)
        strategy_info = model.strategy_info() if hasattr(model, "strategy_info") else {}

        core = env.core
        act_t = torch.as_tensor(act, dtype=torch.long, device=core.device).unsqueeze(0)
        with torch.no_grad():
            beh_t = compute_behavior_telemetry_batch(core, act_t)
        beh_np = beh_t.detach().cpu().numpy()[0]

        positions = _team_positions(core)

        blue_score = int(getattr(core, "blue_score", torch.zeros((1,), dtype=torch.long))[0].item())
        red_score = int(getattr(core, "red_score", torch.zeros((1,), dtype=torch.long))[0].item())
        blue_carrier_count = int(sum(positions["blue_carrying"]))
        red_carrier_count = int(sum(positions["red_carrying"]))
        any_blue_carrying = blue_carrier_count > 0
        any_red_carrying = red_carrier_count > 0

        blue_picked_up_now = bool(any_blue_carrying and not prev_blue_carrying_any)
        red_picked_up_now = bool(any_red_carrying and not prev_red_carrying_any)
        blue_dropped_now = bool(not any_blue_carrying and prev_blue_carrying_any)
        red_dropped_now = bool(not any_red_carrying and prev_red_carrying_any)
        blue_score_delta = int(blue_score - prev_blue_score)
        red_score_delta = int(red_score - prev_red_score)

        row: dict[str, Any] = {
            "opponent": opponent_label,
            "mode": mode,
            "fixed_z_id": int(fixed_z_id),
            "episode_idx": int(episode_idx),
            "step": int(step),
            "z_active": int(strategy_info.get("strategy", -1)),
            "z_resampled": int(bool(strategy_info.get("strategy_resampled", False))),
            "q_phi_entropy": float(strategy_info.get("strategy_entropy", 0.0)),
            "blue_score": blue_score,
            "red_score": red_score,
            "blue_score_delta": blue_score_delta,
            "red_score_delta": red_score_delta,
            "blue_carrier_count": blue_carrier_count,
            "red_carrier_count": red_carrier_count,
            "blue_picked_up_now": int(blue_picked_up_now),
            "red_picked_up_now": int(red_picked_up_now),
            "blue_dropped_now": int(blue_dropped_now),
            "red_dropped_now": int(red_dropped_now),
        }
        for k_idx in range(latent_k):
            row[f"q_phi_prob_{k_idx}"] = float(
                strategy_info.get(f"strategy_prob_{k_idx}", 0.0)
            )
        for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
            row[name] = float(beh_np[j])
        for arr_name, vals in positions.items():
            for ai, val in enumerate(vals):
                row[f"{arr_name}_{ai}"] = float(val)

        rec.rows.append(row)

        env.step_async(act)
        obs, _, done, infos = env.step_wait()

        # Latch this step's observed state so the next iteration's deltas
        # measure the change between consecutive observations.
        prev_blue_score = blue_score
        prev_red_score = red_score
        prev_blue_carrying_any = any_blue_carrying
        prev_red_carrying_any = any_red_carrying

        if bool(done[0]):
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info) or {}
            rec.outcome_blue_score = int(ep_res.get("blue_score", blue_score))
            rec.outcome_red_score = int(ep_res.get("red_score", red_score))
            rec.outcome_blue_won = bool(rec.outcome_blue_score > rec.outcome_red_score)
            rec.outcome_decision_steps = int(
                ep_res.get("decision_steps", info.get("decision_steps", step + 1))
            )
            rec.n_steps = step + 1
            return rec

    rec.outcome_blue_score = prev_blue_score
    rec.outcome_red_score = prev_red_score
    rec.outcome_blue_won = bool(prev_blue_score > prev_red_score)
    rec.outcome_decision_steps = max_steps
    rec.n_steps = max_steps
    return rec


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_by_z(records: list[EpisodeRecord]) -> list[dict[str, Any]]:
    """Per (opponent, mode, z) aggregate of episode count, WR, dwell, behavior means."""
    # Build a flat list of (key, row) for streaming aggregation.
    # key = (opponent, mode, z_active_or_fixed)
    by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for ep in records:
        # For fixed_z mode the z is forced; for natural mode we group rows by
        # the z that was *actually* active at that step.
        for r in ep.rows:
            z_eff = int(r["fixed_z_id"]) if ep.mode == "fixed_z" else int(r["z_active"])
            key = (ep.opponent, ep.mode, z_eff)
            slot = by_key.setdefault(
                key,
                {
                    "map_layout": str(r.get("map_layout", "")),
                    "opponent": ep.opponent,
                    "mode": ep.mode,
                    "z": z_eff,
                    "step_count": 0,
                    "episodes_touched": set(),
                    "blue_wins": 0,
                    "blue_score_total": 0,
                    "red_score_total": 0,
                    "blue_picks": 0,
                    "red_picks": 0,
                    "blue_score_deltas": 0,
                    "red_score_deltas": 0,
                    "behavior_sum": np.zeros(len(BEHAVIOR_TELEMETRY_NAMES), dtype=np.float64),
                },
            )
            slot["step_count"] += 1
            slot["episodes_touched"].add(int(ep.episode_idx))
            for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
                slot["behavior_sum"][j] += float(r[name])
            slot["blue_picks"] += int(r["blue_picked_up_now"])
            slot["red_picks"] += int(r["red_picked_up_now"])
            if int(r["blue_score_delta"]) > 0:
                slot["blue_score_deltas"] += 1
            if int(r["red_score_delta"]) > 0:
                slot["red_score_deltas"] += 1

    # Attach episode-level outcomes per key. WR = (episodes where this z
    # was active for >0 steps AND blue won) / (episodes where this z was active).
    # NB: in fixed_z mode this naturally collapses to per-z episode WR.
    episode_outcomes_per_key: dict[tuple[str, str, int], list[bool]] = {}
    for ep in records:
        z_seen: set[int] = set()
        if ep.mode == "fixed_z":
            z_seen.add(int(ep.fixed_z_id))
        else:
            for r in ep.rows:
                z_seen.add(int(r["z_active"]))
        for z in z_seen:
            key = (ep.opponent, ep.mode, z)
            episode_outcomes_per_key.setdefault(key, []).append(ep.outcome_blue_won)

    rows_out: list[dict[str, Any]] = []
    for key, slot in by_key.items():
        steps = max(1, int(slot["step_count"]))
        outcomes = episode_outcomes_per_key.get(key, [])
        n_eps = len(outcomes)
        wr = float(sum(outcomes) / n_eps) if n_eps else 0.0
        row = {
            "map_layout": slot.get("map_layout", ""),
            "opponent": slot["opponent"],
            "mode": slot["mode"],
            "z": slot["z"],
            "n_episodes_touched": n_eps,
            "n_steps": int(slot["step_count"]),
            "blue_win_rate": wr,
            "blue_pickups_per_episode": float(slot["blue_picks"] / max(n_eps, 1)),
            "red_pickups_per_episode": float(slot["red_picks"] / max(n_eps, 1)),
            "blue_scores_per_episode": float(
                slot["blue_score_deltas"] / max(n_eps, 1)
            ),
            "red_scores_per_episode": float(
                slot["red_score_deltas"] / max(n_eps, 1)
            ),
        }
        for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
            row[f"{name}_mean"] = float(slot["behavior_sum"][j] / steps)
        rows_out.append(row)

    rows_out.sort(key=lambda r: (r["opponent"], r["mode"], r["z"]))
    return rows_out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _step_csv_fieldnames(latent_k: int, n_blue: int, n_red: int) -> list[str]:
    base = [
        "map_layout", "opponent", "mode", "fixed_z_id", "episode_idx", "step",
        "z_active", "z_resampled", "q_phi_entropy",
        "blue_score", "red_score", "blue_score_delta", "red_score_delta",
        "blue_carrier_count", "red_carrier_count",
        "blue_picked_up_now", "red_picked_up_now",
        "blue_dropped_now", "red_dropped_now",
    ]
    base += [f"q_phi_prob_{k}" for k in range(latent_k)]
    base += list(BEHAVIOR_TELEMETRY_NAMES)
    for prefix in ("blue_x", "blue_y", "blue_alive", "blue_carrying"):
        base += [f"{prefix}_{i}" for i in range(n_blue)]
    for prefix in ("red_x", "red_y", "red_alive", "red_carrying"):
        base += [f"{prefix}_{i}" for i in range(n_red)]
    return base


def _write_steps_csv(path: Path, records: list[EpisodeRecord], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ep in records:
            for r in ep.rows:
                writer.writerow(r)


def _write_rollout_by_z_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_strategy_evidence_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _forced_behavior_spread(rows: list[dict[str, Any]]) -> float:
    if len(rows) < 2:
        return 0.0
    ranges: list[float] = []
    for name in BEHAVIOR_TELEMETRY_NAMES:
        vals = [float(r.get(f"{name}_mean", 0.0)) for r in rows]
        ranges.append(max(vals) - min(vals))
    return float(np.mean(ranges)) if ranges else 0.0


def _strategy_spread_label(spread: float) -> str:
    if spread >= 0.25:
        return "high"
    if spread >= 0.10:
        return "medium"
    return "low"


def _strategy_interpretation(perf_spread: float, behavior_spread: float) -> str:
    if behavior_spread >= 0.10 and perf_spread >= 0.05:
        return "latent specialization"
    if behavior_spread >= 0.10:
        return "behavior differs; performance unclear"
    if perf_spread >= 0.05:
        return "performance differs; strategy meaning unclear"
    return "no causal strategy"


def _build_strategy_evidence_rows(
    records: list[EpisodeRecord],
    agg_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    natural_by_opp: dict[str, list[EpisodeRecord]] = {}
    for ep in records:
        if ep.mode == "natural":
            natural_by_opp.setdefault(ep.opponent, []).append(ep)

    fixed_by_opp: dict[str, list[dict[str, Any]]] = {}
    for row in agg_rows:
        if row.get("mode") == "fixed_z":
            fixed_by_opp.setdefault(str(row.get("opponent", "")), []).append(row)

    rows_out: list[dict[str, Any]] = []
    for opponent in sorted(set(natural_by_opp) | set(fixed_by_opp)):
        natural_eps = natural_by_opp.get(opponent, [])
        natural_wr = (
            float(sum(1 for ep in natural_eps if ep.outcome_blue_won) / len(natural_eps))
            if natural_eps
            else float("nan")
        )
        forced_rows = sorted(fixed_by_opp.get(opponent, []), key=lambda r: int(r.get("z", -1)))
        if not forced_rows:
            continue
        best = max(forced_rows, key=lambda r: float(r.get("blue_win_rate", 0.0)))
        worst = min(forced_rows, key=lambda r: float(r.get("blue_win_rate", 0.0)))
        best_wr = float(best.get("blue_win_rate", 0.0))
        worst_wr = float(worst.get("blue_win_rate", 0.0))
        behavior_spread = _forced_behavior_spread(forced_rows)
        perf_spread = best_wr - worst_wr
        rows_out.append(
            {
                "opponent": opponent,
                "natural_win_rate": natural_wr,
                "best_z": int(best.get("z", -1)),
                "best_forced_z_win_rate": best_wr,
                "worst_z": int(worst.get("z", -1)),
                "worst_forced_z_win_rate": worst_wr,
                "forced_z_performance_spread": perf_spread,
                "forced_z_behavior_spread": behavior_spread,
                "strategy_spread": _strategy_spread_label(behavior_spread),
                "interpretation": _strategy_interpretation(perf_spread, behavior_spread),
            }
        )
    return rows_out


def _markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_(empty)_\n"
    head = "| " + " | ".join(cols) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |\n"
    body_lines: list[str] = []
    for r in rows:
        cells: list[str] = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        body_lines.append("| " + " | ".join(cells) + " |")
    return head + sep + "\n".join(body_lines) + "\n"


def _write_summary_md(
    path: Path,
    *,
    records: list[EpisodeRecord],
    agg_rows: list[dict[str, Any]],
    checkpoint: Path,
    latent_k: int,
    n_blue: int,
    n_red: int,
    opponents: list[str],
    deterministic: bool,
    seed: int,
    is_latent: bool = True,
    map_layout: str = "map_a_open",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    title_suffix = "" if is_latent else " (baseline -- no latent strategy)"
    lines.append(f"# Qualitative rollout: `{checkpoint.name}`{title_suffix}\n")
    lines.append(
        f"Pure evaluation -- no training objectives, no supervised labels, no "
        f"backward passes. Tool: `tools/qualitative_rollout.py`.\n"
    )
    latent_line = (
        f"- latent_k: **{latent_k}**" if is_latent else "- latent_k: **n/a (no_latent baseline)**"
    )
    lines.append(
        f"{latent_line} | blue agents: **{n_blue}** | red agents: **{n_red}**\n"
        f"- map layout: **{map_layout}**\n"
        f"- opponents evaluated: **{', '.join(opponents)}**\n"
        f"- deterministic policy: **{deterministic}** | seed: **{seed}**\n"
        f"- total episodes: **{len(records)}**\n"
    )

    # Episode-level summary per (opponent, mode, z) — natural vs fixed.
    natural_rows = [r for r in agg_rows if r["mode"] == "natural"]
    fixed_rows = [r for r in agg_rows if r["mode"] == "fixed_z"]
    evidence_rows = _build_strategy_evidence_rows(records, agg_rows)

    if is_latent:
        lines.append("## Strategy evidence table\n")
        lines.append(
            "Natural router win rate is compared against the best and worst "
            "forced-z rollouts for the same opponent. ``forced_z_behavior_spread`` "
            "is the mean per-feature range across the 13 behavior telemetry "
            "signals. Latent strategy evidence requires both forced-z behavior "
            "differences and opponent-dependent performance or macro-behavior "
            "differences.\n"
        )
        lines.append(
            _markdown_table(
                evidence_rows,
                [
                    "opponent",
                    "natural_win_rate",
                    "best_z",
                    "best_forced_z_win_rate",
                    "worst_z",
                    "worst_forced_z_win_rate",
                    "forced_z_behavior_spread",
                    "strategy_spread",
                    "interpretation",
                ],
            )
        )

        lines.append("## Win rate by (opponent, z) -- fixed-z mode\n")
        lines.append(
            "Each row forces ``z`` to a single value for the entire episode. "
            "If the WR rows differ meaningfully across z for the same opponent, "
            "the actor is genuinely sensitive to z. If they look identical, the "
            "actor learned to ignore z.\n"
        )
        lines.append(
            _markdown_table(
                fixed_rows,
                ["opponent", "z", "n_episodes_touched", "blue_win_rate",
                 "blue_scores_per_episode", "red_scores_per_episode", "n_steps"],
            )
        )

        lines.append("## Natural q_phi routing -- per-z dwell + WR\n")
        lines.append(
            "In natural mode q_phi picks z. The WR column here is per-episode "
            "(an episode contributes to every z it visited). ``n_steps`` is the "
            "total dwell that z accumulated across all episodes for that opponent.\n"
        )
        lines.append(
            _markdown_table(
                natural_rows,
                ["opponent", "z", "n_episodes_touched", "n_steps",
                 "blue_win_rate", "blue_scores_per_episode", "red_scores_per_episode"],
            )
        )
    else:
        lines.append("## Per-opponent WR -- baseline (no z)\n")
        lines.append(
            "Non-latent checkpoint: there is no ``z`` to fix or route. Each "
            "row is the aggregate over all natural-mode episodes for that "
            "opponent. ``z`` is reported as -1 to keep the schema consistent "
            "with latent runs for side-by-side comparison.\n"
        )
        lines.append(
            _markdown_table(
                natural_rows,
                ["opponent", "z", "n_episodes_touched", "n_steps",
                 "blue_win_rate", "blue_scores_per_episode", "red_scores_per_episode"],
            )
        )

    # Behavioral fingerprint -- per z for latent, per opponent for baseline.
    if is_latent:
        # Average behavior signals across all opponents + modes, weighted by
        # the number of steps that z was active.
        by_z_global: dict[int, dict[str, float]] = {}
        by_z_count: dict[int, int] = {}
        for r in agg_rows:
            z = int(r["z"])
            if z < 0:
                continue
            if z not in by_z_global:
                by_z_global[z] = {name: 0.0 for name in BEHAVIOR_TELEMETRY_NAMES}
                by_z_count[z] = 0
            w = int(r["n_steps"])
            for name in BEHAVIOR_TELEMETRY_NAMES:
                by_z_global[z][name] += float(r.get(f"{name}_mean", 0.0)) * w
            by_z_count[z] += w

        behavior_rows: list[dict[str, Any]] = []
        for z in sorted(by_z_global):
            denom = max(by_z_count[z], 1)
            row: dict[str, Any] = {"z": z, "total_steps": by_z_count[z]}
            for name in BEHAVIOR_TELEMETRY_NAMES:
                row[name] = by_z_global[z][name] / denom
            behavior_rows.append(row)

        lines.append("## Behavioral fingerprint per z (step-weighted, all opponents + modes)\n")
        lines.append(
            "Means of the 13 ``BEHAVIOR_TELEMETRY_NAMES`` signals, weighted by "
            "the number of steps that z was active. These are *observed* "
            "behaviors -- no labels were used to compute them.\n"
        )
        lines.append(_markdown_table(behavior_rows, ["z", "total_steps", *BEHAVIOR_TELEMETRY_NAMES]))

        if len(behavior_rows) > 1:
            means = np.asarray(
                [[r[name] for name in BEHAVIOR_TELEMETRY_NAMES] for r in behavior_rows],
                dtype=np.float64,
            )
            avg = means.mean(axis=0, keepdims=True)
            dev = means - avg
            lines.append("## Top 3 distinguishing behaviors per z\n")
            for i, r in enumerate(behavior_rows):
                order = np.argsort(-np.abs(dev[i]))[:3]
                picks = ", ".join(
                    f"`{BEHAVIOR_TELEMETRY_NAMES[j]}` "
                    f"({means[i, j]:+.3f} vs avg {avg[0, j]:+.3f})"
                    for j in order
                )
                lines.append(f"- **z{int(r['z'])}**: {picks}")
            lines.append("")
    else:
        # Baseline fingerprint: one row per opponent, step-weighted means.
        by_opp: dict[str, dict[str, float]] = {}
        by_opp_count: dict[str, int] = {}
        for r in agg_rows:
            opp = str(r["opponent"])
            if opp not in by_opp:
                by_opp[opp] = {name: 0.0 for name in BEHAVIOR_TELEMETRY_NAMES}
                by_opp_count[opp] = 0
            w = int(r["n_steps"])
            for name in BEHAVIOR_TELEMETRY_NAMES:
                by_opp[opp][name] += float(r.get(f"{name}_mean", 0.0)) * w
            by_opp_count[opp] += w
        behavior_rows = []
        for opp in sorted(by_opp):
            denom = max(by_opp_count[opp], 1)
            row = {"opponent": opp, "total_steps": by_opp_count[opp]}
            for name in BEHAVIOR_TELEMETRY_NAMES:
                row[name] = by_opp[opp][name] / denom
            behavior_rows.append(row)

        lines.append("## Behavioral fingerprint per opponent (step-weighted)\n")
        lines.append(
            "Baseline reference: means of the 13 ``BEHAVIOR_TELEMETRY_NAMES`` "
            "signals per opponent. Use these as the no-latent comparison "
            "point when reading a latent checkpoint's per-z fingerprint.\n"
        )
        lines.append(_markdown_table(behavior_rows, ["opponent", "total_steps", *BEHAVIOR_TELEMETRY_NAMES]))

    lines.append("## Summer-faithful audit\n")
    lines.append(
        "- No phase / flag / outcome prediction loss applied. "
        "- No opponent-id loss applied. "
        "- No backward pass at any point. "
        "- All metrics in this report are read-only observations of the "
        "trained checkpoint stepping the environment.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(
    *,
    checkpoint: Path,
    opponents: list[str],
    episodes_per_mode: int,
    agents: int,
    device: str,
    seed: int,
    out_dir: Path,
    modes: list[str],
    deterministic: bool,
    max_steps: int,
    map_layout: str = "map_a_open",
) -> dict[str, Path]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    meta = read_custom_ppo_metadata(str(checkpoint))
    is_latent = bool(meta.get("use_latent_strategy", False))
    latent_k = int(meta.get("latent_k", 4)) if is_latent else 0
    n_blue = int(meta.get("n_blue", agents))
    n_red = n_blue  # CTF is symmetric

    effective_modes = list(modes)
    if not is_latent and "fixed_z" in effective_modes:
        warnings.warn(
            "[qualitative] checkpoint has no latent strategy "
            "(use_latent_strategy=False); skipping fixed_z mode."
        )
        effective_modes = [m for m in effective_modes if m != "fixed_z"]
    if not effective_modes:
        raise SystemExit(
            "[qualitative] no modes left to run "
            "(non-latent checkpoint with only fixed_z requested)."
        )

    cfg = PPOConfig()
    cfg.use_latent_strategy = is_latent
    cfg.n_envs = 1
    cfg.seed = int(seed)
    cfg.device = str(device)
    cfg.max_blue_agents = n_blue
    cfg.n_agents_per_team = n_blue
    cfg.map_layout = str(map_layout).strip().lower()
    if is_latent:
        cfg.latent_k = latent_k

    flavour = f"latent_k={latent_k}" if is_latent else "no_latent (baseline)"
    print(f"[qualitative] checkpoint: {checkpoint}")
    print(
        f"[qualitative] {flavour}  agents={n_blue}v{n_red}  "
        f"map_layout={cfg.map_layout}  device={device}  seed={seed}"
    )
    print(f"[qualitative] modes={effective_modes}  episodes_per_mode={episodes_per_mode}")
    print(f"[qualitative] opponents={opponents}")

    first_opp_env = _env_opponent_tag(opponents[0])
    env = build_training_env(cfg, initial_phase="PHASE1", initial_opponent_tag=first_opp_env)
    try:
        model = load_custom_ppo_policy(
            str(checkpoint),
            env.observation_space,
            env.action_space,
            device=device,
        )

        records: list[EpisodeRecord] = []
        global_ep_counter = 0
        for opp_label in opponents:
            opp_env = _env_opponent_tag(opp_label)
            try:
                env.env_method("set_next_opponent", "SCRIPTED", opp_env)
            except Exception as exc:
                warnings.warn(
                    f"[qualitative] could not set opponent={opp_label} (env tag={opp_env}): "
                    f"{exc}. Skipping."
                )
                continue
            # NB: opponent only switches on the next env.reset(), so the very
            # first reset of a (new opponent) block hooks up the correct red AI.
            for mode in effective_modes:
                if mode == "natural":
                    for ep_idx in range(episodes_per_mode):
                        torch.manual_seed(seed + global_ep_counter)
                        np.random.seed(seed + global_ep_counter)
                        rec = _run_episode(
                            env=env,
                            model=model,
                            opponent_label=opp_label,
                            mode="natural",
                            fixed_z_id=-1,
                            episode_idx=ep_idx,
                            deterministic=deterministic,
                            max_steps=max_steps,
                            latent_k=latent_k,
                        )
                        for row in rec.rows:
                            row["map_layout"] = cfg.map_layout
                        records.append(rec)
                        global_ep_counter += 1
                        print(
                            f"[qualitative] {opp_label} natural ep{ep_idx} "
                            f"-> WR={int(rec.outcome_blue_won)} "
                            f"score={rec.outcome_blue_score}-{rec.outcome_red_score} "
                            f"steps={rec.n_steps}"
                        )
                elif mode == "fixed_z":
                    for z_force in range(latent_k):
                        for ep_idx in range(episodes_per_mode):
                            torch.manual_seed(seed + global_ep_counter)
                            np.random.seed(seed + global_ep_counter)
                            rec = _run_episode(
                                env=env,
                                model=model,
                                opponent_label=opp_label,
                                mode="fixed_z",
                                fixed_z_id=z_force,
                                episode_idx=ep_idx,
                                deterministic=deterministic,
                                max_steps=max_steps,
                                latent_k=latent_k,
                            )
                            for row in rec.rows:
                                row["map_layout"] = cfg.map_layout
                            records.append(rec)
                            global_ep_counter += 1
                            print(
                                f"[qualitative] {opp_label} fixed_z={z_force} ep{ep_idx} "
                                f"-> WR={int(rec.outcome_blue_won)} "
                                f"score={rec.outcome_blue_score}-{rec.outcome_red_score} "
                                f"steps={rec.n_steps}"
                            )
                else:
                    warnings.warn(f"[qualitative] unknown mode: {mode}")

        # --- Write outputs ---
        stem = checkpoint.stem
        steps_csv = out_dir / f"{stem}_qualitative_steps.csv"
        by_z_csv = out_dir / f"{stem}_qualitative_rollout_by_z.csv"
        evidence_csv = out_dir / f"{stem}_strategy_evidence.csv"
        summary_md = out_dir / f"{stem}_qualitative_rollout_summary.md"

        fieldnames = _step_csv_fieldnames(latent_k, n_blue, n_red)
        _write_steps_csv(steps_csv, records, fieldnames)

        agg_rows = _aggregate_by_z(records)
        _write_rollout_by_z_csv(by_z_csv, agg_rows)
        evidence_rows = _build_strategy_evidence_rows(records, agg_rows)
        _write_strategy_evidence_csv(evidence_csv, evidence_rows)
        _write_summary_md(
            summary_md,
            records=records,
            agg_rows=agg_rows,
            checkpoint=checkpoint,
            latent_k=latent_k,
            n_blue=n_blue,
            n_red=n_red,
            opponents=opponents,
            deterministic=deterministic,
            seed=seed,
            is_latent=is_latent,
            map_layout=cfg.map_layout,
        )

        outputs = {
            "rollout_by_z": by_z_csv,
            "strategy_evidence": evidence_csv,
            "rollout_summary": summary_md,
            "steps": steps_csv,
        }
        print("[qualitative] wrote:")
        for name, p in outputs.items():
            print(f"  - {name}: {p}")
        return outputs
    finally:
        try:
            env.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"[qualitative] WARNING: env.close() raised: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a custom PPO checkpoint .zip")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["OP3", "OP5", "OP6", "OP4"],
        help="Opponent labels (OP3, OP5, OP6, OP4). Order respected; missing ones are skipped with a warning.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["natural", "fixed_z"],
        choices=["natural", "fixed_z"],
        help="Which evaluation modes to run.",
    )
    parser.add_argument(
        "--episodes-per-mode",
        type=int,
        default=5,
        help="Episodes per (opponent, mode) block. In fixed_z mode this is per forced z too.",
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=None,
        help="Agents per team (default: read from checkpoint metadata).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--map-layout",
        type=str,
        default="map_a_open",
        choices=["map_a_open", "map_b_split_lane", "map_b_split_lane_v2", "open", "split_lane", "split_lane_v2"],
        help="GPUFieldConfig.map_layout used for the evaluation environment.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <checkpoint_dir>/qualitative/).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="Hard cap on per-episode decision steps (safety belt).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of argmax. Default is deterministic (greedy) eval.",
    )
    args = parser.parse_args(argv)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.suffix:
        ckpt_path = ckpt_path.with_suffix(".zip")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (ckpt_path.parent / "qualitative")
    agents = int(args.agents) if args.agents is not None else None
    if agents is None:
        meta = read_custom_ppo_metadata(str(ckpt_path))
        agents = int(meta.get("n_blue", 4))
    run(
        checkpoint=ckpt_path,
        opponents=list(args.opponents),
        episodes_per_mode=int(args.episodes_per_mode),
        agents=agents,
        device=str(args.device),
        seed=int(args.seed),
        out_dir=out_dir,
        modes=list(args.modes),
        deterministic=not bool(args.stochastic),
        max_steps=int(args.max_steps),
        map_layout=str(args.map_layout),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
