"""High-level orchestration: extracts data from trainer/buffer and delegates to pure modules."""

from __future__ import annotations

import csv
import math
import os
from typing import Any

import torch

from rl.behavior_telemetry import N_ATTACK_DEFENSE_RATIO_BUCKET, N_ROLE_BUCKET_MI
from rl.custom_ppo.csv_writers import (
    SCRIPTED_OPPONENT_MI_COUNT,
    V3I3_REFRESH_REASON_LABELS,
    _ensure_additive_csv_header,
    _strategy_experience_fieldnames,
    _v3i3_refresh_log_fieldnames,
)
from rl.latent_phase_labels import TEAM_PHASES
from rl.custom_ppo.diagnostics.entropy import (
    _bucket_z_fracs,
    _fill_zero_z_fracs,
    _flat_float_np,
    _flat_long_np,
    _mi_z_vs,
    _shannon_entropy_nats,
)
from rl.custom_ppo.diagnostics.occupancy import compute_occupancy_stats
from rl.custom_ppo.diagnostics.specialization import (
    _behavior_diversity_stats,
    _flag_state_per_step,
    _phase_block,
    _q_phi_probs_and_entropy,
)
from rl.custom_ppo.diagnostics.switching import (
    _reward_sum_after_switch_5,
    _switch_proximity_fracs,
)

import numpy as np


def _latent_rollout_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Summarize strategy occupancy and switching for the latest rollout.

    Existing keys (CSV-pinned, do not rename or remove):

    * ``strategy_unique_count``, ``strategy_dominant``,
      ``strategy_switch_count``, ``strategy_switch_fraction``,
      ``strategy_resample_count``, ``strategy_resample_fraction_rollout``,
      ``strategy_occupancy_{0..K-1}``.

    Added for v5i5 (occupancy-collapse diagnostics requested with the
    entropy-floor experiment; no new gradient channel, no new objective):

    * ``effective_num_latents`` -- ``exp(H)`` of the **sampled-z**
      rollout-marginal distribution (one categorical sample per state,
      ``argmax``-style). Equals ``K`` for uniform sampled occupancy;
      equals 1 for full collapse.
    * ``latent_marginal_entropy_nats`` -- the underlying ``H`` of the
      **sampled-z empirical histogram** (NOT ``H(E_s[q_phi(z|s)])``;
      see ``router_rollout_soft_marginal_entropy_nats`` for the soft
      analogue computed from differentiable router probabilities).
    * ``latent_occupancy_min`` / ``latent_occupancy_max`` --
      **sampled-z** per-z population fractions; one categorical sample
      per state. Distinct from ``router_rollout_soft_argmax_occupancy_*``
      which is computed from ``argmax_z q_phi(z|s)`` over the same
      states (no sampling noise but still a hard decision; both labels
      are kept for cross-checking).
    * ``latent_occupancy_ratio`` -- ``max / max(min, 1e-8)`` of the
      sampled-z occupancy. Large = severe sampled imbalance; ``1.0`` =
      perfect uniform sampling.
    * ``mean_strategy_duration`` -- mean dwell length in decision steps
      between latent switches. Computed as
      ``num_decision_steps / max(1, num_arc_boundaries)`` where
      ``num_arc_boundaries = strategy_resample_count``. Stays a
      diagnostic; not used in any loss.
    """
    if not trainer.use_latent_strategy or "z" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    z = buffer.fields["z"][:length].reshape(-1).long()
    prev_z = buffer.fields["prev_z"][:length].reshape(-1).long()
    if z.numel() == 0:
        return {}
    counts = torch.bincount(z.clamp(min=0, max=trainer.latent_k - 1), minlength=trainer.latent_k).float()
    persist_mask = buffer.fields["z_persist_mask"][:length].reshape(-1).bool()
    resample_field = "z_resampled_actual" if "z_resampled_actual" in buffer.fields else "z_resampled"
    resampled = buffer.fields[resample_field][:length].reshape(-1).bool()
    switched = persist_mask & (z != prev_z)
    resample_count = float(resampled.sum().detach().cpu().item())
    persistence_valid = (
        buffer.fields["persistence_valid"][:length].reshape(-1).bool()
        if "persistence_valid" in buffer.fields
        else persist_mask
    )
    out: dict[str, float] = {
        "strategy_unique_count": float((counts > 0).sum().detach().cpu().item()),
        "strategy_dominant": float(torch.argmax(counts).detach().cpu().item()),
        "strategy_switch_count": float(switched.sum().detach().cpu().item()),
        "strategy_switch_fraction": float(
            (switched.float().sum() / persist_mask.float().sum().clamp_min(1.0)).detach().cpu().item()
        ),
        "strategy_resample_count": resample_count,
        "z_resampled_actual": resample_count,
        "router_opportunity_count": resample_count,
        "persistence_valid_pair_count": float(persistence_valid.sum().detach().cpu().item()),
        "strategy_resample_fraction_rollout": float(resampled.float().mean().detach().cpu().item()),
    }

    # Delegate pure occupancy stats to the occupancy module.
    occ_stats = compute_occupancy_stats(counts, trainer.latent_k)
    out["latent_marginal_entropy_nats"] = float(occ_stats["latent_marginal_entropy_nats"])
    out["effective_num_latents"] = float(occ_stats["effective_num_latents"])
    out["latent_occupancy_min"] = float(occ_stats["latent_occupancy_min"])
    out["latent_occupancy_max"] = float(occ_stats["latent_occupancy_max"])
    out["latent_occupancy_ratio"] = float(occ_stats["latent_occupancy_ratio"])
    for k in range(trainer.latent_k):
        out[f"strategy_occupancy_{k}"] = float(occ_stats.get(f"strategy_occupancy_{k}", 0.0))

    total_decisions = float(z.numel())
    arc_boundaries = max(1.0, resample_count)
    out["mean_strategy_duration"] = total_decisions / arc_boundaries
    return out


def _latent_opponent_rollout_diag(trainer: Any, buffer: Any) -> dict[str, float]:
    """Per-opponent z occupancy plus MI(z; opponent/phase/outcome) and phase / behavior bucket rollups."""
    if not trainer.use_latent_strategy or "z" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}

    K = int(trainer.latent_k)
    z = _flat_long_np(buffer, "z", length)
    prev_z = _flat_long_np(buffer, "prev_z", length)
    assert z is not None and prev_z is not None
    sw = (z != prev_z).astype(np.float64)
    out: dict[str, float] = {}

    q_probs, q_entropy = _q_phi_probs_and_entropy(buffer, length, K)

    # MI(z; categorical context fields) plus per-opponent occupancy
    oid = _flat_long_np(buffer, "opponent_id", length)
    if oid is not None:
        out["latent_mi_z_opponent_nats"] = _mi_z_vs(z, K, oid, SCRIPTED_OPPONENT_MI_COUNT)
        _bucket_z_fracs(out, z, K, oid, SCRIPTED_OPPONENT_MI_COUNT,
                        lambda o, k: f"strategy_occupancy_op{o}_z{k}")
    else:
        out["latent_mi_z_opponent_nats"] = 0.0

    pid = _flat_long_np(buffer, "phase_id", length)
    out["latent_mi_z_phase_nats"] = _mi_z_vs(z, K, pid, len(TEAM_PHASES))
    out["MI_executed_z_phase"] = out["latent_mi_z_phase_nats"]

    yid = _flat_long_np(buffer, "outcome_id", length)
    out["latent_mi_z_outcome_nats"] = _mi_z_vs(z, K, yid, 3)
    out["MI_executed_z_outcome"] = out["latent_mi_z_outcome_nats"]

    # Phase-conditioned rollups + ahead/trail switch rates
    ba = _flat_float_np(buffer, "blue_ahead", length)
    rsp_flat = _flat_float_np(buffer, "reward_sparse_points", length)
    rsp_bin = (np.abs(rsp_flat) > 1e-5).astype(np.float64) if rsp_flat is not None else None

    if pid is not None:
        _phase_block(out, z, K, sw, ba, rsp_bin, pid, q_probs, q_entropy)

    if ba is not None:
        ahead = ba > 0.5
        trail = ~ahead
        out["latent_switch_rate_blue_ahead"] = float(sw[ahead].mean()) if bool(ahead.any()) else 0.0
        out["latent_switch_rate_blue_trail"] = float(sw[trail].mean()) if bool(trail.any()) else 0.0
    else:
        out["latent_switch_rate_blue_ahead"] = 0.0
        out["latent_switch_rate_blue_trail"] = 0.0

    _reward_sum_after_switch_5(out, buffer, length)

    # MI(z; situation bucket) for spread / role / pressure / ADR
    n_sb, n_rb, n_pb = 3, int(N_ROLE_BUCKET_MI), 3
    n_adr = int(N_ATTACK_DEFENSE_RATIO_BUCKET)

    sb = _flat_long_np(buffer, "spread_bucket_id", length)
    out["latent_mi_z_spread_bucket_nats"] = _mi_z_vs(z, K, sb, n_sb)

    rb = _flat_long_np(buffer, "role_bucket_id", length)
    if rb is not None:
        out["latent_mi_z_role_bucket_nats"] = _mi_z_vs(z, K, rb, n_rb)
        _bucket_z_fracs(out, z, K, rb, n_rb, lambda r, k: f"latent_role{r}_z{k}_frac")
        for r in range(n_rb):
            mask = rb == r
            out[f"latent_role{r}_switch_mean"] = float(sw[mask].mean()) if bool(mask.any()) else 0.0
    else:
        out["latent_mi_z_role_bucket_nats"] = 0.0
        _fill_zero_z_fracs(out, K, n_rb, lambda r, k: f"latent_role{r}_z{k}_frac")
        for r in range(n_rb):
            out[f"latent_role{r}_switch_mean"] = 0.0

    pb = _flat_long_np(buffer, "pressure_bucket_id", length)
    out["latent_mi_z_pressure_bucket_nats"] = _mi_z_vs(z, K, pb, n_pb)

    adb = _flat_long_np(buffer, "attack_defense_ratio_bucket_id", length)
    out["latent_mi_z_attack_defense_ratio_bucket_nats"] = _mi_z_vs(z, K, adb, n_adr)

    # Flag-state derived from global_state
    flag_state = _flag_state_per_step(buffer, length)
    out["latent_mi_z_flag_state_nats"] = _mi_z_vs(z, K, flag_state, 4)
    out["MI_executed_z_flag"] = out["latent_mi_z_flag_state_nats"]
    _bucket_z_fracs(out, z, K, flag_state, 4, lambda f, k: f"latent_flag_state{f}_z{k}_frac")

    # Per-bucket occupancy distributions (spread, ADR)
    if sb is not None:
        _bucket_z_fracs(out, z, K, sb, 3, lambda s, k: f"latent_spread{s}_z{k}_frac")
    else:
        _fill_zero_z_fracs(out, K, 3, lambda s, k: f"latent_spread{s}_z{k}_frac")

    if adb is not None:
        _bucket_z_fracs(out, z, K, adb, 3, lambda a, k: f"latent_adr{a}_z{k}_frac")
    else:
        _fill_zero_z_fracs(out, K, 3, lambda a, k: f"latent_adr{a}_z{k}_frac")

    # Per-phase Shannon entropy over z
    for p in range(len(TEAM_PHASES)):
        if pid is None:
            out[f"latent_phase{p}_entropy"] = 0.0
            continue
        mask = pid == p
        if bool(mask.any()):
            out[f"latent_phase{p}_entropy"] = _shannon_entropy_nats(
                np.clip(z[mask], 0, K - 1), K
            )
        else:
            out[f"latent_phase{p}_entropy"] = 0.0

    # Overall diversity (Shannon entropy of the bucket distribution)
    out["latent_role_diversity"] = _shannon_entropy_nats(rb, n_rb)
    out["latent_spread_diversity"] = _shannon_entropy_nats(sb, n_sb)
    out["latent_pressure_diversity"] = _shannon_entropy_nats(pb, n_pb)
    out["latent_adr_diversity"] = _shannon_entropy_nats(adb, n_adr)

    # Normalized MI = I(z; x) / H(z)
    z_valid = z[(z >= 0) & (z < K)] if z is not None else None
    h_z = _shannon_entropy_nats(z_valid.astype(np.int64), K) if z_valid is not None else 0.0
    if h_z > 1e-12:
        out["latent_normalized_mi_z_opponent"] = float(out["latent_mi_z_opponent_nats"] / h_z)
        out["latent_normalized_mi_z_phase"] = float(out["latent_mi_z_phase_nats"] / h_z)
        out["latent_normalized_mi_z_outcome"] = float(out["latent_mi_z_outcome_nats"] / h_z)
        out["latent_normalized_mi_z_flag_state"] = float(out["latent_mi_z_flag_state_nats"] / h_z)
    else:
        out["latent_normalized_mi_z_opponent"] = 0.0
        out["latent_normalized_mi_z_phase"] = 0.0
        out["latent_normalized_mi_z_outcome"] = 0.0
        out["latent_normalized_mi_z_flag_state"] = 0.0
    out["latent_z_marginal_entropy_nats"] = float(h_z)

    _switch_proximity_fracs(out, buffer, length)

    return out


def _write_strategy_experience_table(trainer: Any) -> dict[str, float]:
    if not trainer.strategy_experience_csv_path or not trainer.use_latent_strategy or trainer.latent_k <= 0:
        return {"strategy_bucket_best_match_frac": 0.0, "strategy_experience_records": 0.0, "strategy_experience_buckets": 0.0}
    records = list(trainer.latent_state.rollout_strategy_episode_records)
    if not records:
        return {"strategy_bucket_best_match_frac": 0.0, "strategy_experience_records": 0.0, "strategy_experience_buckets": 0.0}

    by_bucket: dict[int, list[dict[str, Any]]] = {}
    for r in records:
        by_bucket.setdefault(int(r["bucket_id"]), []).append(r)

    rows: list[dict[str, Any]] = []
    best_match = 0
    total = 0
    for bucket_id, bucket_records in sorted(by_bucket.items()):
        bucket_count = len(bucket_records)
        total += bucket_count
        returns_by_z: dict[int, list[float]] = {z: [] for z in range(trainer.latent_k)}
        wins_by_z: dict[int, list[int]] = {z: [] for z in range(trainer.latent_k)}
        for r in bucket_records:
            z = int(r["z"])
            if 0 <= z < trainer.latent_k:
                returns_by_z[z].append(float(r["episode_return"]))
                wins_by_z[z].append(int(r["episode_win"]))
        best_candidates = [
            (float(np.mean(vals)), z) for z, vals in returns_by_z.items() if vals
        ]
        best_z = max(best_candidates)[1] if best_candidates else -1
        if best_z >= 0:
            best_match += len(returns_by_z[best_z])
        best_z_match_frac = float(len(returns_by_z.get(best_z, []))) / float(max(1, bucket_count))
        for z in range(trainer.latent_k):
            z_returns = returns_by_z[z]
            z_wins = wins_by_z[z]
            prob_vals = [
                float(r["q_phi_probs"][z])
                for r in bucket_records
                if z < len(r.get("q_phi_probs", []))
            ]
            count = len(z_returns)
            rows.append(
                {
                    "update": int(trainer._updates_completed),
                    "run_id": trainer.run_id,
                    "run_pid": trainer.run_pid,
                    "timesteps": int(trainer.global_step),
                    "bucket_id": int(bucket_id),
                    "z": int(z),
                    "count": int(count),
                    "bucket_count": int(bucket_count),
                    "mean_return": "" if count <= 0 else float(np.mean(z_returns)),
                    "win_rate": "" if count <= 0 else float(np.mean(z_wins)),
                    "q_phi_prob_mean": float(np.mean(prob_vals)) if prob_vals else "",
                    "chosen_freq": float(count) / float(max(1, bucket_count)),
                    "best_z": int(best_z),
                    "best_z_match_frac": best_z_match_frac,
                }
            )

    fieldnames = _strategy_experience_fieldnames()
    path = trainer.strategy_experience_csv_path
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _ensure_additive_csv_header(path, fieldnames)
    nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not nonempty:
            writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    return {
        "strategy_bucket_best_match_frac": float(best_match) / float(max(1, total)),
        "strategy_experience_records": float(total),
        "strategy_experience_buckets": float(len(by_bucket)),
    }


def _write_refresh_log_table(trainer: Any) -> dict[str, float]:
    """Write one CSV row per finalized v3i3 refresh event for this rollout."""
    if not bool(getattr(trainer, "latent_v3i3_refresh_log_enabled", False)):
        return {"latent_v3i3_refresh_log_rows": 0.0}
    path = str(getattr(trainer, "latent_v3i3_refresh_log_path", "") or "")
    if not path:
        return {"latent_v3i3_refresh_log_rows": 0.0}
    if not getattr(trainer, "use_latent_strategy", False):
        return {"latent_v3i3_refresh_log_rows": 0.0}
    latent_state = getattr(trainer, "latent_state", None)
    if latent_state is None:
        return {"latent_v3i3_refresh_log_rows": 0.0}
    records = list(getattr(latent_state, "rollout_refresh_records", []) or [])
    if not records:
        return {"latent_v3i3_refresh_log_rows": 0.0}
    fieldnames = _v3i3_refresh_log_fieldnames()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    _ensure_additive_csv_header(path, fieldnames)
    nonempty = os.path.isfile(path) and os.path.getsize(path) > 0
    rows: list[dict[str, Any]] = []
    update_idx = int(getattr(trainer, "_updates_completed", 0))
    timesteps = int(getattr(trainer, "global_step", 0))
    for rec in records:
        reason_id = int(rec.get("reason_id", -1))
        reason_label = (
            V3I3_REFRESH_REASON_LABELS[reason_id]
            if 0 <= reason_id < len(V3I3_REFRESH_REASON_LABELS)
            else "unknown"
        )
        rows.append(
            {
                "update": update_idx,
                "run_id": getattr(trainer, "run_id", ""),
                "run_pid": getattr(trainer, "run_pid", 0),
                "timesteps": timesteps,
                "env_id": int(rec.get("env_id", -1)),
                "episode_id": int(rec.get("episode_id", -1)),
                "decision_step": int(rec.get("decision_step", -1)),
                "reason_id": reason_id,
                "reason": reason_label,
                "prev_z": int(rec.get("prev_z", -1)),
                "next_z": int(rec.get("next_z", -1)),
                "opponent_id": int(rec.get("opponent_id", -1)),
                "flag_state_bucket": int(rec.get("flag_state_bucket", -1)),
                "carrier_progress_bucket": int(rec.get("carrier_progress_bucket", -1)),
                "return_at_refresh": float(rec.get("return_at_refresh", 0.0)),
                "return_from_now_to_end": float(
                    rec.get("return_from_now_to_end", rec.get("future_return", 0.0))
                ),
            }
        )
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not nonempty:
            writer.writeheader()
        writer.writerows({key: r.get(key, "") for key in fieldnames} for r in rows)
    return {"latent_v3i3_refresh_log_rows": float(len(rows))}


latent_rollout_stats = _latent_rollout_stats
latent_opponent_rollout_diag = _latent_opponent_rollout_diag
write_strategy_experience_table = _write_strategy_experience_table
write_refresh_log_table = _write_refresh_log_table

__all__ = [
    "_latent_rollout_stats",
    "_latent_opponent_rollout_diag",
    "_behavior_diversity_stats",
    "_write_strategy_experience_table",
    "_write_refresh_log_table",
    "latent_rollout_stats",
    "latent_opponent_rollout_diag",
    "write_strategy_experience_table",
    "write_refresh_log_table",
]
