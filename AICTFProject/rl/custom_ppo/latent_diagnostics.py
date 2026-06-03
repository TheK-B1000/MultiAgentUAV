from __future__ import annotations

import csv
import os
from typing import Any, Callable

import numpy as np
import torch

from rl.behavior_telemetry import (
    BEHAVIOR_TELEMETRY_NAMES,
    N_ATTACK_DEFENSE_RATIO_BUCKET,
    N_ROLE_BUCKET_MI,
    N_TELEMETRY,
)
from rl.custom_ppo.csv_writers import (
    SCRIPTED_OPPONENT_MI_COUNT,
    V3I3_REFRESH_REASON_LABELS,
    _ensure_additive_csv_header,
    _strategy_experience_fieldnames,
    _v3i3_refresh_log_fieldnames,
)
from rl.custom_ppo.inference import FORCED_Z_MACRO_ACTIONS, FORCED_Z_PROFILE_MAX_ROWS
from rl.discrete_mi import discrete_mi_plugin
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_phase_labels import TEAM_PHASES


# ---------------------------------------------------------------------------
# Internal helpers shared by the rollout diagnostics below.
#
# These exist because the same three patterns appear ~8-10 times each inside
# `_latent_opponent_rollout_diag`:
#   1. pull an optional integer/float field out of the rollout buffer,
#   2. compute plug-in MI(z; x) for some categorical x,
#   3. write P(z=k | bucket=b) for every (b, k).
# Centralising them keeps the diagnostic readable as a checklist and replaces
# the Python-level joint-histogram loops with a single vectorised bincount.
# ---------------------------------------------------------------------------


def _flat_long_np(buffer: Any, name: str, length: int) -> np.ndarray | None:
    """Return ``buffer.fields[name][:length]`` flattened to int64 numpy, or None."""
    if name not in buffer.fields:
        return None
    return buffer.fields[name][:length].reshape(-1).long().cpu().numpy()


def _flat_float_np(buffer: Any, name: str, length: int) -> np.ndarray | None:
    """Return ``buffer.fields[name][:length]`` flattened to float32 numpy, or None."""
    if name not in buffer.fields:
        return None
    return buffer.fields[name][:length].reshape(-1).float().cpu().numpy()


def _mi_z_vs(z: np.ndarray, K: int, x: np.ndarray | None, n_x: int) -> float:
    """Plug-in MI(z; x) in nats. Returns 0.0 when ``x`` is missing or empty."""
    if x is None:
        return 0.0
    valid = (z >= 0) & (z < K) & (x >= 0) & (x < n_x)
    if not bool(valid.any()):
        return 0.0
    idx = z[valid].astype(np.int64) * n_x + x[valid].astype(np.int64)
    joint = np.bincount(idx, minlength=K * n_x).reshape(K, n_x).astype(np.float64)
    return float(discrete_mi_plugin(joint))


def _bucket_z_fracs(
    out: dict[str, float],
    z: np.ndarray,
    K: int,
    bucket: np.ndarray,
    n_buckets: int,
    key: Callable[[int, int], str],
) -> None:
    """Write ``out[key(b, k)] = P(z=k | bucket=b)`` for every (b, k); zeros when empty."""
    for b in range(n_buckets):
        mask = bucket == b
        if bool(mask.any()):
            z_sub = np.clip(z[mask], 0, K - 1)
            for k in range(K):
                out[key(b, k)] = float((z_sub == k).mean())
        else:
            for k in range(K):
                out[key(b, k)] = 0.0


def _fill_zero_z_fracs(
    out: dict[str, float], K: int, n_buckets: int, key: Callable[[int, int], str]
) -> None:
    """Default branch for ``_bucket_z_fracs`` when the bucket field is absent."""
    for b in range(n_buckets):
        for k in range(K):
            out[key(b, k)] = 0.0


def _shannon_entropy_nats(arr: np.ndarray | None, num_categories: int) -> float:
    """Plug-in Shannon entropy in nats. Returns 0.0 when ``arr`` is missing/empty."""
    if arr is None or arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=num_categories).astype(np.float64)
    total = counts.sum()
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


# ---------------------------------------------------------------------------
# Public diagnostics. The function names below are imported by the trainer
# (``rl.custom_ppo.ppo_updater``) and by ``tests/test_latent_strategy_alignment``;
# their signatures, return-dict keys, and numeric outputs are part of the
# trainer's CSV contract and must not change.
# ---------------------------------------------------------------------------


def _latent_rollout_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Summarize strategy occupancy and switching for the latest rollout."""
    if not trainer.use_latent_strategy or "z" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    z = buffer.fields["z"][:length].reshape(-1).long()
    prev_z = buffer.fields["prev_z"][:length].reshape(-1).long()
    if z.numel() == 0:
        return {}
    counts = torch.bincount(z.clamp(min=0, max=trainer.latent_k - 1), minlength=trainer.latent_k).float()
    occupancy = counts / counts.sum().clamp_min(1.0)
    persist_mask = buffer.fields["z_persist_mask"][:length].reshape(-1).bool()
    resampled = buffer.fields["z_resampled"][:length].reshape(-1).bool()
    switched = persist_mask & (z != prev_z)
    out = {
        "strategy_unique_count": float((counts > 0).sum().detach().cpu().item()),
        "strategy_dominant": float(torch.argmax(counts).detach().cpu().item()),
        "strategy_switch_count": float(switched.sum().detach().cpu().item()),
        "strategy_switch_fraction": float(
            (switched.float().sum() / persist_mask.float().sum().clamp_min(1.0)).detach().cpu().item()
        ),
        "strategy_resample_count": float(resampled.sum().detach().cpu().item()),
        "strategy_resample_fraction_rollout": float(resampled.float().mean().detach().cpu().item()),
    }
    for idx, value in enumerate(occupancy.detach().cpu().tolist()):
        out[f"strategy_occupancy_{idx}"] = float(value)
    return out


def _q_phi_probs_and_entropy(
    buffer: Any, length: int, K: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Softmax probs and per-step entropy of stored q_phi logits, or (None, None)."""
    if "z_logits" not in buffer.fields:
        return None, None
    logits = buffer.fields["z_logits"][:length].reshape(-1, K).float()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return probs.detach().cpu().numpy(), entropy.detach().cpu().numpy()


def _flag_state_per_step(buffer: Any, length: int) -> np.ndarray:
    """Encode (blue_has_red_flag, red_has_blue_flag) as a 4-valued bucket per step."""
    gs_all = buffer.fields["global_state"][:length].cpu().numpy()
    gs_flat = gs_all.reshape(-1, gs_all.shape[-1])
    blue_cap = (gs_flat[:, 10] > 0.5).astype(np.int64)
    red_cap = (gs_flat[:, 11] > 0.5).astype(np.int64)
    return blue_cap + 2 * red_cap


def _phase_block(
    out: dict[str, float],
    z: np.ndarray,
    K: int,
    sw: np.ndarray,
    ba: np.ndarray | None,
    rsp_bin: np.ndarray | None,
    pid: np.ndarray,
    q_probs: np.ndarray | None,
    q_entropy: np.ndarray | None,
) -> None:
    """Per-phase z fractions, switch / blue-ahead / capture means, and q_phi means."""
    for p in range(len(TEAM_PHASES)):
        mask = pid == p
        if not bool(mask.any()):
            for k in range(K):
                out[f"latent_phase{p}_z{k}_frac"] = 0.0
            out[f"latent_phase{p}_switch_mean"] = 0.0
            out[f"latent_phase{p}_blue_ahead_mean"] = 0.0
            out[f"latent_phase{p}_capture_step_mean"] = 0.0
            out[f"q_phi_phase{p}_entropy_mean"] = 0.0
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
            continue

        z_sub = np.clip(z[mask], 0, K - 1)
        for k in range(K):
            out[f"latent_phase{p}_z{k}_frac"] = float((z_sub == k).mean())
        out[f"latent_phase{p}_switch_mean"] = float(sw[mask].mean())
        out[f"latent_phase{p}_blue_ahead_mean"] = float(ba[mask].mean()) if ba is not None else 0.0
        out[f"latent_phase{p}_capture_step_mean"] = (
            float(rsp_bin[mask].mean()) if rsp_bin is not None else 0.0
        )

        if q_probs is None or q_entropy is None:
            out[f"q_phi_phase{p}_entropy_mean"] = 0.0
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
        else:
            out[f"q_phi_phase{p}_entropy_mean"] = float(q_entropy[mask].mean())
            q_phase = q_probs[mask]
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = float(q_phase[:, k].mean())


def _reward_sum_after_switch_5(out: dict[str, float], buffer: Any, length: int) -> None:
    """Mean of the next-5-step reward sum following a z-switch (excludes terminal switches)."""
    rewards = buffer.fields["rewards"][:length].detach().cpu().numpy()
    z_tb = buffer.fields["z"][:length].detach().cpu().numpy()
    pz_tb = buffer.fields["prev_z"][:length].detach().cpu().numpy()
    Tn, Bn = int(z_tb.shape[0]), int(z_tb.shape[1])
    sums: list[float] = []
    for t in range(Tn):
        h = min(5, Tn - 1 - t)
        if h <= 0:
            continue
        for b in range(Bn):
            if int(z_tb[t, b]) != int(pz_tb[t, b]):
                sums.append(float(rewards[t + 1 : t + 1 + h, b].sum()))
    out["latent_reward_sum_5_after_z_switch_mean"] = float(np.mean(sums)) if sums else 0.0


def _flag_return_indices(
    blue_cap_col: np.ndarray, red_cap_col: np.ndarray, abs_rsp_col: np.ndarray
) -> np.ndarray:
    """Time indices where a flag was returned (carrier dropped it) without scoring."""
    if blue_cap_col.shape[0] < 2:
        return np.empty(0, dtype=np.int64)
    # blue_cap is bool; for the time-stepped diff we need integer logic against ~.
    blue_ret = blue_cap_col[:-1] & ~blue_cap_col[1:]
    red_ret = red_cap_col[:-1] & ~red_cap_col[1:]
    no_score = abs_rsp_col[1:] < 1.0
    hit = (blue_ret | red_ret) & no_score
    return np.where(hit)[0] + 1  # +1 because index t in original referred to the second step.


def _switch_proximity_fracs(out: dict[str, float], buffer: Any, length: int) -> None:
    """Fraction of z-switches within 3 steps of a capture / kill / flag-return event."""
    rsp = buffer.fields["reward_sparse_points"][:length].cpu().numpy()
    z_env = buffer.fields["z"][:length].cpu().numpy()
    pz_env = buffer.fields["prev_z"][:length].cpu().numpy()
    persist_env = buffer.fields["z_persist_mask"][:length].cpu().numpy()
    sw_env = persist_env & (z_env != pz_env)
    total = float(sw_env.sum())
    if total <= 0.0:
        out["latent_switch_near_capture_frac"] = 0.0
        out["latent_switch_near_kill_frac"] = 0.0
        out["latent_switch_near_return_frac"] = 0.0
        return

    gs_env = buffer.fields["global_state"][:length].cpu().numpy()
    blue_cap_env = gs_env[:, :, 10] > 0.5
    red_cap_env = gs_env[:, :, 11] > 0.5
    near_capture = near_kill = near_return = 0.0
    for b in range(int(buffer.n_envs)):
        switch_idx = np.where(sw_env[:, b])[0]
        if switch_idx.size == 0:
            continue
        abs_rsp = np.abs(rsp[:, b])
        capture_idx = np.where(abs_rsp > 50.0)[0]
        kill_idx = np.where((abs_rsp > 1.0) & (abs_rsp < 40.0))[0]
        return_idx = _flag_return_indices(blue_cap_env[:, b], red_cap_env[:, b], abs_rsp)
        for idx in switch_idx:
            if capture_idx.size and int(np.min(np.abs(capture_idx - idx))) <= 3:
                near_capture += 1.0
            if kill_idx.size and int(np.min(np.abs(kill_idx - idx))) <= 3:
                near_kill += 1.0
            if return_idx.size and int(np.min(np.abs(return_idx - idx))) <= 3:
                near_return += 1.0
    out["latent_switch_near_capture_frac"] = near_capture / total
    out["latent_switch_near_kill_frac"] = near_kill / total
    out["latent_switch_near_return_frac"] = near_return / total


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
    assert z is not None and prev_z is not None  # guarded by the "z in fields" check above
    sw = (z != prev_z).astype(np.float64)
    out: dict[str, float] = {}

    q_probs, q_entropy = _q_phi_probs_and_entropy(buffer, length, K)

    # --- MI(z; categorical context fields) plus per-opponent occupancy --------
    oid = _flat_long_np(buffer, "opponent_id", length)
    if oid is not None:
        out["latent_mi_z_opponent_nats"] = _mi_z_vs(z, K, oid, SCRIPTED_OPPONENT_MI_COUNT)
        _bucket_z_fracs(out, z, K, oid, SCRIPTED_OPPONENT_MI_COUNT,
                        lambda o, k: f"strategy_occupancy_op{o}_z{k}")
    else:
        out["latent_mi_z_opponent_nats"] = 0.0

    pid = _flat_long_np(buffer, "phase_id", length)
    out["latent_mi_z_phase_nats"] = _mi_z_vs(z, K, pid, len(TEAM_PHASES))

    yid = _flat_long_np(buffer, "outcome_id", length)
    out["latent_mi_z_outcome_nats"] = _mi_z_vs(z, K, yid, 3)

    # --- Phase-conditioned rollups + ahead/trail switch rates ----------------
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

    # --- MI(z; situation bucket) for spread / role / pressure / ADR ----------
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

    # --- Flag-state derived from global_state --------------------------------
    flag_state = _flag_state_per_step(buffer, length)
    out["latent_mi_z_flag_state_nats"] = _mi_z_vs(z, K, flag_state, 4)
    _bucket_z_fracs(out, z, K, flag_state, 4, lambda f, k: f"latent_flag_state{f}_z{k}_frac")

    # --- Per-bucket occupancy distributions (spread, ADR) --------------------
    if sb is not None:
        _bucket_z_fracs(out, z, K, sb, 3, lambda s, k: f"latent_spread{s}_z{k}_frac")
    else:
        _fill_zero_z_fracs(out, K, 3, lambda s, k: f"latent_spread{s}_z{k}_frac")

    if adb is not None:
        _bucket_z_fracs(out, z, K, adb, 3, lambda a, k: f"latent_adr{a}_z{k}_frac")
    else:
        _fill_zero_z_fracs(out, K, 3, lambda a, k: f"latent_adr{a}_z{k}_frac")

    # --- Per-phase Shannon entropy over z ------------------------------------
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

    # --- Overall diversity (Shannon entropy of the bucket distribution) ------
    out["latent_role_diversity"] = _shannon_entropy_nats(rb, n_rb)
    out["latent_spread_diversity"] = _shannon_entropy_nats(sb, n_sb)
    out["latent_pressure_diversity"] = _shannon_entropy_nats(pb, n_pb)
    out["latent_adr_diversity"] = _shannon_entropy_nats(adb, n_adr)

    _switch_proximity_fracs(out, buffer, length)

    return out


def _behavior_diversity_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Post-hoc behavior spread by sampled z; diagnostics only, no labels or losses."""
    if not trainer.use_latent_strategy or "z" not in buffer.fields or "behavior_telemetry" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    z = buffer.fields["z"][:length].reshape(-1).long()
    beh = buffer.fields["behavior_telemetry"][:length].reshape(-1, N_TELEMETRY).float()
    out: dict[str, float] = {}
    means: list[torch.Tensor] = []
    for k in range(int(trainer.latent_k)):
        mask = z == k
        if bool(mask.any().item()):
            mean_k = beh[mask].mean(dim=0)
            means.append(mean_k)
        else:
            mean_k = torch.zeros((N_TELEMETRY,), dtype=torch.float32, device=beh.device)
        for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
            out[f"latent_z{k}_behavior_{name}_mean"] = float(mean_k[j].detach().cpu().item())

    if len(means) >= 2:
        pairwise: list[float] = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                pairwise.append(float(torch.linalg.vector_norm(means[i] - means[j]).detach().cpu().item()))
        out["latent_behavior_diversity_l2_mean"] = float(np.mean(pairwise)) if pairwise else 0.0
    else:
        out["latent_behavior_diversity_l2_mean"] = 0.0
    return out


def _macro_probs_from_logits(trainer: Any, logits: torch.Tensor) -> torch.Tensor:
    """Return macro-action probabilities with shape (B, n_agents, macro_dim)."""
    macro_chunks: list[torch.Tensor] = []
    offset = 0
    for _agent_idx in range(int(trainer.model.n_agents)):
        for head_idx in range(int(trainer.model.heads_per_agent)):
            dim = int(trainer.model.per_agent_action_dims[head_idx])
            chunk = logits[:, offset : offset + dim]
            if head_idx == 0:
                macro_chunks.append(torch.softmax(chunk, dim=-1))
            offset += dim
    if not macro_chunks:
        raise AssertionError("could not find macro-action heads for forced-z profiling")
    return torch.stack(macro_chunks, dim=1)


def _forced_z_behavior_profile(trainer: Any, buffer: Any) -> dict[str, float]:
    """Profile actor macro preferences under every forced z on the same rollout observations."""
    if not trainer.use_latent_strategy:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    total = length * int(buffer.n_envs)
    if total <= 0:
        return {}
    if total > FORCED_Z_PROFILE_MAX_ROWS:
        row_idx = torch.linspace(
            0,
            total - 1,
            steps=FORCED_Z_PROFILE_MAX_ROWS,
            device=trainer.device,
        ).long()
    else:
        row_idx = torch.arange(total, device=trainer.device)
    obs_batch = {
        "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, row_idx),
        "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, row_idx),
        "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, row_idx),
        "mask": buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, row_idx),
    }
    out: dict[str, float] = {}
    mean_macros: list[torch.Tensor] = []
    with torch.no_grad():
        for z_id in range(int(trainer.latent_k)):
            z_idx = torch.full((int(row_idx.numel()),), z_id, dtype=torch.long, device=trainer.device)
            logits = trainer.model.policy_logits(obs_batch, z_idx=z_idx)
            logits = trainer.model._mask_logits(logits, obs_batch.get("mask"))
            macro_probs = _macro_probs_from_logits(trainer, logits)
            mean_macro = macro_probs.mean(dim=(0, 1))
            mean_macros.append(mean_macro)
            macro_entropy = -(
                macro_probs.clamp_min(1e-8) * macro_probs.clamp_min(1e-8).log()
            ).sum(dim=-1).mean()
            for action_id, action_name in FORCED_Z_MACRO_ACTIONS:
                if action_id < int(mean_macro.numel()):
                    out[f"forced_z{z_id}_macro_{action_name}_prob"] = float(mean_macro[action_id].detach().cpu().item())
                else:
                    out[f"forced_z{z_id}_macro_{action_name}_prob"] = 0.0
            out[f"forced_z{z_id}_macro_entropy"] = float(macro_entropy.detach().cpu().item())

    if len(mean_macros) >= 2:
        js_vals: list[float] = []
        for i in range(len(mean_macros)):
            for j in range(i + 1, len(mean_macros)):
                p = mean_macros[i].clamp_min(1e-8)
                q = mean_macros[j].clamp_min(1e-8)
                m = 0.5 * (p + q)
                js = 0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()
                js_vals.append(float(js.detach().cpu().item()))
        out["forced_z_macro_jsd_mean"] = float(np.mean(js_vals)) if js_vals else 0.0
    else:
        out["forced_z_macro_jsd_mean"] = 0.0
    return out


def _strategy_resample_advantage_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Per-z mean/std of raw GAE advantages at z-resample steps (pre-minibatch normalization)."""
    if not trainer.use_latent_strategy or trainer.fixed_latent_strategy:
        return {}
    length = int(buffer.pos)
    if length <= 0 or "advantages" not in buffer.fields or "z" not in buffer.fields:
        return {}
    adv = buffer.fields["advantages"][:length]
    z = buffer.fields["z"][:length].long()
    rs = buffer.fields["z_resampled"][:length].bool()
    flat_adv = adv.reshape(-1).float()
    flat_z = z.reshape(-1)
    flat_rs = rs.reshape(-1)
    out: dict[str, float] = {}
    K = int(trainer.latent_k)
    for k in range(K):
        m = flat_rs & (flat_z == k)
        n = int(m.sum().item())
        out[f"strategy_resample_adv_n_z{k}"] = float(n)
        if n > 0:
            vals = flat_adv[m]
            out[f"strategy_resample_adv_mean_z{k}"] = float(vals.mean().item())
            out[f"strategy_resample_adv_std_z{k}"] = (
                float(vals.std(unbiased=False).item()) if n > 1 else 0.0
            )
        else:
            out[f"strategy_resample_adv_mean_z{k}"] = 0.0
            out[f"strategy_resample_adv_std_z{k}"] = 0.0
    return out


def _rollout_advantage_diagnostics(trainer: Any, buffer: Any) -> dict[str, float]:
    """Raw GAE advantage scale and split at latent z-segment starts (t>0, z[t]!=z[t-1])."""
    length = int(buffer.pos)
    if length <= 0 or "advantages" not in buffer.fields:
        return {}
    adv = buffer.fields["advantages"][:length].detach().float()
    flat = adv.reshape(-1)
    out: dict[str, float] = {
        "rollout_adv_std": float(flat.std(unbiased=False).item()) if flat.numel() > 1 else 0.0,
    }
    if (
        not trainer.use_latent_strategy
        or trainer.fixed_latent_strategy
        or "z" not in buffer.fields
        or length < 2
    ):
        return out
    z = buffer.fields["z"][:length].long()
    z_switch = torch.zeros((length, z.shape[1]), dtype=torch.bool, device=z.device)
    z_switch[1:] = z[1:] != z[:-1]
    flat_sw = z_switch.reshape(-1)
    if flat_sw.any() and (~flat_sw).any():
        out["rollout_adv_std_at_z_switch"] = float(flat[flat_sw].std(unbiased=False).item())
        out["rollout_adv_std_not_z_switch"] = float(flat[~flat_sw].std(unbiased=False).item())
    else:
        out["rollout_adv_std_at_z_switch"] = float(out["rollout_adv_std"])
        out["rollout_adv_std_not_z_switch"] = float(out["rollout_adv_std"])
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
    """Write one CSV row per finalized v3i3 refresh event for this rollout.

    Schema (see ``csv_writers._v3i3_refresh_log_fieldnames``):
        update, run_id, run_pid, timesteps,
        env_id, episode_id, decision_step, reason_id, reason,
        prev_z, next_z, opponent_id, flag_state_bucket,
        return_at_refresh, return_from_now_to_end

    No-op when v3i3 logging is disabled, the path is empty, or no refresh
    records were finalized this rollout. Returns a stats dict the caller
    can merge into ``last_stats`` for surfacing in the per-update metrics.
    """
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


_ZERO_OPT_ADV: dict[str, float] = {
    "latent_q_phi_option_advantage_mean": 0.0,
    "latent_q_phi_option_advantage_std": 0.0,
    "latent_q_phi_option_advantage_count": 0.0,
}


def _latent_option_advantage_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Calculate mean, std, and count of option advantages at resampled steps."""
    if not trainer.use_latent_strategy or trainer.fixed_latent_strategy:
        return dict(_ZERO_OPT_ADV)
    length = int(buffer.pos)
    if length <= 0 or "option_advantages" not in buffer.fields or "z_resampled" not in buffer.fields:
        return dict(_ZERO_OPT_ADV)

    opt_adv = buffer.fields["option_advantages"][:length].reshape(-1).float()
    rs = buffer.fields["z_resampled"][:length].reshape(-1).bool()
    vals = opt_adv[rs]
    count = int(vals.numel())
    if count == 0:
        return dict(_ZERO_OPT_ADV)
    return {
        "latent_q_phi_option_advantage_mean": float(vals.mean().item()),
        "latent_q_phi_option_advantage_std": (
            float(vals.std(unbiased=False).item()) if count > 1 else 0.0
        ),
        "latent_q_phi_option_advantage_count": float(count),
    }


def _policy_z_sensitivity_kl(trainer: Any, buffer: Any) -> dict[str, float]:
    """Audit policy z-sensitivity KL divergence across different latent strategy assignments."""
    if not trainer.use_latent_strategy or trainer.latent_k <= 1:
        return {"policy_z_sensitivity_KL": 0.0}
    length = int(buffer.pos)
    if length <= 0:
        return {"policy_z_sensitivity_KL": 0.0}
    total = length * int(buffer.n_envs)
    if total <= 0:
        return {"policy_z_sensitivity_KL": 0.0}

    from rl.custom_ppo.inference import FORCED_Z_PROFILE_MAX_ROWS
    if total > FORCED_Z_PROFILE_MAX_ROWS:
        row_idx = torch.linspace(
            0,
            total - 1,
            steps=FORCED_Z_PROFILE_MAX_ROWS,
            device=trainer.device,
        ).long()
    else:
        row_idx = torch.arange(total, device=trainer.device)

    obs_batch = {
        "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, row_idx),
        "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, row_idx),
        "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, row_idx),
        "mask": buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, row_idx),
    }

    dists_by_z = []
    with torch.no_grad():
        for z_id in range(int(trainer.latent_k)):
            z_idx = torch.full((int(row_idx.numel()),), z_id, dtype=torch.long, device=trainer.device)
            logits = trainer.model.policy_logits(obs_batch, z_idx=z_idx)
            logits = trainer.model._mask_logits(logits, obs_batch.get("mask"))
            dists = list(trainer.model._categoricals(logits))
            dists_by_z.append(dists)

    kl_values = []
    K = int(trainer.latent_k)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            dists_i = dists_by_z[i]
            dists_j = dists_by_z[j]
            kl_sum = torch.zeros((int(row_idx.numel()),), device=trainer.device)
            for di, dj in zip(dists_i, dists_j):
                kl_sum += torch.distributions.kl.kl_divergence(di, dj)
            kl_values.append(float(kl_sum.mean().item()))

    mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
    return {"policy_z_sensitivity_KL": mean_kl}
