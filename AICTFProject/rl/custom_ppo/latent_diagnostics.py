from __future__ import annotations

import csv
import math
import os
from typing import Any

import numpy as np
import torch

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES, N_TELEMETRY, N_ROLE_BUCKET_MI, N_ATTACK_DEFENSE_RATIO_BUCKET
from rl.discrete_mi import discrete_mi_plugin
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_phase_labels import TEAM_PHASES
from rl.custom_ppo.inference import FORCED_Z_PROFILE_MAX_ROWS, FORCED_Z_MACRO_ACTIONS
from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT, _strategy_experience_fieldnames, _ensure_additive_csv_header


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


def _latent_opponent_rollout_diag(trainer: Any, buffer: Any) -> dict[str, float]:
    """Per-opponent z occupancy plus MI(z; opponent/phase/outcome) and phase / behavior bucket rollups."""
    if not trainer.use_latent_strategy or "z" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    z = buffer.fields["z"][:length].reshape(-1).long().cpu().numpy()
    K = int(trainer.latent_k)
    out: dict[str, float] = {}
    q_probs_np: Optional[np.ndarray] = None
    q_entropy_np: Optional[np.ndarray] = None
    if "z_logits" in buffer.fields:
        z_logits_t = buffer.fields["z_logits"][:length].reshape(-1, K).float()
        q_probs_t = torch.softmax(z_logits_t, dim=-1)
        q_entropy_t = -(q_probs_t.clamp_min(1e-8) * q_probs_t.clamp_min(1e-8).log()).sum(dim=-1)
        q_probs_np = q_probs_t.detach().cpu().numpy()
        q_entropy_np = q_entropy_t.detach().cpu().numpy()

    if "opponent_id" in buffer.fields:
        oid = buffer.fields["opponent_id"][:length].reshape(-1).long().cpu().numpy()
        joint = np.zeros((K, SCRIPTED_OPPONENT_MI_COUNT), dtype=np.float64)
        for i in range(z.size):
            zi = int(z[i])
            oi = int(oid[i])
            if 0 <= zi < K and 0 <= oi < SCRIPTED_OPPONENT_MI_COUNT:
                joint[zi, oi] += 1.0
        out["latent_mi_z_opponent_nats"] = float(discrete_mi_plugin(joint))
        for o in range(SCRIPTED_OPPONENT_MI_COUNT):
            mask = oid == o
            if not np.any(mask):
                for k in range(K):
                    out[f"strategy_occupancy_op{o}_z{k}"] = 0.0
                continue
            z_sub = np.clip(z[mask], 0, K - 1)
            cnt = np.bincount(z_sub, minlength=K).astype(np.float64)
            total = float(cnt.sum())
            occ = cnt / max(total, 1.0)
            for k in range(K):
                out[f"strategy_occupancy_op{o}_z{k}"] = float(occ[k])
    else:
        out["latent_mi_z_opponent_nats"] = 0.0

    pid_flat: Optional[np.ndarray] = None
    if "phase_id" in buffer.fields:
        pid_flat = buffer.fields["phase_id"][:length].reshape(-1).long().cpu().numpy()
        n_p = len(TEAM_PHASES)
        joint_p = np.zeros((K, n_p), dtype=np.float64)
        for i in range(z.size):
            zi = int(z[i])
            pi = int(pid_flat[i])
            if 0 <= zi < K and 0 <= pi < n_p:
                joint_p[zi, pi] += 1.0
        out["latent_mi_z_phase_nats"] = float(discrete_mi_plugin(joint_p))
    else:
        out["latent_mi_z_phase_nats"] = 0.0

    if "outcome_id" in buffer.fields:
        yid = buffer.fields["outcome_id"][:length].reshape(-1).long().cpu().numpy()
        joint_y = np.zeros((K, 3), dtype=np.float64)
        for i in range(z.size):
            zi = int(z[i])
            yi = int(yid[i])
            if 0 <= zi < K and 0 <= yi < 3:
                joint_y[zi, yi] += 1.0
        out["latent_mi_z_outcome_nats"] = float(discrete_mi_plugin(joint_y))
    else:
        out["latent_mi_z_outcome_nats"] = 0.0

    prev_np = buffer.fields["prev_z"][:length].reshape(-1).long().cpu().numpy()
    sw = (z != prev_np).astype(np.float64)
    ba: Optional[np.ndarray] = None
    if "blue_ahead" in buffer.fields:
        ba = buffer.fields["blue_ahead"][:length].reshape(-1).float().cpu().numpy()
    rsp_bin: Optional[np.ndarray] = None
    if "reward_sparse_points" in buffer.fields:
        rsp_bin = (
            (np.abs(buffer.fields["reward_sparse_points"][:length].reshape(-1).float().cpu().numpy()) > 1e-5)
            .astype(np.float64)
        )

    if pid_flat is not None:
        n_p = len(TEAM_PHASES)
        for p in range(n_p):
            mask = pid_flat == p
            cnt_m = float(mask.sum())
            if cnt_m <= 0.0:
                for k in range(K):
                    out[f"latent_phase{p}_z{k}_frac"] = 0.0
                out[f"latent_phase{p}_switch_mean"] = 0.0
                out[f"latent_phase{p}_blue_ahead_mean"] = 0.0
                out[f"latent_phase{p}_capture_step_mean"] = 0.0
            else:
                z_sub = np.clip(z[mask], 0, K - 1)
                for k in range(K):
                    out[f"latent_phase{p}_z{k}_frac"] = float((z_sub == k).mean())
                out[f"latent_phase{p}_switch_mean"] = float(sw[mask].mean())
                out[f"latent_phase{p}_blue_ahead_mean"] = (
                    float(ba[mask].mean()) if ba is not None else 0.0
                )
                out[f"latent_phase{p}_capture_step_mean"] = (
                    float(rsp_bin[mask].mean()) if rsp_bin is not None else 0.0
                )
            if q_entropy_np is None or q_probs_np is None or not np.any(mask):
                out[f"q_phi_phase{p}_entropy_mean"] = 0.0
                for k in range(K):
                    out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
            else:
                out[f"q_phi_phase{p}_entropy_mean"] = float(q_entropy_np[mask].mean())
                q_phase = q_probs_np[mask]
                for k in range(K):
                    out[f"q_phi_phase{p}_z{k}_prob_mean"] = float(q_phase[:, k].mean())

    if ba is not None:
        ahead = ba > 0.5
        trail = ~ahead
        out["latent_switch_rate_blue_ahead"] = float(sw[ahead].mean()) if bool(ahead.any()) else 0.0
        out["latent_switch_rate_blue_trail"] = float(sw[trail].mean()) if bool(trail.any()) else 0.0
    else:
        out["latent_switch_rate_blue_ahead"] = 0.0
        out["latent_switch_rate_blue_trail"] = 0.0

    Rtb = buffer.fields["rewards"][:length].detach().cpu().numpy()
    Ztb = buffer.fields["z"][:length].detach().cpu().numpy()
    Ptb = buffer.fields["prev_z"][:length].detach().cpu().numpy()
    Tn, Bn = int(Ztb.shape[0]), int(Ztb.shape[1])
    sums: list[float] = []
    for t in range(Tn):
        for b in range(Bn):
            if int(Ztb[t, b]) != int(Ptb[t, b]):
                h = min(5, Tn - 1 - t)
                if h > 0:
                    sums.append(float(Rtb[t + 1 : t + 1 + h, b].sum()))
    out["latent_reward_sum_5_after_z_switch_mean"] = float(np.mean(sums)) if sums else 0.0

    n_sb, n_rb, n_pb = 3, int(N_ROLE_BUCKET_MI), 3
    n_adr = int(N_ATTACK_DEFENSE_RATIO_BUCKET)

    if "spread_bucket_id" in buffer.fields:
        sb = buffer.fields["spread_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        j = np.zeros((K, n_sb), dtype=np.float64)
        for i in range(z.size):
            zi, si = int(z[i]), int(sb[i])
            if 0 <= zi < K and 0 <= si < n_sb:
                j[zi, si] += 1.0
        out["latent_mi_z_spread_bucket_nats"] = float(discrete_mi_plugin(j))
    else:
        out["latent_mi_z_spread_bucket_nats"] = 0.0

    if "role_bucket_id" in buffer.fields:
        rb = buffer.fields["role_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        j = np.zeros((K, n_rb), dtype=np.float64)
        for i in range(z.size):
            zi, ri = int(z[i]), int(rb[i])
            if 0 <= zi < K and 0 <= ri < n_rb:
                j[zi, ri] += 1.0
        out["latent_mi_z_role_bucket_nats"] = float(discrete_mi_plugin(j))
        for r in range(n_rb):
            mask = rb == r
            if not np.any(mask):
                for k_idx in range(K):
                    out[f"latent_role{r}_z{k_idx}_frac"] = 0.0
                out[f"latent_role{r}_switch_mean"] = 0.0
            else:
                z_sub = np.clip(z[mask], 0, K - 1)
                for k_idx in range(K):
                    out[f"latent_role{r}_z{k_idx}_frac"] = float((z_sub == k_idx).mean())
                out[f"latent_role{r}_switch_mean"] = float(sw[mask].mean())
    else:
        out["latent_mi_z_role_bucket_nats"] = 0.0
        for r in range(n_rb):
            for k_idx in range(K):
                out[f"latent_role{r}_z{k_idx}_frac"] = 0.0
            out[f"latent_role{r}_switch_mean"] = 0.0

    if "pressure_bucket_id" in buffer.fields:
        pb = buffer.fields["pressure_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        j = np.zeros((K, n_pb), dtype=np.float64)
        for i in range(z.size):
            zi, pi2 = int(z[i]), int(pb[i])
            if 0 <= zi < K and 0 <= pi2 < n_pb:
                j[zi, pi2] += 1.0
        out["latent_mi_z_pressure_bucket_nats"] = float(discrete_mi_plugin(j))
    else:
        out["latent_mi_z_pressure_bucket_nats"] = 0.0

    if "attack_defense_ratio_bucket_id" in buffer.fields:
        adb = buffer.fields["attack_defense_ratio_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        j = np.zeros((K, n_adr), dtype=np.float64)
        for i in range(z.size):
            zi, ai = int(z[i]), int(adb[i])
            if 0 <= zi < K and 0 <= ai < n_adr:
                j[zi, ai] += 1.0
        out["latent_mi_z_attack_defense_ratio_bucket_nats"] = float(discrete_mi_plugin(j))
    else:
        out["latent_mi_z_attack_defense_ratio_bucket_nats"] = 0.0

    def shannon_entropy(arr, num_categories):
        if arr.size == 0:
            return 0.0
        counts = np.bincount(arr, minlength=num_categories).astype(np.float64)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    gs_all = buffer.fields["global_state"][:length].cpu().numpy()
    gs_flat = gs_all.reshape(-1, gs_all.shape[-1])
    gs_raw = gs_flat[:, :19]
    blue_flag_captured = (gs_raw[:, 10] > 0.5).astype(int)
    red_flag_captured = (gs_raw[:, 11] > 0.5).astype(int)
    flag_state = blue_flag_captured + 2 * red_flag_captured
    joint_f = np.zeros((K, 4), dtype=np.float64)
    for i in range(z.size):
        zi = int(z[i])
        fi = int(flag_state[i])
        if 0 <= zi < K and 0 <= fi < 4:
            joint_f[zi, fi] += 1.0
    out["latent_mi_z_flag_state_nats"] = float(discrete_mi_plugin(joint_f))
    
    for f in range(4):
        f_mask = (flag_state == f)
        if not np.any(f_mask):
            for k in range(K):
                out[f"latent_flag_state{f}_z{k}_frac"] = 0.0
        else:
            z_f = np.clip(z[f_mask], 0, K - 1)
            for k in range(K):
                out[f"latent_flag_state{f}_z{k}_frac"] = float((z_f == k).mean())

    if "spread_bucket_id" in buffer.fields:
        sb = buffer.fields["spread_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        for s in range(3):
            s_mask = (sb == s)
            if not np.any(s_mask):
                for k in range(K):
                    out[f"latent_spread{s}_z{k}_frac"] = 0.0
            else:
                z_s = np.clip(z[s_mask], 0, K - 1)
                for k in range(K):
                    out[f"latent_spread{s}_z{k}_frac"] = float((z_s == k).mean())
    else:
        for s in range(3):
            for k in range(K):
                out[f"latent_spread{s}_z{k}_frac"] = 0.0

    if "attack_defense_ratio_bucket_id" in buffer.fields:
        adr = buffer.fields["attack_defense_ratio_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        for a in range(3):
            a_mask = (adr == a)
            if not np.any(a_mask):
                for k in range(K):
                    out[f"latent_adr{a}_z{k}_frac"] = 0.0
            else:
                z_a = np.clip(z[a_mask], 0, K - 1)
                for k in range(K):
                    out[f"latent_adr{a}_z{k}_frac"] = float((z_a == k).mean())
    else:
        for a in range(3):
            for k in range(K):
                out[f"latent_adr{a}_z{k}_frac"] = 0.0

    if pid_flat is not None:
        n_p = len(TEAM_PHASES)
        for p in range(n_p):
            mask = (pid_flat == p)
            if not np.any(mask):
                out[f"latent_phase{p}_entropy"] = 0.0
            else:
                z_sub = np.clip(z[mask], 0, K - 1)
                out[f"latent_phase{p}_entropy"] = shannon_entropy(z_sub, K)
    else:
        for p in range(len(TEAM_PHASES)):
            out[f"latent_phase{p}_entropy"] = 0.0

    if "role_bucket_id" in buffer.fields:
        rb = buffer.fields["role_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        out["latent_role_diversity"] = shannon_entropy(rb, n_rb)
    else:
        out["latent_role_diversity"] = 0.0

    if "spread_bucket_id" in buffer.fields:
        sb = buffer.fields["spread_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        out["latent_spread_diversity"] = shannon_entropy(sb, n_sb)
    else:
        out["latent_spread_diversity"] = 0.0

    if "pressure_bucket_id" in buffer.fields:
        pb = buffer.fields["pressure_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        out["latent_pressure_diversity"] = shannon_entropy(pb, n_pb)
    else:
        out["latent_pressure_diversity"] = 0.0

    if "attack_defense_ratio_bucket_id" in buffer.fields:
        adr = buffer.fields["attack_defense_ratio_bucket_id"][:length].reshape(-1).long().cpu().numpy()
        out["latent_adr_diversity"] = shannon_entropy(adr, n_adr)
    else:
        out["latent_adr_diversity"] = 0.0

    n_envs = int(buffer.n_envs)
    rsp = buffer.fields["reward_sparse_points"][:length].cpu().numpy()
    z_env = buffer.fields["z"][:length].cpu().numpy()
    pz_env = buffer.fields["prev_z"][:length].cpu().numpy()
    persist_env = buffer.fields["z_persist_mask"][:length].cpu().numpy()
    sw_env = persist_env & (z_env != pz_env)
    total_switches = float(sw_env.sum())

    switches_near_capture = 0.0
    switches_near_kill = 0.0
    switches_near_return = 0.0

    if total_switches > 0:
        gs_env = buffer.fields["global_state"][:length].cpu().numpy()
        blue_cap_env = gs_env[:, :, 10] > 0.5
        red_cap_env = gs_env[:, :, 11] > 0.5
        for b in range(n_envs):
            switch_indices = np.where(sw_env[:, b])[0]
            if len(switch_indices) == 0:
                continue
            capture_indices = np.where(np.abs(rsp[:, b]) > 50.0)[0]
            kill_indices = np.where((np.abs(rsp[:, b]) > 1.0) & (np.abs(rsp[:, b]) < 40.0))[0]
            
            blue_cap = blue_cap_env[:, b]
            red_cap = red_cap_env[:, b]
            return_indices = []
            for t in range(1, length):
                blue_ret = blue_cap[t-1] and (not blue_cap[t])
                red_ret = red_cap[t-1] and (not red_cap[t])
                no_score = np.abs(rsp[t, b]) < 1.0
                if (blue_ret or red_ret) and no_score:
                    return_indices.append(t)
            return_indices = np.array(return_indices)

            for idx in switch_indices:
                if len(capture_indices) > 0 and np.min(np.abs(capture_indices - idx)) <= 3:
                    switches_near_capture += 1.0
                if len(kill_indices) > 0 and np.min(np.abs(kill_indices - idx)) <= 3:
                    switches_near_kill += 1.0
                if len(return_indices) > 0 and np.min(np.abs(return_indices - idx)) <= 3:
                    switches_near_return += 1.0

        out["latent_switch_near_capture_frac"] = switches_near_capture / total_switches
        out["latent_switch_near_kill_frac"] = switches_near_kill / total_switches
        out["latent_switch_near_return_frac"] = switches_near_return / total_switches
    else:
        out["latent_switch_near_capture_frac"] = 0.0
        out["latent_switch_near_kill_frac"] = 0.0
        out["latent_switch_near_return_frac"] = 0.0

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
    mask = flat_rs
    out: dict[str, float] = {}
    K = int(trainer.latent_k)
    for k in range(K):
        m = mask & (flat_z == k)
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


def _strategy_experience_bucket_ids(context_state: torch.Tensor) -> torch.Tensor:
    """Coarse post-hoc situation buckets for diagnostics only; never used as training labels."""
    if context_state.dim() != 2:
        raise ValueError(f"context_state must be 2-D, got {tuple(context_state.shape)}")
    raw = context_state[:, :GLOBAL_STATE_DIM].float()
    enemy_has_our_flag = (raw[:, 10] > 0.5).long()
    we_have_enemy_flag = (raw[:, 11] > 0.5).long()
    dist_edges = torch.tensor([0.20, 0.50], dtype=torch.float32, device=raw.device)
    closest_ally_to_enemy_flag = torch.bucketize(raw[:, 8].contiguous(), dist_edges).long().clamp(0, 2)
    closest_enemy_to_our_flag = torch.bucketize(raw[:, 9].contiguous(), dist_edges).long().clamp(0, 2)
    spread = torch.sqrt(torch.clamp(raw[:, 2].pow(2) + raw[:, 3].pow(2), min=0.0))
    spread_bin = (spread > 0.15).long()
    score = raw[:, 16]
    score_state = torch.where(
        score < -0.05,
        torch.zeros_like(score, dtype=torch.long),
        torch.where(score > 0.05, torch.full_like(score, 2, dtype=torch.long), torch.ones_like(score, dtype=torch.long)),
    )
    bucket = enemy_has_our_flag
    bucket = bucket * 2 + we_have_enemy_flag
    bucket = bucket * 3 + closest_ally_to_enemy_flag
    bucket = bucket * 3 + closest_enemy_to_our_flag
    bucket = bucket * 2 + spread_bin
    bucket = bucket * 3 + score_state
    return bucket.long()


def _write_strategy_experience_table(trainer: Any) -> dict[str, float]:
    if not trainer.strategy_experience_csv_path or not trainer.use_latent_strategy or trainer.latent_k <= 0:
        return {"strategy_bucket_best_match_frac": 0.0, "strategy_experience_records": 0.0, "strategy_experience_buckets": 0.0}
    records = list(trainer._rollout_strategy_episode_records)
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


def _latent_option_advantage_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Calculate mean, std, and count of option advantages at resampled steps."""
    if not trainer.use_latent_strategy or trainer.fixed_latent_strategy:
        return {
            "latent_q_phi_option_advantage_mean": 0.0,
            "latent_q_phi_option_advantage_std": 0.0,
            "latent_q_phi_option_advantage_count": 0.0,
        }
    length = int(buffer.pos)
    if length <= 0 or "option_advantages" not in buffer.fields or "z_resampled" not in buffer.fields:
        return {
            "latent_q_phi_option_advantage_mean": 0.0,
            "latent_q_phi_option_advantage_std": 0.0,
            "latent_q_phi_option_advantage_count": 0.0,
        }
    
    opt_adv = buffer.fields["option_advantages"][:length]
    rs = buffer.fields["z_resampled"][:length].bool()
    
    flat_opt_adv = opt_adv.reshape(-1).float()
    flat_rs = rs.reshape(-1)
    
    resampled_opt_adv = flat_opt_adv[flat_rs]
    count = int(resampled_opt_adv.numel())
    
    if count > 0:
        mean_val = float(resampled_opt_adv.mean().item())
        std_val = float(resampled_opt_adv.std(unbiased=False).item()) if count > 1 else 0.0
    else:
        mean_val = 0.0
        std_val = 0.0
        
    return {
        "latent_q_phi_option_advantage_mean": mean_val,
        "latent_q_phi_option_advantage_std": std_val,
        "latent_q_phi_option_advantage_count": float(count),
    }

