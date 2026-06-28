"""Counterfactual latent probing: forced-z profiling, JSD, KL sensitivity."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import torch

from rl.custom_ppo.inference import FORCED_Z_MACRO_ACTIONS, FORCED_Z_PROFILE_MAX_ROWS
from rl.forced_z_behavior_vectors import (
    behavior_vector_from_macro_probs,
    build_behavior_distance_profile,
)


# ---------------------------------------------------------------------------
# Pure mathematical primitives
# ---------------------------------------------------------------------------


def _jsd_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Jensen-Shannon divergence for matching categorical logits."""
    log_p = torch.log_softmax(logits_a.float(), dim=dim)
    log_q = torch.log_softmax(logits_b.float(), dim=dim)
    p = log_p.exp()
    q = log_q.exp()
    mixture = 0.5 * (p + q)
    log_mixture = torch.log(mixture.clamp_min(float(eps)))
    kl_pm = torch.sum(p * (log_p - log_mixture), dim=dim)
    kl_qm = torch.sum(q * (log_q - log_mixture), dim=dim)
    return 0.5 * (kl_pm + kl_qm)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


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


def _batched_policy_trunk_features(
    trainer: Any, obs_batch: dict[str, torch.Tensor], z_idx: torch.Tensor
) -> torch.Tensor:
    """Batched ``policy_trunk_features`` with the same chunking as policy logits."""
    total = z_idx.shape[0]
    batch_size = min(1024, int(getattr(trainer.cfg, "batch_size", 1024)))
    if total <= batch_size:
        return trainer.model.policy_trunk_features(obs_batch, z_idx=z_idx)
    chunks: list[torch.Tensor] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        slice_obs = {k: v[start:end] for k, v in obs_batch.items()}
        slice_z = z_idx[start:end]
        chunks.append(trainer.model.policy_trunk_features(slice_obs, z_idx=slice_z))
    return torch.cat(chunks, dim=0)


def _batched_policy_logits(
    trainer: Any, obs_batch: dict[str, torch.Tensor], z_idx: torch.Tensor
) -> torch.Tensor:
    """Run model.policy_logits in smaller mini-batches to prevent CUDA OOM on large state spaces."""
    total = z_idx.shape[0]
    batch_size = min(1024, int(getattr(trainer.cfg, "batch_size", 1024)))
    if total <= batch_size:
        return trainer.model.policy_logits(obs_batch, z_idx=z_idx)
    logits_list = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        slice_obs = {k: v[start:end] for k, v in obs_batch.items()}
        slice_z = z_idx[start:end]
        slice_logits = trainer.model.policy_logits(slice_obs, z_idx=slice_z)
        logits_list.append(slice_logits)
    return torch.cat(logits_list, dim=0)


# ---------------------------------------------------------------------------
# Trainer/buffer-coupled diagnostics
# ---------------------------------------------------------------------------


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
    row_idx = torch.clamp(row_idx, 0, total - 1)
    obs_batch = {
        "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, row_idx),
        "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, row_idx),
        "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, row_idx),
        "mask": buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, row_idx),
    }
    out: dict[str, float] = {}
    mean_macros: list[torch.Tensor] = []
    behavior_vectors: list[np.ndarray] = []
    with torch.no_grad():
        for z_id in range(int(trainer.latent_k)):
            z_idx = torch.full((int(row_idx.numel()),), z_id, dtype=torch.long, device=trainer.device)
            logits = _batched_policy_logits(trainer, obs_batch, z_idx=z_idx)
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
            behavior_vectors.append(behavior_vector_from_macro_probs(mean_macro))

    out.update(
        build_behavior_distance_profile(
            behavior_vectors,
            source="macro",
            pair_count=int(trainer.latent_k) * (int(trainer.latent_k) - 1) // 2,
            latent_k=int(trainer.latent_k),
        )
    )

    if len(mean_macros) >= 2:
        js_vals: list[float] = []
        pair_idx = 0
        for i in range(len(mean_macros)):
            for j in range(i + 1, len(mean_macros)):
                p = mean_macros[i].clamp_min(1e-8)
                q = mean_macros[j].clamp_min(1e-8)
                m = 0.5 * (p + q)
                js = 0.5 * (p * (p.log() - m.log())).sum() + 0.5 * (q * (q.log() - m.log())).sum()
                js_val = float(js.detach().cpu().item())
                js_vals.append(js_val)
                out[f"forced_z_pair_jsd_{pair_idx}"] = js_val
                pair_idx += 1
        out["forced_z_macro_jsd_mean"] = float(np.mean(js_vals)) if js_vals else 0.0
    else:
        out["forced_z_macro_jsd_mean"] = 0.0
    out["forced_z_macro_jsd"] = out["forced_z_macro_jsd_mean"]
    return out


def _policy_z_sensitivity_kl(trainer: Any, buffer: Any) -> dict[str, Any]:
    """Probe actor behavior under every forced z on the same observations."""
    zero_stats: dict[str, Any] = {
        "policy_z_sensitivity_KL": 0.0,
        "actor_z_jsd_mean": 0.0,
        "actor_z_jsd_min": 0.0,
        "actor_z_jsd_max": 0.0,
        "actor_z_pairs_total": 0.0,
        "actor_z_pairs_above_margin": 0.0,
        "actor_z_pairs_above_margin_fraction": 0.0,
        "actor_z_eval_state_count": 0.0,
        "actor_z_eval_pair_count": 0.0,
        "actor_z_jsd_per_head": "",
        "actor_z_argmax_disagree": 0.0,
        "actor_z_logit_l2": 0.0,
        "actor_z_entropy_by_z": "",
        "actor_z_trunk_l2": 0.0,
        "actor_z_film_mod_l2": 0.0,
    }
    if not trainer.use_latent_strategy or trainer.latent_k <= 1:
        return zero_stats
    length = int(buffer.pos)
    if length <= 0:
        return zero_stats
    total = length * int(buffer.n_envs)
    if total <= 0:
        return zero_stats

    if total > FORCED_Z_PROFILE_MAX_ROWS:
        row_idx = torch.linspace(
            0,
            total - 1,
            steps=FORCED_Z_PROFILE_MAX_ROWS,
            device=trainer.device,
        ).long()
    else:
        row_idx = torch.arange(total, device=trainer.device)
    row_idx = torch.clamp(row_idx, 0, total - 1)

    obs_batch = {
        "grid": buffer.fields["obs_grid"][:length].reshape(total, *buffer.fields["obs_grid"].shape[2:]).index_select(0, row_idx),
        "vec": buffer.fields["obs_vec"][:length].reshape(total, *buffer.fields["obs_vec"].shape[2:]).index_select(0, row_idx),
        "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(total, *buffer.fields["obs_agent_mask"].shape[2:]).index_select(0, row_idx),
        "mask": buffer.fields["obs_mask"][:length].reshape(total, *buffer.fields["obs_mask"].shape[2:]).index_select(0, row_idx),
    }

    logits_by_z: list[torch.Tensor] = []
    trunk_by_z: list[torch.Tensor] = []
    dists_by_z: list[list[torch.distributions.Categorical]] = []
    with torch.no_grad():
        for z_id in range(int(trainer.latent_k)):
            z_idx = torch.full((int(row_idx.numel()),), z_id, dtype=torch.long, device=trainer.device)
            logits = _batched_policy_logits(trainer, obs_batch, z_idx=z_idx)
            logits = trainer.model._mask_logits(logits, obs_batch.get("mask"))
            logits_by_z.append(logits.float())
            trunk_by_z.append(_batched_policy_trunk_features(trainer, obs_batch, z_idx=z_idx).float())
            dists_by_z.append(list(trainer.model._categoricals(logits)))

    kl_values: list[float] = []
    latent_k = int(trainer.latent_k)
    for i in range(latent_k):
        for j in range(latent_k):
            if i == j:
                continue
            dists_i = dists_by_z[i]
            dists_j = dists_by_z[j]
            kl_sum = torch.zeros((int(row_idx.numel()),), device=trainer.device)
            for di, dj in zip(dists_i, dists_j):
                kl_sum += torch.distributions.kl.kl_divergence(di, dj)
            kl_values.append(float(kl_sum.mean().item()))

    mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
    action_dims = tuple(int(dim) for dim in trainer.model.action_dims)
    heads_per_agent = int(trainer.model.heads_per_agent)
    agent_mask = obs_batch.get("agent_mask")
    if agent_mask is None:
        agent_mask = torch.ones(
            (int(row_idx.numel()), int(trainer.model.n_agents)),
            dtype=torch.float32,
            device=trainer.device,
        )
    else:
        agent_mask = agent_mask.to(device=trainer.device).float()

    offsets: list[tuple[int, int]] = []
    offset = 0
    for dim in action_dims:
        offsets.append((offset, offset + dim))
        offset += dim

    entropy_by_z: list[float] = []
    for z_id, dists in enumerate(dists_by_z):
        entropy_values: list[torch.Tensor] = []
        for action_idx, dist in enumerate(dists):
            agent_idx = action_idx // heads_per_agent
            valid = agent_mask[:, agent_idx] > 0.5
            if bool(valid.any()):
                entropy_values.append(dist.entropy()[valid])
        entropy = (
            torch.cat(entropy_values).mean()
            if entropy_values
            else torch.zeros((), device=trainer.device)
        )
        entropy_by_z.append(float(entropy.item()))
        zero_stats[f"actor_z_entropy_z{z_id}"] = float(entropy.item())

    jsd_values: list[torch.Tensor] = []
    argmax_disagreements: list[torch.Tensor] = []
    logit_l2_values: list[torch.Tensor] = []
    jsd_by_head: list[list[torch.Tensor]] = [
        [] for _ in range(heads_per_agent)
    ]
    for i in range(latent_k):
        for j in range(i + 1, latent_k):
            logits_i = logits_by_z[i]
            logits_j = logits_by_z[j]
            for action_idx, (start, end) in enumerate(offsets):
                agent_idx = action_idx // heads_per_agent
                head_idx = action_idx % heads_per_agent
                valid = agent_mask[:, agent_idx] > 0.5
                if not bool(valid.any()):
                    continue
                head_i = logits_i[valid, start:end]
                head_j = logits_j[valid, start:end]
                jsd = _jsd_from_logits(head_i, head_j)
                jsd_values.append(jsd)
                jsd_by_head[head_idx].append(jsd)
                argmax_disagreements.append(
                    (head_i.argmax(dim=-1) != head_j.argmax(dim=-1)).float()
                )
                logit_l2_values.append(
                    torch.linalg.vector_norm(head_i - head_j, dim=-1)
                )

    all_jsd = (
        torch.cat(jsd_values)
        if jsd_values
        else torch.zeros((1,), device=trainer.device)
    )
    pair_count = max(0, latent_k * (latent_k - 1) // 2)
    actor_margin = float(getattr(trainer.cfg, "actor_jsd_margin", 0.001) or 0.001)
    pair_means: list[float] = []
    for i in range(latent_k):
        for j in range(i + 1, latent_k):
            logits_i = logits_by_z[i]
            logits_j = logits_by_z[j]
            values: list[torch.Tensor] = []
            for action_idx, (start, end) in enumerate(offsets):
                agent_idx = action_idx // heads_per_agent
                valid = agent_mask[:, agent_idx] > 0.5
                if bool(valid.any()):
                    values.append(_jsd_from_logits(logits_i[valid, start:end], logits_j[valid, start:end]))
            if values:
                pair_means.append(float(torch.cat(values).mean().detach().cpu().item()))
    pairs_above = sum(1 for value in pair_means if value >= actor_margin)
    all_disagree = (
        torch.cat(argmax_disagreements)
        if argmax_disagreements
        else torch.zeros((1,), device=trainer.device)
    )
    all_logit_l2 = (
        torch.cat(logit_l2_values)
        if logit_l2_values
        else torch.zeros((1,), device=trainer.device)
    )
    per_head_jsd = [
        float(torch.cat(values).mean().item()) if values else 0.0
        for values in jsd_by_head
    ]
    for head_idx, value in enumerate(per_head_jsd):
        zero_stats[f"actor_z_jsd_head_{head_idx}"] = value

    trunk_l2_values: list[torch.Tensor] = []
    for i in range(latent_k):
        for j in range(i + 1, latent_k):
            diff = trunk_by_z[i] - trunk_by_z[j]
            trunk_l2_values.append(torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=-1))
    trunk_l2_mean = (
        float(torch.cat(trunk_l2_values).mean().item())
        if trunk_l2_values
        else 0.0
    )
    latent_actor = getattr(trainer.model, "latent_actor", None)
    film_mod_l2 = 0.0
    if latent_actor is not None and hasattr(latent_actor, "film_modulation_l2"):
        z0 = torch.zeros((1,), dtype=torch.long, device=trainer.device)
        z1 = torch.ones((1,), dtype=torch.long, device=trainer.device)
        film_mod_l2 = float(latent_actor.film_modulation_l2(z0, z1))

    zero_stats.update(
        {
            "policy_z_sensitivity_KL": mean_kl,
            "actor_z_jsd_mean": float(all_jsd.mean().item()),
            "actor_z_jsd_min": float(all_jsd.min().item()),
            "actor_z_jsd_max": float(all_jsd.max().item()),
            "actor_z_pairs_total": float(pair_count),
            "actor_z_pairs_above_margin": float(pairs_above),
            "actor_z_pairs_above_margin_fraction": float(pairs_above) / float(max(1, pair_count)),
            "actor_z_eval_state_count": float(int(row_idx.numel())),
            "actor_z_eval_pair_count": float(pair_count),
            "actor_z_jsd_per_head": ",".join(
                f"{value:.8e}" for value in per_head_jsd
            ),
            "actor_z_argmax_disagree": float(all_disagree.mean().item()),
            "actor_z_logit_l2": float(all_logit_l2.mean().item()),
            "actor_z_entropy_by_z": ",".join(
                f"{value:.8e}" for value in entropy_by_z
            ),
            "actor_z_trunk_l2": trunk_l2_mean,
            "actor_z_film_mod_l2": film_mod_l2,
        }
    )
    return zero_stats


# ---------------------------------------------------------------------------
# V6I7 public API
# ---------------------------------------------------------------------------


def compute_pairwise_actor_jsd(
    model: "SharedActorCentralizedCritic",
    local_features: torch.Tensor,
) -> dict[str, float]:
    """Compute mean/min/max pairwise actor JSD and argmax disagreement.

    ``local_features`` must have shape ``(N, local_feature_dim)`` — a batch of
    pre-encoded local observations (CNN + vec, no z concatenated).  The
    function evaluates all K*(K-1)/2 pairs and returns a flat stats dict
    suitable for the metrics CSV.
    """
    K = int(model.latent_k)
    if K < 2 or not model.uses_latent_strategy:
        return {
            "actor_jsd_mean": float("nan"),
            "actor_jsd_min": float("nan"),
            "actor_jsd_max": float("nan"),
            "actor_argmax_disagree": float("nan"),
        }
    device = local_features.device
    N = local_features.shape[0]

    with torch.no_grad():
        logits_by_z = []
        for k in range(K):
            z_t = torch.full((N,), k, dtype=torch.long, device=device)
            logits_k = model.latent_actor(local_features, z_t)
            logits_by_z.append(logits_k)

    pair_jsds: list[float] = []
    pair_disagrees: list[float] = []
    for i, j in combinations(range(K), 2):
        jsd = _jsd_from_logits(logits_by_z[i], logits_by_z[j]).mean().item()
        pair_jsds.append(float(jsd))
        argmax_i = logits_by_z[i].argmax(dim=-1)
        argmax_j = logits_by_z[j].argmax(dim=-1)
        pair_disagrees.append(float((argmax_i != argmax_j).float().mean().item()))

    return {
        "actor_jsd_mean": float(sum(pair_jsds) / len(pair_jsds)),
        "actor_jsd_min": float(min(pair_jsds)),
        "actor_jsd_max": float(max(pair_jsds)),
        "actor_argmax_disagree": float(sum(pair_disagrees) / len(pair_disagrees)),
    }


jsd_from_logits = _jsd_from_logits
macro_probs_from_logits = _macro_probs_from_logits
batched_policy_logits = _batched_policy_logits
batched_policy_trunk_features = _batched_policy_trunk_features
forced_z_behavior_profile = _forced_z_behavior_profile
policy_z_sensitivity_kl = _policy_z_sensitivity_kl

__all__ = [
    "_jsd_from_logits",
    "_macro_probs_from_logits",
    "_batched_policy_logits",
    "_batched_policy_trunk_features",
    "_forced_z_behavior_profile",
    "_policy_z_sensitivity_kl",
    "jsd_from_logits",
    "macro_probs_from_logits",
    "batched_policy_logits",
    "batched_policy_trunk_features",
    "forced_z_behavior_profile",
    "policy_z_sensitivity_kl",
    "compute_pairwise_actor_jsd",
]
