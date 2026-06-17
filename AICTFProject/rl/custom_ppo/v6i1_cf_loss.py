"""V6I1 counterfactual separation loss L_cf with competence-weighted pair hinges."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl.custom_ppo.curriculum_gates import PAIR_ORDER


class _FaithfulObsGuard(dict):
    disallowed_keys = {
        "opponent_id", "phase_id", "phase", "outcome_id", "role_bucket_id",
        "spread_bucket_id", "pressure_bucket_id", "attack_defense_ratio_bucket_id",
        "role_bucket", "spread_bucket", "pressure_bucket", "attack_defense_ratio_bucket",
        "opponent", "outcome",
    }

    def __getitem__(self, key):
        if key in self.disallowed_keys:
            raise AssertionError(f"Disallowed key '{key}' in CF separation obs path.")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.disallowed_keys:
            raise AssertionError(f"Disallowed key '{key}' in CF separation obs path.")
        return super().get(key, default)


def v6i1_pair_suffix(pair_idx: int) -> str:
    """Fixed pair label for CSV columns, e.g. pair 0 -> ``01``."""
    i, j = PAIR_ORDER[int(pair_idx)]
    return f"{int(i)}{int(j)}"


def _actor_trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and ("actor_cnn" in name or "latent_actor" in name)
    ]


def actor_cf_grad_norm(cf_loss: torch.Tensor, model: torch.nn.Module) -> float:
    """L2 norm of gradients from the CF term alone (no full backward)."""
    params = _actor_trainable_params(model)
    if not params or not cf_loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(
        cf_loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    sq = 0.0
    for grad in grads:
        if grad is not None:
            sq += float(grad.detach().pow(2).sum().cpu().item())
    return float(sq**0.5)


def v6i1_cf_separation_loss(
    model: Any,
    obs_batch: dict[str, torch.Tensor],
    *,
    latent_k: int,
    margin: float,
    competence: np.ndarray,
    competence_ready: bool,
    subsample_generator: torch.Generator | None = None,
    max_rows: int = 512,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Competence-weighted counterfactual separation on full per-head categorical policies.

    For each batch row the same observations are evaluated under all ``K`` forced ``z``
    values. For each unordered pair ``(z, z')`` the divergence ``D`` is the mean JS
  divergence across all agents and action heads. The loss is

      sum_{z<z'} sqrt(c_z c_z') ReLU(m - D) / (sum_{z<z'} sqrt(c_z c_z') + eps)

    Competence scores are detached; they do not receive gradients.
    """
    device = obs_batch.get("mask", torch.tensor(0.0)).device
    if int(latent_k) <= 1:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, {"jsd": zero, "min_jsd": zero, "active": zero}

    obs_batch = _FaithfulObsGuard(obs_batch)
    batch_size = 0
    for value in obs_batch.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            batch_size = int(value.shape[0])
            break
    if batch_size <= 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, {"jsd": zero, "min_jsd": zero, "active": zero}

    if batch_size > max_rows:
        indices = torch.randperm(batch_size, device=device, generator=subsample_generator)[
            :max_rows
        ]
        obs_sub: dict[str, Any] = {}
        for key, value in obs_batch.items():
            if isinstance(value, torch.Tensor) and int(value.shape[0]) == batch_size:
                obs_sub[key] = value.index_select(0, indices)
            else:
                obs_sub[key] = value
        obs_batch = _FaithfulObsGuard(obs_sub)
        curr_batch_size = max_rows
    else:
        curr_batch_size = batch_size

    logits_list = []
    for k in range(int(latent_k)):
        z_k = torch.full((curr_batch_size,), k, dtype=torch.long, device=device)
        logits_k = model._mask_logits(
            model.policy_logits(obs_batch, z_idx=z_k),
            obs_batch.get("mask"),
        )
        logits_list.append(logits_k)

    pair_count = len(PAIR_ORDER)
    pair_js_sum = [logits_list[0].new_zeros((curr_batch_size,)) for _ in range(pair_count)]
    pair_js_count = [0 for _ in range(pair_count)]

    offset = 0
    for _agent_idx in range(int(model.n_agents)):
        for dim in model.per_agent_action_dims:
            width = int(dim)
            p_stacked = []
            for k in range(int(latent_k)):
                a_k = logits_list[k][:, offset : offset + width]
                p_stacked.append(torch.softmax(a_k, dim=-1).clamp_min(1e-8))
            p_stacked_t = torch.stack(p_stacked, dim=0)
            p_i = p_stacked_t.unsqueeze(1)
            p_j = p_stacked_t.unsqueeze(0)
            m = 0.5 * (p_i + p_j)
            kl_i = (p_i * (p_i.log() - m.log())).sum(dim=-1)
            kl_j = (p_j * (p_j.log() - m.log())).sum(dim=-1)
            js_matrix = 0.5 * kl_i + 0.5 * kl_j
            for pair_idx, (zi, zj) in enumerate(PAIR_ORDER):
                if zi >= int(latent_k) or zj >= int(latent_k):
                    continue
                pair_js_sum[pair_idx] = pair_js_sum[pair_idx] + js_matrix[zi, zj]
                pair_js_count[pair_idx] += 1
            offset += width

    if not any(pair_js_count):
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, {"jsd": zero, "min_jsd": zero, "active": zero}

    pair_d = []
    for pair_idx in range(pair_count):
        denom = max(1, pair_js_count[pair_idx])
        pair_d.append(pair_js_sum[pair_idx] / float(denom))
    pair_d_t = torch.stack(pair_d, dim=0)

    if competence_ready:
        c = torch.as_tensor(competence, device=device, dtype=torch.float32).reshape(-1)
    else:
        c = torch.ones((int(latent_k),), device=device, dtype=torch.float32)
    c = c.detach()

    margin_t = pair_d_t.new_tensor(float(max(0.0, margin)))
    weights = []
    penalties = []
    for pair_idx, (zi, zj) in enumerate(PAIR_ORDER):
        if zi >= int(latent_k) or zj >= int(latent_k):
            continue
        weight = torch.sqrt(c[int(zi)] * c[int(zj)]).clamp_min(0.0)
        weights.append(weight)
        penalties.append(weight * F.relu(margin_t - pair_d_t[pair_idx]).mean())

    if not penalties:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return zero, {"jsd": zero, "min_jsd": zero, "active": zero}

    weight_sum = torch.stack(weights).sum().clamp_min(1e-8)
    loss = torch.stack(penalties).sum() / weight_sum
    jsd = pair_d_t.mean()
    return loss, {
        "jsd": jsd.detach(),
        "min_jsd": pair_d_t.min().detach(),
        "active": pair_d_t.new_tensor(1.0),
    }


__all__ = [
    "actor_cf_grad_norm",
    "v6i1_cf_separation_loss",
    "v6i1_pair_suffix",
]
