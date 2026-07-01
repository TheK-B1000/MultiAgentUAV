"""V6I1 counterfactual separation loss L_cf with competence-weighted pair hinges."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl.custom_ppo.curriculum_gates import PAIR_ORDER

# Rollout forced-z profile must expose all six unordered pairs before the
# intervention-gate EMA may advance. ``forced_z_macro_jsd_mean`` alone is
# legacy diagnostic only and must never substitute for missing pair keys.
REQUIRED_FORCED_Z_PAIR_KEYS: tuple[str, ...] = tuple(
    f"forced_z_pair_jsd_{idx}" for idx in range(len(PAIR_ORDER))
)


def forced_z_pairwise_profile_available(profile_stats: dict[str, Any]) -> bool:
    """True when every enforced pair key is present (values may be zero)."""
    return all(key in profile_stats for key in REQUIRED_FORCED_Z_PAIR_KEYS)


def extract_forced_z_pair_values(profile_stats: dict[str, Any]) -> list[float] | None:
    """Return the six measured pair JSDs, or None when the profile is incomplete/invalid."""
    if not forced_z_pairwise_profile_available(profile_stats):
        return None
    vals = [float(profile_stats[key]) for key in REQUIRED_FORCED_Z_PAIR_KEYS]
    if not all(np.isfinite(v) for v in vals):
        return None
    return vals


def _zero_cf_diag(device: torch.device) -> dict[str, torch.Tensor]:
    zero = torch.zeros((), dtype=torch.float32, device=device)
    return {
        "jsd": zero,
        "max_jsd": zero,
        "min_jsd": zero,
        "active": zero,
        "pair_jsd": zero.new_zeros((len(PAIR_ORDER),)),
        "cf_pair_hinge": zero.new_zeros((len(PAIR_ORDER),)),
        "cf_pair_weight": zero.new_zeros((len(PAIR_ORDER),)),
        "pairs_below_margin": zero,
        "cf_hinge_active": zero,
        "cf_hinge_effective": zero,
        "cf_valid_team_groups": zero,
        "cf_weight_sum": zero,
        "cf_effective_pairs": zero,
        "cf_mean_pair_hinge": zero,
        "cf_worst_pair_hinge": zero,
        "cf_worst_pair_index": zero,
        "cf_worst_pair_coef": zero,
        "cf_weak_pair_boost": zero,
        "cf_competence_required": zero,
        "cf_competence_ready": zero,
        "cf_competence_min": zero,
        "cf_competence_gate_blocks_cf": zero,
    }


def _build_cf_diag_stats(
    *,
    device: torch.device,
    pair_batch_means: torch.Tensor,
    margin: float,
    competence: torch.Tensor,
    latent_k: int,
    valid_team_groups: int,
    weight_sum: torch.Tensor,
    pair_hinge_means: torch.Tensor,
    pair_weights: torch.Tensor,
    worst_pair_hinge: torch.Tensor,
    worst_pair_index: int,
    worst_pair_coef: float,
    weak_pair_boost: float,
    competence_ready: bool,
    competence_required: bool,
) -> dict[str, torch.Tensor]:
    """Diagnostics for CF hinge / gradient interpretation (see tests)."""
    margin_f = float(margin)
    pairs_below = int(sum(1 for v in pair_batch_means if float(v.item()) < margin_f))
    cf_hinge_active = pairs_below > 0
    effective_pairs = 0
    for pair_idx, (zi, zj) in enumerate(PAIR_ORDER):
        if zi >= int(latent_k) or zj >= int(latent_k):
            continue
        w = float(pair_weights[pair_idx].detach().cpu().item())
        if w > 1e-8 and float(pair_batch_means[pair_idx].item()) < margin_f:
            effective_pairs += 1
    weight_sum_f = float(weight_sum.item())
    cf_hinge_effective = (
        cf_hinge_active and weight_sum_f > 1e-8 and effective_pairs > 0
    )
    return {
        "jsd": pair_batch_means.mean().detach(),
        "max_jsd": pair_batch_means.max().detach(),
        "min_jsd": pair_batch_means.min().detach(),
        "active": pair_batch_means.new_tensor(1.0),
        "pair_jsd": pair_batch_means.detach(),
        "cf_pair_hinge": pair_hinge_means.detach(),
        "cf_pair_weight": pair_weights.detach(),
        "pairs_below_margin": pair_batch_means.new_tensor(float(pairs_below)),
        "cf_hinge_active": pair_batch_means.new_tensor(1.0 if cf_hinge_active else 0.0),
        "cf_hinge_effective": pair_batch_means.new_tensor(1.0 if cf_hinge_effective else 0.0),
        "cf_valid_team_groups": pair_batch_means.new_tensor(float(valid_team_groups)),
        "cf_weight_sum": pair_batch_means.new_tensor(weight_sum_f),
        "cf_effective_pairs": pair_batch_means.new_tensor(float(effective_pairs)),
        "cf_mean_pair_hinge": pair_hinge_means.mean().detach(),
        "cf_worst_pair_hinge": worst_pair_hinge.detach(),
        "cf_worst_pair_index": pair_batch_means.new_tensor(float(worst_pair_index)),
        "cf_worst_pair_coef": pair_batch_means.new_tensor(float(worst_pair_coef)),
        "cf_weak_pair_boost": pair_batch_means.new_tensor(float(weak_pair_boost)),
        "cf_competence_required": pair_batch_means.new_tensor(
            1.0 if competence_required else 0.0
        ),
        "cf_competence_ready": pair_batch_means.new_tensor(1.0 if competence_ready else 0.0),
        "cf_competence_min": (
            competence.detach().float().min()
            if int(competence.numel()) > 0
            else pair_batch_means.new_tensor(0.0)
        ),
        "cf_competence_gate_blocks_cf": pair_batch_means.new_tensor(0.0),
    }


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


def global_grad_norm(grads: tuple[torch.Tensor | None, ...] | list[torch.Tensor | None]) -> float:
    """L2 norm over a list of per-parameter gradient tensors."""
    sq = 0.0
    for grad in grads:
        if grad is not None:
            sq += float(grad.detach().pow(2).sum().cpu().item())
    return float(sq**0.5)


def actor_diagnostic_grad_norm(
    loss: torch.Tensor,
    actor_parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
) -> float:
    """Gradient norm for one actor loss term via ``autograd.grad`` (no ``.grad`` mutation)."""
    params = [p for p in actor_parameters if p.requires_grad]
    if not params or not loss.requires_grad:
        return 0.0
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    return global_grad_norm(grads)


def actor_cf_grad_norm(
    cf_loss: torch.Tensor,
    actor_parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
) -> float:
    """L2 norm of gradients from the scaled CF term alone (no ``.grad`` mutation)."""
    return actor_diagnostic_grad_norm(cf_loss, actor_parameters)


def actor_cf_ppo_grad_diagnostics(
    *,
    scaled_cf_loss: torch.Tensor,
    ppo_actor_loss: torch.Tensor,
    actor_parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    ratio_epsilon: float = 1e-8,
) -> tuple[float, float, float]:
    """Independent CF / PPO-actor gradient norms for telemetry (diagnostics only)."""
    cf_actor_grad_norm = actor_diagnostic_grad_norm(scaled_cf_loss, actor_parameters)
    ppo_actor_grad_norm = actor_diagnostic_grad_norm(ppo_actor_loss, actor_parameters)
    cf_to_ppo_grad_ratio = cf_actor_grad_norm / max(ppo_actor_grad_norm, float(ratio_epsilon))
    return cf_actor_grad_norm, ppo_actor_grad_norm, cf_to_ppo_grad_ratio


def v6i1_cf_separation_loss(
    model: Any,
    obs_batch: dict[str, torch.Tensor],
    *,
    latent_k: int,
    margin: float,
    competence: np.ndarray,
    competence_ready: bool,
    weak_pair_ema: np.ndarray | None = None,
    weak_pair_boost: float = 0.0,
    worst_pair_coef: float = 0.0,
    require_competence: bool = False,
    subsample_generator: torch.Generator | None = None,
    max_rows: int = 512,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Competence-weighted counterfactual separation on full per-head categorical policies.

    For each batch row the same observations are evaluated under all ``K`` forced ``z``
    values. For each unordered pair ``(z, z')`` the divergence ``D`` is the mean JS
    divergence across all agents and action heads. The mean-pair loss is

      sum_{z<z'} sqrt(c_z c_z') ReLU(m - D) / (sum_{z<z'} sqrt(c_z c_z') + eps)

    v6i2 may add a worst-pair hinge term and persistent weak-pair multipliers.
    Competence scores and weak-pair EMA values are detached.
    """
    device = obs_batch.get("mask", torch.tensor(0.0)).device
    if bool(require_competence) and not bool(competence_ready):
        diag = _zero_cf_diag(device)
        diag["cf_competence_required"] = torch.ones((), dtype=torch.float32, device=device)
        diag["cf_competence_gate_blocks_cf"] = torch.ones((), dtype=torch.float32, device=device)
        c = torch.as_tensor(competence, device=device, dtype=torch.float32).reshape(-1)
        if int(c.numel()) > 0:
            diag["cf_competence_min"] = c.detach().float().min()
        return torch.zeros((), dtype=torch.float32, device=device), diag
    if int(latent_k) <= 1:
        return torch.zeros((), dtype=torch.float32, device=device), _zero_cf_diag(device)

    obs_batch = _FaithfulObsGuard(obs_batch)
    batch_size = 0
    for value in obs_batch.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            batch_size = int(value.shape[0])
            break
    if batch_size <= 0:
        return torch.zeros((), dtype=torch.float32, device=device), _zero_cf_diag(device)

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
            model.policy_logits(obs_batch, z_idx=z_k, detach_local_features=True),
            obs_batch.get("mask"),
        )
        logits_list.append(logits_k)

    pair_count = len(PAIR_ORDER)
    pair_js_sum = [logits_list[0].new_zeros((curr_batch_size,)) for _ in range(pair_count)]
    pair_js_count = [0 for _ in range(pair_count)]
    n_heads = len(model.per_agent_action_dims)
    head_pair_js_sum = [
        [logits_list[0].new_zeros((curr_batch_size,)) for _ in range(pair_count)]
        for _ in range(n_heads)
    ]
    head_pair_js_count = [[0 for _ in range(pair_count)] for _ in range(n_heads)]

    offset = 0
    for _agent_idx in range(int(model.n_agents)):
        for head_idx, dim in enumerate(model.per_agent_action_dims):
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
                head_pair_js_sum[head_idx][pair_idx] = (
                    head_pair_js_sum[head_idx][pair_idx] + js_matrix[zi, zj]
                )
                head_pair_js_count[head_idx][pair_idx] += 1
            offset += width

    if not any(pair_js_count):
        return torch.zeros((), dtype=torch.float32, device=device), _zero_cf_diag(device)

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
    weights: list[torch.Tensor] = []
    pair_hinge_means: list[torch.Tensor] = []
    penalties: list[torch.Tensor] = []
    zero_pair = pair_d_t.new_zeros(())
    pair_weights_full: list[torch.Tensor] = [zero_pair for _ in range(pair_count)]
    pair_hinge_full: list[torch.Tensor] = [zero_pair for _ in range(pair_count)]
    weak_pair_ema_t: torch.Tensor | None = None
    if weak_pair_ema is not None:
        weak_pair_ema_t = torch.as_tensor(
            weak_pair_ema, device=device, dtype=torch.float32
        ).reshape(-1)
    for pair_idx, (zi, zj) in enumerate(PAIR_ORDER):
        if zi >= int(latent_k) or zj >= int(latent_k):
            continue
        weight = torch.sqrt(c[int(zi)] * c[int(zj)]).clamp_min(0.0)
        if weak_pair_ema_t is not None and pair_idx < int(weak_pair_ema_t.numel()):
            denom = margin_t.clamp_min(1e-8)
            weakness = ((margin_t - weak_pair_ema_t[pair_idx].detach()) / denom).clamp(
                min=0.0,
                max=1.0,
            )
            weight = weight * (1.0 + float(max(0.0, weak_pair_boost)) * weakness)
        pair_hinge = F.relu(margin_t - pair_d_t[pair_idx]).mean()
        weights.append(weight)
        pair_hinge_means.append(pair_hinge)
        pair_weights_full[pair_idx] = weight
        pair_hinge_full[pair_idx] = pair_hinge
        penalties.append(weight * pair_hinge)

    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device), _zero_cf_diag(device)

    pair_weights_t = torch.stack(weights)
    pair_hinge_means_t = torch.stack(pair_hinge_means)
    pair_weights_full_t = torch.stack(pair_weights_full)
    pair_hinge_full_t = torch.stack(pair_hinge_full)
    weight_sum_raw = pair_weights_t.sum()
    weight_sum = weight_sum_raw.clamp_min(1e-8)
    loss = torch.stack(penalties).sum() / weight_sum
    worst_pair_hinge, worst_pair_local_idx = pair_hinge_means_t.max(dim=0)
    if float(worst_pair_coef) > 0.0:
        loss = loss + float(worst_pair_coef) * worst_pair_hinge
    pair_batch_means = pair_d_t.mean(dim=-1)
    head_jsd_means: list[float] = []
    for head_idx in range(n_heads):
        head_pair_d = []
        for pair_idx in range(pair_count):
            denom = max(1, head_pair_js_count[head_idx][pair_idx])
            head_pair_d.append(head_pair_js_sum[head_idx][pair_idx] / float(denom))
        if head_pair_d:
            head_jsd_means.append(float(torch.stack(head_pair_d).mean().item()))
        else:
            head_jsd_means.append(0.0)
    diag = _build_cf_diag_stats(
        device=device,
        pair_batch_means=pair_batch_means,
        margin=float(margin),
        competence=c,
        latent_k=int(latent_k),
        valid_team_groups=int(curr_batch_size),
        weight_sum=weight_sum_raw,
        pair_hinge_means=pair_hinge_full_t,
        pair_weights=pair_weights_full_t,
        worst_pair_hinge=worst_pair_hinge,
        worst_pair_index=int(worst_pair_local_idx.item()),
        worst_pair_coef=float(worst_pair_coef),
        weak_pair_boost=float(weak_pair_boost),
        competence_ready=bool(competence_ready),
        competence_required=bool(require_competence),
    )
    if head_jsd_means:
        diag["cf_batch_macro_jsd"] = pair_d_t.new_tensor(head_jsd_means[0])
        if len(head_jsd_means) > 1:
            diag["cf_batch_waypoint_jsd"] = pair_d_t.new_tensor(
                float(np.mean(head_jsd_means[1:]))
            )
        else:
            diag["cf_batch_waypoint_jsd"] = pair_d_t.new_tensor(0.0)
    return loss, diag


def v6i1_cf_separation_loss_for_action_head(
    model: Any,
    obs_batch: dict[str, torch.Tensor],
    *,
    action_head_idx: int,
    latent_k: int,
    margin: float,
    competence: np.ndarray,
    competence_ready: bool,
    subsample_generator: torch.Generator | None = None,
    max_rows: int = 512,
) -> torch.Tensor:
    """CF separation restricted to one per-agent action head (macro=0, waypoint=1+)."""
    device = obs_batch.get("mask", torch.tensor(0.0)).device
    if int(latent_k) <= 1:
        return torch.zeros((), dtype=torch.float32, device=device)

    obs_batch = _FaithfulObsGuard(obs_batch)
    batch_size = 0
    for value in obs_batch.values():
        if isinstance(value, torch.Tensor) and value.dim() >= 1:
            batch_size = int(value.shape[0])
            break
    if batch_size <= 0:
        return torch.zeros((), dtype=torch.float32, device=device)

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
            model.policy_logits(obs_batch, z_idx=z_k, detach_local_features=True),
            obs_batch.get("mask"),
        )
        logits_list.append(logits_k)

    pair_count = len(PAIR_ORDER)
    pair_js_sum = [logits_list[0].new_zeros((curr_batch_size,)) for _ in range(pair_count)]
    pair_js_count = [0 for _ in range(pair_count)]
    head_idx = int(action_head_idx)

    offset = 0
    for _agent_idx in range(int(model.n_agents)):
        for local_head_idx, dim in enumerate(model.per_agent_action_dims):
            width = int(dim)
            if local_head_idx != head_idx:
                offset += width
                continue
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
        return torch.zeros((), dtype=torch.float32, device=device)

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
    penalties: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    for pair_idx, (zi, zj) in enumerate(PAIR_ORDER):
        if zi >= int(latent_k) or zj >= int(latent_k):
            continue
        weight = torch.sqrt(c[int(zi)] * c[int(zj)]).clamp_min(0.0)
        weights.append(weight)
        penalties.append(weight * F.relu(margin_t - pair_d_t[pair_idx]).mean())

    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device)

    weight_sum = torch.stack(weights).sum().clamp_min(1e-8)
    return torch.stack(penalties).sum() / weight_sum


__all__ = [
    "REQUIRED_FORCED_Z_PAIR_KEYS",
    "actor_cf_grad_norm",
    "actor_cf_ppo_grad_diagnostics",
    "actor_diagnostic_grad_norm",
    "extract_forced_z_pair_values",
    "forced_z_pairwise_profile_available",
    "global_grad_norm",
    "v6i1_cf_separation_loss",
    "v6i1_cf_separation_loss_for_action_head",
    "v6i1_pair_suffix",
]
