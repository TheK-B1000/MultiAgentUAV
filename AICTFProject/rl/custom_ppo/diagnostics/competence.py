"""Adapter, critic, and advantage competence diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from rl.custom_ppo.diagnostics.counterfactual import _jsd_from_logits


_ZERO_OPT_ADV: dict[str, float] = {
    "latent_q_phi_option_advantage_mean": 0.0,
    "latent_q_phi_option_advantage_std": 0.0,
    "latent_q_phi_option_advantage_count": 0.0,
}


def compute_adapter_grad_norms(model: "SharedActorCentralizedCritic") -> dict[str, float]:
    """Return per-latent adapter and action-bias gradient L2 norms.

    Call after ``loss.backward()`` but before ``optimizer.zero_grad()``.
    Returns an empty dict when residual adapters are not enabled.
    """
    la = getattr(model, "latent_actor", None)
    if la is None or not getattr(la, "enable_latent_z_residual", False):
        return {}
    out: dict[str, float] = {}
    adapters = getattr(la, "latent_adapters", None)
    if adapters is not None:
        for k, adapter in enumerate(adapters):
            total_sq = sum(
                p.grad.pow(2).sum().item()
                for p in adapter.parameters()
                if p.grad is not None
            )
            out[f"adapter_grad_norm_z{k}"] = float(total_sq ** 0.5)
    gates = getattr(la, "latent_adapter_gates", None)
    if gates is not None and gates.grad is not None:
        for k in range(int(gates.shape[0])):
            out[f"adapter_gate_grad_z{k}"] = float(gates.grad[k].abs().item())
    biases = getattr(la, "latent_action_biases", None)
    if biases is not None and biases.grad is not None:
        for k in range(int(biases.shape[0])):
            out[f"action_bias_grad_norm_z{k}"] = float(biases.grad[k].norm().item())
    return out


def compute_critic_value_variance(
    model: "SharedActorCentralizedCritic",
    global_state: torch.Tensor,
) -> dict[str, float]:
    """Return Var_z[V(s, z)] for identical global states across all K latents.

    ``global_state`` has shape ``(N, global_state_dim)``.  Returns nan when
    latent strategy is not enabled.
    """
    K = int(model.latent_k)
    if K < 2 or not model.uses_latent_strategy:
        return {"critic_value_var_z": float("nan")}
    device = global_state.device
    N = global_state.shape[0]

    with torch.no_grad():
        values_by_z = []
        for k in range(K):
            z_t = torch.full((N,), k, dtype=torch.long, device=device)
            v_k = model.values(global_state, z_idx=z_t)
            values_by_z.append(v_k)
        stacked = torch.stack(values_by_z, dim=0)  # (K, N)
        var_across_z = stacked.var(dim=0).mean().item()

    return {"critic_value_var_z": float(var_across_z)}


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


def _v6i8_residual_adapter_stats(runtime: Any, buffer: Any) -> dict[str, float]:
    """Post-update V6I8 adapter diagnostics: pairwise actor JSD and adapter grad norms.

    Pairwise JSD: identical random local features, all K latents — measures
    differentiation from adapters and biases only (z-embedding contribution
    held fixed because the same input tensor is reused across z values).

    Adapter grad norms: a diagnostic forward-backward on the same random
    sample, immediately zeroed after measurement.  These are diagnostic
    gradient magnitudes, not training gradients.

    Returns an empty dict when ``enable_latent_z_residual`` is False.
    """
    from itertools import combinations

    model = getattr(runtime, "model", None)
    if model is None:
        return {}
    la = getattr(model, "latent_actor", None)
    if la is None or not getattr(la, "enable_latent_z_residual", False):
        return {}
    K = int(getattr(model, "latent_k", 0))
    if K < 2:
        return {}

    try:
        device = next(model.parameters()).device
        N = 64
        full_input_dim = int(model.actor_input_dim)
        local_feats = torch.randn(N, full_input_dim, device=device)

        with torch.no_grad():
            logits_by_z = []
            for k in range(K):
                z_t = torch.full((N,), k, dtype=torch.long, device=device)
                logits_k = la(local_feats, z_t)
                logits_by_z.append(logits_k)

        pair_jsds: list[float] = []
        pair_disagrees: list[float] = []
        for i, j in combinations(range(K), 2):
            jsd = _jsd_from_logits(logits_by_z[i], logits_by_z[j]).mean().item()
            pair_jsds.append(float(jsd))
            argmax_i = logits_by_z[i].argmax(dim=-1)
            argmax_j = logits_by_z[j].argmax(dim=-1)
            pair_disagrees.append(float((argmax_i != argmax_j).float().mean().item()))

        out: dict[str, float] = {
            "actor_jsd_mean": float(sum(pair_jsds) / len(pair_jsds)),
            "actor_jsd_min": float(min(pair_jsds)),
            "actor_jsd_max": float(max(pair_jsds)),
            "actor_argmax_disagree": float(sum(pair_disagrees) / len(pair_disagrees)),
        }

        model.zero_grad()
        z_diag = torch.randint(K, (N,), device=device)
        logits_diag = la(local_feats.detach(), z_diag)
        logits_diag.sum().backward()
        out.update(compute_adapter_grad_norms(model))
        model.zero_grad()

        return out
    except Exception:
        return {}


strategy_resample_advantage_stats = _strategy_resample_advantage_stats
rollout_advantage_diagnostics = _rollout_advantage_diagnostics
latent_option_advantage_stats = _latent_option_advantage_stats
v6i8_residual_adapter_stats = _v6i8_residual_adapter_stats

__all__ = [
    "compute_adapter_grad_norms",
    "compute_critic_value_variance",
    "_strategy_resample_advantage_stats",
    "_rollout_advantage_diagnostics",
    "_latent_option_advantage_stats",
    "_v6i8_residual_adapter_stats",
    "strategy_resample_advantage_stats",
    "rollout_advantage_diagnostics",
    "latent_option_advantage_stats",
    "v6i8_residual_adapter_stats",
]
