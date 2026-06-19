"""Per-component actor gradient diagnostics for PPO vs counterfactual separation."""

from __future__ import annotations

from typing import Any

import torch

from rl.custom_ppo.v6i1_cf_loss import actor_diagnostic_grad_norm

PATHWAY_GROUP_NAMES: tuple[str, ...] = (
    "z_embed",
    "film",
    "trunk",
    "action_head",
    "actor_cnn",
)


def _name_in_group(name: str, group: str) -> bool:
    if group == "z_embed":
        return "latent_actor.strategy_embedding" in name
    if group == "film":
        return any(
            token in name
            for token in (
                "latent_actor.film_layer",
                "latent_actor.actor_z_film",
                "latent_actor.z_adapter",
            )
        )
    if group == "trunk":
        return "latent_actor.body" in name
    if group == "action_head":
        return "latent_actor.action_head" in name
    if group == "actor_cnn":
        return "actor_cnn" in name
    return False


def collect_actor_pathway_parameters(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    """Group actor parameters by latent-conditioning pathway stage."""
    groups: dict[str, list[torch.nn.Parameter]] = {name: [] for name in PATHWAY_GROUP_NAMES}
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        if pid in seen:
            continue
        for group in PATHWAY_GROUP_NAMES:
            if _name_in_group(name, group):
                groups[group].append(param)
                seen.add(pid)
                break
    return groups


def actor_pathway_grad_diagnostics(
    *,
    scaled_cf_loss: torch.Tensor,
    ppo_actor_loss: torch.Tensor,
    pathway_params: dict[str, list[torch.nn.Parameter]],
) -> dict[str, float]:
    """Independent gradient norms per pathway group (diagnostics only; no ``.grad`` mutation)."""
    out: dict[str, float] = {}
    for group in PATHWAY_GROUP_NAMES:
        params = pathway_params.get(group, [])
        cf_norm = actor_diagnostic_grad_norm(scaled_cf_loss, params) if params else 0.0
        ppo_norm = actor_diagnostic_grad_norm(ppo_actor_loss, params) if params else 0.0
        out[f"{group}_grad_from_cf"] = float(cf_norm)
        out[f"{group}_grad_from_ppo"] = float(ppo_norm)
    return out


def actor_pathway_grad_diagnostics_for_model(
    *,
    model: torch.nn.Module,
    scaled_cf_loss: torch.Tensor,
    ppo_actor_loss: torch.Tensor,
) -> dict[str, float]:
    return actor_pathway_grad_diagnostics(
        scaled_cf_loss=scaled_cf_loss,
        ppo_actor_loss=ppo_actor_loss,
        pathway_params=collect_actor_pathway_parameters(model),
    )


__all__ = [
    "PATHWAY_GROUP_NAMES",
    "actor_pathway_grad_diagnostics",
    "actor_pathway_grad_diagnostics_for_model",
    "collect_actor_pathway_parameters",
]
