"""Trace z → conditioning → trunk → logits causal leverage through the actor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ActorZPathwayStage:
    name: str
    pair_mean_l2: float
    pair_max_l2: float


@dataclass(frozen=True)
class ActorZPathwayReport:
    conditioning_mode: str
    stages: tuple[ActorZPathwayStage, ...]
    weakest_stage: str
    logits_pairwise_jsd_mean: float

    def as_dict(self) -> dict[str, float | str]:
        out: dict[str, float | str] = {
            "actor_z_pathway_conditioning": self.conditioning_mode,
            "actor_z_pathway_weakest_stage": self.weakest_stage,
            "actor_z_pathway_logits_jsd_mean": float(self.logits_pairwise_jsd_mean),
        }
        for stage in self.stages:
            out[f"actor_z_pathway_{stage.name}_mean_l2"] = float(stage.pair_mean_l2)
            out[f"actor_z_pathway_{stage.name}_max_l2"] = float(stage.pair_max_l2)
        return out


def _pairwise_mean_max_l2(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    flat_a = a.reshape(a.shape[0], -1)
    flat_b = b.reshape(b.shape[0], -1)
    dist = torch.linalg.vector_norm(flat_a - flat_b, dim=-1)
    return float(dist.mean().item()), float(dist.max().item())


def _pairwise_jsd_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    from rl.custom_ppo.latent_diagnostics import _jsd_from_logits

    log_p = torch.log_softmax(logits_a.float(), dim=-1)
    log_q = torch.log_softmax(logits_b.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    js = 0.5 * ((p * (log_p - torch.log(m.clamp_min(1e-8)))).sum(dim=-1))
    js = js + 0.5 * ((q * (log_q - torch.log(m.clamp_min(1e-8)))).sum(dim=-1))
    return float(js.mean().item())


def trace_actor_z_pathway(
    model: Any,
    obs_batch: dict[str, torch.Tensor],
    *,
    z_a: int = 0,
    z_b: int = 1,
) -> ActorZPathwayReport:
    """Measure per-stage divergence for two forced-z values on identical observations."""
    if not bool(getattr(model, "uses_latent_strategy", False)):
        raise ValueError("trace_actor_z_pathway requires latent strategy enabled")
    batch = int(obs_batch["grid"].shape[0])
    device = obs_batch["grid"].device
    z_idx_a = torch.full((batch,), int(z_a), dtype=torch.long, device=device)
    z_idx_b = torch.full((batch,), int(z_b), dtype=torch.long, device=device)

    latent_actor = getattr(model, "latent_actor", None)
    if latent_actor is None or latent_actor.strategy_embedding is None:
        raise ValueError("model has no latent_actor.strategy_embedding")

    with torch.no_grad():
        emb_a = latent_actor.strategy_embedding(z_idx_a) * float(latent_actor.z_embed_scale)
        emb_b = latent_actor.strategy_embedding(z_idx_b) * float(latent_actor.z_embed_scale)
        embed_mean, embed_max = _pairwise_mean_max_l2(emb_a, emb_b)

        mod_mean, mod_max = embed_mean, embed_max
        if hasattr(latent_actor, "film_modulation_l2"):
            mod_mean = mod_max = float(latent_actor.film_modulation_l2(z_idx_a[:1], z_idx_b[:1]))

        trunk_a = model.policy_trunk_features(obs_batch, z_idx=z_idx_a)
        trunk_b = model.policy_trunk_features(obs_batch, z_idx=z_idx_b)
        trunk_mean, trunk_max = _pairwise_mean_max_l2(trunk_a, trunk_b)

        logits_a = model._mask_logits(model.policy_logits(obs_batch, z_idx=z_idx_a), obs_batch.get("mask"))
        logits_b = model._mask_logits(model.policy_logits(obs_batch, z_idx=z_idx_b), obs_batch.get("mask"))
        logit_mean, logit_max = _pairwise_mean_max_l2(logits_a, logits_b)
        jsd_mean = _pairwise_jsd_from_logits(logits_a, logits_b)

    mode = str(getattr(latent_actor, "latent_actor_conditioning", "concat"))
    has_film = mode == "film_v6" or bool(getattr(latent_actor, "actor_z_film", None))
    stage_list: list[ActorZPathwayStage] = [
        ActorZPathwayStage("embed", embed_mean, embed_max),
    ]
    if has_film:
        stage_list.append(ActorZPathwayStage("film", mod_mean, mod_max))
    stage_list.extend(
        (
            ActorZPathwayStage("trunk", trunk_mean, trunk_max),
            ActorZPathwayStage("logits", logit_mean, logit_max),
        )
    )
    stages = tuple(stage_list)
    weakest = min(stages, key=lambda s: s.pair_mean_l2).name
    return ActorZPathwayReport(
        conditioning_mode=mode,
        stages=stages,
        weakest_stage=weakest,
        logits_pairwise_jsd_mean=jsd_mean,
    )


__all__ = [
    "ActorZPathwayReport",
    "ActorZPathwayStage",
    "trace_actor_z_pathway",
]
