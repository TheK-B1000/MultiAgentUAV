"""Fully shared + z distillation student for the cross-scale baseline suite.

One ``SharedActorCentralizedCritic``: specialist CNN / 256-256 body / action
head, with strategy identity only via concat embedding (``latent_k=2``,
``d_z=16``). No router, FiLM, or per-z action heads.

The architecture is derived from a specialist checkpoint's observation/action
spaces and CNN width, then latent concat is switched on. Weights are a fresh
draw under ``seed`` (not a warm-start of the specialist).
"""
from __future__ import annotations

import os
from typing import Any

import torch

from rl.ladder_rung1 import _specialist_arch


def fully_shared_model_kwargs(specialist_kwargs: dict) -> dict:
    """Specialist actor class + concat z only. Drops private-branch flags."""
    kw = dict(specialist_kwargs)
    kw.update(
        latent_k=2,
        z_embed_dim=16,
        strategy_encoder_enabled=False,
        latent_actor_conditioning="concat",
        enable_actor_z_film=False,
        latent_actor_z_adapter_enabled=False,
        latent_population_birth_per_z_action_heads=False,
        exp2c_mode_specific_action_heads=False,
        enable_latent_z_residual=False,
        latent_lro_deep_branches=False,
        use_strategy_aux_return_head=False,
        use_episode_strategy_value_head=False,
        use_recurrent_selector=False,
    )
    return kw


def suite_cfg_for_fully_shared(specialist_cfg: dict) -> dict:
    cfg = dict(specialist_cfg or {})
    cfg["use_latent_strategy"] = True
    cfg["latent_k"] = 2
    cfg["latent_z_embed_dim"] = 16
    cfg["latent_strategy_encoder_enabled"] = False
    cfg["latent_actor_conditioning"] = "concat"
    cfg["enable_actor_z_film"] = False
    cfg["latent_population_birth_per_z_action_heads"] = False
    cfg["exp2c_mode_specific_action_heads"] = False
    cfg["enable_latent_z_residual"] = False
    cfg["suite_arm"] = "fully_shared_z"
    return cfg


def assert_fully_shared_structure(model: Any) -> None:
    if int(getattr(model, "latent_k", 0)) != 2:
        raise RuntimeError(f"fully-shared requires latent_k=2, got {getattr(model, 'latent_k', None)}")
    if getattr(model, "strategy_encoder", None) is not None:
        raise RuntimeError("fully-shared forbids a q_phi / strategy encoder")
    actor = model.latent_actor
    if getattr(actor, "strategy_embedding", None) is None:
        raise RuntimeError("fully-shared requires a z embedding")
    if int(actor.strategy_embedding.num_embeddings) != 2 or int(actor.strategy_embedding.embedding_dim) != 16:
        raise RuntimeError("fully-shared z embedding must be Embedding(2, 16)")
    if getattr(actor, "actor_z_film", None) is not None or getattr(actor, "film_layer1", None) is not None:
        raise RuntimeError("fully-shared forbids FiLM")
    if bool(getattr(actor, "exp2c_mode_specific_action_heads", False)):
        raise RuntimeError("fully-shared forbids mode-specific action heads")
    if bool(getattr(actor, "latent_population_birth_per_z_action_heads", False)):
        raise RuntimeError("fully-shared forbids per-z action heads")


def build_fully_shared_student(spec_ckpt_path: str, observation_space, action_space, *,
                               seed: int, device: str):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload, spec_kw = _specialist_arch(spec_ckpt_path, observation_space, action_space)
    kw = fully_shared_model_kwargs(spec_kw)
    torch.manual_seed(int(seed))
    model = SharedActorCentralizedCritic(observation_space, action_space, **kw).to(device)
    assert_fully_shared_structure(model)
    ref = payload["model_state_dict"]
    # Body input widened by d_z, so names will not match the specialist. Refuse a
    # silent copy of any overlapping tensor that is bit-identical (warm start).
    sd = model.state_dict()
    overlap = [k for k in sd if k in ref and tuple(sd[k].shape) == tuple(ref[k].shape) and sd[k].numel()]
    if overlap:
        diff = max(float((sd[k].detach().cpu().float() - ref[k].float()).abs().max()) for k in overlap)
        if diff == 0.0:
            raise RuntimeError("fully-shared student overlaps the specialist bit-exactly -- silent warm start")
    return model, dict(payload.get("cfg") or {}), kw


def save_fully_shared(model, specialist_cfg: dict, kwargs: dict, out_path: str, provenance: dict) -> None:
    payload = {
        "format": "suite_fully_shared_z_v1",
        "cfg": suite_cfg_for_fully_shared(specialist_cfg),
        "model_kwargs": dict(kwargs),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "suite_fully_shared_z": dict(provenance),
    }
    tmp = f"{out_path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def load_fully_shared(path: str, observation_space, action_space, *, device: str):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "suite_fully_shared_z_v1":
        raise RuntimeError(f"{path}: not a suite fully-shared checkpoint")
    kw = dict(payload["model_kwargs"])
    model = SharedActorCentralizedCritic(observation_space, action_space, **kw)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    assert_fully_shared_structure(model)
    return model, payload
