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


def fully_shared_model_kwargs(specialist_kwargs: dict, *, role_conditioning: bool = False) -> dict:
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
        role_conditioning_enabled=bool(role_conditioning),
        assignment_conditioning_enabled=False,
    )
    return kw


def suite_cfg_for_fully_shared(specialist_cfg: dict, *, role_conditioning: bool = False) -> dict:
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
    cfg["role_conditioning_enabled"] = bool(role_conditioning)
    cfg["suite_arm"] = "fully_shared_z_r" if role_conditioning else "fully_shared_z"
    return cfg


def role_only_model_kwargs(specialist_kwargs: dict) -> dict:
    """pi(a | o, r): same body as Fully Shared+z+r with z removed (Stage 4 Role-only)."""
    kw = fully_shared_model_kwargs(specialist_kwargs, role_conditioning=True)
    kw.update(
        latent_k=0,
        z_embed_dim=0,
        strategy_encoder_enabled=False,
        latent_actor_conditioning="none",
    )
    return kw


def suite_cfg_for_role_only(specialist_cfg: dict) -> dict:
    cfg = dict(specialist_cfg or {})
    cfg["use_latent_strategy"] = False
    cfg["latent_k"] = 0
    cfg["role_conditioning_enabled"] = True
    cfg["suite_arm"] = "role_only"
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
                               seed: int, device: str, role_conditioning: bool = False):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload, spec_kw = _specialist_arch(spec_ckpt_path, observation_space, action_space)
    kw = fully_shared_model_kwargs(spec_kw, role_conditioning=role_conditioning)
    torch.manual_seed(int(seed))
    model = SharedActorCentralizedCritic(observation_space, action_space, **kw).to(device)
    assert_fully_shared_structure(model)
    if bool(role_conditioning) != bool(getattr(model, "role_conditioning_enabled", False)):
        raise RuntimeError("fully-shared+z+r requires role_conditioning_enabled on the student")
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
    role_on = bool(kwargs.get("role_conditioning_enabled", False))
    fmt = "suite_fully_shared_z_r_v1" if role_on else "suite_fully_shared_z_v1"
    payload = {
        "format": fmt,
        "cfg": suite_cfg_for_fully_shared(specialist_cfg, role_conditioning=role_on),
        "model_kwargs": dict(kwargs),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "suite_fully_shared_z": dict(provenance),
    }
    tmp = f"{out_path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def build_role_only_student(spec_ckpt_path: str, observation_space, action_space, *,
                            seed: int, device: str):
    """Stage 4 Role-only: pi(a | o, r). Controlled difference vs Fully Shared+z+r is z removed."""
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload, spec_kw = _specialist_arch(spec_ckpt_path, observation_space, action_space)
    kw = role_only_model_kwargs(spec_kw)
    torch.manual_seed(int(seed))
    model = SharedActorCentralizedCritic(observation_space, action_space, **kw).to(device)
    if int(getattr(model, "latent_k", 0) or 0) != 0:
        raise RuntimeError(f"role-only requires latent_k=0, got {getattr(model, 'latent_k', None)}")
    if not bool(getattr(model, "role_conditioning_enabled", False)):
        raise RuntimeError("role-only requires role_conditioning_enabled=True")
    if getattr(model, "strategy_encoder", None) is not None:
        raise RuntimeError("role-only forbids a q_phi / strategy encoder")
    ref = payload["model_state_dict"]
    sd = model.state_dict()
    same = [k for k in sd if k in ref and tuple(sd[k].shape) == tuple(ref[k].shape) and sd[k].numel()
            and torch.equal(sd[k].detach().cpu().float(), ref[k].float())]
    if any(k.endswith("weight") for k in same):
        raise RuntimeError(f"role-only copies specialist weights bit-exactly -- silent warm start ({same[:3]})")
    return model, dict(payload.get("cfg") or {}), kw


def save_role_only(model, specialist_cfg: dict, kwargs: dict, out_path: str, provenance: dict) -> None:
    payload = {
        "format": "suite_role_only_v1",
        "cfg": suite_cfg_for_role_only(specialist_cfg),
        "model_kwargs": dict(kwargs),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "suite_role_only": dict(provenance),
    }
    tmp = f"{out_path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def load_role_only(path: str, observation_space, action_space, *, device: str):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "suite_role_only_v1":
        raise RuntimeError(f"{path}: not a role-only student checkpoint")
    kw = dict(payload["model_kwargs"])
    model = SharedActorCentralizedCritic(observation_space, action_space, **kw).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, dict(payload.get("cfg") or {}), payload


def assert_generalist_structure(model: Any) -> None:
    """Generalist pi_G(a|o): the specialists' own architecture with no strategy pathway at all."""
    if int(getattr(model, "latent_k", 0) or 0) != 0:
        raise RuntimeError(f"generalist requires latent_k=0, got {getattr(model, 'latent_k', None)}")
    if getattr(model, "strategy_encoder", None) is not None:
        raise RuntimeError("generalist forbids a q_phi / strategy encoder")
    if getattr(getattr(model, "latent_actor", None), "strategy_embedding", None) is not None:
        raise RuntimeError("generalist forbids a z embedding")


def build_generalist_student(spec_ckpt_path: str, observation_space, action_space, *,
                             seed: int, device: str):
    """GENERALIST_DEFINITION_V1: fresh draw of the specialist architecture under ``seed``, no z.

    Trained by the same objective as Fully Shared+z; with no z the two KL terms act on one
    output, so it fits 0.5*KL(pi_A||pi_G) + 0.5*KL(pi_B||pi_G) on every stored state.
    """
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload, kw = _specialist_arch(spec_ckpt_path, observation_space, action_space)
    torch.manual_seed(int(seed))
    model = SharedActorCentralizedCritic(observation_space, action_space, **kw).to(device)
    assert_generalist_structure(model)
    ref = payload["model_state_dict"]
    sd = model.state_dict()
    same = [k for k in sd if k in ref and tuple(sd[k].shape) == tuple(ref[k].shape) and sd[k].numel()
            and torch.equal(sd[k].detach().cpu().float(), ref[k].float())]
    # Every tensor has the specialist's shape here, so any bit-identical weight matrix is a warm start.
    if any(k.endswith("weight") for k in same):
        raise RuntimeError(f"generalist copies specialist weights bit-exactly -- silent warm start ({same[:3]})")
    return model, dict(payload.get("cfg") or {}), kw


def save_generalist(model, specialist_cfg: dict, kwargs: dict, out_path: str, provenance: dict) -> None:
    payload = {
        "format": "suite_generalist_v1",
        "cfg": {**dict(specialist_cfg or {}), "use_latent_strategy": False, "suite_arm": "generalist"},
        "model_kwargs": dict(kwargs),
        "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "suite_generalist": dict(provenance),
    }
    tmp = f"{out_path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def load_generalist(path: str, observation_space, action_space, *, device: str):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "suite_generalist_v1":
        raise RuntimeError(f"{path}: not a suite generalist checkpoint")
    model = SharedActorCentralizedCritic(observation_space, action_space, **dict(payload["model_kwargs"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    assert_generalist_structure(model)
    return model, payload


def load_fully_shared(path: str, observation_space, action_space, *, device: str):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    fmt = payload.get("format")
    if fmt not in ("suite_fully_shared_z_v1", "suite_fully_shared_z_r_v1"):
        raise RuntimeError(f"{path}: not a suite fully-shared checkpoint")
    kw = dict(payload["model_kwargs"])
    model = SharedActorCentralizedCritic(observation_space, action_space, **kw)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    assert_fully_shared_structure(model)
    return model, payload
