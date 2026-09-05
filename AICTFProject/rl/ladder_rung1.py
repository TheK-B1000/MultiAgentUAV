"""Sharing-ladder Rung 1: shared CNN encoder, everything else separate per z.

Implements RUNG1_CONSTRUCTION_AMENDMENT.json. Two complete actors of the SPECIALISTS' own
architecture (one per z) with the z1 branch's ``actor_cnn`` module REPLACED by the z0 branch's
(parameter tying by object identity). Bodies, action heads (macro + target logits) and critics
stay private. The only difference from Rung 0 is the shared encoder.

Training-side interface (used by rl.teacher_distillation unchanged): ``policy_logits(obs, z_idx)``
selects per row between the two branches, ``_mask_logits`` delegates, ``action_dims`` /
``per_agent_action_dims`` / ``n_agents`` mirror the branches.

Inference-side: each branch is wrapped in ``CustomPPOInferencePolicy`` exactly as the specialists
are, then dispatched by z through ``Rung0DispatchPolicy`` -- the path already verified bit-exact.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


class Rung1Model(nn.Module):
    def __init__(self, branch_z0: nn.Module, branch_z1: nn.Module):
        super().__init__()
        branch_z1.actor_cnn = branch_z0.actor_cnn          # tie by identity
        self.branch = nn.ModuleDict({"z0": branch_z0, "z1": branch_z1})
        ref = branch_z0
        self.action_dims = tuple(ref.action_dims)
        self.per_agent_action_dims = tuple(ref.per_agent_action_dims)
        self.n_agents = int(ref.n_agents)
        self.latent_k = 2
        self.uses_latent_strategy = True

    @property
    def shared_cnn(self) -> nn.Module:
        return self.branch["z0"].actor_cnn

    def encoder_is_shared(self) -> bool:
        return self.branch["z0"].actor_cnn is self.branch["z1"].actor_cnn

    def policy_logits(self, obs, z_idx=None, **_):
        if z_idx is None:
            raise ValueError("Rung1Model.policy_logits requires z_idx (per-row dispatch)")
        la = self.branch["z0"].policy_logits(obs, z_idx=None)
        lb = self.branch["z1"].policy_logits(obs, z_idx=None)
        sel = (z_idx.reshape(-1, 1) == 0)
        return torch.where(sel, la, lb)

    def _mask_logits(self, logits, mask):
        return self.branch["z0"]._mask_logits(logits, mask)


# ----------------------------------------------------------------- parameter partitions
def shared_parameters(m: Rung1Model):
    return [(n, p) for n, p in m.named_parameters() if ".actor_cnn." in n]


def critic_parameters(m: Rung1Model):
    return [(n, p) for n, p in m.named_parameters() if ".critic." in n]


def actor_parameters(m: Rung1Model):
    return [(n, p) for n, p in m.named_parameters() if ".critic." not in n]


def private_actor_parameters(m: Rung1Model, z: int):
    tag = f"branch.z{z}."
    return [(n, p) for n, p in m.named_parameters()
            if n.startswith(tag) and ".critic." not in n and ".actor_cnn." not in n]


def sharing_arithmetic(m: Rung1Model) -> dict:
    n_unique = sum(p.numel() for _, p in m.named_parameters())
    n_cnn = sum(p.numel() for p in m.shared_cnn.parameters())
    n_branch = sum(p.numel() for p in m.branch["z0"].parameters())
    expected = n_cnn + 2 * (n_branch - n_cnn)
    return {"n_unique": n_unique, "n_cnn": n_cnn, "n_branch_total": n_branch,
            "expected_unique": expected, "ok": n_unique == expected}


# ----------------------------------------------------------------- build / save / load
def _specialist_arch(spec_ckpt_path: str, observation_space, action_space):
    from rl.custom_ppo.checkpoints.archive import read_checkpoint_payload
    from rl.custom_ppo.checkpoints.loader import _architecture_from_metadata
    from rl.custom_ppo.checkpoints.metadata import parse_checkpoint_metadata
    payload = read_checkpoint_payload(str(spec_ckpt_path), map_location="cpu")
    metadata = parse_checkpoint_metadata(payload, str(spec_ckpt_path), observation_space, action_space)
    arch = _architecture_from_metadata(metadata, observation_space, action_space)
    return payload, dict(arch.model_kwargs)


def build_rung1(spec_ckpt_path: str, observation_space, action_space, *, seeds=(11_961_001, 11_961_002),
                device: str = "cpu"):
    """Fresh Rung 1 with the specialists' architecture per branch. Returns (model, cfg, kwargs,
    specialist_state_dict) so callers can verify names/shapes and non-identity to pi_A."""
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    payload, kwargs = _specialist_arch(spec_ckpt_path, observation_space, action_space)
    torch.manual_seed(int(seeds[0]))
    b0 = SharedActorCentralizedCritic(observation_space, action_space, **kwargs)
    torch.manual_seed(int(seeds[1]))
    b1 = SharedActorCentralizedCritic(observation_space, action_space, **kwargs)
    ref = payload["model_state_dict"]
    for tag, b in (("z0", b0), ("z1", b1)):
        sd = b.state_dict()
        if list(sd.keys()) != list(ref.keys()):
            raise RuntimeError(f"Rung 1 branch {tag} parameter names differ from the specialist architecture")
        bad = [k for k in sd if tuple(sd[k].shape) != tuple(ref[k].shape)]
        if bad:
            raise RuntimeError(f"Rung 1 branch {tag} shapes differ from the specialist architecture at {bad}")
        diff = max(float((sd[k].float() - ref[k].float()).abs().max()) for k in sd if sd[k].numel())
        if diff == 0.0:
            raise RuntimeError(f"Rung 1 branch {tag} is bit-identical to pi_A -- silent warm start")
    model = Rung1Model(b0, b1).to(device)
    return model, dict(payload.get("cfg") or {}), kwargs, ref


def save_rung1(model: Rung1Model, cfg: dict, kwargs: dict, out_path: str, provenance: dict) -> None:
    payload = {"rung": 1, "format": "sharing_ladder_rung1_v1",
               "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               "branch_cfg": cfg, "model_kwargs": kwargs, "provenance": dict(provenance)}
    tmp = f"{out_path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def load_rung1(path: str, observation_space, action_space, *, device: str):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != "sharing_ladder_rung1_v1":
        raise RuntimeError(f"{path}: not a Rung 1 checkpoint")
    kw = payload["model_kwargs"]
    b0 = SharedActorCentralizedCritic(observation_space, action_space, **kw)
    b1 = SharedActorCentralizedCritic(observation_space, action_space, **kw)
    model = Rung1Model(b0, b1)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    if not model.encoder_is_shared():
        raise RuntimeError("loaded Rung 1 model lost encoder sharing")
    return model, dict(payload["branch_cfg"]), payload


def make_dispatch_policy(model: Rung1Model, cfg: dict, *, device: str):
    """Each branch wrapped exactly as a specialist is, then dispatched by z through the
    bit-exact-verified Rung 0 wrapper."""
    from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy
    from rl.rung0_dispatch import Rung0DispatchPolicy
    pa = CustomPPOInferencePolicy(model.branch["z0"], device=device, cfg=cfg)
    pb = CustomPPOInferencePolicy(model.branch["z1"], device=device, cfg=cfg)
    return Rung0DispatchPolicy(pa, pb)


@torch.no_grad()
def logits_for_z(model: Rung1Model, obs: dict, z: int, device: str) -> list[torch.Tensor]:
    from rl.teacher_distillation import head_logits
    b = int(next(iter(obs.values())).shape[0])
    zt = torch.full((b,), int(z), dtype=torch.long, device=device)
    return head_logits(model, obs, z_idx=zt)


# ===================================================================== Rung 2 (generic tying)
SHARED_BY_RUNG = {1: ("actor_cnn",), 2: ("actor_cnn", "latent_actor.body")}


def _get_module(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _set_module(root, dotted: str, value):
    parts = dotted.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


class LadderModel(Rung1Model):
    """Rung 1 generalised: tie an arbitrary set of modules from branch z0 into branch z1."""

    def __init__(self, branch_z0, branch_z1, shared_modules):
        self._shared_names = tuple(shared_modules)
        for name in self._shared_names:
            _set_module(branch_z1, name, _get_module(branch_z0, name))
        super().__init__(branch_z0, branch_z1)

    def modules_are_shared(self) -> bool:
        return all(_get_module(self.branch["z0"], n) is _get_module(self.branch["z1"], n)
                   for n in self._shared_names)

    def encoder_is_shared(self) -> bool:      # keep the Rung-1 check meaningful
        return _get_module(self.branch["z0"], "actor_cnn") is _get_module(self.branch["z1"], "actor_cnn")


def shared_module_parameters(m: LadderModel):
    keys = tuple(f".{n}." for n in m._shared_names)
    return [(n, p) for n, p in m.named_parameters() if any(k in n for k in keys)]


def private_actor_parameters_generic(m: LadderModel, z: int):
    tag = f"branch.z{z}."
    keys = tuple(f".{n}." for n in m._shared_names)
    return [(n, p) for n, p in m.named_parameters()
            if n.startswith(tag) and ".critic." not in n and not any(k in n for k in keys)]


def sharing_arithmetic_generic(m: LadderModel) -> dict:
    n_unique = sum(p.numel() for _, p in m.named_parameters())
    n_shared = sum(p.numel() for p in
                   {id(q): q for name in m._shared_names
                    for q in _get_module(m.branch["z0"], name).parameters()}.values())
    n_branch = sum(p.numel() for p in m.branch["z0"].parameters())
    expected = n_shared + 2 * (n_branch - n_shared)
    return {"n_unique": n_unique, "n_shared": n_shared, "n_branch_total": n_branch,
            "expected_unique": expected, "ok": n_unique == expected,
            "shared_modules": list(m._shared_names)}


def build_rung(rung: int, spec_ckpt_path: str, observation_space, action_space, *, seeds,
               device: str = "cpu"):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    payload, kwargs = _specialist_arch(spec_ckpt_path, observation_space, action_space)
    torch.manual_seed(int(seeds[0]))
    b0 = SharedActorCentralizedCritic(observation_space, action_space, **kwargs)
    torch.manual_seed(int(seeds[1]))
    b1 = SharedActorCentralizedCritic(observation_space, action_space, **kwargs)
    ref = payload["model_state_dict"]
    for tag, b in (("z0", b0), ("z1", b1)):
        sd = b.state_dict()
        if list(sd.keys()) != list(ref.keys()):
            raise RuntimeError(f"Rung {rung} branch {tag} parameter names differ from the specialist architecture")
        if [tuple(sd[k].shape) for k in sd] != [tuple(ref[k].shape) for k in ref]:
            raise RuntimeError(f"Rung {rung} branch {tag} shapes differ from the specialist architecture")
        if max(float((sd[k].float() - ref[k].float()).abs().max()) for k in sd if sd[k].numel()) == 0.0:
            raise RuntimeError(f"Rung {rung} branch {tag} is bit-identical to pi_A -- silent warm start")
    model = LadderModel(b0, b1, SHARED_BY_RUNG[int(rung)]).to(device)
    return model, dict(payload.get("cfg") or {}), kwargs, ref


def save_rung(rung: int, model: LadderModel, cfg: dict, kwargs: dict, out_path: str, provenance: dict) -> None:
    payload = {"rung": int(rung), "format": f"sharing_ladder_rung{int(rung)}_v1",
               "shared_modules": list(model._shared_names),
               "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               "branch_cfg": cfg, "model_kwargs": kwargs, "provenance": dict(provenance)}
    tmp = f"{out_path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, out_path)


def load_rung(rung: int, path: str, observation_space, action_space, *, device: str):
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("format") != f"sharing_ladder_rung{int(rung)}_v1":
        raise RuntimeError(f"{path}: not a Rung {rung} checkpoint (format {payload.get('format')!r})")
    kw = payload["model_kwargs"]
    b0 = SharedActorCentralizedCritic(observation_space, action_space, **kw)
    b1 = SharedActorCentralizedCritic(observation_space, action_space, **kw)
    model = LadderModel(b0, b1, tuple(payload.get("shared_modules") or SHARED_BY_RUNG[int(rung)]))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    if not model.modules_are_shared():
        raise RuntimeError(f"loaded Rung {rung} model lost module sharing")
    return model, dict(payload["branch_cfg"]), payload
