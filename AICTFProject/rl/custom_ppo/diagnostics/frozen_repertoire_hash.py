"""Frozen repertoire tensor hashing for router-stage invariance gates."""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import torch

from rl.custom_ppo.trainer_optimizers import is_shared_frozen_actor_param, is_z_specific_actor_param

_FROZEN_NAME_PARTS = (
  "actor_cnn",
  "latent_actor.body",
  "latent_actor.action_head",
  "latent_adapters",
  "latent_adapter_gates",
  "latent_action_biases",
  "strategy_embedding",
)


def is_frozen_repertoire_param_name(name: str) -> bool:
  if is_shared_frozen_actor_param(name) or is_z_specific_actor_param(name):
    return True
  return any(part in name for part in _FROZEN_NAME_PARTS)


def iter_frozen_repertoire_tensors(model_or_state: Any) -> Iterable[tuple[str, torch.Tensor]]:
  if isinstance(model_or_state, dict):
    for name, tensor in sorted(model_or_state.items()):
      if not isinstance(tensor, torch.Tensor):
        continue
      if is_frozen_repertoire_param_name(name):
        yield name, tensor.detach().cpu()
    return
  for name, param in model_or_state.named_parameters():
    if is_frozen_repertoire_param_name(name):
      yield name, param.detach().cpu()


def hash_frozen_repertoire_tensors(model_or_state: Any) -> str:
  digest = hashlib.sha256()
  for name, tensor in iter_frozen_repertoire_tensors(model_or_state):
    digest.update(name.encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
  return digest.hexdigest()


def compare_frozen_repertoire_hashes(anchor: Any, candidate: Any) -> dict[str, Any]:
  anchor_map = dict(iter_frozen_repertoire_tensors(anchor))
  candidate_map = dict(iter_frozen_repertoire_tensors(candidate))
  shared_names = sorted(set(anchor_map) & set(candidate_map))
  mismatched: list[str] = []
  max_abs_delta = 0.0
  z_max_abs_delta = 0.0
  shared_max_abs_delta = 0.0
  for name in shared_names:
    delta = float((candidate_map[name] - anchor_map[name]).abs().max().item())
    max_abs_delta = max(max_abs_delta, delta)
    if is_z_specific_actor_param(name):
      z_max_abs_delta = max(z_max_abs_delta, delta)
    elif is_shared_frozen_actor_param(name):
      shared_max_abs_delta = max(shared_max_abs_delta, delta)
    if delta > 0.0:
      mismatched.append(name)
  return {
    "frozen_tensor_hash_anchor": hash_frozen_repertoire_tensors(anchor),
    "frozen_tensor_hash_candidate": hash_frozen_repertoire_tensors(candidate),
    "frozen_tensor_hash_match": hash_frozen_repertoire_tensors(anchor)
    == hash_frozen_repertoire_tensors(candidate),
    "shared_actor_max_abs_delta": shared_max_abs_delta,
    "z_specific_max_abs_delta": z_max_abs_delta,
    "frozen_max_abs_delta": max_abs_delta,
    "mismatched_tensor_count": len(mismatched),
    "mismatched_tensors": mismatched[:20],
  }


__all__ = [
  "compare_frozen_repertoire_hashes",
  "hash_frozen_repertoire_tensors",
  "is_frozen_repertoire_param_name",
  "iter_frozen_repertoire_tensors",
]
