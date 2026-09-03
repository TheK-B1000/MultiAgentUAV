"""Shared-trunk freezing for TRUNK_FREEZE_SPEC.json.

Freezes the parameters TRUNK_FREEZE_SPEC.json's PARAMETER_PARTITION identifies as shared
between z0 and z1 (verified empirically -- perturb, check whether the model's own masked
logits move -- not assumed from parameter names). Leaves the 20 private per-latent parameters
(latent_adapters, latent_action_heads, latent_branch_trunks, the private critic heads) fully
trainable under ordinary PPO and, in the TREATMENT arm, the existing causal loss.

This module does not decide WHAT the frozen/trainable sets are -- it reads them from the
frozen spec, so a hand-edit to this file cannot silently diverge from what was actually
verified and committed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "artifacts" / "strategic_demand" / "sppo" / "TRUNK_FREEZE_SPEC.json"


class TrunkFreezeError(RuntimeError):
    pass


def load_partition() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise TrunkFreezeError(f"trunk freeze spec not frozen: {spec['status']!r}")
    p = spec["PARAMETER_PARTITION"]
    return {
        "frozen": set(p["FROZEN"]["names"]) | set(p["VESTIGIAL_LEFT_FROZEN_HARMLESSLY"]["names"]),
        "trainable": set(p["TRAINABLE_PRIVATE"]["names"]),
    }


def apply(model: Any) -> dict:
    """Set requires_grad on every named parameter per the frozen partition. Returns a
    report of what was actually touched, so a caller can verify nothing was missed rather
    than assuming full coverage."""
    partition = load_partition()
    seen_frozen, seen_trainable, unknown = set(), set(), []
    for name, p in model.named_parameters():
        if name in partition["frozen"]:
            p.requires_grad_(False)
            seen_frozen.add(name)
        elif name in partition["trainable"]:
            p.requires_grad_(True)
            seen_trainable.add(name)
        else:
            unknown.append(name)
    missing_frozen = partition["frozen"] - seen_frozen
    missing_trainable = partition["trainable"] - seen_trainable
    if missing_frozen or missing_trainable or unknown:
        raise TrunkFreezeError(
            f"model parameters do not match the frozen partition exactly. "
            f"missing_frozen={sorted(missing_frozen)} "
            f"missing_trainable={sorted(missing_trainable)} "
            f"unaccounted_for={sorted(unknown)} -- refusing rather than freezing a "
            "possibly-wrong subset")
    return {"frozen": sorted(seen_frozen), "trainable": sorted(seen_trainable),
           "n_frozen": len(seen_frozen), "n_trainable": len(seen_trainable)}


def verify_frozen_after_step(before: dict, model: Any) -> dict:
    """Compare a pre-step parameter snapshot against the model NOW. Every frozen parameter
    must be bit-identical; at least one trainable parameter must have moved (positive
    control -- an inert optimizer would otherwise pass the frozen check vacuously)."""
    import torch
    partition = load_partition()
    moved_frozen, moved_trainable = [], []
    for name, p in model.named_parameters():
        if name not in before:
            continue
        same = torch.equal(before[name], p.detach())
        if name in partition["frozen"] and not same:
            moved_frozen.append(name)
        if name in partition["trainable"] and not same:
            moved_trainable.append(name)
    if moved_frozen:
        raise TrunkFreezeError(f"REFUSING: supposedly-frozen parameters moved: {moved_frozen}")
    if not moved_trainable:
        raise TrunkFreezeError(
            "REFUSING: no trainable parameter moved at all -- the positive control failed, "
            "so 'frozen parameters did not move' would be vacuously true rather than "
            "meaningful (same class of bug as CCP-S2's zero-training-steps horizon bug)")
    return {"moved_frozen": moved_frozen, "moved_trainable": moved_trainable}


def snapshot(model: Any) -> dict:
    import torch
    return {n: p.detach().clone() for n, p in model.named_parameters()}
