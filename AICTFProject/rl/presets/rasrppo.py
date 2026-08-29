"""Preset adapters for the frozen RASR-PPO causal ladder.

Proposed Preset Review
----------------------
Names: rasrppo_s0_same_block_control, rasrppo_r1_regime_scorer,
rasrppo_r2_private_critic, rasrppo_r3_directed_identity.
Parents: S0 <- frozen SPPPO V1 production; R1 <- S0; R2 <- R1; R3 <- R2.
Classification: SUMMER-COMPATIBLE EXTENSION (S0 is the same-block control).
Question: which frozen scorer, critic-head, and teacher-identity mechanisms are
required to preserve the confirmed SAPPO repertoire in one persistent K=2
controller?

These presets intentionally have no paper_faithful, summer_faithful, or
plan_faithful aliases. They add supervised semantics and external teachers.
"""
from __future__ import annotations

import dataclasses

from rl.config.ppo_config import PPOConfig


def _apply_arm(cfg: PPOConfig, arm: str) -> PPOConfig:
    # Import lazily: the experiment builder imports training modules that also
    # resolve presets.
    from experiments.run_rasrppo_ladder import build_config

    resolved, _parent_contract = build_config(arm)
    for field in dataclasses.fields(PPOConfig):
        setattr(cfg, field.name, getattr(resolved, field.name))
    return cfg


def apply_rasrppo_s0_same_block_control(cfg: PPOConfig) -> PPOConfig:
    """Apply the exact SPPPO V1 treatment on the fresh RASR TRAIN block."""
    return _apply_arm(cfg, "S0")


def apply_rasrppo_r1_regime_scorer(cfg: PPOConfig) -> PPOConfig:
    """Apply S0 plus the frozen four-regime payoff scorer selector."""
    return _apply_arm(cfg, "R1")


def apply_rasrppo_r2_private_critic(cfg: PPOConfig) -> PPOConfig:
    """Apply R1 plus z-selected final centralized-value heads."""
    return _apply_arm(cfg, "R2")


def apply_rasrppo_r3_directed_identity(cfg: PPOConfig) -> PPOConfig:
    """Apply R2 plus the frozen directed teacher-identity objective."""
    return _apply_arm(cfg, "R3")


__all__ = [
    "apply_rasrppo_s0_same_block_control",
    "apply_rasrppo_r1_regime_scorer",
    "apply_rasrppo_r2_private_critic",
    "apply_rasrppo_r3_directed_identity",
]
