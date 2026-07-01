"""Configuration dataclasses for the local PPO trainer.

Centralizes the pure-data structures that describe a training run so multiple
modules (trainer, presets, CLI, validation, env factory) can share them
without circular imports. Concrete imports live in :mod:`rl.config.ppo_config`.
"""

from rl.config.ppo_config import PPOConfig, TrainMode

__all__ = ["PPOConfig", "TrainMode"]
