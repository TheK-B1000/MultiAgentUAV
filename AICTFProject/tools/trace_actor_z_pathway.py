#!/usr/bin/env python3
"""Trace actor-z causal leverage: z → embed/FiLM → trunk → logits."""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.update.actor_z_pathway import trace_actor_z_pathway
from rl.presets import apply_preset


def _load_model(preset: str, checkpoint: str | None):
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.trainer import CustomPPOTrainer

    cfg = apply_preset(PPOConfig(), preset)
    cfg.device = "cpu"
    cfg.n_envs = 1
    cfg.max_blue_agents = 2
    cfg.n_steps = 8
    cfg.batch_size = 8
    cfg.enable_eval = False
    cfg.verbose_training = False
    env = GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            n_agents_per_team=2,
            max_decision_steps=32,
            device="cpu",
            seed=0,
            map_layout=cfg.map_layout,
        )
    )
    trainer = CustomPPOTrainer(
        env,
        cfg,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        n_epochs=1,
        batch_size=8,
    )
    if checkpoint:
        trainer.load(checkpoint)
    return trainer, env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="v6i2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--z-a", type=int, default=0)
    parser.add_argument("--z-b", type=int, default=1)
    args = parser.parse_args(argv)

    trainer, env = _load_model(args.preset, args.checkpoint)
    try:
        rollout = trainer.collect_rollout()
        buffer = rollout
        length = int(buffer.pos)
        total = length * int(buffer.n_envs)
        obs_batch = {
            "grid": buffer.fields["obs_grid"][:length].reshape(
                total, *buffer.fields["obs_grid"].shape[2:]
            )[:4],
            "vec": buffer.fields["obs_vec"][:length].reshape(
                total, *buffer.fields["obs_vec"].shape[2:]
            )[:4],
            "agent_mask": buffer.fields["obs_agent_mask"][:length].reshape(
                total, *buffer.fields["obs_agent_mask"].shape[2:]
            )[:4],
            "mask": buffer.fields["obs_mask"][:length].reshape(
                total, *buffer.fields["obs_mask"].shape[2:]
            )[:4],
        }
        report = trace_actor_z_pathway(
            trainer.model,
            obs_batch,
            z_a=args.z_a,
            z_b=args.z_b,
        )
        print(f"conditioning={report.conditioning_mode} weakest_stage={report.weakest_stage}")
        print(f"logits_pairwise_jsd_mean={report.logits_pairwise_jsd_mean:.8e}")
        for stage in report.stages:
            print(
                f"  {stage.name:8s} mean_l2={stage.pair_mean_l2:.8e} "
                f"max_l2={stage.pair_max_l2:.8e}"
            )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
