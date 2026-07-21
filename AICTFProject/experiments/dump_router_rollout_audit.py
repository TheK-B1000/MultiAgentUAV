#!/usr/bin/env python3
"""Collect one pre-update router rollout and save it for credit-assignment audit.

Example::

    python experiments/dump_router_rollout_audit.py \\
      --preset v6i9_mapaware_router_feedforward_hardpool \\
      --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
      --device cuda \\
      --seed 1 \\
      --output artifacts/router_credit_audit/update_0001.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.config.ppo_config import PPOConfig  # noqa: E402
from rl.custom_ppo.diagnostics.router_rollout_dump import (  # noqa: E402
    collect_router_rollout_for_audit,
    file_sha256,
    git_commit_hash,
    package_rollout_tensors,
    save_router_rollout_audit,
)
from rl.global_state import GLOBAL_STATE_V6I7_DIM  # noqa: E402
from rl.presets import apply_preset  # noqa: E402
from rl.training.config_validation import normalize_and_validate_training_config  # noqa: E402
from rl.training.factories import build_training_env  # noqa: E402
from rl.training.initialization import build_trainer, maybe_load_checkpoint  # noqa: E402
from rl.training.lifecycle import _ensure_cuda_or_fallback, set_global_seed  # noqa: E402
from rl.training.resolved_config import resolve_training_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump one router rollout for credit audit")
    p.add_argument("--preset", default="v6i9_mapaware_router_feedforward_hardpool")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output", default="artifacts/router_credit_audit/update_0001.pt")
    p.add_argument(
        "--run-update",
        action="store_true",
        help="Run optimizer update after dump (dump always happens first)",
    )
    return p.parse_args()


def _build_audit_trainer(
    *,
    preset: str,
    checkpoint: str,
    device: str,
    seed: int,
):
    cfg = PPOConfig()
    cfg = apply_preset(cfg, preset)
    cfg.load_path = str(checkpoint)
    cfg.seed = int(seed)
    cfg.device = str(device)
    cfg.enable_tensorboard = False
    cfg.enable_checkpoints = False
    cfg.enable_eval = False
    cfg.verbose_training = False
    cfg.fresh_metrics_csv = True
    cfg.load_weights_only = True
    cfg = normalize_and_validate_training_config(cfg)
    _ensure_cuda_or_fallback(cfg)
    set_global_seed(int(cfg.seed))

    resolved = resolve_training_config(cfg)
    env = build_training_env(
        cfg,
        initial_phase=resolved.initial_phase,
        initial_opponent_tag=resolved.initial_opponent_tag,
    )
    trainer = build_trainer(env, cfg, resolved)
    maybe_load_checkpoint(cfg, trainer)
    return cfg, resolved, env, trainer


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cfg, resolved, env, trainer = _build_audit_trainer(
        preset=args.preset,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )

    if not bool(getattr(cfg, "router_reward_enabled", False)):
        raise RuntimeError("Preset must have router_reward_enabled=True")
    if int(getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0) != 0:
        raise RuntimeError("Preset must use feedforward router (recurrent_selector_hidden_dim=0)")

    try:
        buffer = collect_router_rollout_for_audit(trainer)
        tensors, pack_meta = package_rollout_tensors(buffer, cfg=cfg, trainer=trainer)

        metadata = {
            "preset": args.preset,
            "checkpoint_path": str(checkpoint),
            "checkpoint_hash": file_sha256(checkpoint),
            "source_commit": git_commit_hash(PROJECT_ROOT),
            "seed": int(args.seed),
            "global_step": int(getattr(trainer, "global_step", 0)),
            "n_envs": int(buffer.n_envs),
            "rollout_length": int(buffer.pos),
            "router_context_mode": str(getattr(cfg, "router_context_mode", "")),
            "q_phi_input_dimension": int(
                getattr(trainer.model, "router_context_dimension", 0)
                or getattr(trainer.model, "global_state_dim", GLOBAL_STATE_V6I7_DIM)
            ),
            "router_reward_enabled": bool(getattr(cfg, "router_reward_enabled", False)),
            "router_entropy_coefficient": float(getattr(cfg, "router_ent_coef", 0.0) or 0.0),
            "recurrent_selector_hidden_dim": int(getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0),
            "latent_strategy_ppo_coef": float(getattr(cfg, "latent_strategy_ppo_coef", 0.10)),
            "clip_range": float(getattr(trainer, "clip_range", getattr(cfg, "clip_range", 0.2))),
            **pack_meta,
        }

        out = save_router_rollout_audit(
            args.output,
            tensors=tensors,
            metadata=metadata,
            cfg=cfg,
            trainer=trainer,
        )
        print(f"[router-rollout-dump] wrote {out}")
        print(
            f"[router-rollout-dump] decisions={metadata['router_decision_count']} "
            f"advantage_source={metadata['advantage_source_used']} "
            f"router_adv_std={metadata.get('router_advantage_std', 0.0):.6f}"
        )

        if args.run_update:
            stats = trainer.update(buffer, total_timesteps=int(trainer.global_step + buffer.pos))
            print(f"[router-rollout-dump] update complete; keys={sorted(stats.keys())[:8]}...")
    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
