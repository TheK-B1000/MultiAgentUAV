#!/usr/bin/env python3
"""Cross-checkpoint policy-differentiation diagnostic.

experiments/diagnose_v6i26_lro_adapter_divergence.py compares branches *within*
one checkpoint. That's not the right tool when the two branches being compared
live in different files -- e.g. a candidate z0 from a single-branch LRO round
(z1/z2/z3 frozen, stale) versus the real, fully-trained z3 from its own locked
checkpoint. This script loads two checkpoints independently (each via its own
exact resolved_ppo_config, same config-fidelity approach as the sibling
script) and compares branch_a's logits on checkpoint_a against branch_b's
logits on checkpoint_b, on the SAME observation batch.

Read-only: loads checkpoints for inference only, writes nothing.

Usage:
    python experiments/diagnose_v6i26_cross_checkpoint_divergence.py \\
        --checkpoint-a artifacts/.../final_v6i26_lro_kl_ladder_z0_u3_seed1.zip --branch-a 0 \\
        --checkpoint-b artifacts/v6i26_lro_niches_round1_seed1/final_v6i26_lro_z3_r1_25u_seed1.zip --branch-b 3 \\
        --device cuda
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

from rl.training.config_validation import normalize_and_validate_training_config  # noqa: E402
from rl.training.factories import build_training_env  # noqa: E402
from rl.training.initialization import build_trainer, maybe_load_checkpoint  # noqa: E402
from rl.training.lifecycle import _ensure_cuda_or_fallback, set_global_seed  # noqa: E402
from rl.training.resolved_config import resolve_training_config  # noqa: E402

from experiments.diagnose_v6i26_lro_adapter_divergence import (  # noqa: E402
    _build_cfg_from_run_config,
    _find_run_config,
    _flatten_time_env,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-a", required=True)
    p.add_argument("--branch-a", type=int, required=True)
    p.add_argument("--checkpoint-b", required=True)
    p.add_argument("--branch-b", type=int, required=True)
    p.add_argument("--run-config-a", default=None)
    p.add_argument("--run-config-b", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def _build_model(checkpoint: str, run_config_override, *, device: str, seed: int):
    ckpt = Path(checkpoint)
    run_config_path = _find_run_config(ckpt, run_config_override)
    cfg = _build_cfg_from_run_config(run_config_path, checkpoint=str(ckpt), device=device, seed=seed)
    cfg = normalize_and_validate_training_config(cfg)
    _ensure_cuda_or_fallback(cfg)
    set_global_seed(int(cfg.seed))
    resolved = resolve_training_config(cfg)
    env = build_training_env(cfg, initial_phase=resolved.initial_phase, initial_opponent_tag=resolved.initial_opponent_tag)
    trainer = build_trainer(env, cfg, resolved)
    maybe_load_checkpoint(cfg, trainer)
    return trainer, trainer.model


def main() -> None:
    args = _parse_args()
    print(f"Checkpoint A: {args.checkpoint_a}  branch_a=z{args.branch_a}")
    print(f"Checkpoint B: {args.checkpoint_b}  branch_b=z{args.branch_b}")

    trainer_a, model_a = _build_model(args.checkpoint_a, args.run_config_a, device=args.device, seed=args.seed)
    _, model_b = _build_model(args.checkpoint_b, args.run_config_b, device=args.device, seed=args.seed)

    print("\nCollecting one shared observation batch (from checkpoint A's env)...")
    buf = trainer_a.collect_rollout()
    n = int(buf.pos)
    required = ("obs_grid", "obs_vec", "obs_agent_mask", "obs_mask")
    missing = [k for k in required if k not in buf.fields]
    if missing:
        print(f"[warn] Buffer missing expected obs fields {missing} — aborting.")
        return

    obs = {
        "grid": _flatten_time_env(buf.fields["obs_grid"][:n]).to(args.device).float(),
        "vec": _flatten_time_env(buf.fields["obs_vec"][:n]).to(args.device).float(),
        "agent_mask": _flatten_time_env(buf.fields["obs_agent_mask"][:n]).to(args.device).float(),
        "mask": _flatten_time_env(buf.fields["obs_mask"][:n]).to(args.device).float(),
    }
    batch_size = obs["grid"].shape[0]
    print(f"Observation batch: {batch_size} samples")

    with torch.no_grad():
        z_a = torch.full((batch_size,), int(args.branch_a), dtype=torch.long, device=args.device)
        dist_a = model_a.get_distribution(obs, z_idx=z_a)
        heads_a = [h.logits.float() for h in dist_a.heads]

        z_b = torch.full((batch_size,), int(args.branch_b), dtype=torch.long, device=args.device)
        dist_b = model_b.get_distribution(obs, z_idx=z_b)
        heads_b = [h.logits.float() for h in dist_b.heads]

    print(f"A z{args.branch_a}: {len(heads_a)} heads, shapes={[tuple(h.shape) for h in heads_a]}")
    print(f"B z{args.branch_b}: {len(heads_b)} heads, shapes={[tuple(h.shape) for h in heads_b]}")

    la_cat = torch.cat(heads_a, dim=-1)
    lb_cat = torch.cat(heads_b, dim=-1)
    l2 = (la_cat - lb_cat).norm(dim=-1)

    argmax_differ, jsd_vals = [], []
    for ha, hb in zip(heads_a, heads_b):
        aa, ab = ha.argmax(-1), hb.argmax(-1)
        argmax_differ.append(float((aa != ab).float().mean()))
        pa, pb = torch.softmax(ha, -1), torch.softmax(hb, -1)
        m = 0.5 * (pa + pb)
        jsd = (0.5 * (pa * (pa / (m + 1e-10)).log()).sum(-1) +
               0.5 * (pb * (pb / (m + 1e-10)).log()).sum(-1)).mean()
        jsd_vals.append(float(jsd))

    print()
    print("=" * 72)
    print(f"A(z{args.branch_a}) vs B(z{args.branch_b}) on shared observation batch:")
    print(f"  logit_L2 mean={l2.mean():.5f}  median={l2.median():.5f}")
    print(f"  per_head_argmax_differ={[f'{v:.4f}' for v in argmax_differ]}")
    print(f"  per_head_JSD={[f'{v:.6f}' for v in jsd_vals]}")
    print("=" * 72)
    print("Reference from same-checkpoint diagnostic (z0/z1/z2 mutually ~0.3-0.4 L2,")
    print("z3 vs any of them ~2.9-3.0 L2, JSD ~0.008-0.02): use these bands to judge")
    print("whether A vs B here reads as 'genuinely distinct' or 'still redundant'.")


if __name__ == "__main__":
    main()
