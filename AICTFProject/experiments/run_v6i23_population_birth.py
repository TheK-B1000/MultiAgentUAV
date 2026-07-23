#!/usr/bin/env python3
"""V6I23: Summer-compatible population birth (independent per-z specialists).

Hypothesis
----------
V6I22–V6I22E achieved Stage-C oracle complementarity but failed CF action-JSD
because Stage-2 freezes the shared ``action_head``. Adapters moved in hidden
space without stably separating π(a|s,z).

Fix (population birth, Summer-compatible — not paper-faithful):
  * keep fixed-alpha residual adapters (α=0.1, Kaiming)
  * active-z-only residual forward
  * independent per-z Linear action heads (Stage-2 trainable)
  * forced ``balanced_episode`` z, router off, no soft diversity rewards,
    no opponent-ID routing

Success gate
------------
CF action-JSD pair mean > 0.05 on ≥2 oracle-hot cells
  OR head0 argmax disagree > 0.2 with non-tie check.
Then freeze specialists and train router.

Usage (from AICTFProject/)
-------------------------
Train 5u smoke:

    uv run python rl/train_ppo.py \\
        --preset v6i23 \\
        --load checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip \\
        --load-weights-only \\
        --additional-steps 5120 \\
        --n-envs 4 --n-steps 256 --n-epochs 1 \\
        --device cuda \\
        --run-tag v6i23_population_birth_5u_seed1 \\
        --checkpoint-dir artifacts/v6i23_population_birth_5u_seed1 \\
        --fresh-metrics-csv --episode-log-every 0 \\
        --periodic-checkpoint-steps 0 --no-progress-bar

CF action-JSD probe:

    uv run python experiments/run_per_cell_action_jsd_probe.py \\
        --checkpoint artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip \\
        --out-dir artifacts/v6i23_population_birth_5u_seed1/action_jsd_probe \\
        --device cuda \\
        --opponents OP8 OP9 OP10 OP11 OP12 \\
        --maps map_b_split_lane map_b_split_lane_v2

Forced-z fingerprint:

    uv run python experiments/run_forced_z_eval.py \\
        --checkpoint artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip \\
        --preset v6i23 \\
        --episodes-per-cell 8 \\
        --output-dir artifacts/v6i23_population_birth_5u_seed1/forced_z_fingerprint
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

from experiments.dump_router_rollout_audit import _build_audit_trainer  # noqa: E402

_PRESET = "v6i23"
_LATENT_K = 4


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I23 population-birth diagnostic")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def _find_actor(model):
    for attr in ("latent_actor", "policy_net", "actor"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "latent_adapters"):
            return sub
    for _, mod in model.named_modules():
        if hasattr(mod, "latent_adapters"):
            return mod
    return None


def main() -> int:
    args = _parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: checkpoint not found: {ckpt}")
        return 2

    trainer = _build_audit_trainer(
        checkpoint=str(ckpt),
        preset=_PRESET,
        device=args.device,
        seed=int(args.seed),
    )
    if isinstance(trainer, tuple):
        _, _, _, trainer = trainer
    model = trainer.model
    actor = _find_actor(model)
    if actor is None:
        print("ERROR: could not locate LatentConditionedActor")
        return 2

    print("=" * 72)
    print("V6I23 population-birth diagnostic")
    print("=" * 72)
    print(f"checkpoint: {ckpt}")
    print(f"active_z_only: {getattr(actor, '_population_birth_active_z_only', False)}")
    heads = getattr(actor, "latent_action_heads", None)
    print(f"per_z_action_heads: {heads is not None}")
    if heads is not None:
        for k, head in enumerate(heads):
            w = head.weight.detach().float()
            b = head.bias.detach().float()
            print(
                f"  head[{k}] weight_L2={float(w.norm()):.4f} "
                f"bias_L2={float(b.norm()):.4f} max_abs={float(w.abs().max()):.4f}"
            )
            # Pairwise head distance vs head 0
            if k > 0:
                d = float((w - heads[0].weight.detach().float()).norm())
                print(f"    L2(head[{k}]-head[0])={d:.4f}")

    adapters = getattr(actor, "latent_adapters", None)
    if adapters is not None:
        print("adapters:")
        for k, ad in enumerate(adapters):
            w = ad.weight.detach().float()
            print(f"  A[{k}] weight_L2={float(w.norm()):.4f} max_abs={float(w.abs().max()):.5f}")

    shared = actor.action_head.weight.detach().float()
    print(f"shared action_head weight_L2={float(shared.norm()):.4f}")

    # Quick forced-z logit separation on random local features
    local = torch.randn(8, actor.local_feature_dim, device=next(actor.parameters()).device)
    with torch.no_grad():
        logits = [actor(local, torch.full((8,), k, dtype=torch.long, device=local.device)) for k in range(_LATENT_K)]
    print("forced-z logit pairwise max-abs (random local):")
    pairs_above = 0
    for i in range(_LATENT_K):
        for j in range(i + 1, _LATENT_K):
            d = float((logits[i] - logits[j]).abs().max())
            print(f"  ({i},{j}) max_abs={d:.4f}")
            if d > 0.5:
                pairs_above += 1
    print(f"pairs with max_abs>0.5: {pairs_above}")
    print()
    print("Next: run CF action-JSD probe (see module docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
