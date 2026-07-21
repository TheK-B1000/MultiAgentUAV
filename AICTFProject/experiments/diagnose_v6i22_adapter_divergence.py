#!/usr/bin/env python3
"""V6I22 adapter divergence diagnostic.

Answers: has z actually differentiated the adapter weights, and if so,
how does that differentiation manifest in action logits?

Two stages:
  1. Adapter weight geometry — pairwise cosine similarity and L2 norms
     across all (z_i, z_j) pairs.  Near-1 cosine = collapsed.
  2. Per-obs logit profile — on a shared batch of observations (one
     collect_rollout with balanced z), compute per-obs action logits under
     all 4 z values and report:
       * distribution of pairwise logit L2 across the batch
       * distribution of action-dimension-wise absolute logit diff
       * fraction of obs where argmax changes across any z pair
       * per-(z_i, z_j) JSD averaged over the batch

Usage
-----
    uv run python experiments/diagnose_v6i22_adapter_divergence.py \\
        --checkpoint artifacts/v6i22_adaptive_hardpool_repertoire_birth_25u_seed1/\\
final_v6i22_adaptive_hardpool_repertoire_birth_25u_seed1_2v2.zip \\
        --device cuda

Interpretation guide
--------------------
    Adapters collapsed (cos_sim > 0.95 for most pairs):
        z has not differentiated at the weight level.  WR gaps are likely
        an artifact of the stochastic training trajectory rather than a
        stable strategy encoding.  Remedy: increase adapter capacity
        (V6I22E) or add a contrastive weight-separation penalty.

    Adapters differentiated (cos_sim < 0.80) but logit L2 low:
        Weights differ but the shared-trunk activation pattern flattens them.
        The adapter output is overwhelmed by the frozen-trunk logits.
        Remedy: scale up adapter gate values, or unfreeze top trunk layers.

    Adapters differentiated AND logit L2 substantial:
        z is doing something; the per-step JSD staying low means the logit
        shifts are on non-argmax dimensions (confidence reranking, not
        direction flipping).  This is the "sub-threshold logit shift"
        hypothesis.  Check which action dimensions differ most.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dump_router_rollout_audit import _build_audit_trainer  # noqa: E402

_PRESET = "v6i22_adaptive_hardpool_repertoire_birth"
_LATENT_K = 4


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I22 adapter weight and logit divergence diagnostic")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stage 1: adapter weight geometry
# ---------------------------------------------------------------------------

def _find_adapter_module(model) -> object | None:
    """Walk the model to find the latent_actor / latent adapter container."""
    for attr in ("latent_actor", "policy_net", "actor"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "latent_adapters"):
            return sub
    # fallback: search named modules
    for name, mod in model.named_modules():
        if hasattr(mod, "latent_adapters"):
            return mod
    return None


def _stage1_adapter_geometry(model, latent_k: int) -> None:
    print()
    print("=" * 72)
    print("STAGE 1: Adapter weight geometry")
    print("=" * 72)

    actor = _find_adapter_module(model)
    if actor is None:
        print("[warn] Could not locate latent_adapters — skipping weight geometry.")
        return

    adapters = actor.latent_adapters
    print(f"Found latent_adapters: {type(adapters).__name__} with {len(adapters)} entries")

    # Per-z weight norms.
    print()
    print("Per-z adapter weight norms:")
    weights: list[torch.Tensor] = []
    for z in range(latent_k):
        m = adapters[z]
        w = m.weight.detach().float()
        b = m.bias.detach().float() if m.bias is not None else torch.zeros(w.shape[0])
        weights.append(w)
        print(f"  z{z}: weight shape={tuple(w.shape)}  L2={w.norm():.5f}  "
              f"max_abs={w.abs().max():.5f}  bias_L2={b.norm():.5f}")

    # Pairwise cosine similarity of flattened weight vectors.
    print()
    print("Pairwise cosine similarity (weight matrices, flattened):")
    for i, j in combinations(range(latent_k), 2):
        wi = weights[i].flatten()
        wj = weights[j].flatten()
        cos = F.cosine_similarity(wi.unsqueeze(0), wj.unsqueeze(0)).item()
        l2 = (wi - wj).norm().item()
        print(f"  z{i} vs z{j}: cos_sim={cos:+.5f}  delta_L2={l2:.5f}  "
              f"{'COLLAPSED' if cos > 0.95 else 'DIFFERENTIATED' if cos < 0.80 else 'borderline'}")

    # Action biases.
    biases = getattr(actor, "latent_action_biases", None)
    if biases is not None:
        b = biases.detach().float()
        print()
        print(f"latent_action_biases: shape={tuple(b.shape)}")
        for z in range(latent_k):
            print(f"  z{z}: {b[z].cpu().numpy().round(4)}")

    # Gates.
    gates = getattr(actor, "latent_adapter_gates", None)
    if gates is not None:
        g = torch.sigmoid(gates).detach().float()
        print()
        print(f"latent_adapter_gates (post-sigmoid): shape={tuple(g.shape)}")
        for z in range(latent_k):
            print(f"  z{z}: mean={g[z].mean():.4f}  std={g[z].std():.4f}  "
                  f"min={g[z].min():.4f}  max={g[z].max():.4f}")


# ---------------------------------------------------------------------------
# Stage 2: per-obs logit profile
# ---------------------------------------------------------------------------

def _get_logits_for_z(model, obs_tensor: torch.Tensor, z_idx: int,
                       latent_state, device: str) -> torch.Tensor:
    """Return action logits for all obs in obs_tensor under a fixed z."""
    # Set the latent index on the latent_state so the model uses it.
    n = obs_tensor.shape[0]
    z_t = torch.full((n,), z_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        # Try standard predict-like interface first.
        try:
            dist = model.get_distribution(obs_tensor, latent_z=z_t)
            return dist.distribution.logits.detach()
        except TypeError:
            pass
        # Fall back: set z on latent_state and call forward.
        if hasattr(latent_state, "z_idx"):
            latent_state.z_idx = z_t
        try:
            dist = model.get_distribution(obs_tensor)
            return dist.distribution.logits.detach()
        except Exception:
            pass
    return torch.zeros(n, 1, device=device)


def _stage2_logit_profile(trainer, device: str, latent_k: int) -> None:
    print()
    print("=" * 72)
    print("STAGE 2: Per-obs logit profile (one rollout)")
    print("=" * 72)

    # Collect one rollout to get a batch of observations.
    print("Collecting one rollout for observation batch...")
    buf = trainer.collect_rollout()

    # Extract observations from the rollout buffer.
    # Different buffer types expose observations differently.
    obs_raw = None
    for attr in ("observations", "obs", "_observations"):
        val = getattr(buf, attr, None)
        if val is not None:
            obs_raw = val
            break
    if obs_raw is None:
        print("[warn] Could not access rollout buffer observations — skipping stage 2.")
        return

    if isinstance(obs_raw, dict):
        obs_raw = obs_raw.get("obs", next(iter(obs_raw.values())))
    obs_tensor = torch.as_tensor(obs_raw, dtype=torch.float32, device=device)
    # Flatten time/env dims: shape [T, N_env, obs_dim] → [T*N_env, obs_dim]
    if obs_tensor.dim() == 3:
        T, N, D = obs_tensor.shape
        obs_tensor = obs_tensor.reshape(T * N, D)
    elif obs_tensor.dim() == 2:
        pass  # already [B, D]
    print(f"Observation batch: {tuple(obs_tensor.shape)}")

    model = trainer.model
    latent_state = getattr(trainer, "latent_state", None)

    # Try to get logits under each z.
    all_logits: dict[int, torch.Tensor] = {}
    for z in range(latent_k):
        logits = _get_logits_for_z(model, obs_tensor, z, latent_state, device)
        if logits.shape[-1] > 1:
            all_logits[z] = logits
            print(f"  z{z}: logits shape={tuple(logits.shape)}  "
                  f"mean_max_logit={logits.max(dim=-1).values.mean():.4f}")

    if len(all_logits) < 2:
        print("[warn] Could not retrieve logits for multiple z values — "
              "the policy interface may require a different approach.")
        print("  Hint: check model.get_distribution() signature or use policy.predict().")
        return

    # Pairwise analysis.
    print()
    print("Pairwise logit analysis across obs batch:")
    for i, j in combinations(range(latent_k), 2):
        if i not in all_logits or j not in all_logits:
            continue
        li = all_logits[i].float()
        lj = all_logits[j].float()
        delta = (li - lj).abs()
        l2 = (li - lj).norm(dim=-1)                 # [B]
        argmax_i = li.argmax(dim=-1)
        argmax_j = lj.argmax(dim=-1)
        argmax_differ = (argmax_i != argmax_j).float().mean().item()
        pi = torch.softmax(li, dim=-1)
        pj = torch.softmax(lj, dim=-1)
        m = 0.5 * (pi + pj)
        jsd = (0.5 * (pi * (pi / (m + 1e-10)).log()).sum(-1) +
               0.5 * (pj * (pj / (m + 1e-10)).log()).sum(-1)).mean().item()
        print(f"  z{i} vs z{j}:")
        print(f"    logit_L2 p50={l2.median():.5f}  p95={torch.quantile(l2, 0.95):.5f}  "
              f"mean={l2.mean():.5f}")
        print(f"    argmax_differ={argmax_differ:.4f}  mean_JSD={jsd:.6f}")
        # Which action dimensions differ most?
        top3 = delta.mean(dim=0).topk(min(3, delta.shape[-1])).indices.tolist()
        top3_vals = delta.mean(dim=0).topk(min(3, delta.shape[-1])).values.tolist()
        print(f"    top-3 differing action dims: {list(zip(top3, [f'{v:.4f}' for v in top3_vals]))}")

    # Per-z logit confidence: are some z values systematically more/less confident?
    print()
    print("Per-z logit confidence (max_logit - second_max):")
    for z, logits in all_logits.items():
        top2 = logits.float().topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1])
        print(f"  z{z}: margin mean={margin.mean():.4f}  p10={torch.quantile(margin, 0.1):.4f}  "
              f"p90={torch.quantile(margin, 0.9):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Checkpoint : {checkpoint}")
    print(f"Preset     : {_PRESET}")
    print(f"Device     : {args.device}")

    cfg, resolved, env, trainer = _build_audit_trainer(
        preset=_PRESET,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )

    latent_k = int(getattr(cfg, "latent_k", _LATENT_K) or _LATENT_K)
    model = trainer.model

    _stage1_adapter_geometry(model, latent_k)
    _stage2_logit_profile(trainer, args.device, latent_k)

    print()
    print("=" * 72)
    print("Interpretation:")
    print("  Stage 1 cos_sim > 0.95 → adapters collapsed → increase capacity")
    print("  Stage 1 cos_sim < 0.80, Stage 2 L2 high → logit shift real but sub-threshold")
    print("  Stage 2 argmax_differ ≈ 0.077 matches training telemetry → adapter effect is")
    print("    concentrated; top action dims will show what z is actually changing")
    print("=" * 72)


if __name__ == "__main__":
    main()
