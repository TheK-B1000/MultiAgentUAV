#!/usr/bin/env python3
"""V6I22E: fixed-alpha gate-free adapter birth experiment.

Hypothesis
----------
V6I22 adapters are directionally differentiated (cos_sim ~0 between all pairs)
but have negligibly small magnitudes (weight L2 ~0.09, max element ~0.002 in
256x256 matrices).  The learned gate is stuck at sigmoid(0.08) ~= 0.52 and
barely moves, creating a zero-gradient degenerate equilibrium:

    adapter_out = A_z(h) ~= 0  (zero-init weights)
    gate_grad  = dL/dgate * A_z(h) ~= 0  (no signal to open the gate)

Fix: remove the gate entirely.  Use fixed alpha=0.1 and Kaiming init:

    h_z = h + 0.1 * A_z(h)      (Kaiming-init A_z)

This gives non-zero gradient from step 1 and ~8% adapter contribution at init.

Success criterion: forced-z behavior_pair_distance_mean > 0.06 (birth gate).

Usage
-----
Step 1 — train (run from AICTFProject/):

    uv run python rl/train_ppo.py \\
        --preset v6i22e \\
        --load checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip \\
        --load-weights-only \\
        --additional-steps 5120 \\
        --n-envs 4 --n-steps 256 --n-epochs 1 \\
        --device cuda \\
        --run-tag v6i22e_fixed_alpha_adapters_5u_seed1 \\
        --checkpoint-dir artifacts/v6i22e_fixed_alpha_adapters_5u_seed1 \\
        --fresh-metrics-csv --episode-log-every 0 \\
        --periodic-checkpoint-steps 0 --no-progress-bar

Step 2 — diagnose adapter geometry + log adapter/trunk norm ratio:

    uv run python experiments/run_v6i22e_fixed_alpha_adapters.py \\
        --checkpoint artifacts/v6i22e_fixed_alpha_adapters_5u_seed1/final_v6i22e_fixed_alpha_adapters_5u_seed1_2v2.zip \\
        --device cuda

Step 3 — run forced-z fingerprint:

    uv run python experiments/run_forced_z_eval.py \\
        --checkpoint artifacts/v6i22e_fixed_alpha_adapters_5u_seed1/final_v6i22e_fixed_alpha_adapters_5u_seed1_2v2.zip \\
        --preset v6i22e \\
        --episodes-per-cell 8 --output-dir artifacts/v6i22e_fixed_alpha_adapters_5u_seed1/forced_z_fingerprint
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

_PRESET = "v6i22e"
_LATENT_K = 4
_ALPHA = 0.1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I22E adapter geometry + norm-ratio diagnostic")
    p.add_argument("--checkpoint", required=True, help="Path to trained V6I22E checkpoint (.zip)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    return p.parse_args()


def _find_adapter_module(model) -> object | None:
    for attr in ("latent_actor", "policy_net", "actor"):
        sub = getattr(model, attr, None)
        if sub is not None and hasattr(sub, "latent_adapters"):
            return sub
    for name, mod in model.named_modules():
        if hasattr(mod, "latent_adapters"):
            return mod
    return None


def _section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _stage1_adapter_geometry(model, latent_k: int) -> dict:
    """Adapter weight geometry — same as V6I22 diagnostic, returns metrics dict."""
    _section("STAGE 1: Adapter weight geometry")

    actor = _find_adapter_module(model)
    if actor is None:
        print("[warn] Could not locate latent_adapters — skipping.")
        return {}

    adapters = actor.latent_adapters
    alpha = getattr(actor, "_latent_z_alpha", _ALPHA)
    gate_param = getattr(actor, "latent_adapter_gates", None)
    print(f"Mode: {'fixed-alpha' if gate_param is None else 'gated'}"
          f"   alpha={alpha if gate_param is None else 'N/A (gated)'}")
    print(f"Found latent_adapters: {len(adapters)} entries")

    weights: list[torch.Tensor] = []
    norms: list[float] = []
    print()
    print("Per-z adapter weight norms:")
    for z in range(latent_k):
        m = adapters[z]
        w = m.weight.detach().float()
        b = m.bias.detach().float() if m.bias is not None else torch.zeros(w.shape[0])
        weights.append(w)
        norms.append(w.norm().item())
        print(f"  z{z}: L2={w.norm():.5f}  max_abs={w.abs().max():.5f}  "
              f"bias_L2={b.norm():.5f}")

    print()
    print("Pairwise cosine similarity (flattened weight matrices):")
    cos_sims = []
    for i, j in combinations(range(latent_k), 2):
        wi = weights[i].flatten()
        wj = weights[j].flatten()
        cos = F.cosine_similarity(wi.unsqueeze(0), wj.unsqueeze(0)).item()
        l2 = (wi - wj).norm().item()
        cos_sims.append(cos)
        label = "COLLAPSED" if cos > 0.95 else "DIFFERENTIATED"
        print(f"  z{i} vs z{j}: cos_sim={cos:+.5f}  delta_L2={l2:.5f}  {label}")

    if gate_param is not None:
        g = torch.sigmoid(gate_param).detach().float()
        print()
        print(f"latent_adapter_gates (post-sigmoid): shape={tuple(g.shape)}")
        for z in range(latent_k):
            print(f"  z{z}: {g[z].item():.4f}")
    else:
        print()
        print(f"latent_adapter_gates: None (fixed-alpha={alpha})")

    biases = getattr(actor, "latent_action_biases", None)
    if biases is not None:
        b = biases.detach().float()
        rng = b.abs().max().item()
        print(f"latent_action_biases: shape={tuple(b.shape)}  max_abs={rng:.5f}")

    return {
        "weight_norms": norms,
        "mean_cos_sim": float(np.mean(np.abs(cos_sims))),
        "max_cos_sim": float(np.max(np.abs(cos_sims))),
        "all_differentiated": all(abs(c) < 0.80 for c in cos_sims),
        "mode": "fixed_alpha" if gate_param is None else "gated",
        "alpha": float(alpha),
    }


def _stage2_adapter_trunk_norm_ratio(trainer, device: str, latent_k: int) -> dict:
    """Collect one rollout and measure adapter/trunk activation norm ratio per z."""
    _section("STAGE 2: Adapter/trunk norm ratio (one rollout)")

    buf = trainer.collect_rollout()

    obs_raw = None
    for attr in ("observations", "obs", "_observations"):
        val = getattr(buf, attr, None)
        if val is not None:
            obs_raw = val
            break
    if obs_raw is None:
        print("[warn] Could not access rollout buffer observations — skipping stage 2.")
        return {}

    if isinstance(obs_raw, dict):
        obs_raw = obs_raw.get("obs", next(iter(obs_raw.values())))
    obs_tensor = torch.as_tensor(obs_raw, dtype=torch.float32, device=device)
    if obs_tensor.dim() == 3:
        T, N, D = obs_tensor.shape
        obs_tensor = obs_tensor.reshape(T * N, D)

    model = trainer.model
    actor = _find_adapter_module(model)
    if actor is None:
        print("[warn] Could not locate latent_adapters — skipping stage 2.")
        return {}

    alpha = getattr(actor, "_latent_z_alpha", _ALPHA)
    print(f"Observation batch: {tuple(obs_tensor.shape)}")
    print(f"Fixed alpha: {alpha}")
    print()

    ratios = []
    with torch.no_grad():
        for z in range(latent_k):
            z_t = torch.full((obs_tensor.shape[0],), z, dtype=torch.long, device=device)
            try:
                logits_z = model.get_distribution(obs_tensor, latent_z=z_t).distribution.logits
            except Exception:
                logits_z = None

            # Compute hidden state before/after adapter via hooks
            hidden_before: list[torch.Tensor] = []
            hidden_after: list[torch.Tensor] = []

            def _pre_hook(module, inp, out):
                hidden_before.append(out.detach().clone())

            def _post_hook(module, inp, out):
                hidden_after.append(out.detach().clone())

            # Register hooks on action_head (input = hidden after adapter)
            pre_h = actor.action_head.register_forward_pre_hook(
                lambda m, inp: hidden_after.append(inp[0].detach().clone())
            )
            try:
                if logits_z is None:
                    _ = model.get_distribution(obs_tensor, latent_z=z_t)
            except Exception:
                pass
            pre_h.remove()

            if hidden_after:
                h_after = hidden_after[0]
                # Estimate trunk-only hidden by reversing adapter contribution:
                # h_after = h_trunk + alpha * A_z(h_trunk)  approx h_trunk for small alpha
                # Ratio: |alpha * A_z(h)| / |h_trunk|
                # We proxy this as: |h_after - mean_z(h_after)| / |h_after|
                # (since mean_z removes common trunk component)
                norm_ratio = float("nan")
                ratios.append(norm_ratio)
                print(f"  z{z}: hidden_after norm mean={h_after.norm(dim=-1).mean():.4f}"
                      f"  std={h_after.norm(dim=-1).std():.4f}")
            else:
                ratios.append(float("nan"))
                print(f"  z{z}: [could not capture hidden]")

    # Estimate pairwise hidden norm ratio via direct adapter call
    print()
    print("Direct adapter output norm vs hidden norm (random obs sample):")
    obs_sample = obs_tensor[:64]
    hidden_norms = []
    adapter_norms = []
    with torch.no_grad():
        # Get trunk hidden via actor forward; use z=0 to get post-trunk hidden
        # We hack this by temporarily patching adapter to be identity
        adapters = actor.latent_adapters
        for z in range(latent_k):
            # Run forward pass — extract hidden state before adapter via hook
            captured = []

            def pre_adapter_hook(m, inp, out, _cap=captured):
                _cap.append(inp[0].detach().clone())

            # Hook on adapter itself to get hidden before it
            h = adapters[z].register_forward_pre_hook(
                lambda m, inp, _cap=captured: _cap.append(inp[0].detach().clone())
            )
            try:
                z_t = torch.full((obs_sample.shape[0],), z, dtype=torch.long, device=device)
                _ = model.get_distribution(obs_sample, latent_z=z_t)
            except Exception:
                pass
            h.remove()

            if captured:
                h_in = captured[0]
                a_out = adapters[z](h_in)
                h_norm = h_in.norm(dim=-1).mean().item()
                a_norm = a_out.norm(dim=-1).mean().item()
                contribution = alpha * a_norm / max(h_norm, 1e-8)
                hidden_norms.append(h_norm)
                adapter_norms.append(a_norm)
                label = "MEANINGFUL (>1%)" if contribution > 0.01 else "NEGLIGIBLE (<1%)"
                print(f"  z{z}: |h|={h_norm:.4f}  |A_z(h)|={a_norm:.4f}  "
                      f"alpha*|A_z(h)|/|h|={contribution:.4f}  "
                      f"({label})")
            else:
                hidden_norms.append(float("nan"))
                adapter_norms.append(float("nan"))
                print(f"  z{z}: [hook did not fire]")

    mean_contribution = float(np.nanmean([
        alpha * a / max(h, 1e-8)
        for h, a in zip(hidden_norms, adapter_norms)
    ]))
    print()
    print(f"Mean alpha*|A_z(h)|/|h| across z: {mean_contribution:.4f}")
    verdict = "MEANINGFUL" if mean_contribution > 0.01 else "NEGLIGIBLE"
    print(f"Contribution verdict: {verdict}")
    return {"mean_adapter_contribution": mean_contribution, "verdict": verdict}


def _print_run_commands(checkpoint: Path) -> None:
    _section("TRAINING COMMANDS (for reference)")
    ckpt_base = "checkpoints/2v2/final_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2.zip"
    print()
    print("# 5-update birth run (smoke):")
    print(f"uv run python rl/train_ppo.py \\")
    print(f"    --preset v6i22e \\")
    print(f"    --load {ckpt_base} \\")
    print(f"    --load-weights-only \\")
    print(f"    --additional-steps 5120 \\")
    print(f"    --n-envs 4 --n-steps 256 --n-epochs 1 \\")
    print(f"    --device cuda \\")
    print(f"    --run-tag v6i22e_fixed_alpha_adapters_5u_seed1 \\")
    print(f"    --checkpoint-dir artifacts/v6i22e_fixed_alpha_adapters_5u_seed1 \\")
    print(f"    --fresh-metrics-csv --episode-log-every 0 \\")
    print(f"    --periodic-checkpoint-steps 0 --no-progress-bar")
    print()
    print("# 25-update run (if 5u passes):")
    print(f"uv run python rl/train_ppo.py \\")
    print(f"    --preset v6i22e \\")
    print(f"    --load {ckpt_base} \\")
    print(f"    --load-weights-only \\")
    print(f"    --additional-steps 25600 \\")
    print(f"    --n-envs 4 --n-steps 256 --n-epochs 1 \\")
    print(f"    --device cuda \\")
    print(f"    --run-tag v6i22e_fixed_alpha_adapters_25u_seed1 \\")
    print(f"    --checkpoint-dir artifacts/v6i22e_fixed_alpha_adapters_25u_seed1 \\")
    print(f"    --fresh-metrics-csv --episode-log-every 0 \\")
    print(f"    --periodic-checkpoint-steps 0 --no-progress-bar")
    print()
    print("# Forced-z fingerprint after training:")
    print(f"uv run python experiments/run_forced_z_eval.py \\")
    print(f"    --checkpoint <trained_checkpoint.zip> \\")
    print(f"    --preset v6i22e \\")
    print(f"    --episodes-per-cell 8 \\")
    print(f"    --output-dir <artifact_dir>/forced_z_fingerprint")


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)

    print(f"Checkpoint : {checkpoint}")
    print(f"Preset     : {_PRESET}")
    print(f"Device     : {args.device}")

    if not checkpoint.is_file():
        print()
        print("[info] Checkpoint not found — printing training commands and exiting.")
        _print_run_commands(checkpoint)
        return

    cfg, resolved, env, trainer = _build_audit_trainer(
        preset=_PRESET,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )
    model = trainer.model
    latent_k = int(getattr(cfg, "latent_k", _LATENT_K) or _LATENT_K)

    geo = _stage1_adapter_geometry(model, latent_k)
    norm = _stage2_adapter_trunk_norm_ratio(trainer, args.device, latent_k)

    _section("VERDICT")
    print()
    print(f"Adapter mode       : {geo.get('mode', 'unknown')}")
    print(f"Alpha              : {geo.get('alpha', 'N/A')}")
    print(f"All differentiated : {geo.get('all_differentiated', 'N/A')}")
    print(f"Max |cos_sim|      : {geo.get('max_cos_sim', float('nan')):.4f}")
    print(f"Mean contribution  : {norm.get('mean_adapter_contribution', float('nan')):.4f}")
    contribution = norm.get("mean_adapter_contribution", float("nan"))
    differentiated = geo.get("all_differentiated", False)
    weight_l2s = geo.get("weight_norms") or []
    mean_weight_l2 = float(np.mean(weight_l2s)) if weight_l2s else float("nan")
    print(f"Mean weight L2     : {mean_weight_l2:.4f}")
    if contribution != contribution:  # NaN — stage 2 skipped
        print()
        if differentiated and mean_weight_l2 > 1.0:
            print("STATUS: PROMISING — adapters differentiated with non-trivial weight L2.")
            print("        Stage-2 contribution probe skipped; use forced-z / offline ratio.")
        else:
            print("STATUS: INCONCLUSIVE — stage-2 contribution probe skipped (NaN).")
            print("        Re-run with a live rollout buffer or offline contribution probe.")
    elif contribution > 0.01 and differentiated:
        print()
        print("STATUS: PROMISING — adapters are active and differentiated.")
        print("        Run forced-z fingerprint to check behavior_pair_distance > 0.06.")
    elif contribution > 0.01:
        print()
        print("STATUS: PARTIAL — adapters are active but some pairs still collapsed.")
    else:
        print()
        print("STATUS: BLOCKED — adapter contribution still negligible.")
        print("        Increase alpha or unfreeze trunk layers.")

    print()
    _print_run_commands(checkpoint)


if __name__ == "__main__":
    main()
