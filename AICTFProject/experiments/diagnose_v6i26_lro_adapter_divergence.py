#!/usr/bin/env python3
"""V6I26 LRO adapter divergence diagnostic — root-causes why z2/z3's forced-z
behavior vectors nearly collapsed (nearest_behavior_distance=0.082 vs a 0.35
threshold) even though their oracle win-margins differ (Phase-2 confirm,
2026-07-24). Same question as experiments/diagnose_v6i22_adapter_divergence.py
asked for V6I22: has z actually differentiated the adapter *weights*, or is
the payoff gap an artifact of noise on top of near-identical policies?

Unlike diagnose_v6i22_adapter_divergence.py, this script does NOT go through
rl.presets.apply_preset -- the V6I26 LRO run (experiments/run_v6i26_lro_oracle_round.py)
was launched with a direct PPOConfig, not a named preset (run_config.json's
cli_preset is null). Reusing a same-family-but-different preset here would
risk building a PPOConfig whose latent/adapter architecture doesn't match
what the checkpoint was actually trained with, which would either crash the
weights-only load or (worse) silently load onto a mismatched architecture.
Instead this script reconstructs PPOConfig directly from the checkpoint's own
sibling *_run_config.json::resolved_ppo_config, guaranteeing an exact match.

Stage 2 also does NOT reuse diagnose_v6i22_adapter_divergence.py's
_stage2_logit_profile: that helper assumes a flat-tensor rollout buffer
(``buf.observations`` / ``.obs``), but this trainer's buffer is
``rl.ppo_core.TensorDictRolloutBuffer`` with dict-style fields
(``obs_grid``/``obs_vec``/``obs_agent_mask``/``obs_mask``), and its policy's
``get_distribution(obs: dict, *, z_idx=...)`` returns a multi-head
distribution (``dist.heads[i].logits``, not ``dist.distribution.logits``).
Stage 2 here is a from-scratch reimplementation against that real interface.

Read-only: loads a checkpoint for inference only, writes nothing.

Usage:
    python experiments/diagnose_v6i26_lro_adapter_divergence.py \\
        --checkpoint artifacts/v6i26_lro_niches_round1_seed1/final_v6i26_lro_z3_r1_25u_seed1.zip \\
        --device cuda
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.config.ppo_config import PPOConfig  # noqa: E402
from rl.training.config_validation import normalize_and_validate_training_config  # noqa: E402
from rl.training.factories import build_training_env  # noqa: E402
from rl.training.initialization import build_trainer, maybe_load_checkpoint  # noqa: E402
from rl.training.lifecycle import _ensure_cuda_or_fallback, set_global_seed  # noqa: E402
from rl.training.resolved_config import resolve_training_config  # noqa: E402

from experiments.diagnose_v6i22_adapter_divergence import _stage1_adapter_geometry  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run-config", default=None, help="Override sibling *_run_config.json path")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--skip-stage2", action="store_true", help="Skip the rollout-based logit profile (weights-only check)")
    return p.parse_args()


def _find_run_config(checkpoint: Path, override: str | None) -> Path:
    if override:
        p = Path(override)
        if not p.is_file():
            raise FileNotFoundError(f"--run-config not found: {p}")
        return p
    # Checkpoint is final_<run_tag>_<NvN>.zip; run_config is <run_tag>_run_config.json.
    stem = checkpoint.stem
    if stem.startswith("final_"):
        stem = stem[len("final_"):]
    # Strip a trailing _NvN team-size suffix if present (e.g. _2v2).
    parts = stem.rsplit("_", 1)
    candidates = [checkpoint.parent / f"{stem}_run_config.json"]
    if len(parts) == 2 and "v" in parts[1] and parts[1][0].isdigit():
        candidates.append(checkpoint.parent / f"{parts[0]}_run_config.json")
    for c in candidates:
        if c.is_file():
            return c
    found = sorted(checkpoint.parent.glob("*_run_config.json"))
    if len(found) == 1:
        return found[0]
    raise FileNotFoundError(
        f"Could not find a unique *_run_config.json next to {checkpoint} "
        f"(tried {candidates}, found {found}). Pass --run-config explicitly."
    )


def _build_cfg_from_run_config(run_config_path: Path, *, checkpoint: str, device: str, seed: int) -> PPOConfig:
    payload = json.loads(run_config_path.read_text(encoding="utf-8"))
    resolved = payload.get("resolved_ppo_config")
    if not isinstance(resolved, dict):
        raise ValueError(f"{run_config_path} has no resolved_ppo_config block")

    cfg = PPOConfig()
    valid_fields = {f.name for f in dataclasses.fields(cfg)}
    applied, skipped = 0, []
    for key, value in resolved.items():
        if key in valid_fields:
            setattr(cfg, key, value)
            applied += 1
        else:
            skipped.append(key)
    print(f"[cfg] applied {applied} fields from {run_config_path.name}; "
          f"{len(skipped)} keys not present on current PPOConfig (likely renamed/removed): "
          f"{skipped[:10]}{'...' if len(skipped) > 10 else ''}")

    cfg.load_path = str(checkpoint)
    cfg.load_weights_only = True
    cfg.seed = int(seed)
    cfg.device = str(device)
    cfg.enable_tensorboard = False
    cfg.enable_checkpoints = False
    cfg.enable_eval = False
    cfg.verbose_training = False
    cfg.fresh_metrics_csv = True
    return cfg


def _flatten_time_env(t: torch.Tensor) -> torch.Tensor:
    """(T, N_env, ...) -> (T*N_env, ...)."""
    return t.reshape((t.shape[0] * t.shape[1],) + tuple(t.shape[2:]))


def _stage2_action_logit_profile(trainer, *, device: str, latent_k: int) -> None:
    """Query the policy directly under every z on one real observation batch,
    bypassing the rollout buffer's generic-attribute assumptions entirely."""
    print()
    print("=" * 72)
    print("STAGE 2: Per-obs action-logit profile (one rollout, direct policy query)")
    print("=" * 72)

    print("Collecting one rollout for observation batch...")
    buf = trainer.collect_rollout()
    n = int(buf.pos)
    required = ("obs_grid", "obs_vec", "obs_agent_mask", "obs_mask")
    missing = [k for k in required if k not in buf.fields]
    if missing:
        print(f"[warn] Buffer is missing expected obs fields {missing} — skipping stage 2. "
              f"Available fields: {sorted(buf.fields.keys())}")
        return

    obs = {
        "grid": _flatten_time_env(buf.fields["obs_grid"][:n]).to(device).float(),
        "vec": _flatten_time_env(buf.fields["obs_vec"][:n]).to(device).float(),
        "agent_mask": _flatten_time_env(buf.fields["obs_agent_mask"][:n]).to(device).float(),
        "mask": _flatten_time_env(buf.fields["obs_mask"][:n]).to(device).float(),
    }
    batch_size = obs["grid"].shape[0]
    print(f"Observation batch: {batch_size} samples")

    model = trainer.model
    per_z_heads: dict[int, list[torch.Tensor]] = {}
    for z in range(latent_k):
        z_idx = torch.full((batch_size,), z, dtype=torch.long, device=device)
        with torch.no_grad():
            dist = model.get_distribution(obs, z_idx=z_idx)
        heads = [h.logits.float() for h in dist.heads]
        per_z_heads[z] = heads
        print(f"  z{z}: {len(heads)} action heads, shapes={[tuple(h.shape) for h in heads]}")

    print()
    print("Pairwise action-logit analysis (all heads concatenated for logit_L2; "
          "argmax_differ/JSD reported per head):")
    summary_l2: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(latent_k), 2):
        li_cat = torch.cat(per_z_heads[i], dim=-1)
        lj_cat = torch.cat(per_z_heads[j], dim=-1)
        l2 = (li_cat - lj_cat).norm(dim=-1)
        summary_l2[(i, j)] = float(l2.mean())

        argmax_differ, jsd_vals = [], []
        for hi, hj in zip(per_z_heads[i], per_z_heads[j]):
            ai, aj = hi.argmax(-1), hj.argmax(-1)
            argmax_differ.append(float((ai != aj).float().mean()))
            pi, pj = torch.softmax(hi, -1), torch.softmax(hj, -1)
            m = 0.5 * (pi + pj)
            jsd = (0.5 * (pi * (pi / (m + 1e-10)).log()).sum(-1) +
                   0.5 * (pj * (pj / (m + 1e-10)).log()).sum(-1)).mean()
            jsd_vals.append(float(jsd))
        print(f"  z{i} vs z{j}: logit_L2 mean={l2.mean():.5f}  "
              f"per_head_argmax_differ={[f'{v:.4f}' for v in argmax_differ]}  "
              f"per_head_JSD={[f'{v:.6f}' for v in jsd_vals]}")

    print()
    print("Cluster check (mean logit_L2 to every other z):")
    for z in range(latent_k):
        others = [summary_l2[(min(z, o), max(z, o))] for o in range(latent_k) if o != z]
        print(f"  z{z}: mean_L2_to_others={sum(others) / len(others):.4f}  per_pair={[f'{v:.3f}' for v in others]}")


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    run_config_path = _find_run_config(checkpoint, args.run_config)
    print(f"Checkpoint : {checkpoint}")
    print(f"Run config : {run_config_path}")
    print(f"Device     : {args.device}")

    cfg = _build_cfg_from_run_config(run_config_path, checkpoint=str(checkpoint), device=args.device, seed=args.seed)
    cfg = normalize_and_validate_training_config(cfg)
    _ensure_cuda_or_fallback(cfg)
    set_global_seed(int(cfg.seed))

    resolved = resolve_training_config(cfg)
    env = build_training_env(cfg, initial_phase=resolved.initial_phase, initial_opponent_tag=resolved.initial_opponent_tag)
    trainer = build_trainer(env, cfg, resolved)
    maybe_load_checkpoint(cfg, trainer)

    latent_k = int(getattr(cfg, "latent_k", 4) or 4)
    model = trainer.model

    _stage1_adapter_geometry(model, latent_k)

    if not args.skip_stage2:
        _stage2_action_logit_profile(trainer, device=args.device, latent_k=latent_k)

    print()
    print("=" * 72)
    print("V6I26-specific interpretation:")
    print("  Phase-2 confirm found branch_nearest_behavior_distance(z3,z2)=0.0824")
    print("  against a 0.35 threshold, on the coarse 7-dim per-episode-mean")
    print("  behavior vector, alongside a real oracle win-margin gap. Two")
    print("  distinct failure modes produce a low behavior-vector distance and")
    print("  must not be conflated:")
    print()
    print("  (a) COLLAPSED: the candidate's weights/logits are near-identical to")
    print("      its neighbor specifically. Check Stage 1's z(candidate) row --")
    print("      cos_sim > 0.95 vs that one neighbor (but not others) confirms it.")
    print()
    print("  (b) CLUSTERED-VS-OUTLIER: the candidate is genuinely different from")
    print("      EVERY other z (Stage 2 argmax_differ/JSD large for all its")
    print("      pairs), but the OTHER z's are mutually near-identical to each")
    print("      other (Stage 2 'Cluster check' shows near-equal, large L2 from")
    print("      the candidate to all others, and near-zero L2 among the rest).")
    print("      Here the low behavior-vector distance to 'nearest neighbor' is")
    print("      an artifact of comparing against an undifferentiated cluster,")
    print("      not evidence the candidate lacks a real strategy. This is what")
    print("      z3 vs {z0,z1,z2} showed on 2026-07-24: z3's per-pair L2 to each")
    print("      of z0/z1/z2 was uniformly ~2.9-3.0, while z0/z1/z2's per-pair L2")
    print("      *among themselves* was only ~0.3-0.4 (their mean_L2_to_others is")
    print("      pulled up to ~1.2-1.3 only because the z3 pair drags it there).")
    print("      Fix target shifts from the candidate to the undifferentiated")
    print("      cluster -- they need their own birth/LRO round, not the")
    print("      candidate.")
    print("=" * 72)


if __name__ == "__main__":
    main()
