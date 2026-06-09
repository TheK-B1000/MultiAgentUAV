"""v4i2 / v4i2b: Return-Ranked Router Distillation from q_probe Data.

This is **offline** supervised distillation of the latent-strategy router
``q_phi`` from the matched-start returns produced by ``tools/q_probe.py``
(v4i1).

Scope guard (mirrors the v4i2 plan):

* This script does **not** touch the actor, critic, value heads, reward,
  opponent pool, environment, arc-credit math, entropy schedule, or the PPO
  trainer. The PPO loop is not invoked at all here.
* It only trains the ``strategy_encoder`` (q_phi) sub-module of the loaded
  checkpoint, with the rest of the model **frozen**.
* No role / phase / opponent-ID labels are introduced. The supervision is
  task return.

Two supervision modes are supported (use one, not both):

* **v4i2 episode-start mode** (default): supervision is per matched starting
  world ``(opponent, probe_seed)``. The K returns are full-episode returns
  under each forced z; targets are ``softmax(((R - mean) / std) / temp)``.
  Inputs: ``--qprobe-csv`` + ``--contexts`` (one context per (opp, seed)).

* **v4i2b arc-boundary mode**: supervision is per arc boundary
  ``(opponent, probe_seed, arc_idx)``. The K targets are the SAME soft-max
  distribution built from the K remaining-returns-from-here values, but
  each of the K examples carries the **per-z** q_phi context (different
  across z because trajectories diverge after step 0). This is logged
  contextual-bandit data labelled ``approx_remaining_return_supervision``
  -- NOT a clone-and-replay counterfactual.
  Inputs: ``--arc-csv`` + ``--arc-contexts``. When both are supplied, the
  script ignores ``--qprobe-csv`` / ``--contexts`` and runs in arc mode.

Inputs
------

* ``--checkpoint``       v4i1 (or later) PPO checkpoint .zip with q_phi.
* ``--qprobe-csv``       Raw q_probe episode CSV (``<run_tag>_qprobe.csv``).
                         Required for episode-start mode.
* ``--contexts``         Contexts NPZ (``<run_tag>_qprobe_contexts.npz``)
                         produced by ``q_probe.py --save-contexts``.
                         Required for episode-start mode.
* ``--arc-csv``          Arc-boundary CSV (``<run_tag>_qprobe_arcs.csv``).
                         Provide together with ``--arc-contexts`` to enable
                         arc-boundary supervision.
* ``--arc-contexts``     Arc-boundary contexts NPZ
                         (``<run_tag>_qprobe_arc_contexts.npz``).
* ``--out``              Output path for the router-distilled checkpoint.

Output
------

A new checkpoint at ``--out`` that is byte-identical to the source
checkpoint EXCEPT the ``strategy_encoder.*`` weights inside
``model_state_dict``. The actor, critic, value heads, and config are
preserved untouched.

Metrics reported (before and after distillation, on train and val splits):

* ``router_top1_accuracy``  fraction of contexts where ``q_phi`` argmax
  equals the q_probe-best z.
* ``router_regret``          mean ``R_best_z - R_qphi_argmax_z``. **Best
  metric** for whether the router is choosing useful z's.
* ``q_phi_mean_max_prob``    average peak probability of q_phi's output
  distribution (router confidence).
* ``ce_loss``                cross-entropy of soft-target vs q_phi probs;
  what we optimize, up to a constant.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Re-route imports the same way tools/q_probe.py does so the script can be
# run from anywhere inside AICTFProject without requiring a pip install.
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.inference import (
    _torch_load_checkpoint,
    load_custom_ppo_policy,
    read_custom_ppo_metadata,
)
from rl.training.env_factory import build_training_env


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ContextRow:
    """One matched-starting-world training example for the router."""

    key: str             # f"{steps}|{opp}|{seed}"
    steps: int
    opponent: str
    seed: int
    context: np.ndarray  # shape [170], float32
    returns: np.ndarray  # shape [K],  float32 (per-z episode return)


def _load_contexts_npz(path: Path) -> dict[str, np.ndarray]:
    """Load the q_probe contexts NPZ (parallel ``keys`` / ``contexts``)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Contexts NPZ not found: {path}. Re-run q_probe.py with --save-contexts."
        )
    with np.load(path, allow_pickle=True) as data:
        keys = [str(k) for k in data["keys"].tolist()]
        ctx = np.asarray(data["contexts"], dtype=np.float32)
    if ctx.shape[0] != len(keys):
        raise ValueError(
            f"Contexts NPZ malformed: {len(keys)} keys vs {ctx.shape[0]} context rows."
        )
    return {keys[i]: ctx[i].astype(np.float32) for i in range(len(keys))}


def _load_qprobe_rows(path: Path) -> list[dict[str, Any]]:
    """Load the raw q_probe episode CSV into a list of dicts (typed strings)."""
    if not path.exists():
        raise FileNotFoundError(f"qprobe CSV not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(dict(row))
    return out


def build_examples(
    *,
    qprobe_rows: list[dict[str, Any]],
    contexts: dict[str, np.ndarray],
    checkpoint_path: str,
    latent_k: int,
) -> list[_ContextRow]:
    """Group q_probe rows into one ``_ContextRow`` per ``(steps, opp, seed)``.

    Only rows whose ``checkpoint_path`` matches the loaded checkpoint are
    used (so returns come from the same model being distilled). Groups that
    are missing any z in ``[0, latent_k)`` or whose context is absent from
    the NPZ are silently skipped (printed as a count at the end).
    """
    ckpt_str = str(Path(checkpoint_path).resolve())
    ckpt_name = Path(checkpoint_path).name

    # Group returns by (steps, opp, seed) -> {z: R}. Filter to rows tied to
    # this checkpoint (by absolute path OR by basename for portability when
    # the CSV captured a different cwd).
    grouped: dict[tuple[int, str, int], dict[int, float]] = {}
    for r in qprobe_rows:
        rp = str(r.get("checkpoint_path", ""))
        if not rp:
            continue
        try:
            rp_resolved = str(Path(rp).resolve())
        except OSError:
            rp_resolved = rp
        if rp_resolved != ckpt_str and Path(rp).name != ckpt_name:
            continue
        try:
            steps = int(r["checkpoint_steps"])
            opp = str(r["opponent"]).upper()
            seed = int(r["probe_seed"])
            z = int(r["z"])
            ret = float(r["episode_return"])
        except (KeyError, ValueError):
            continue
        if not (0 <= z < latent_k):
            continue
        grouped.setdefault((steps, opp, seed), {})[z] = ret

    out: list[_ContextRow] = []
    missing_z = 0
    missing_ctx = 0
    for (steps, opp, seed), zmap in grouped.items():
        if len(zmap) < latent_k or any(z not in zmap for z in range(latent_k)):
            missing_z += 1
            continue
        key = f"{steps}|{opp}|{seed}"
        ctx = contexts.get(key)
        if ctx is None:
            missing_ctx += 1
            continue
        returns = np.array(
            [zmap[z] for z in range(latent_k)], dtype=np.float32
        )
        out.append(
            _ContextRow(
                key=key,
                steps=steps,
                opponent=opp,
                seed=seed,
                context=np.asarray(ctx, dtype=np.float32).reshape(-1),
                returns=returns,
            )
        )
    if missing_z or missing_ctx:
        print(
            f"[v4i2] build_examples: skipped {missing_z} groups missing z, "
            f"{missing_ctx} groups missing context, kept {len(out)}."
        )
    return out


# ---------------------------------------------------------------------------
# Arc-boundary (v4i2b) data loading
# ---------------------------------------------------------------------------

def _load_arc_contexts_npz(path: Path) -> dict[str, np.ndarray]:
    """Load ``<run_tag>_qprobe_arc_contexts.npz`` (same layout as v4i2)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arc-contexts NPZ not found: {path}. Re-run q_probe.py with "
            "--save-arc-contexts."
        )
    with np.load(path, allow_pickle=True) as data:
        keys = [str(k) for k in data["keys"].tolist()]
        ctx = np.asarray(data["contexts"], dtype=np.float32)
    if ctx.shape[0] != len(keys):
        raise ValueError(
            f"Arc-contexts NPZ malformed: {len(keys)} keys vs {ctx.shape[0]} rows."
        )
    return {keys[i]: ctx[i].astype(np.float32) for i in range(len(keys))}


def _load_arc_rows(path: Path) -> list[dict[str, Any]]:
    """Load the arc-boundary CSV into a list of dicts (typed strings)."""
    if not path.exists():
        raise FileNotFoundError(f"Arc CSV not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(dict(row))
    return out


def build_arc_examples(
    *,
    arc_rows: list[dict[str, Any]],
    arc_contexts: dict[str, np.ndarray],
    checkpoint_path: str,
    latent_k: int,
) -> tuple[list[_ContextRow], dict[str, Any]]:
    """v4i2b: emit K examples per K-complete arc-boundary scene.

    Group ``arc_rows`` by ``(checkpoint_steps, opponent, probe_seed, arc_idx)``.
    A scene is "K-complete" when all ``latent_k`` forced-z rollouts reached
    this arc boundary AND all K context-keys are present in the NPZ.

    For each K-complete scene we build ONE returns vector
    ``[R_rem(z0), R_rem(z1), ..., R_rem(z_{K-1})]`` from the per-z
    ``approx_remaining_return_supervision`` values and emit K training rows
    -- one per forced_z. Each emitted row carries the **per-z** q_phi
    context at this arc boundary in the z-th rollout (different across z,
    because trajectories diverge), and the SAME K-vector ``returns``.

    The matched-target / per-z-context pairing is the v4i2b approximation:
    we are saying "from this arc-state reached via z=k, the relative
    desirability of the K z's at this scene is the K remaining-returns
    we measured". This is logged contextual-bandit data, NOT a
    clone-and-replay counterfactual.

    Returns ``(rows, stats)``.
    """
    ckpt_str = str(Path(checkpoint_path).resolve())
    ckpt_name = Path(checkpoint_path).name

    grouped: dict[
        tuple[int, str, int, int], dict[int, dict[str, Any]]
    ] = {}
    n_csv_total = 0
    n_csv_filtered_path = 0
    for r in arc_rows:
        n_csv_total += 1
        rp = str(r.get("checkpoint_path", ""))
        if not rp:
            n_csv_filtered_path += 1
            continue
        try:
            rp_resolved = str(Path(rp).resolve())
        except OSError:
            rp_resolved = rp
        if rp_resolved != ckpt_str and Path(rp).name != ckpt_name:
            n_csv_filtered_path += 1
            continue
        try:
            steps = int(r["checkpoint_steps"])
            opp = str(r["opponent"]).upper()
            seed = int(r["probe_seed"])
            z = int(r["forced_z"])
            arc_idx = int(r["arc_idx"])
            rem_R = float(r["approx_remaining_return_supervision"])
            ctx_key = str(r["context_key"])
        except (KeyError, ValueError):
            continue
        if not (0 <= z < latent_k):
            continue
        grouped.setdefault((steps, opp, seed, arc_idx), {})[z] = {
            "remaining_R": rem_R,
            "context_key": ctx_key,
        }

    out: list[_ContextRow] = []
    n_scenes = len(grouped)
    n_complete = 0
    n_skipped_missing_z = 0
    n_skipped_missing_ctx = 0
    arc_idx_hist: dict[int, int] = {}
    # v4i2b diagnostic: dispersion of the K per-z contexts within each
    # K-complete arc scene. Low dispersion = rollouts have not yet
    # meaningfully diverged at this arc, so grouping a single K-vector
    # target across the K contexts is close to valid. High dispersion =
    # rollouts have diverged, so the target is noisy. Reported overall
    # and broken out per arc_idx (expectation: ~0 at arc_idx=0 because
    # the matched-start contract makes step-0 global state identical
    # across z, and grows with arc_idx).
    per_scene_disp: list[dict[str, Any]] = []
    for (steps, opp, seed, arc_idx), zmap in grouped.items():
        if len(zmap) < latent_k or any(
            z not in zmap for z in range(latent_k)
        ):
            n_skipped_missing_z += 1
            continue
        ctxs: list[np.ndarray] = []
        any_missing = False
        for z in range(latent_k):
            ck = zmap[z]["context_key"]
            ctx = arc_contexts.get(ck)
            if ctx is None:
                any_missing = True
                break
            ctxs.append(np.asarray(ctx, dtype=np.float32).reshape(-1))
        if any_missing:
            n_skipped_missing_ctx += 1
            continue
        n_complete += 1
        arc_idx_hist[arc_idx] = arc_idx_hist.get(arc_idx, 0) + 1
        returns_arr = np.array(
            [zmap[z]["remaining_R"] for z in range(latent_k)],
            dtype=np.float32,
        )
        ctx_stack = np.stack(ctxs, axis=0).astype(np.float32)  # [K, 170]
        centroid = ctx_stack.mean(axis=0)
        diffs = ctx_stack - centroid[None, :]
        per_z_l2 = np.linalg.norm(diffs, axis=1)
        scene_l2 = float(per_z_l2.mean())
        scene_l2_max = float(per_z_l2.max())
        centroid_norm = float(np.linalg.norm(centroid))
        scene_l2_norm = scene_l2 / (centroid_norm + 1e-8)
        ctx_norms = np.linalg.norm(ctx_stack, axis=1)
        if centroid_norm > 1e-8:
            cos_sims = (ctx_stack @ centroid) / (
                ctx_norms * centroid_norm + 1e-8
            )
            scene_cos_disp = float(1.0 - cos_sims.mean())
        else:
            scene_cos_disp = 0.0
        per_scene_disp.append(
            {
                "checkpoint_steps": int(steps),
                "opponent": str(opp),
                "probe_seed": int(seed),
                "arc_idx": int(arc_idx),
                "centroid_l2_norm": float(centroid_norm),
                "context_l2_dispersion": scene_l2,
                "context_l2_dispersion_max": scene_l2_max,
                "context_l2_dispersion_normalized": float(scene_l2_norm),
                "context_cos_dispersion": float(scene_cos_disp),
            }
        )
        for z in range(latent_k):
            key = f"{steps}|{opp}|{seed}|{z}|{arc_idx}"
            out.append(
                _ContextRow(
                    key=key,
                    steps=steps,
                    opponent=opp,
                    seed=seed,
                    context=ctxs[z],
                    returns=returns_arr,
                )
            )
    dispersion_summary = _summarize_arc_dispersion(per_scene_disp)
    stats = {
        "n_csv_rows": int(n_csv_total),
        "n_csv_filtered_by_checkpoint": int(n_csv_filtered_path),
        "n_scenes_total": int(n_scenes),
        "n_scenes_complete_K": int(n_complete),
        "n_scenes_missing_z": int(n_skipped_missing_z),
        "n_scenes_missing_ctx": int(n_skipped_missing_ctx),
        "n_rows_emitted": int(len(out)),
        "arc_idx_histogram": dict(sorted(arc_idx_hist.items())),
        "context_dispersion_across_forced_z": dispersion_summary,
        "_per_scene_dispersion_rows": per_scene_disp,
    }
    return out, stats


def _summarize_arc_dispersion(
    per_scene: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-scene dispersion into overall + per-arc_idx summaries."""
    if not per_scene:
        return {
            "n_scenes": 0,
            "overall": {},
            "per_arc_idx": {},
        }
    l2 = np.array(
        [r["context_l2_dispersion"] for r in per_scene], dtype=np.float64
    )
    l2_norm = np.array(
        [r["context_l2_dispersion_normalized"] for r in per_scene],
        dtype=np.float64,
    )
    cos_disp = np.array(
        [r["context_cos_dispersion"] for r in per_scene], dtype=np.float64
    )
    overall = {
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "max_l2": float(l2.max()),
        "mean_l2_normalized": float(l2_norm.mean()),
        "median_l2_normalized": float(np.median(l2_norm)),
        "max_l2_normalized": float(l2_norm.max()),
        "mean_cos_dispersion": float(cos_disp.mean()),
        "median_cos_dispersion": float(np.median(cos_disp)),
        "max_cos_dispersion": float(cos_disp.max()),
    }
    per_arc: dict[int, dict[str, float]] = {}
    arc_indices = sorted({int(r["arc_idx"]) for r in per_scene})
    for ai in arc_indices:
        bucket = [r for r in per_scene if int(r["arc_idx"]) == ai]
        bl2_n = np.array(
            [r["context_l2_dispersion_normalized"] for r in bucket],
            dtype=np.float64,
        )
        bcos = np.array(
            [r["context_cos_dispersion"] for r in bucket],
            dtype=np.float64,
        )
        per_arc[ai] = {
            "n_scenes": int(len(bucket)),
            "mean_l2_normalized": float(bl2_n.mean()),
            "max_l2_normalized": float(bl2_n.max()),
            "mean_cos_dispersion": float(bcos.mean()),
            "max_cos_dispersion": float(bcos.max()),
        }
    return {
        "n_scenes": int(len(per_scene)),
        "overall": overall,
        "per_arc_idx": per_arc,
    }


# ---------------------------------------------------------------------------
# Target distribution + train/val split
# ---------------------------------------------------------------------------

def soft_targets_from_returns(returns: np.ndarray, *, temperature: float) -> np.ndarray:
    """Per-row centered/scaled softmax target. Input shape ``[N, K]``."""
    mean = returns.mean(axis=-1, keepdims=True)
    std = returns.std(axis=-1, keepdims=True)
    adv = (returns - mean) / (std + 1e-8)
    scaled = adv / max(float(temperature), 1e-8)
    # Numerically stable softmax (subtract row max).
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    expv = np.exp(scaled)
    return (expv / expv.sum(axis=-1, keepdims=True)).astype(np.float32)


def split_train_val(
    rows: list[_ContextRow],
    *,
    val_seeds_start: int | None,
    val_frac: float,
) -> tuple[list[_ContextRow], list[_ContextRow]]:
    """Seed-based train/val split. Val seeds are the largest seeds.

    If ``val_seeds_start`` is given, seeds ``>=`` that value go to val.
    Otherwise the last ``val_frac`` fraction of the sorted unique seed list
    becomes val. The split is computed over ALL opponents together so the
    same set of seeds is held out across opponents.
    """
    if not rows:
        return [], []
    all_seeds = sorted({r.seed for r in rows})
    if val_seeds_start is None:
        n_val = max(1, int(round(val_frac * len(all_seeds))))
        if n_val >= len(all_seeds):
            n_val = max(1, len(all_seeds) - 1)
        val_set = set(all_seeds[-n_val:])
    else:
        val_set = {s for s in all_seeds if s >= int(val_seeds_start)}
        if not val_set:
            print(
                f"[v4i2] WARNING: --val-seeds-start={val_seeds_start} excludes all "
                f"seeds (range {all_seeds[0]}..{all_seeds[-1]}); falling back to "
                f"--val-frac={val_frac}."
            )
            n_val = max(1, int(round(val_frac * len(all_seeds))))
            val_set = set(all_seeds[-n_val:])

    train: list[_ContextRow] = []
    val: list[_ContextRow] = []
    for r in rows:
        (val if r.seed in val_set else train).append(r)
    return train, val


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class _RouterMetrics:
    n: int
    top1_accuracy: float
    regret_mean: float
    regret_median: float
    q_phi_mean_max_prob: float
    mean_entropy_nats: float
    ce_loss: float

    def as_row(self, split: str, phase: str) -> dict[str, Any]:
        return {
            "split": split,
            "phase": phase,
            "n": int(self.n),
            "router_top1_accuracy": float(self.top1_accuracy),
            "router_regret_mean": float(self.regret_mean),
            "router_regret_median": float(self.regret_median),
            "q_phi_mean_max_prob": float(self.q_phi_mean_max_prob),
            "q_phi_mean_entropy_nats": float(self.mean_entropy_nats),
            "ce_loss": float(self.ce_loss),
        }


def _compute_metrics(
    *,
    logits: torch.Tensor,            # [N, K]
    returns: np.ndarray,             # [N, K]
    targets: torch.Tensor,           # [N, K]
) -> _RouterMetrics:
    n = int(logits.shape[0])
    if n == 0:
        return _RouterMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        argmax = probs.argmax(dim=-1).cpu().numpy()
        max_prob = probs.max(dim=-1).values.mean().item()
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()
        ce = -(targets * log_probs).sum(dim=-1).mean().item()
    best_z = returns.argmax(axis=-1)
    top1 = float((argmax == best_z).mean())
    r_best = returns[np.arange(n), best_z]
    r_pick = returns[np.arange(n), argmax]
    regret = r_best - r_pick
    return _RouterMetrics(
        n=n,
        top1_accuracy=top1,
        regret_mean=float(regret.mean()),
        regret_median=float(np.median(regret)),
        q_phi_mean_max_prob=float(max_prob),
        mean_entropy_nats=float(entropy),
        ce_loss=float(ce),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_dummy_env_for_spaces(*, cfg_meta: dict[str, Any], device: str):
    """Build a single env just to grab observation_space and action_space."""
    cfg = PPOConfig()
    cfg.use_latent_strategy = True
    cfg.latent_k = int(cfg_meta.get("latent_k", 4))
    cfg.n_envs = 1
    cfg.seed = 0
    cfg.device = str(device)
    n_blue = int(cfg_meta.get("n_blue", 4))
    cfg.max_blue_agents = n_blue
    cfg.n_agents_per_team = n_blue
    cfg.opponent_randomize = False
    cfg.fixed_opponent_tag = "OP5"
    env = build_training_env(
        cfg,
        initial_phase="PHASE1",
        initial_opponent_tag="OP5",
    )
    return env


def _freeze_all_except_strategy_encoder(model: torch.nn.Module) -> tuple[list[str], list[str]]:
    """Freeze every parameter NOT under ``strategy_encoder.*``.

    Returns ``(trainable_names, frozen_names)``.
    """
    if getattr(model, "strategy_encoder", None) is None:
        raise RuntimeError(
            "Checkpoint has no strategy_encoder (use_latent_strategy=False?). "
            "Router distillation only makes sense for latent-strategy models."
        )
    trainable: list[str] = []
    frozen: list[str] = []
    for name, p in model.named_parameters():
        if name.startswith("strategy_encoder."):
            p.requires_grad_(True)
            trainable.append(name)
        else:
            p.requires_grad_(False)
            frozen.append(name)
    return trainable, frozen


def train_distill(
    *,
    model: torch.nn.Module,
    train_rows: list[_ContextRow],
    val_rows: list[_ContextRow],
    temperature: float,
    epochs: int,
    lr: float,
    weight_decay: float,
    log_every: int,
    device: torch.device,
) -> dict[str, Any]:
    """Train ``strategy_encoder`` to match the soft return-ranked targets.

    Returns a history dict with pre/post metrics and the per-epoch log.
    """
    if not train_rows:
        raise ValueError("No training examples; cannot distill.")

    def _pack(rows: list[_ContextRow]) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        if not rows:
            empty_ctx = torch.zeros((0, 170), dtype=torch.float32, device=device)
            empty_ret = np.zeros((0, model.latent_k), dtype=np.float32)
            empty_tgt = torch.zeros((0, model.latent_k), dtype=torch.float32, device=device)
            return empty_ctx, empty_ret, empty_tgt
        ctx_np = np.stack([r.context for r in rows], axis=0)
        ret_np = np.stack([r.returns for r in rows], axis=0)
        tgt_np = soft_targets_from_returns(ret_np, temperature=temperature)
        ctx_t = torch.as_tensor(ctx_np, dtype=torch.float32, device=device)
        tgt_t = torch.as_tensor(tgt_np, dtype=torch.float32, device=device)
        return ctx_t, ret_np, tgt_t

    train_ctx, train_ret, train_tgt = _pack(train_rows)
    val_ctx, val_ret, val_tgt = _pack(val_rows)

    # Pre-distill metrics.
    model.eval()
    with torch.no_grad():
        pre_train_logits = model.strategy_logits(train_ctx)
        pre_train = _compute_metrics(
            logits=pre_train_logits, returns=train_ret, targets=train_tgt
        )
        pre_val: _RouterMetrics
        if val_rows:
            pre_val_logits = model.strategy_logits(val_ctx)
            pre_val = _compute_metrics(
                logits=pre_val_logits, returns=val_ret, targets=val_tgt
            )
        else:
            pre_val = _RouterMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    print(f"[v4i2] PRE  train: {pre_train.as_row('train', 'pre')}")
    print(f"[v4i2] PRE  val:   {pre_val.as_row('val', 'pre')}")

    # Optimizer over only the unfrozen (strategy_encoder.*) params.
    train_params = [p for p in model.parameters() if p.requires_grad]
    n_train_params = sum(int(p.numel()) for p in train_params)
    print(f"[v4i2] trainable params: {n_train_params} (strategy_encoder only)")
    opt = torch.optim.AdamW(train_params, lr=float(lr), weight_decay=float(weight_decay))

    history: list[dict[str, Any]] = []

    for epoch in range(int(epochs)):
        model.train()
        # Full-batch update; q_phi context dataset is small (typically <100).
        logits = model.strategy_logits(train_ctx)
        log_probs = F.log_softmax(logits, dim=-1)
        # CE(target || q_phi); minimizing CE == minimizing KL(target||q_phi).
        loss = -(train_tgt * log_probs).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if log_every > 0 and ((epoch + 1) % log_every == 0 or epoch == 0):
            model.eval()
            with torch.no_grad():
                tr_logits = model.strategy_logits(train_ctx)
                tr_m = _compute_metrics(
                    logits=tr_logits, returns=train_ret, targets=train_tgt
                )
                if val_rows:
                    val_logits = model.strategy_logits(val_ctx)
                    val_m = _compute_metrics(
                        logits=val_logits, returns=val_ret, targets=val_tgt
                    )
                else:
                    val_m = _RouterMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(loss.item()),
                    "train_top1": tr_m.top1_accuracy,
                    "train_regret": tr_m.regret_mean,
                    "train_max_prob": tr_m.q_phi_mean_max_prob,
                    "val_top1": val_m.top1_accuracy,
                    "val_regret": val_m.regret_mean,
                    "val_max_prob": val_m.q_phi_mean_max_prob,
                }
            )
            print(
                f"[v4i2] epoch={epoch + 1:4d} loss={loss.item():.4f} "
                f"train(top1={tr_m.top1_accuracy:.3f}, regret={tr_m.regret_mean:+.3f}, "
                f"max_prob={tr_m.q_phi_mean_max_prob:.3f}) "
                f"val(top1={val_m.top1_accuracy:.3f}, regret={val_m.regret_mean:+.3f}, "
                f"max_prob={val_m.q_phi_mean_max_prob:.3f})"
            )

    # Post-distill metrics.
    model.eval()
    with torch.no_grad():
        post_train_logits = model.strategy_logits(train_ctx)
        post_train = _compute_metrics(
            logits=post_train_logits, returns=train_ret, targets=train_tgt
        )
        if val_rows:
            post_val_logits = model.strategy_logits(val_ctx)
            post_val = _compute_metrics(
                logits=post_val_logits, returns=val_ret, targets=val_tgt
            )
        else:
            post_val = _RouterMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    print(f"[v4i2] POST train: {post_train.as_row('train', 'post')}")
    print(f"[v4i2] POST val:   {post_val.as_row('val', 'post')}")

    return {
        "pre_train": pre_train,
        "pre_val": pre_val,
        "post_train": post_train,
        "post_val": post_val,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def save_distilled_checkpoint(
    *,
    src_checkpoint: Path,
    model: torch.nn.Module,
    out_path: Path,
) -> None:
    """Save a router-distilled checkpoint.

    Strategy: load the original payload, overwrite ``model_state_dict``
    with ``model.state_dict()`` (which has the trained strategy_encoder
    AND all original weights since everything else was frozen), tag the
    payload with a small ``v4i2_router_distill`` provenance dict, and
    write it via ``torch.save``.
    """
    payload = _torch_load_checkpoint(str(src_checkpoint), map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Source checkpoint malformed: {src_checkpoint}")
    new_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    payload["model_state_dict"] = new_sd
    prov = dict(payload.get("v4i2_router_distill") or {})
    prov.update(
        {
            "source_checkpoint": str(src_checkpoint),
            "frozen_modules_except": "strategy_encoder",
        }
    )
    payload["v4i2_router_distill"] = prov
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".tmp" + out_path.suffix)
    torch.save(payload, tmp)
    tmp.replace(out_path)
    print(f"[v4i2] wrote distilled checkpoint: {out_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_arc_dispersion_csv(
    csv_path: Path, per_scene_rows: list[dict[str, Any]]
) -> None:
    """Write per-(opp, seed, arc_idx) context dispersion stats for inspection."""
    if not per_scene_rows:
        return
    fieldnames = [
        "checkpoint_steps",
        "opponent",
        "probe_seed",
        "arc_idx",
        "centroid_l2_norm",
        "context_l2_dispersion",
        "context_l2_dispersion_max",
        "context_l2_dispersion_normalized",
        "context_cos_dispersion",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_sorted = sorted(
        per_scene_rows,
        key=lambda r: (
            int(r.get("checkpoint_steps", 0)),
            str(r.get("opponent", "")),
            int(r.get("probe_seed", 0)),
            int(r.get("arc_idx", 0)),
        ),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _write_metrics_csv(metrics_path: Path, results: dict[str, Any]) -> None:
    rows = [
        results["pre_train"].as_row("train", "pre"),
        results["pre_val"].as_row("val", "pre"),
        results["post_train"].as_row("train", "post"),
        results["post_val"].as_row("val", "post"),
    ]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_report(
    report_path: Path,
    *,
    results: dict[str, Any],
    args: argparse.Namespace,
    n_train: int,
    n_val: int,
    latent_k: int,
    arc_mode: bool = False,
    arc_stats: dict[str, Any] | None = None,
) -> None:
    lines: list[str] = []
    if arc_mode:
        lines.append("# v4i2b Router Distillation Report (arc-boundary supervision)")
    else:
        lines.append("# v4i2 Router Distillation Report (episode-start supervision)")
    lines.append("")
    lines.append(f"- source checkpoint: `{Path(args.checkpoint).name}`")
    if arc_mode:
        lines.append(
            f"- supervision: **approx_remaining_return_supervision** "
            "(remaining return of the forced-z trajectory from each arc onward; "
            "NOT a clone-and-replay counterfactual)"
        )
        lines.append(f"- arc CSV:           `{Path(args.arc_csv).name}`")
        lines.append(f"- arc contexts NPZ:  `{Path(args.arc_contexts).name}`")
    else:
        lines.append(
            "- supervision: full-episode return per matched (opponent, probe_seed)"
        )
        lines.append(f"- qprobe CSV:        `{Path(args.qprobe_csv).name}`")
        lines.append(f"- contexts NPZ:      `{Path(args.contexts).name}`")
    lines.append(f"- output checkpoint: `{Path(args.out).name}`")
    lines.append(f"- temperature: {args.temperature}, epochs: {args.epochs}, "
                 f"lr: {args.lr}, weight_decay: {args.weight_decay}")
    lines.append(f"- n_train: {n_train}, n_val: {n_val}, latent_k: {latent_k}, "
                 f"random_baseline_top1: {1.0 / max(1, latent_k):.3f}")
    if arc_mode and arc_stats is not None:
        lines.append(
            f"- arc scenes: total={arc_stats['n_scenes_total']}, "
            f"K-complete={arc_stats['n_scenes_complete_K']}, "
            f"missing_z={arc_stats['n_scenes_missing_z']}, "
            f"missing_ctx={arc_stats['n_scenes_missing_ctx']}, "
            f"rows_emitted={arc_stats['n_rows_emitted']}"
        )
        lines.append(
            f"- arc_idx histogram (K-complete only): "
            f"{arc_stats['arc_idx_histogram']}"
        )
    lines.append("")
    if arc_mode and arc_stats is not None:
        disp = arc_stats.get("context_dispersion_across_forced_z", {})
        if disp.get("n_scenes", 0) > 0:
            lines.append("## context_dispersion_across_forced_z (diagnostic)")
            lines.append("")
            lines.append(
                "How far the K per-z q_phi contexts have drifted apart at each "
                "K-complete arc scene. Computed against the per-scene centroid:"
            )
            lines.append("")
            lines.append(
                "- `l2_normalized` = mean_k ||ctx_k - centroid||_2 / "
                "(||centroid||_2 + 1e-8)"
            )
            lines.append(
                "- `cos_dispersion` = 1 - mean_k cos(ctx_k, centroid)"
            )
            lines.append("")
            lines.append(
                "Low values mean rollouts have not meaningfully diverged at "
                "this arc, so grouping a single K-vector target across the K "
                "contexts is close to valid (clean labels). High values mean "
                "rollouts have already diverged, so the target is noisy "
                "(swampy labels)."
            )
            lines.append("")
            overall = disp["overall"]
            lines.append("### Overall (K-complete scenes)")
            lines.append("")
            lines.append("| metric | mean | median | max |")
            lines.append("|---|---|---|---|")
            lines.append(
                f"| l2 | {overall['mean_l2']:.4f} | "
                f"{overall['median_l2']:.4f} | "
                f"{overall['max_l2']:.4f} |"
            )
            lines.append(
                f"| l2_normalized | {overall['mean_l2_normalized']:.4f} | "
                f"{overall['median_l2_normalized']:.4f} | "
                f"{overall['max_l2_normalized']:.4f} |"
            )
            lines.append(
                f"| cos_dispersion | "
                f"{overall['mean_cos_dispersion']:.4f} | "
                f"{overall['median_cos_dispersion']:.4f} | "
                f"{overall['max_cos_dispersion']:.4f} |"
            )
            lines.append("")
            lines.append("### Per arc_idx (K-complete scenes only)")
            lines.append("")
            lines.append(
                "| arc_idx | n_scenes | mean_l2_norm | max_l2_norm | "
                "mean_cos_disp | max_cos_disp |"
            )
            lines.append("|---|---|---|---|---|---|")
            for ai, b in disp["per_arc_idx"].items():
                lines.append(
                    f"| {ai} | {b['n_scenes']} | "
                    f"{b['mean_l2_normalized']:.4f} | "
                    f"{b['max_l2_normalized']:.4f} | "
                    f"{b['mean_cos_dispersion']:.4f} | "
                    f"{b['max_cos_dispersion']:.4f} |"
                )
            lines.append("")
            lines.append(
                "Sanity check: at `arc_idx=0` the matched-start contract "
                "forces identical global state across z, so dispersion "
                "should be exactly 0.0. Dispersion should grow monotonically "
                "with `arc_idx` as rollouts diverge. Per-scene values are in "
                "the companion `*_distill_arc_dispersion.csv`."
            )
            lines.append("")
    lines.append("## Metrics (pre vs post)")
    lines.append("")
    lines.append("| split | phase | n | top1 | regret_mean | max_prob | entropy(nats) | ce |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for split in ("train", "val"):
        for phase in ("pre", "post"):
            m: _RouterMetrics = results[f"{phase}_{split}"]
            lines.append(
                f"| {split} | {phase} | {m.n} | {m.top1_accuracy:.3f} | "
                f"{m.regret_mean:+.3f} | {m.q_phi_mean_max_prob:.3f} | "
                f"{m.mean_entropy_nats:.3f} | {m.ce_loss:.3f} |"
            )
    lines.append("")
    pre_v: _RouterMetrics = results["pre_val"]
    post_v: _RouterMetrics = results["post_val"]
    pre_t: _RouterMetrics = results["pre_train"]
    post_t: _RouterMetrics = results["post_train"]
    ln_k = math.log(max(2, latent_k))
    lines.append("## Pass/Fail vs the v4i2 plan thresholds")
    lines.append("")
    lines.append(f"- train top1 > 0.70: **{'PASS' if post_t.top1_accuracy > 0.70 else 'FAIL'}** "
                 f"(post_train_top1 = {post_t.top1_accuracy:.3f})")
    if n_val > 0:
        lines.append(f"- val top1 > 0.45-0.50: **{'PASS' if post_v.top1_accuracy > 0.45 else 'CHECK'}** "
                     f"(post_val_top1 = {post_v.top1_accuracy:.3f})")
        lines.append(f"- val regret drops vs pre: **{'PASS' if post_v.regret_mean < pre_v.regret_mean else 'FAIL'}** "
                     f"(pre={pre_v.regret_mean:+.3f}, post={post_v.regret_mean:+.3f})")
        lines.append(f"- val max_prob > 0.25 (rose above uniform): **{'PASS' if post_v.q_phi_mean_max_prob > 0.25 else 'FAIL'}** "
                     f"(post_val_max_prob = {post_v.q_phi_mean_max_prob:.3f})")
        lines.append(f"- val max_prob <= 0.80 (not brittle collapse): **{'PASS' if post_v.q_phi_mean_max_prob <= 0.80 else 'CHECK'}** "
                     f"(post_val_max_prob = {post_v.q_phi_mean_max_prob:.3f})")
        lines.append(
            f"- val entropy < ln(K)={ln_k:.3f} (committed): "
            f"**{'PASS' if post_v.mean_entropy_nats < ln_k else 'FAIL'}** "
            f"(post_val_entropy = {post_v.mean_entropy_nats:.3f} nats)"
        )
        lines.append(
            f"- val entropy > 0.05 (not collapsed to zero): "
            f"**{'PASS' if post_v.mean_entropy_nats > 0.05 else 'CHECK'}** "
            f"(post_val_entropy = {post_v.mean_entropy_nats:.3f} nats)"
        )
    else:
        lines.append("- val metrics: SKIPPED (no val rows; consider rerunning q_probe with more seeds).")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Soft target per row: target = softmax(((R - R.mean) / (R.std + 1e-8)) / temperature).")
    lines.append("- Loss: CE(target_probs, q_phi_probs) = KL(target||q_phi) + H(target).")
    lines.append("- Only `strategy_encoder.*` parameters were trained. Actor, critic, and ")
    lines.append("  value heads stayed byte-identical to the source checkpoint.")
    lines.append("- This script does NOT change reward, opponents, maps, arc-credit math,")
    lines.append("  or the PPO trainer (v4i2 scope guard).")
    if arc_mode:
        lines.append(
            "- v4i2b note: the per-arc supervision is the return of the "
            "**forced-z** trajectory from that arc onward (logged data), not a "
            "clone-and-replay counterfactual. The K-vector target for the "
            "scene is built from the K such measurements (one per z), and "
            "each emitted example carries the per-z arc-state q_phi context."
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v4i2: offline q_phi router distillation from q_probe returns."
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Source PPO checkpoint (e.g. final_v4i1_*.zip).")
    p.add_argument("--qprobe-csv", type=Path, default=None,
                   help=("Raw q_probe episodes CSV (<run_tag>_qprobe.csv). "
                         "Required unless --arc-csv is given (v4i2b mode)."))
    p.add_argument("--contexts", type=Path, default=None,
                   help=("q_probe contexts NPZ (<run_tag>_qprobe_contexts.npz). "
                         "Required unless --arc-contexts is given (v4i2b mode)."))
    p.add_argument(
        "--arc-csv",
        type=Path,
        default=None,
        help=(
            "v4i2b arc-boundary CSV (<run_tag>_qprobe_arcs.csv). When both "
            "--arc-csv and --arc-contexts are provided, the script runs in "
            "arc-boundary mode (approx_remaining_return_supervision) and "
            "ignores --qprobe-csv / --contexts."
        ),
    )
    p.add_argument(
        "--arc-contexts",
        type=Path,
        default=None,
        help=(
            "v4i2b arc-boundary contexts NPZ "
            "(<run_tag>_qprobe_arc_contexts.npz). See --arc-csv."
        ),
    )
    p.add_argument("--out", type=Path, required=True,
                   help="Output path for the router-distilled checkpoint.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Soft-target temperature (default 1.0).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--val-seeds-start", type=int, default=None,
                   help="Seeds >= this value go to validation. Overrides --val-frac.")
    p.add_argument("--val-frac", type=float, default=0.25,
                   help="Used when --val-seeds-start is not given.")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the optimizer (PyTorch global seed).")
    p.add_argument("--report-suffix", type=str, default="distill",
                   help="Suffix used for the report/metrics files next to --out.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.checkpoint.exists():
        print(f"[v4i2] FATAL: checkpoint not found: {args.checkpoint}")
        return 2

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    meta = read_custom_ppo_metadata(str(args.checkpoint))
    if not bool(meta.get("use_latent_strategy", False)):
        print(f"[v4i2] FATAL: checkpoint is not latent (use_latent_strategy=False).")
        return 2
    latent_k = int(meta.get("latent_k", 4))
    print(f"[v4i2] checkpoint: {args.checkpoint}")
    print(f"[v4i2] latent_k={latent_k}, n_blue={int(meta.get('n_blue', 4))}, "
          f"device={args.device}")

    arc_mode = bool(args.arc_csv) and bool(args.arc_contexts)
    if arc_mode:
        if not Path(args.arc_csv).exists():
            print(f"[v4i2b] FATAL: arc CSV not found: {args.arc_csv}")
            return 2
        if not Path(args.arc_contexts).exists():
            print(f"[v4i2b] FATAL: arc contexts NPZ not found: {args.arc_contexts}")
            return 2
        arc_contexts_map = _load_arc_contexts_npz(args.arc_contexts)
        arc_rows_raw = _load_arc_rows(args.arc_csv)
        print(
            f"[v4i2b] arc-mode: contexts NPZ {len(arc_contexts_map)} keys; "
            f"arc CSV {len(arc_rows_raw)} rows"
        )
        rows, arc_stats = build_arc_examples(
            arc_rows=arc_rows_raw,
            arc_contexts=arc_contexts_map,
            checkpoint_path=str(args.checkpoint),
            latent_k=latent_k,
        )
        print(
            "[v4i2b] arc-mode stats: "
            f"scenes_total={arc_stats['n_scenes_total']}, "
            f"K-complete={arc_stats['n_scenes_complete_K']}, "
            f"missing_z={arc_stats['n_scenes_missing_z']}, "
            f"missing_ctx={arc_stats['n_scenes_missing_ctx']}, "
            f"rows_emitted={arc_stats['n_rows_emitted']}, "
            f"arc_idx_hist={arc_stats['arc_idx_histogram']}"
        )
        disp = arc_stats.get("context_dispersion_across_forced_z", {})
        if disp.get("n_scenes", 0) > 0:
            overall = disp["overall"]
            print(
                "[v4i2b] context_dispersion_across_forced_z (overall, K-complete scenes): "
                f"mean_l2={overall['mean_l2']:.4f}, "
                f"mean_l2_normalized={overall['mean_l2_normalized']:.4f}, "
                f"mean_cos_disp={overall['mean_cos_dispersion']:.4f} | "
                f"max_l2_norm={overall['max_l2_normalized']:.4f}, "
                f"max_cos_disp={overall['max_cos_dispersion']:.4f}"
            )
            print(
                "[v4i2b] context_dispersion_across_forced_z per arc_idx "
                "(n / mean_l2_norm / mean_cos_disp):"
            )
            for ai, b in disp["per_arc_idx"].items():
                print(
                    f"[v4i2b]   arc_idx={ai}: n={b['n_scenes']}, "
                    f"mean_l2_norm={b['mean_l2_normalized']:.4f}, "
                    f"mean_cos_disp={b['mean_cos_dispersion']:.4f}"
                )
        if not rows:
            print(
                "[v4i2b] FATAL: no K-complete arc scenes. Verify that the "
                "arc CSV and arc contexts NPZ correspond to this checkpoint, "
                "and that latent_k matches."
            )
            return 1
    else:
        if args.qprobe_csv is None or args.contexts is None:
            print(
                "[v4i2] FATAL: episode-start mode requires both --qprobe-csv "
                "and --contexts (or pass --arc-csv + --arc-contexts for v4i2b)."
            )
            return 2
        contexts = _load_contexts_npz(args.contexts)
        qprobe_rows = _load_qprobe_rows(args.qprobe_csv)
        print(
            f"[v4i2] contexts: {len(contexts)} rows; qprobe CSV: "
            f"{len(qprobe_rows)} rows"
        )
        rows = build_examples(
            qprobe_rows=qprobe_rows,
            contexts=contexts,
            checkpoint_path=str(args.checkpoint),
            latent_k=latent_k,
        )
        arc_stats = None
        if not rows:
            print(
                "[v4i2] FATAL: no usable (steps, opp, seed) groups. Double-check "
                "that the qprobe CSV and contexts NPZ correspond to this checkpoint."
            )
            return 1

    train_rows, val_rows = split_train_val(
        rows,
        val_seeds_start=args.val_seeds_start,
        val_frac=float(args.val_frac),
    )
    print(
        f"[v4i2] split: n_train={len(train_rows)}, n_val={len(val_rows)} "
        f"(unique train seeds={len({r.seed for r in train_rows})}, "
        f"val seeds={sorted({r.seed for r in val_rows})})"
    )

    env = _make_dummy_env_for_spaces(cfg_meta=meta, device=str(args.device))
    try:
        policy = load_custom_ppo_policy(
            str(args.checkpoint),
            env.observation_space,
            env.action_space,
            device=str(args.device),
        )
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[v4i2] WARNING: env.close() raised: {exc}")

    model = policy.model
    trainable, frozen = _freeze_all_except_strategy_encoder(model)
    print(
        f"[v4i2] froze {len(frozen)} params, trainable {len(trainable)} "
        f"(first trainable: {trainable[:3]!r}{'...' if len(trainable) > 3 else ''})"
    )

    results = train_distill(
        model=model,
        train_rows=train_rows,
        val_rows=val_rows,
        temperature=float(args.temperature),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        log_every=int(args.log_every),
        device=torch.device(str(args.device)),
    )

    save_distilled_checkpoint(
        src_checkpoint=args.checkpoint,
        model=model,
        out_path=args.out,
    )

    base_name = args.out.with_suffix("").name
    metrics_csv = args.out.parent / f"{base_name}_{args.report_suffix}_metrics.csv"
    report_md = args.out.parent / f"{base_name}_{args.report_suffix}_report.md"
    _write_metrics_csv(metrics_csv, results)
    _write_report(
        report_md,
        results=results,
        args=args,
        n_train=len(train_rows),
        n_val=len(val_rows),
        latent_k=latent_k,
        arc_mode=arc_mode,
        arc_stats=arc_stats,
    )
    print(f"[v4i2] wrote metrics: {metrics_csv}")
    print(f"[v4i2] wrote report:  {report_md}")
    if arc_mode and arc_stats is not None:
        per_scene_disp = arc_stats.get("_per_scene_dispersion_rows", [])
        if per_scene_disp:
            disp_csv = (
                args.out.parent
                / f"{base_name}_{args.report_suffix}_arc_dispersion.csv"
            )
            _write_arc_dispersion_csv(disp_csv, per_scene_disp)
            print(f"[v4i2b] wrote arc dispersion CSV: {disp_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
