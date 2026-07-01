"""v4i3 Local Counterfactual Q(s, z) Probe.

This is the **gold-standard local counterfactual** probe used by the
v4i3 Summer-Faithful Proof Suite. For each ``(opponent, probe_seed)``
matched starting world, the natural q_phi-driven episode is rolled
forward, and at every arc boundary listed in ``--branch-arc-indices``
the env+policy state is snapshotted. From that snapshot we force each
``z`` in ``[0, latent_k)`` and roll to episode termination, summing
remaining reward. The snapshot is then restored exactly and the natural
rollout continues toward the next arc boundary.

This differs from ``tools/q_probe.py``'s arc-boundary mode (v4i2b) in
exactly one way that matters: that one is **logged-bandit** data --
every measurement corresponds to "the return of the trajectory the
probe actually walked". This tool is a true counterfactual: at the
SAME local env state, what if we had taken z=0 vs z=1 vs z=2 vs z=3?
The K remaining returns at one arc boundary share the same state, the
same opponent at the same step, the same temporal-tracker EMAs, and
the same sampling-RNG point. Only ``z`` varies.

Outputs (in ``--output-dir``):

* ``<run_tag>_qprobe_local_cf.csv``        -- one row per (opp, seed, arc_idx, forced_z)
* ``<run_tag>_qprobe_local_cf_summary.csv`` -- per (opp, seed, arc_idx): best_z, Q-contrast,
                                                R(z0..z_{K-1})
* ``<run_tag>_qprobe_local_cf_contexts.npz`` -- one 170-d context per
                                                (opp, seed, arc_idx); same across z by construction
* ``<run_tag>_qprobe_local_cf_report.md``   -- Gate 3 pass/fail report

Determinism contract
--------------------

The probe begins by running a self-test: snapshot the env+policy at the
first arc boundary, force z=2, roll to completion, get R_A; restore;
force z=2 again, roll to completion, get R_B. If ``|R_A - R_B| > 1e-5``,
the snapshot/restore is incomplete and the probe aborts. Pass
``--allow-determinism-drift`` to continue past a failed self-test
(useful only for diagnosing what's missing from the snapshot).

Usage::

    .\\.venv\\Scripts\\python.exe tools/q_probe_local_counterfactual.py \\
        --checkpoint checkpoints/4v4/final_v4i3_summer_proof_OP5_OP6_OP7_4v4.zip \\
        --opponents OP5 OP6 OP7 --n-seeds 8 --base-seed 1000 \\
        --branch-arc-indices 0 1 \\
        --device cpu --output-dir checkpoints/4v4/v4i3_local_cf_smoke
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.inference import (
    apply_deterministic_sampling_generators,
    load_custom_ppo_policy,
    read_custom_ppo_metadata,
)
from rl.training.env_factory import build_training_env


# ---------------------------------------------------------------------------
# Constants / opponent tag mapping (mirrors tools/q_probe.py)
# ---------------------------------------------------------------------------

_OPPONENT_ENV_TAG: dict[str, str] = {
    "OP5": "OP5_RUSHER",
    "OP6": "OP6_TURTLE",
    "OP7": "OP7_SWITCHER",
}


def _env_opponent_tag(label: str) -> str:
    return _OPPONENT_ENV_TAG.get(
        str(label).strip().upper(), str(label).strip().upper()
    )


_DEFAULT_MAX_STEPS = 1500
_DETERMINISM_TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Checkpoint discovery (single-shot mode only -- watcher is overkill here)
# ---------------------------------------------------------------------------

_RUN_TAG_PREFIX_RE = re.compile(r"^(ckpt|interrupt|final)_")


def _resolve_run_tag(arg_run_tag: str | None, arg_checkpoint: Path) -> str:
    if arg_run_tag:
        return str(arg_run_tag).strip()
    stem = arg_checkpoint.stem
    stem = _RUN_TAG_PREFIX_RE.sub("", stem)
    # If the checkpoint stem ends with ``_<digits>`` (periodic checkpoint),
    # strip the steps suffix so the run tag matches its training run.
    stem = re.sub(r"_\d+$", "", stem)
    return stem


# ---------------------------------------------------------------------------
# Snapshot / restore primitives
#
# The env stack is ``GPUCTFVecEnv`` over ``BatchedCTFCore``. The core's
# mutable state is:
#   - all ``torch.Tensor`` attributes on the core (alive, x/y/heading/speed,
#     score, flag, mines, commit timers, step counters, done/truncated,
#     rt_* DR scratch, etc.)
#   - a single ``torch.Generator`` (core._rng)
#   - a handful of plain Python ``list[str]`` / ``dict`` fields
#     (_phase, _opponent_kind, _opponent_key, _phase_tensor_cache,
#     _red_control_mask*, blue_scripted)
# No pygame surfaces, no file handles, no sockets.
#
# The policy's mutable state lives on ``CustomPPOInferencePolicy``:
#   - _prev_z, _strategy_age, fixed_latent_strategy{,_id}
#   - _temporal_tracker.{ema_short, ema_long, initialized}
#   - model._sampling_gen_{strategy,action} (torch.Generator)
# ---------------------------------------------------------------------------


def _snapshot_env(env: Any) -> dict[str, Any]:
    core = env.core
    snap: dict[str, Any] = {
        "rng_state": core._rng.get_state().clone(),
        "tensors": {},
        "lists": {},
        "scalars": {},
    }
    for name, val in vars(core).items():
        if isinstance(val, torch.Tensor):
            snap["tensors"][name] = val.detach().clone()
    # Capture phase / opponent identity lists (per-env-batch labels). At B=1
    # these are length-1 lists; we still snapshot to be future-proof.
    for name in ("_phase", "_opponent_kind", "_opponent_key"):
        if hasattr(core, name):
            snap["lists"][name] = list(getattr(core, name))
    # The two scalar booleans actually consulted at step time.
    for name in ("blue_scripted",):
        if hasattr(core, name):
            snap["scalars"][name] = bool(getattr(core, name))
    return snap


def _restore_env(env: Any, snap: dict[str, Any]) -> None:
    core = env.core
    core._rng.set_state(snap["rng_state"])
    # Restore tensor attributes in-place to preserve external aliasing of
    # tensor identity from other mixins.
    for name, t in snap["tensors"].items():
        dst = getattr(core, name, None)
        if isinstance(dst, torch.Tensor) and dst.shape == t.shape and dst.dtype == t.dtype:
            dst.copy_(t)
        else:
            setattr(core, name, t.clone())
    for name, v in snap["lists"].items():
        setattr(core, name, list(v))
    for name, v in snap["scalars"].items():
        setattr(core, name, v)
    # Derived caches: drop and let the env rebuild on demand.
    if hasattr(core, "_phase_tensor_cache"):
        core._phase_tensor_cache = {}
    if hasattr(core, "_red_control_mask"):
        core._red_control_mask = None
    if hasattr(core, "_red_control_mask_dirty"):
        core._red_control_mask_dirty = True


def _snapshot_policy(model: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "_prev_z": (
            None if model._prev_z is None else model._prev_z.detach().clone()
        ),
        "_strategy_age": int(model._strategy_age),
        "_last_strategy_resampled": bool(model._last_strategy_resampled),
        "fixed_latent_strategy": bool(model.fixed_latent_strategy),
        "fixed_latent_strategy_id": int(model.fixed_latent_strategy_id),
        "tracker": None,
        "gen_strategy": None,
        "gen_action": None,
    }
    tracker = getattr(model, "_temporal_tracker", None)
    if tracker is not None:
        snap["tracker"] = {
            "ema_short": tracker.ema_short.detach().clone(),
            "ema_long": tracker.ema_long.detach().clone(),
            "initialized": tracker.initialized.detach().clone(),
        }
    inner = getattr(model, "model", None)
    gs = getattr(inner, "_sampling_gen_strategy", None)
    if gs is not None:
        snap["gen_strategy"] = gs.get_state().clone()
    ga = getattr(inner, "_sampling_gen_action", None)
    if ga is not None:
        snap["gen_action"] = ga.get_state().clone()
    return snap


def _restore_policy(model: Any, snap: dict[str, Any]) -> None:
    model._prev_z = (
        None if snap["_prev_z"] is None else snap["_prev_z"].clone()
    )
    model._strategy_age = int(snap["_strategy_age"])
    model._last_strategy_resampled = bool(snap["_last_strategy_resampled"])
    model.fixed_latent_strategy = bool(snap["fixed_latent_strategy"])
    model.fixed_latent_strategy_id = int(snap["fixed_latent_strategy_id"])
    tracker = getattr(model, "_temporal_tracker", None)
    if tracker is not None and snap["tracker"] is not None:
        tracker.ema_short.copy_(snap["tracker"]["ema_short"])
        tracker.ema_long.copy_(snap["tracker"]["ema_long"])
        tracker.initialized.copy_(snap["tracker"]["initialized"])
    inner = getattr(model, "model", None)
    gs = getattr(inner, "_sampling_gen_strategy", None)
    if gs is not None and snap["gen_strategy"] is not None:
        gs.set_state(snap["gen_strategy"])
    ga = getattr(inner, "_sampling_gen_action", None)
    if ga is not None and snap["gen_action"] is not None:
        ga.set_state(snap["gen_action"])


def _set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def _single_obs(obs: dict[str, np.ndarray], env: Any) -> dict[str, np.ndarray]:
    """Mirror q_probe's per-env obs slicing for a 1-env vec env."""
    single = {
        k: (
            v[0]
            if hasattr(v, "shape") and v.ndim >= 2 and v.shape[0] == 1
            else v
        )
        for k, v in obs.items()
    }
    try:
        single["global_state"] = env.state()[0]
    except Exception:
        pass
    return single


def _roll_to_end_with_forced_z(
    env: Any,
    model: Any,
    obs: dict[str, np.ndarray],
    *,
    forced_z: int,
    starting_step: int,
    max_steps: int,
) -> tuple[float, int, int, int]:
    """Force ``z`` from this state and roll to episode termination.

    Returns ``(remaining_return, terminal_step, blue_score, red_score)``.
    """
    model.fixed_latent_strategy = True
    model.fixed_latent_strategy_id = int(forced_z)
    remaining = 0.0
    blue_score = 0
    red_score = 0
    terminal_step = starting_step
    cur_obs = obs
    for step in range(starting_step, max_steps):
        single = _single_obs(cur_obs, env)
        action, _ = model.predict(single, deterministic=False)
        env.step_async(action)
        cur_obs, rewards, dones, infos = env.step_wait()
        r = float(np.asarray(rewards).reshape(-1)[0])
        remaining += r
        terminal_step = step + 1
        if bool(np.asarray(dones).reshape(-1)[0]):
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info) or {}
            blue_score = int(ep_res.get("blue_score", 0))
            red_score = int(ep_res.get("red_score", 0))
            break
    return remaining, terminal_step, blue_score, red_score


# ---------------------------------------------------------------------------
# Determinism self-test
# ---------------------------------------------------------------------------


def _run_determinism_self_test(
    *,
    env: Any,
    model: Any,
    opp_env_tag: str,
    probe_seed: int,
    device: str,
    latent_k: int,
    arc_interval: int,
    branch_arc_indices: list[int],
    max_steps: int,
    tol: float,
) -> tuple[bool, float, float]:
    """At the first branch arc boundary, force z=2 twice with restore between.

    Returns ``(passed, R_A, R_B)``. ``passed`` requires ``|R_A - R_B| < tol``.

    This is the gating safety check: if it fails, snapshot/restore is
    incomplete and the rest of the probe's results would be untrustworthy.
    """
    first_branch = min(branch_arc_indices)
    target_step = int(first_branch) * int(arc_interval)
    test_z = min(2, max(0, latent_k - 1))

    # Drive a fresh episode forward to ``target_step`` under natural q_phi
    # (so the env + policy are in a non-trivial state at the test point,
    # not just the deterministic reset state).
    _set_global_seeds(probe_seed)
    try:
        env.seed(int(probe_seed))
    except Exception:
        pass
    env.env_method("set_next_opponent", "SCRIPTED", opp_env_tag)
    model.reset_strategy()
    model.fixed_latent_strategy = False
    apply_deterministic_sampling_generators(
        model.model, int(probe_seed), device=device
    )
    obs = env.reset()
    for step in range(target_step):
        single = _single_obs(obs, env)
        action, _ = model.predict(single, deterministic=False)
        env.step_async(action)
        obs, _rewards, dones, _infos = env.step_wait()
        if bool(np.asarray(dones).reshape(-1)[0]):
            # Episode ended before we reached the target step; bail out --
            # caller should pick an earlier branch_arc_index or smaller seed.
            print(
                f"[local_cf] determinism self-test: episode terminated at step "
                f"{step + 1} < target_step {target_step}; cannot test snapshot. "
                "Increase --base-seed range, lower --branch-arc-indices, or "
                "investigate why episodes are so short."
            )
            return False, float("nan"), float("nan")

    env_snap = _snapshot_env(env)
    pol_snap = _snapshot_policy(model)

    # Run A
    R_A, _, _, _ = _roll_to_end_with_forced_z(
        env, model, obs, forced_z=test_z, starting_step=target_step, max_steps=max_steps
    )

    # Restore and Run B
    _restore_env(env, env_snap)
    _restore_policy(model, pol_snap)
    R_B, _, _, _ = _roll_to_end_with_forced_z(
        env, model, obs, forced_z=test_z, starting_step=target_step, max_steps=max_steps
    )

    diff = abs(R_A - R_B)
    passed = diff < float(tol)
    return passed, float(R_A), float(R_B)


# ---------------------------------------------------------------------------
# Main per-(opp, seed) probe
# ---------------------------------------------------------------------------


@dataclass
class _BranchResult:
    arc_idx: int
    timestep: int
    context: np.ndarray  # shape [170], float32
    remaining_return_per_z: list[float]
    terminal_step_per_z: list[int]
    blue_score_per_z: list[int]
    red_score_per_z: list[int]


def _run_one_probe_episode(
    *,
    env: Any,
    model: Any,
    opp_env_tag: str,
    probe_seed: int,
    device: str,
    latent_k: int,
    arc_interval: int,
    branch_arc_indices: list[int],
    max_steps: int,
) -> dict[str, Any]:
    """Run the natural episode, branching at each requested arc boundary."""
    branch_set = set(int(i) for i in branch_arc_indices)
    _set_global_seeds(probe_seed)
    try:
        env.seed(int(probe_seed))
    except Exception:
        pass
    env.env_method("set_next_opponent", "SCRIPTED", opp_env_tag)
    model.reset_strategy()
    model.fixed_latent_strategy = False
    apply_deterministic_sampling_generators(
        model.model, int(probe_seed), device=device
    )

    obs = env.reset()
    natural_return = 0.0
    natural_blue = 0
    natural_red = 0
    natural_episode_length = 0
    branches: list[_BranchResult] = []

    step = 0
    while step < max_steps:
        on_arc_boundary = (step % arc_interval == 0)
        arc_idx = step // arc_interval

        if on_arc_boundary and arc_idx in branch_set:
            # Snapshot BEFORE predict. Each branch's predict advances the
            # temporal tracker exactly once from the same pre-step state,
            # which preserves the matched-context guarantee.
            env_snap = _snapshot_env(env)
            pol_snap = _snapshot_policy(model)

            rem_R = [0.0] * int(latent_k)
            term_step = [step] * int(latent_k)
            blue_s = [0] * int(latent_k)
            red_s = [0] * int(latent_k)
            context_vec: np.ndarray | None = None

            for z in range(int(latent_k)):
                _restore_env(env, env_snap)
                _restore_policy(model, pol_snap)
                model.fixed_latent_strategy = True
                model.fixed_latent_strategy_id = int(z)
                # The first predict at this boundary computes the context
                # from the snapshot's tracker EMAs + the current env's
                # global state -- same across z since both are restored.
                # Sampling RNG also matches across z; only z varies.
                r_z, term_z, b_z, rd_z = _roll_to_end_with_forced_z(
                    env,
                    model,
                    obs,
                    forced_z=z,
                    starting_step=step,
                    max_steps=max_steps,
                )
                rem_R[z] = float(r_z)
                term_step[z] = int(term_z)
                blue_s[z] = int(b_z)
                red_s[z] = int(rd_z)
                if context_vec is None:
                    ctx_t = getattr(model, "_last_context_gs", None)
                    if ctx_t is not None:
                        context_vec = (
                            ctx_t.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                            .reshape(-1)
                        )

            if context_vec is None:
                # Fallback to zeros if the model never wrote
                # _last_context_gs (should not happen for latent models).
                context_vec = np.zeros((170,), dtype=np.float32)
            branches.append(
                _BranchResult(
                    arc_idx=int(arc_idx),
                    timestep=int(step),
                    context=context_vec,
                    remaining_return_per_z=list(rem_R),
                    terminal_step_per_z=list(term_step),
                    blue_score_per_z=list(blue_s),
                    red_score_per_z=list(red_s),
                )
            )

            # Restore for natural continuation.
            _restore_env(env, env_snap)
            _restore_policy(model, pol_snap)
            model.fixed_latent_strategy = False

        # Natural rollout step.
        single = _single_obs(obs, env)
        action, _ = model.predict(single, deterministic=False)
        env.step_async(action)
        obs, rewards, dones, infos = env.step_wait()
        r_t = float(np.asarray(rewards).reshape(-1)[0])
        natural_return += r_t
        natural_episode_length = step + 1
        if bool(np.asarray(dones).reshape(-1)[0]):
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info) or {}
            natural_blue = int(ep_res.get("blue_score", 0))
            natural_red = int(ep_res.get("red_score", 0))
            break
        step += 1

    return {
        "branches": branches,
        "natural_return": float(natural_return),
        "natural_blue": int(natural_blue),
        "natural_red": int(natural_red),
        "natural_blue_won": int(natural_blue > natural_red),
        "natural_episode_length": int(natural_episode_length),
    }


# ---------------------------------------------------------------------------
# I/O: CSVs + NPZ
# ---------------------------------------------------------------------------

_CF_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "checkpoint_steps",
    "opponent",
    "probe_seed",
    "arc_idx",
    "timestep",
    "forced_z",
    "remaining_return",
    "terminal_step",
    "branch_blue_score",
    "branch_red_score",
    "branch_blue_won",
    "natural_return",
    "natural_episode_length",
    "natural_blue_score",
    "natural_red_score",
    "context_key",
)


_CF_SUMMARY_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "checkpoint_steps",
    "opponent",
    "probe_seed",
    "arc_idx",
    "timestep",
    "best_z",
    "worst_z",
    "Q_contrast",
    "best_minus_uniform_avg",
    "argmax_R_z",
    "R_z0",
    "R_z1",
    "R_z2",
    "R_z3",
    "best_z_terminal_step",
)


def _write_cf_rows(
    path: Path, rows: list[dict[str, Any]], fieldnames: tuple[str, ...]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        if is_new:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _read_existing_cf_keys(path: Path) -> set[tuple[str, str, int]]:
    """Return ``(ckpt_path, opp, seed)`` tuples already in the CSV (resume)."""
    out: set[tuple[str, str, int]] = set()
    if not path.exists():
        return out
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out.add(
                    (
                        str(row["checkpoint_path"]),
                        str(row["opponent"]).upper(),
                        int(row["probe_seed"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return out


def _read_contexts_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=True) as data:
            keys = [str(k) for k in data["keys"].tolist()]
            ctx = np.asarray(data["contexts"], dtype=np.float32)
        if ctx.shape[0] != len(keys):
            return {}
        return {keys[i]: ctx[i] for i in range(len(keys))}
    except (OSError, KeyError, ValueError):
        return {}


def _write_contexts_npz(path: Path, merged: dict[str, np.ndarray]) -> None:
    if not merged:
        return
    keys = sorted(merged.keys())
    arr = np.stack(
        [np.asarray(merged[k], dtype=np.float32) for k in keys], axis=0
    )
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, keys=np.array(keys, dtype=object), contexts=arr)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Report writer (Gate 3 pass/fail)
# ---------------------------------------------------------------------------


def _per_opponent_Q_contrast(
    rows: list[dict[str, Any]]
) -> dict[str, dict[str, float]]:
    """Compute Q-contrast aggregates per opponent across all branches.

    Q_contrast(scene) = max_z(R) - min_z(R) at one (opp, seed, arc_idx).
    Per-opponent: mean / median / min across all scenes for that opponent.
    """
    by_opp: dict[str, list[float]] = {}
    by_opp_best_z: dict[str, list[int]] = {}
    for r in rows:
        opp = str(r.get("opponent", "")).upper()
        q = float(r.get("Q_contrast", 0.0))
        bz = int(r.get("best_z", 0))
        by_opp.setdefault(opp, []).append(q)
        by_opp_best_z.setdefault(opp, []).append(bz)
    out: dict[str, dict[str, float]] = {}
    for opp, qs in sorted(by_opp.items()):
        if not qs:
            continue
        bzs = by_opp_best_z[opp]
        # best_z occupancy entropy in nats (uniform over the observed K).
        counts: dict[int, int] = {}
        for bz in bzs:
            counts[bz] = counts.get(bz, 0) + 1
        n = sum(counts.values())
        ent = 0.0
        for c in counts.values():
            p = c / n
            if p > 0:
                ent -= p * math.log(p)
        out[opp] = {
            "n_scenes": float(len(qs)),
            "mean_Q_contrast": float(statistics.fmean(qs)),
            "median_Q_contrast": float(statistics.median(qs)),
            "min_Q_contrast": float(min(qs)),
            "max_Q_contrast": float(max(qs)),
            "best_z_entropy_nats": float(ent),
            "best_z_max_count_frac": float(max(counts.values()) / n) if n else 0.0,
        }
    return out


def _write_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    latent_k: int,
    determinism: tuple[bool, float, float] | None,
    success_threshold: float,
) -> None:
    per_opp = _per_opponent_Q_contrast(summary_rows)
    overall_mean = (
        statistics.fmean(float(r["Q_contrast"]) for r in summary_rows)
        if summary_rows
        else float("nan")
    )
    worst_opp_mean = (
        min(d["mean_Q_contrast"] for d in per_opp.values())
        if per_opp
        else float("nan")
    )
    lines: list[str] = []
    lines.append("# v4i3 Local Counterfactual Q(s, z) Probe Report")
    lines.append("")
    lines.append(f"- checkpoint: `{Path(args.checkpoint).name}`")
    lines.append(
        f"- opponents: {list(args.opponents)}, n_seeds: {args.n_seeds}, "
        f"base_seed: {args.base_seed}"
    )
    lines.append(
        f"- branch arc indices: {list(args.branch_arc_indices)}, "
        f"latent_k: {latent_k}, device: {args.device}"
    )
    lines.append("")
    if determinism is not None:
        passed, ra, rb = determinism
        lines.append("## Determinism self-test")
        lines.append("")
        lines.append(
            f"- R_A (force z=2, roll) = {ra:+.6f}, "
            f"R_B (restore, force z=2, roll) = {rb:+.6f}"
        )
        lines.append(
            f"- |R_A - R_B| = {abs(ra - rb):.2e}, tolerance = "
            f"{_DETERMINISM_TOLERANCE:.0e}"
        )
        lines.append(f"- **{'PASS' if passed else 'FAIL'}**: snapshot/restore is "
                     f"{'complete' if passed else 'INCOMPLETE'}.")
        if not passed:
            lines.append(
                "- WARNING: subsequent counterfactual returns may not be "
                "reproducible. Investigate which env or policy state is "
                "missing from the snapshot before trusting the gates."
            )
        lines.append("")
    lines.append("## Gate 3: True local Q(s, z) consequence")
    lines.append("")
    lines.append("Per the Summer Proof spec, Gate 3 passes when:")
    lines.append("")
    lines.append(f"- `mean local Q contrast > {success_threshold:.2f}` AND")
    lines.append(f"- `worst-opponent local Q contrast > {success_threshold:.2f}` AND")
    lines.append("- `best_z entropy not zero` AND")
    lines.append("- `best_z varies across tactical states`")
    lines.append("")
    lines.append("### Per-opponent local Q-contrast aggregates")
    lines.append("")
    lines.append(
        "| opponent | n_scenes | mean Q-contrast | median | min | "
        "max | best_z entropy (nats) | best_z max share |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for opp, d in per_opp.items():
        lines.append(
            f"| {opp} | {int(d['n_scenes'])} | "
            f"{d['mean_Q_contrast']:+.4f} | {d['median_Q_contrast']:+.4f} | "
            f"{d['min_Q_contrast']:+.4f} | {d['max_Q_contrast']:+.4f} | "
            f"{d['best_z_entropy_nats']:.3f} | {d['best_z_max_count_frac']:.3f} |"
        )
    lines.append("")
    lines.append(f"- overall mean Q-contrast: {overall_mean:+.4f}")
    lines.append(f"- worst-opponent mean Q-contrast: {worst_opp_mean:+.4f}")
    lines.append("")
    gate_3a = overall_mean > float(success_threshold)
    gate_3b = (
        worst_opp_mean > float(success_threshold) if per_opp else False
    )
    gate_3c = all(d["best_z_entropy_nats"] > 0.0 for d in per_opp.values())
    gate_3d = all(d["best_z_max_count_frac"] < 0.95 for d in per_opp.values())
    overall_pass = gate_3a and gate_3b and gate_3c and gate_3d
    lines.append("### Gate 3 verdict")
    lines.append("")
    lines.append(
        f"- mean local Q-contrast > {success_threshold:.2f}: "
        f"**{'PASS' if gate_3a else 'FAIL'}** (got {overall_mean:+.4f})"
    )
    lines.append(
        f"- worst-opponent Q-contrast > {success_threshold:.2f}: "
        f"**{'PASS' if gate_3b else 'FAIL'}** (got {worst_opp_mean:+.4f})"
    )
    lines.append(
        f"- best_z entropy > 0 on every opponent: "
        f"**{'PASS' if gate_3c else 'FAIL'}**"
    )
    lines.append(
        f"- best_z varies (no single z >=95% of scenes per opponent): "
        f"**{'PASS' if gate_3d else 'FAIL'}**"
    )
    lines.append("")
    lines.append(f"**Gate 3 overall: {'PASS' if overall_pass else 'FAIL'}**")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Each row in the per-scene summary CSV represents one "
        "`(opp, seed, arc_idx)` triple. K = latent_k forced-z rollouts "
        "share the same env+policy snapshot; only `z` varies."
    )
    lines.append(
        "- The per-row counterfactual CSV captures every individual "
        "`(opp, seed, arc_idx, forced_z)` rollout."
    )
    lines.append(
        "- This probe does NOT change the model; it only reads. Each "
        "snapshot/restore is in-place on the live env and policy state."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v4i3 Local Counterfactual Q(s, z) Probe."
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Source PPO checkpoint (e.g. final_v4i3_summer_proof_*.zip).",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Run tag for output filenames (default: derived from checkpoint).",
    )
    p.add_argument(
        "--opponents",
        nargs="+",
        default=["OP5", "OP6", "OP7"],
        help="Opponent labels to probe.",
    )
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Probe seeds = base_seed, base_seed+1, ..., base_seed+n_seeds-1.",
    )
    p.add_argument(
        "--branch-arc-indices",
        nargs="+",
        type=int,
        default=[0, 1],
        help=(
            "Arc boundaries (0-based) at which to snapshot+branch each forced "
            "z. The natural rollout continues through these to drive forward."
        ),
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--agents", type=int, default=4)
    p.add_argument(
        "--max-steps",
        type=int,
        default=_DEFAULT_MAX_STEPS,
        help="Per-rollout safety cap on env steps.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write CSVs/NPZ/report (default: same dir as checkpoint).",
    )
    p.add_argument(
        "--success-threshold",
        type=float,
        default=0.10,
        help="Gate 3 mean / worst-opp Q-contrast threshold (default 0.10).",
    )
    p.add_argument(
        "--determinism-tolerance",
        type=float,
        default=_DETERMINISM_TOLERANCE,
        help="Max |R_A - R_B| allowed in the snapshot self-test.",
    )
    p.add_argument(
        "--allow-determinism-drift",
        action="store_true",
        help=(
            "Continue past a failed self-test. Use only for debugging what's "
            "missing from the snapshot."
        ),
    )
    p.add_argument(
        "--skip-determinism-test",
        action="store_true",
        help="Skip the snapshot self-test entirely (NOT recommended).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"[local_cf] FATAL: checkpoint not found: {ckpt_path}")
        return 2
    run_tag = _resolve_run_tag(args.run_tag, ckpt_path)
    output_dir = Path(args.output_dir) if args.output_dir else ckpt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cf_csv = output_dir / f"{run_tag}_qprobe_local_cf.csv"
    summary_csv = output_dir / f"{run_tag}_qprobe_local_cf_summary.csv"
    contexts_npz = output_dir / f"{run_tag}_qprobe_local_cf_contexts.npz"
    report_md = output_dir / f"{run_tag}_qprobe_local_cf_report.md"

    print(f"[local_cf] run_tag={run_tag}")
    print(
        f"[local_cf] opponents={list(args.opponents)} n_seeds={int(args.n_seeds)} "
        f"base_seed={int(args.base_seed)} branch_arc_indices={list(args.branch_arc_indices)} "
        f"device={args.device}"
    )
    print(f"[local_cf] output_dir={output_dir}")

    meta = read_custom_ppo_metadata(str(ckpt_path))
    if not bool(meta.get("use_latent_strategy", False)):
        print(
            f"[local_cf] FATAL: checkpoint is not latent (use_latent_strategy=False)."
        )
        return 2
    latent_k = int(meta.get("latent_k", 4))
    n_blue = int(meta.get("n_blue", args.agents))
    ckpt_steps = int(meta.get("global_step", -1) or -1)
    cfg_meta = meta.get("cfg") or {}
    arc_interval = int(cfg_meta.get("latent_resample_every_n", 0) or 0)
    if arc_interval <= 0:
        print(
            "[local_cf] WARNING: latent_resample_every_n is 0 in the checkpoint "
            "config; defaulting to 64 (v3i19/v4i1 family)."
        )
        arc_interval = 64
    print(
        f"[local_cf] latent_k={latent_k}, n_blue={n_blue}, arc_interval={arc_interval}, "
        f"ckpt_steps={ckpt_steps}"
    )

    # Build env + policy (mirrors q_probe.probe_checkpoint).
    cfg = PPOConfig()
    cfg.use_latent_strategy = True
    cfg.latent_k = latent_k
    cfg.n_envs = 1
    cfg.seed = int(args.base_seed)
    cfg.device = str(args.device)
    cfg.max_blue_agents = n_blue
    cfg.n_agents_per_team = n_blue
    cfg.opponent_randomize = False
    cfg.fixed_opponent_tag = _env_opponent_tag(args.opponents[0])
    initial_env_tag = _env_opponent_tag(args.opponents[0])
    env = build_training_env(
        cfg,
        initial_phase="PHASE1",
        initial_opponent_tag=initial_env_tag,
    )
    try:
        model = load_custom_ppo_policy(
            str(ckpt_path),
            env.observation_space,
            env.action_space,
            device=str(args.device),
        )
        # Determinism self-test (against the first opponent + first seed,
        # branching at the smallest requested arc boundary).
        determinism: tuple[bool, float, float] | None = None
        if not args.skip_determinism_test:
            opp0 = _env_opponent_tag(args.opponents[0])
            print(
                "[local_cf] determinism self-test: opp="
                f"{args.opponents[0]} seed={int(args.base_seed)} "
                f"branch_arc={min(args.branch_arc_indices)} test_z="
                f"{min(2, max(0, latent_k - 1))}"
            )
            passed, ra, rb = _run_determinism_self_test(
                env=env,
                model=model,
                opp_env_tag=opp0,
                probe_seed=int(args.base_seed),
                device=str(args.device),
                latent_k=int(latent_k),
                arc_interval=int(arc_interval),
                branch_arc_indices=list(args.branch_arc_indices),
                max_steps=int(args.max_steps),
                tol=float(args.determinism_tolerance),
            )
            determinism = (passed, ra, rb)
            diff = abs(ra - rb) if not (math.isnan(ra) or math.isnan(rb)) else float("nan")
            print(
                f"[local_cf] determinism self-test: "
                f"R_A={ra:+.6f}, R_B={rb:+.6f}, |dR|={diff:.2e}, "
                f"{'PASS' if passed else 'FAIL'}"
            )
            if not passed and not args.allow_determinism_drift:
                print(
                    "[local_cf] ABORTING: snapshot/restore is incomplete and "
                    "--allow-determinism-drift is off. Pass that flag only for "
                    "debugging; the probe's counterfactual returns are not "
                    "trustworthy until this test passes."
                )
                return 3

        # Existing CSV resume keys.
        existing_keys = _read_existing_cf_keys(cf_csv)
        contexts_merged = _read_contexts_npz(contexts_npz)

        ckpt_path_s = str(ckpt_path)
        total_combos = len(args.opponents) * int(args.n_seeds)
        combo_idx = 0
        new_rows: list[dict[str, Any]] = []
        new_summary_rows: list[dict[str, Any]] = []
        new_contexts: dict[str, np.ndarray] = {}

        for opp_label in args.opponents:
            opp_env_tag = _env_opponent_tag(opp_label)
            for seed_offset in range(int(args.n_seeds)):
                probe_seed = int(args.base_seed) + int(seed_offset)
                combo_idx += 1
                resume_key = (ckpt_path_s, str(opp_label).upper(), int(probe_seed))
                if resume_key in existing_keys:
                    continue
                t_start = time.time()
                ep = _run_one_probe_episode(
                    env=env,
                    model=model,
                    opp_env_tag=opp_env_tag,
                    probe_seed=probe_seed,
                    device=str(args.device),
                    latent_k=int(latent_k),
                    arc_interval=int(arc_interval),
                    branch_arc_indices=list(args.branch_arc_indices),
                    max_steps=int(args.max_steps),
                )
                elapsed = time.time() - t_start
                branches: list[_BranchResult] = ep["branches"]
                print(
                    f"[local_cf] {opp_label} seed={probe_seed} natural_R={ep['natural_return']:+.3f} "
                    f"len={ep['natural_episode_length']} score={ep['natural_blue']}-{ep['natural_red']} "
                    f"branches={len(branches)} ({combo_idx}/{total_combos}, {elapsed:.1f}s)"
                )

                # Emit per-row counterfactual rows + per-scene summary rows.
                for br in branches:
                    ctx_key = (
                        f"{ckpt_steps}|{str(opp_label).upper()}|"
                        f"{int(probe_seed)}|arc{int(br.arc_idx)}"
                    )
                    if ctx_key not in contexts_merged and ctx_key not in new_contexts:
                        new_contexts[ctx_key] = np.asarray(
                            br.context, dtype=np.float32
                        )
                    for z in range(int(latent_k)):
                        new_rows.append(
                            {
                                "checkpoint_path": ckpt_path_s,
                                "checkpoint_steps": int(ckpt_steps),
                                "opponent": str(opp_label).upper(),
                                "probe_seed": int(probe_seed),
                                "arc_idx": int(br.arc_idx),
                                "timestep": int(br.timestep),
                                "forced_z": int(z),
                                "remaining_return": float(
                                    br.remaining_return_per_z[z]
                                ),
                                "terminal_step": int(
                                    br.terminal_step_per_z[z]
                                ),
                                "branch_blue_score": int(
                                    br.blue_score_per_z[z]
                                ),
                                "branch_red_score": int(
                                    br.red_score_per_z[z]
                                ),
                                "branch_blue_won": int(
                                    br.blue_score_per_z[z]
                                    > br.red_score_per_z[z]
                                ),
                                "natural_return": float(ep["natural_return"]),
                                "natural_episode_length": int(
                                    ep["natural_episode_length"]
                                ),
                                "natural_blue_score": int(ep["natural_blue"]),
                                "natural_red_score": int(ep["natural_red"]),
                                "context_key": ctx_key,
                            }
                        )
                    rets = np.array(
                        br.remaining_return_per_z, dtype=np.float64
                    )
                    best_z = int(np.argmax(rets))
                    worst_z = int(np.argmin(rets))
                    q_contrast = float(rets.max() - rets.min())
                    uniform_avg = float(rets.mean())
                    summary_row = {
                        "checkpoint_path": ckpt_path_s,
                        "checkpoint_steps": int(ckpt_steps),
                        "opponent": str(opp_label).upper(),
                        "probe_seed": int(probe_seed),
                        "arc_idx": int(br.arc_idx),
                        "timestep": int(br.timestep),
                        "best_z": int(best_z),
                        "worst_z": int(worst_z),
                        "Q_contrast": float(q_contrast),
                        "best_minus_uniform_avg": float(
                            rets[best_z] - uniform_avg
                        ),
                        "argmax_R_z": int(best_z),
                        "best_z_terminal_step": int(
                            br.terminal_step_per_z[best_z]
                        ),
                    }
                    for zi in range(int(latent_k)):
                        summary_row[f"R_z{zi}"] = float(
                            br.remaining_return_per_z[zi]
                        )
                    # Pad missing latent_k slots in the fixed column schema.
                    for zi in range(int(latent_k), 4):
                        summary_row[f"R_z{zi}"] = ""
                    new_summary_rows.append(summary_row)

                existing_keys.add(resume_key)

                # Flush every (opp, seed) so an interrupted run is resumable.
                if new_rows:
                    _write_cf_rows(cf_csv, new_rows, _CF_FIELDS)
                    new_rows = []
                if new_summary_rows:
                    _write_cf_rows(summary_csv, new_summary_rows, _CF_SUMMARY_FIELDS)
                    new_summary_rows = []
                if new_contexts:
                    contexts_merged.update(new_contexts)
                    _write_contexts_npz(contexts_npz, contexts_merged)
                    new_contexts = {}
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[local_cf] WARNING: env.close() raised: {exc}")

    # Build the report from the full CSVs (including any rows from prior runs).
    all_summary: list[dict[str, Any]] = []
    if summary_csv.exists():
        with summary_csv.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    row["Q_contrast"] = float(row.get("Q_contrast", 0.0))
                    row["best_z"] = int(row.get("best_z", 0))
                except (TypeError, ValueError):
                    continue
                all_summary.append(row)
    _write_report(
        report_md,
        summary_rows=all_summary,
        args=args,
        latent_k=int(latent_k),
        determinism=determinism,
        success_threshold=float(args.success_threshold),
    )
    print(f"[local_cf] wrote: {cf_csv}")
    print(f"[local_cf] wrote: {summary_csv}")
    if contexts_merged:
        print(f"[local_cf] wrote: {contexts_npz} ({len(contexts_merged)} keys)")
    print(f"[local_cf] wrote: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
