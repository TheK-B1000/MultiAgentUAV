"""v4i1 Offline Return Contrast Probe.

For each checkpoint of a v4i1 (or any latent) run, this tool freezes the
policy and, for every opponent in the strategic-pressure pool, runs matched
episodes with each forced ``z`` value. The undiscounted episode return is
collected per ``(checkpoint, opponent, z, probe_seed)`` and aggregated into

    return_contrast = max_z(mean_R) - min_z(mean_R)         (per opponent)
    return_contrast_overall = mean_opponent(return_contrast_per_opponent)
    return_contrast_worst   = min_opponent(return_contrast_per_opponent)

This is the v4i1 primary metric. Thresholds (per the v4i1 spec):

    return_contrast < 0.05   -> FAIL: environment does not care about strategy
    return_contrast >= 0.10  -> PASS: different z choices create different outcomes

The probe is **out-of-band**: it never touches ``trainer.learn()`` and never
modifies a checkpoint. It only reads ``.zip`` files from ``--checkpoint-dir``
and writes three artifacts next to them:

* ``<run_tag>_qprobe.csv``         -- one row per matched episode (raw)
* ``<run_tag>_qprobe_summary.csv`` -- aggregated per (checkpoint, opponent, z)
* ``<run_tag>_qprobe_report.md``   -- latest-checkpoint contrast table + verdict

``--watch`` polls the directory and processes newly-discovered checkpoints
incrementally; entries already present in the raw CSV are skipped. Use a
separate process (and/or ``--device cpu``) so the probe does not contend
with training for the GPU.

Usage::

    # Single-shot against one checkpoint:
    python tools/q_probe.py \\
        --checkpoint checkpoints/4v4/ckpt_v4i1_..._500000.zip \\
        --opponents OP5 OP6 OP7 --n-seeds 8 --device cuda

    # Continuous watcher (kicks off and stays running):
    python tools/q_probe.py \\
        --checkpoint-dir checkpoints/4v4 \\
        --run-tag v4i1_strategic_pressure_qprobe_OP5_OP6_OP7_2m_4v4 \\
        --opponents OP5 OP6 OP7 --n-seeds 8 --device cuda --watch
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
from typing import Any, Iterable

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
# Constants / config
# ---------------------------------------------------------------------------

_OPPONENT_ENV_TAG: dict[str, str] = {
    "OP5": "OP5_RUSHER",
    "OP6": "OP6_TURTLE",
    "OP7": "OP7_SWITCHER",
}

# Mirrors qualitative_rollout._env_opponent_tag pass-through behavior.
def _env_opponent_tag(label: str) -> str:
    return _OPPONENT_ENV_TAG.get(str(label).strip().upper(), str(label).strip().upper())


# Hard cap on per-episode rollout length; the env naturally terminates well
# before this for 4v4 (decision-step budget is typically ~200). We just want
# a safety net in case ``done`` never fires.
_DEFAULT_MAX_STEPS = 1500


_RAW_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "checkpoint_steps",
    "checkpoint_kind",
    "opponent",
    "z",
    "probe_seed",
    "episode_return",
    "episode_length",
    "blue_won",
    "blue_score",
    "red_score",
    "blue_score_minus_red",
)

_SUMMARY_FIELDS: tuple[str, ...] = (
    # row identity
    "checkpoint_path",
    "checkpoint_steps",
    "checkpoint_kind",
    "opponent",
    "z",
    # per (opp, z) episode aggregates
    "n_seeds",
    "n_complete_seeds",
    "mean_return",
    "std_return",
    "min_return",
    "max_return",
    "win_rate",
    "mean_blue_minus_red",
    "mean_blue_score",
    "mean_red_score",
    "mean_episode_length",
    "scoring_rate",
    "best_z_count",
    "best_z_frac",
    # per (opp) cross-z metrics (repeated across z rows)
    "contrast_within_opponent_paired",       # PRIMARY: mean over seeds of (max_z R - min_z R)
    "contrast_within_opponent_paired_std",
    "contrast_within_opponent_unpaired",     # DIAGNOSTIC: max_z(mean_R) - min_z(mean_R)
    "win_rate_contrast_paired",              # mean over seeds of (max_z won - min_z won), in {0,1}
    "win_rate_contrast_unpaired",            # max_z(WR_z) - min_z(WR_z)
    "best_z_entropy_nats",                   # H over per-opp best_z histogram
    "best_z_entropy_norm",                   # H / log(K_eff)
    # per-checkpoint rollup (repeated across all rows of same ckpt)
    "return_contrast_overall_paired",        # mean over opps of paired
    "return_contrast_worst_paired",          # min over opps of paired
    "return_contrast_overall_unpaired",      # mean over opps of unpaired
    "return_contrast_worst_unpaired",        # min over opps of unpaired
    "win_rate_contrast_overall_paired",
    "best_z_entropy_overall",
)


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointEntry:
    path: Path
    steps: int
    kind: str  # "ckpt" | "interrupt" | "final"


_CKPT_NUMERIC_RE = re.compile(r"(ckpt|interrupt|final)_(.+?)(?:_(\d+))?\.zip$", re.IGNORECASE)


def _parse_checkpoint(path: Path, run_tag: str) -> CheckpointEntry | None:
    """Return (steps, kind) if ``path`` is a checkpoint for ``run_tag``, else None."""
    name = path.name
    m = _CKPT_NUMERIC_RE.match(name)
    if m is None:
        return None
    kind = m.group(1).lower()
    middle = m.group(2)
    steps_s = m.group(3)
    # ``middle`` is everything between ``<kind>_`` and the trailing ``_<steps>``
    # (for ckpt/interrupt) or the bare end (for final). For ``final_<tag>.zip``
    # steps_s is None and ``middle`` is the run tag itself; for the others
    # ``middle`` is the run tag.
    if str(middle).strip() != str(run_tag).strip():
        return None
    if kind == "final":
        steps = -1  # unknown / terminal
    else:
        steps = int(steps_s) if steps_s is not None else -1
    return CheckpointEntry(path=path, steps=steps, kind=kind)


def discover_checkpoints(directory: Path, run_tag: str) -> list[CheckpointEntry]:
    """List all checkpoints for ``run_tag`` under ``directory``, sorted by step ascending."""
    if not directory.is_dir():
        return []
    out: list[CheckpointEntry] = []
    for child in directory.iterdir():
        if not child.is_file() or child.suffix.lower() != ".zip":
            continue
        entry = _parse_checkpoint(child, run_tag)
        if entry is not None:
            out.append(entry)
    out.sort(key=lambda e: (e.steps, e.kind, e.path.name))
    return out


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _read_existing_raw_keys(csv_path: Path) -> set[tuple[str, int, str, int]]:
    """Set of (checkpoint_path, z, opponent, probe_seed) tuples already in the raw CSV."""
    if not csv_path.exists():
        return set()
    keys: set[tuple[str, int, str, int]] = set()
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    keys.add(
                        (
                            str(row["checkpoint_path"]),
                            int(row["z"]),
                            str(row["opponent"]),
                            int(row["probe_seed"]),
                        )
                    )
                except (KeyError, ValueError):
                    continue
    except OSError:
        return set()
    return keys


def _append_raw_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_RAW_FIELDS))
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _RAW_FIELDS})


def _rewrite_summary(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically overwrite the summary CSV with the rolling state."""
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_SUMMARY_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _SUMMARY_FIELDS})
    os.replace(tmp, csv_path)


def _read_raw_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            out.append(dict(row))
    return out


# ---------------------------------------------------------------------------
# v4i2b: arc-boundary CSV + NPZ I/O (router distillation richer input)
#
# These artifacts are gated by ``--save-arc-contexts``. They are written
# alongside the existing v4i2 ``_qprobe_contexts.npz`` (which stays one
# row per (opp, seed)) and do NOT replace it. The supervision target
# encoded in the arc CSV is ``remaining_return_from_here`` --
# strictly an *approximation* of the local optimal-z target: it is the
# return of the z that was actually forced from that arc onward, NOT a
# clone-and-replay counterfactual for each z'. See the v4i2b plan for
# the distinction.
# ---------------------------------------------------------------------------

_ARC_FIELDS: tuple[str, ...] = (
    "checkpoint_path",
    "checkpoint_steps",
    "checkpoint_kind",
    "opponent",
    "probe_seed",
    "forced_z",
    "arc_idx",
    "timestep",
    "approx_remaining_return_supervision",
    "final_episode_return",
    "episode_length",
    "blue_score",
    "red_score",
    "blue_won",
    "context_key",
)


def _arc_context_key(*, steps: int, opp: str, seed: int, z: int, arc_idx: int) -> str:
    return f"{int(steps)}|{str(opp).upper()}|{int(seed)}|{int(z)}|{int(arc_idx)}"


# Debug toggle for matched-start contract verification: when set to a truthy
# env var, ``_run_one_episode`` dumps ``env.state()`` slices at step 0 of each
# forced-z roll so dispersion across z within (opp, seed) can be eyeballed.
# Default off; only useful when chasing matched-start regressions.
_QPROBE_DEBUG_MATCHED_START: bool = bool(os.environ.get("QPROBE_DEBUG_MATCHED_START", ""))


def _read_existing_arc_keys(
    csv_path: Path,
) -> set[tuple[str, str, int, int]]:
    """Return ``(ckpt_path, opp, seed, z)`` tuples already represented in the arcs CSV.

    Resume contract: if a (ckpt, opp, seed, z) tuple already has ANY arc
    row in the CSV, we treat that episode as done -- we do not re-run it.
    Per-arc-idx granularity is unnecessary because arc-idx coverage is
    deterministic for a fixed (model, seed, z): re-running the same
    matched-start episode would produce the same set of arc indices.
    """
    out: set[tuple[str, str, int, int]] = set()
    if not csv_path.exists():
        return out
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out.add(
                    (
                        str(row["checkpoint_path"]),
                        str(row["opponent"]).upper(),
                        int(row["probe_seed"]),
                        int(row["forced_z"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return out


def _append_arc_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_ARC_FIELDS))
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _ARC_FIELDS})


def _read_arc_contexts_npz(path: Path) -> dict[str, np.ndarray]:
    """Load ``<run_tag>_qprobe_arc_contexts.npz`` if present; else {}.

    Same two-parallel-arrays layout as the v4i2 contexts file. Keys are
    ``f"{steps}|{opp}|{seed}|{z}|{arc_idx}"``.
    """
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


def _write_arc_contexts_npz(
    path: Path, merged: dict[str, np.ndarray]
) -> None:
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
# Contexts NPZ I/O (v4i2 router distillation input)
# ---------------------------------------------------------------------------

def _read_contexts_npz(path: Path) -> dict[str, np.ndarray]:
    """Load an existing ``_qprobe_contexts.npz`` if present; else return {}.

    File layout (two parallel arrays):
      * ``keys``     : object array of strings ``f"{steps}|{opp}|{seed}"``
      * ``contexts`` : float32 array shaped ``(N, 170)``
    """
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
    """Atomically write the merged contexts dict to ``path``."""
    if not merged:
        return
    keys = sorted(merged.keys())
    arr = np.stack([np.asarray(merged[k], dtype=np.float32) for k in keys], axis=0)
    # ``np.savez`` auto-appends ``.npz`` to filenames that do not already end
    # in ``.npz``, so we must build a tmp path that already ends in ``.npz``.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, keys=np.array(keys, dtype=object), contexts=arr)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Probe core
# ---------------------------------------------------------------------------

def _set_global_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFF_FFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _run_one_episode(
    *,
    env: Any,
    model: Any,
    opponent_env_tag: str,
    forced_z: int,
    probe_seed: int,
    device: str,
    max_steps: int,
    arc_interval: int = 0,
) -> dict[str, Any]:
    """Run a single matched-start episode with z forced to ``forced_z``.

    Returns a row dict matching the raw CSV schema for one (opp, z, seed).
    Matched-start contract: callers reseed the env BEFORE calling this so
    every z variant within a given (opponent, probe_seed) sees the same
    initial state.

    Optional ``arc_interval`` (>0) turns on v4i2b arc-boundary capture:
    in addition to the per-episode aggregates, the returned row carries
    ``_arc_captures``, a list of one dict per capture-time, with:

      * ``arc_idx``: 0, 1, 2, ... -- a contiguous counter
      * ``timestep``: env step where the capture happened
      * ``context``: float32 np.ndarray, shape (170,), the q_phi input
        ``model._last_context_gs`` at that step (= what q_phi would see
        if it had been resampling z here under ``latent_resample_every_n``)
      * ``remaining_return_from_here``: float, ``sum(step_rewards[t:])``
        i.e. the return of the (forced_z) trajectory from this arc onward.

    Captures are taken AT decision time (right after ``model.predict``)
    on steps ``0, arc_interval, 2*arc_interval, ...`` so the temporal
    positions match where q_phi is normally queried under
    ``latent_resample_every_n=arc_interval`` -- even though q_probe
    forces z and so q_phi is never actually consulted here.
    """
    # Matched-start contract for forced-z probing:
    #
    #   Within one (opp, probe_seed) group we run K = latent_k episodes that
    #   differ ONLY in the forced ``z``. For arc_idx=0 dispersion to round
    #   to zero (the contract on which v4i2b distillation + v4i3 local CF
    #   both rely), three things must be true at the first ``model.predict``:
    #
    #     1. ``core._rng`` is in the *same* deterministic state across z, so
    #        ``env.reset()`` produces the same initial global_state.
    #     2. The policy's TemporalStateTracker EMAs are zeroed, so the
    #        first ``tracker.update`` produces ``concat([s, s, s, 0, 0])``
    #        from the matched initial state ``s``.
    #     3. The action / strategy sampling generators are reseeded so the
    #        only thing differing across z is z itself, not the RNG.
    #
    # The pitfall: ``env.env_method("set_next_opponent", ...)`` triggers
    # ``_apply_opponent_params_for_mask``, which calls
    # ``sample_batched_opponent_params(..., generator=self._rng)`` and so
    # CONSUMES ``core._rng``. If we seed the env BEFORE set_next_opponent,
    # ``env.reset()`` runs with a partly-consumed RNG that depends on the
    # opponent transition (and on cached mask state inside the core).
    # Therefore we (a) flip the env-opponent BEFORE re-seeding, then
    # (b) re-seed the env so ``env.reset()`` always sees a fresh ``_rng``.
    _set_global_seeds(probe_seed)
    env.env_method("set_next_opponent", "SCRIPTED", opponent_env_tag)
    try:
        env.seed(int(probe_seed))
    except Exception:
        pass
    # Reset the in-policy episode-state caches so q_phi's prev_z + temporal
    # tracker do not bleed across (opp, seed, z) combos. Then clamp z.
    if hasattr(model, "reset_strategy"):
        model.reset_strategy()
    if hasattr(model, "fixed_latent_strategy"):
        model.fixed_latent_strategy = True
    if hasattr(model, "fixed_latent_strategy_id"):
        model.fixed_latent_strategy_id = int(forced_z)
    apply_deterministic_sampling_generators(
        model.model, int(probe_seed), device=device
    )

    obs = env.reset()
    # Defensive second reset of the temporal tracker AFTER env.reset() but
    # BEFORE the first predict. The tracker auto-resets on
    # ``decision_frac < 1e-5`` inside its update path, but only if that
    # column is exactly zero at step 0; defensively zeroing here makes the
    # contract independent of the env's exact initial decision_frac value
    # and matches the way the trainer reinitialises the tracker between
    # rollouts.
    if hasattr(model, "_temporal_tracker") and model._temporal_tracker is not None:
        model._temporal_tracker.reset()
    episode_return = 0.0
    episode_length = 0
    blue_score = 0
    red_score = 0
    blue_won = False
    # First-step q_phi context (170-d global-state temporal stack), captured
    # right after the very first ``model.predict`` so the temporal tracker has
    # been primed once. Same value for every z within a matched (opp, seed)
    # pair, so v4i2 router distillation can key it as one row per (opp, seed).
    first_context: np.ndarray | None = None
    # v4i2b arc-boundary capture state. The captures are filled in during
    # the loop (with the context-at-decision-time but no remaining_return
    # yet); ``remaining_return_from_here`` is back-filled after the loop
    # so it can use the eventual step_rewards.
    arc_interval_eff = int(arc_interval) if arc_interval and int(arc_interval) > 0 else 0
    arc_captures: list[dict[str, Any]] = []
    step_rewards: list[float] = []

    for step in range(max_steps):
        # ``CustomPPOInferencePolicy.predict`` expects per-env (un-batched)
        # obs dicts. ``env.reset()`` / ``step_wait()`` return arrays whose
        # leading axis is num_envs=1, so slice to row 0 before predict.
        single = {
            k: (v[0] if hasattr(v, "shape") and v.ndim >= 2 and v.shape[0] == 1 else v)
            for k, v in obs.items()
        }
        try:
            single["global_state"] = env.state()[0]
        except Exception:
            pass
        if step == 0 and _QPROBE_DEBUG_MATCHED_START:
            import sys
            gs0 = single.get("global_state")
            if gs0 is not None:
                _arr = np.asarray(gs0, dtype=np.float32).reshape(-1)
                print(
                    f"[q_probe-debug] opp={opponent_env_tag} seed={probe_seed} z={forced_z} step0 "
                    f"gs[:6]={_arr[:6].tolist()} gs[12:14]={_arr[12:14].tolist()}",
                    file=sys.stderr, flush=True,
                )
        # Stochastic policy so the only difference across z is z itself
        # (deterministic sampling generators give matched RNG sequences).
        action, _ = model.predict(single, deterministic=False)
        if step == 0 and first_context is None:
            ctx_t = getattr(model, "_last_context_gs", None)
            if ctx_t is not None:
                arr = ctx_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
                first_context = arr
        # v4i2b: capture q_phi context at every arc boundary (including
        # step 0). ``model._last_context_gs`` is set on every predict() and
        # is the exact tensor q_phi consumes at this decision boundary.
        if arc_interval_eff > 0 and (step % arc_interval_eff == 0):
            ctx_t = getattr(model, "_last_context_gs", None)
            if ctx_t is not None:
                arc_captures.append(
                    {
                        "arc_idx": int(len(arc_captures)),
                        "timestep": int(step),
                        "context": ctx_t.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                        .reshape(-1),
                    }
                )
        env.step_async(action)
        obs, rewards, dones, infos = env.step_wait()
        r_t = float(np.asarray(rewards).reshape(-1)[0])
        step_rewards.append(r_t)
        episode_return += r_t
        episode_length = step + 1
        if bool(np.asarray(dones).reshape(-1)[0]):
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info) or {}
            blue_score = int(ep_res.get("blue_score", 0))
            red_score = int(ep_res.get("red_score", 0))
            blue_won = bool(blue_score > red_score)
            break

    # Back-fill ``remaining_return_from_here`` for each arc capture using
    # the (forced-z) per-step rewards collected above. ``step_rewards[t]``
    # is the reward emitted by the env in response to the action chosen
    # at step ``t`` (i.e., the reward consequent to the decision the
    # capture was taken at), so ``sum(step_rewards[t:])`` is exactly the
    # remaining-return-from-here under the forced-z policy.
    if arc_captures:
        # Pre-compute cumulative tail sum for O(N) total cost.
        n = len(step_rewards)
        tail = [0.0] * (n + 1)
        running = 0.0
        for i in range(n - 1, -1, -1):
            running += step_rewards[i]
            tail[i] = running
        episode_R_f = float(episode_return)
        for cap in arc_captures:
            t = int(cap["timestep"])
            if t < n:
                cap["remaining_return_from_here"] = float(tail[t])
            else:
                # Capture timestep is past the last collected reward
                # (episode ended exactly at the arc boundary). Remaining
                # return from that point is 0.0 by definition.
                cap["remaining_return_from_here"] = 0.0
            cap["final_episode_return"] = episode_R_f

    return {
        "z": int(forced_z),
        "probe_seed": int(probe_seed),
        "episode_return": float(episode_return),
        "episode_length": int(episode_length),
        "blue_won": int(blue_won),
        "blue_score": int(blue_score),
        "red_score": int(red_score),
        "blue_score_minus_red": int(blue_score - red_score),
        "_first_context": first_context,
        "_arc_captures": arc_captures,
    }


def probe_checkpoint(
    *,
    entry: CheckpointEntry,
    opponents: list[str],
    n_seeds: int,
    base_seed: int,
    device: str,
    agents: int,
    max_steps: int,
    skip_existing_keys: set[tuple[str, int, str, int]] | None = None,
    save_contexts: bool = False,
    existing_context_keys: set[str] | None = None,
    save_arc_contexts: bool = False,
    existing_arc_keys: set[tuple[str, str, int, int]] | None = None,
    existing_arc_context_keys: set[str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, np.ndarray],
    list[dict[str, Any]],
    dict[str, np.ndarray],
]:
    """Run the full (opponent x z x seed) probe matrix on one checkpoint.

    Returns ``(new_rows, new_contexts, new_arc_rows, new_arc_contexts)``:

    * ``new_rows``: list of raw per-episode dicts (does NOT write them).
    * ``new_contexts``: when ``save_contexts`` is set, maps
      ``f"{checkpoint_steps}|{opponent}|{probe_seed}"`` -> 170-d float32
      context vector. Only keys not already present in
      ``existing_context_keys`` are populated. Caller writes/merges the NPZ.
    * ``new_arc_rows``: when ``save_arc_contexts`` is set, one dict per
      arc-boundary capture matching ``_ARC_FIELDS``. Schema documented at
      the top of the v4i2b section.
    * ``new_arc_contexts``: when ``save_arc_contexts`` is set, maps each
      arc-row ``context_key`` -> 170-d float32 vector.
    """
    skip = skip_existing_keys or set()
    ctx_skip = existing_context_keys or set()
    arc_skip = existing_arc_keys or set()
    arc_ctx_skip = existing_arc_context_keys or set()
    new_contexts: dict[str, np.ndarray] = {}
    new_arc_rows: list[dict[str, Any]] = []
    new_arc_contexts: dict[str, np.ndarray] = {}
    ckpt_path_s = str(entry.path)

    meta = read_custom_ppo_metadata(ckpt_path_s)
    if not bool(meta.get("use_latent_strategy", False)):
        print(
            f"[q_probe] SKIP {entry.path.name}: not a latent checkpoint "
            "(use_latent_strategy=False)."
        )
        return [], new_contexts, new_arc_rows, new_arc_contexts
    latent_k = int(meta.get("latent_k", 4))
    # v4i2b: pick the arc-capture interval. Prefer the checkpoint's
    # ``latent_resample_every_n`` (so the captured contexts match where
    # q_phi would actually be queried under PPO). Fall back to 64 (the
    # v3i19 / v4i1 default) if missing or zero.
    cfg_meta = meta.get("cfg") or {}
    arc_interval = int(cfg_meta.get("latent_resample_every_n", 0) or 0)
    if save_arc_contexts and arc_interval <= 0:
        print(
            "[q_probe] WARNING: latent_resample_every_n is 0 in the checkpoint "
            "config; defaulting --save-arc-contexts interval to 64 to match the "
            "v3i19 / v4i1 family."
        )
        arc_interval = 64
    n_blue = int(meta.get("n_blue", agents))

    cfg = PPOConfig()
    cfg.use_latent_strategy = True
    cfg.latent_k = latent_k
    cfg.n_envs = 1
    cfg.seed = int(base_seed)
    cfg.device = str(device)
    cfg.max_blue_agents = n_blue
    cfg.n_agents_per_team = n_blue
    # Probe never relies on the in-trainer opponent-randomization hook; we
    # set the opponent explicitly per (opp, seed, z) below.
    cfg.opponent_randomize = False
    cfg.fixed_opponent_tag = _env_opponent_tag(opponents[0])

    initial_env_tag = _env_opponent_tag(opponents[0])
    env = build_training_env(
        cfg,
        initial_phase="PHASE1",
        initial_opponent_tag=initial_env_tag,
    )
    new_rows: list[dict[str, Any]] = []
    try:
        model = load_custom_ppo_policy(
            ckpt_path_s,
            env.observation_space,
            env.action_space,
            device=device,
        )

        total_combos = len(opponents) * latent_k * n_seeds
        combo_idx = 0
        for opp_label in opponents:
            opp_env_tag = _env_opponent_tag(opp_label)
            for seed_offset in range(n_seeds):
                probe_seed = int(base_seed) + int(seed_offset)
                for z in range(latent_k):
                    combo_idx += 1
                    key = (ckpt_path_s, int(z), str(opp_label).upper(), int(probe_seed))
                    if key in skip:
                        continue
                    ep = _run_one_episode(
                        env=env,
                        model=model,
                        opponent_env_tag=opp_env_tag,
                        forced_z=z,
                        probe_seed=probe_seed,
                        device=device,
                        max_steps=max_steps,
                        arc_interval=int(arc_interval) if save_arc_contexts else 0,
                    )
                    # Capture first-step q_phi context once per
                    # (entry.steps, opp, seed). It is identical across z
                    # for matched seeds because the env is reset to the
                    # same state, so we only record it on the first z that
                    # finishes for this (opp, seed) pair.
                    if save_contexts:
                        ctx_key = (
                            f"{int(entry.steps)}|{str(opp_label).upper()}|{int(probe_seed)}"
                        )
                        ctx_arr = ep.pop("_first_context", None)
                        if (
                            ctx_arr is not None
                            and ctx_key not in ctx_skip
                            and ctx_key not in new_contexts
                        ):
                            new_contexts[ctx_key] = np.asarray(
                                ctx_arr, dtype=np.float32
                            )
                    else:
                        ep.pop("_first_context", None)
                    # v4i2b: emit per-arc rows + contexts for this episode.
                    arc_caps = ep.pop("_arc_captures", []) or []
                    if save_arc_contexts and arc_caps:
                        arc_resume_key = (
                            ckpt_path_s,
                            str(opp_label).upper(),
                            int(probe_seed),
                            int(z),
                        )
                        # The resume contract is per-episode: if this
                        # (ckpt, opp, seed, z) already has any arc rows
                        # in the CSV, skip them entirely on re-run to
                        # avoid duplicates. (Probe is deterministic.)
                        if arc_resume_key not in arc_skip:
                            for cap in arc_caps:
                                ctx_key = _arc_context_key(
                                    steps=int(entry.steps),
                                    opp=str(opp_label),
                                    seed=int(probe_seed),
                                    z=int(z),
                                    arc_idx=int(cap["arc_idx"]),
                                )
                                if (
                                    ctx_key not in arc_ctx_skip
                                    and ctx_key not in new_arc_contexts
                                ):
                                    new_arc_contexts[ctx_key] = np.asarray(
                                        cap["context"], dtype=np.float32
                                    )
                                new_arc_rows.append(
                                    {
                                        "checkpoint_path": ckpt_path_s,
                                        "checkpoint_steps": int(entry.steps),
                                        "checkpoint_kind": entry.kind,
                                        "opponent": str(opp_label).upper(),
                                        "probe_seed": int(probe_seed),
                                        "forced_z": int(z),
                                        "arc_idx": int(cap["arc_idx"]),
                                        "timestep": int(cap["timestep"]),
                                        "approx_remaining_return_supervision": float(
                                            cap.get(
                                                "remaining_return_from_here", 0.0
                                            )
                                        ),
                                        "final_episode_return": float(
                                            cap.get(
                                                "final_episode_return",
                                                ep.get("episode_return", 0.0),
                                            )
                                        ),
                                        "episode_length": int(
                                            ep.get("episode_length", 0)
                                        ),
                                        "blue_score": int(ep.get("blue_score", 0)),
                                        "red_score": int(ep.get("red_score", 0)),
                                        "blue_won": int(ep.get("blue_won", 0)),
                                        "context_key": ctx_key,
                                    }
                                )
                    row = {
                        "checkpoint_path": ckpt_path_s,
                        "checkpoint_steps": int(entry.steps),
                        "checkpoint_kind": entry.kind,
                        "opponent": str(opp_label).upper(),
                        **ep,
                    }
                    new_rows.append(row)
                    print(
                        f"[q_probe] {entry.path.name} {opp_label} "
                        f"seed={probe_seed} z={z}: "
                        f"return={ep['episode_return']:+.3f} "
                        f"score={ep['blue_score']}-{ep['red_score']} "
                        f"len={ep['episode_length']} "
                        f"({combo_idx}/{total_combos})"
                    )
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[q_probe] WARNING: env.close() raised: {exc}")
    return new_rows, new_contexts, new_arc_rows, new_arc_contexts


# ---------------------------------------------------------------------------
# Aggregation: build summary rows + contrast values
# ---------------------------------------------------------------------------

def _shannon_nats(counts: dict[int, int]) -> float:
    """Plug-in Shannon entropy in nats from an integer-count histogram."""
    total = sum(int(c) for c in counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = float(c) / float(total)
        h -= p * math.log(p)
    return float(h)


def aggregate_summary(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the summary CSV rows with paired contrast as the primary metric.

    Sharper v4i1 metric (vs the prior unpaired version): for each
    ``(opp, seed)``, run every ``z`` from the same initial state and compute

        paired_contrast_seed = max_z R(opp, seed, z) - min_z R(opp, seed, z)

    then average across seeds. This preserves the "same starting world"
    comparison and dominates the older

        unpaired_contrast = max_z mean_seed R(z) - min_z mean_seed R(z)

    which can wash out genuine per-state strategy differentiation when
    seed-to-seed noise is large. Both are emitted (paired primary,
    unpaired diagnostic) so a low paired contrast paired with a high
    unpaired contrast tells us "z matters in aggregate but not per state",
    while the reverse tells us "z matters per state but seeds wash it out".

    Also emits:

      * Win-rate contrast (paired and unpaired)
      * best_z histogram + entropy per opponent (in nats and normalized)
      * Episode-level behavior summary: mean episode length, mean blue/red
        score per opponent x z, scoring_rate.
      * Per-checkpoint rollups: overall/worst paired and unpaired contrast,
        overall win_rate_contrast, overall best_z entropy.
    """
    # ------------------------------------------------------------------
    # Step 1: index raw rows by (ckpt, opp, z, seed) -> episode metrics
    # ------------------------------------------------------------------
    indexed: dict[tuple[str, str, int, int], dict[str, float | int]] = {}
    ckpt_meta: dict[str, tuple[int, str]] = {}

    for r in raw_rows:
        try:
            ckpt_path = str(r["checkpoint_path"])
            opp = str(r["opponent"]).upper()
            z = int(r["z"])
            seed = int(r["probe_seed"])
            ret = float(r["episode_return"])
            won = int(float(r.get("blue_won", 0)))
            bmr = int(float(r.get("blue_score_minus_red", 0)))
            blue_score = int(float(r.get("blue_score", 0)))
            red_score = int(float(r.get("red_score", 0)))
            ep_len = int(float(r.get("episode_length", 0)))
            steps_val = r.get("checkpoint_steps", "")
            steps = int(steps_val) if str(steps_val).strip() not in ("", "None") else -1
            kind = str(r.get("checkpoint_kind", "ckpt"))
        except (KeyError, ValueError, TypeError):
            continue
        indexed[(ckpt_path, opp, z, seed)] = {
            "return": ret,
            "won": won,
            "bmr": bmr,
            "blue_score": blue_score,
            "red_score": red_score,
            "episode_length": ep_len,
        }
        ckpt_meta.setdefault(ckpt_path, (steps, kind))

    if not indexed:
        return []

    # ------------------------------------------------------------------
    # Step 2: gather z + seed sets per (ckpt, opp) and per (ckpt, opp, z)
    # ------------------------------------------------------------------
    z_set_by_opp: dict[tuple[str, str], set[int]] = {}
    seeds_by_opp: dict[tuple[str, str], set[int]] = {}
    for (ckpt_path, opp, z, seed) in indexed.keys():
        z_set_by_opp.setdefault((ckpt_path, opp), set()).add(z)
        seeds_by_opp.setdefault((ckpt_path, opp), set()).add(seed)

    # Per (ckpt, opp): seeds where every observed z has a row (paired rows).
    complete_seeds: dict[tuple[str, str], list[int]] = {}
    for (ckpt_path, opp), zs in z_set_by_opp.items():
        zs_sorted = sorted(zs)
        all_seeds = sorted(seeds_by_opp[(ckpt_path, opp)])
        comp: list[int] = []
        for seed in all_seeds:
            if all((ckpt_path, opp, z, seed) in indexed for z in zs_sorted):
                comp.append(seed)
        complete_seeds[(ckpt_path, opp)] = comp

    # ------------------------------------------------------------------
    # Step 3: compute paired metrics + best_z histograms per (ckpt, opp)
    # ------------------------------------------------------------------
    paired_R_contrast: dict[tuple[str, str], list[float]] = {}
    paired_W_contrast: dict[tuple[str, str], list[float]] = {}
    best_z_hist: dict[tuple[str, str], dict[int, int]] = {}

    for (ckpt_path, opp), seeds in complete_seeds.items():
        zs_sorted = sorted(z_set_by_opp[(ckpt_path, opp)])
        if len(zs_sorted) < 2 or not seeds:
            paired_R_contrast[(ckpt_path, opp)] = []
            paired_W_contrast[(ckpt_path, opp)] = []
            best_z_hist[(ckpt_path, opp)] = {z: 0 for z in zs_sorted}
            continue
        pc_list: list[float] = []
        pw_list: list[float] = []
        hist = {z: 0 for z in zs_sorted}
        for seed in seeds:
            rets = [
                float(indexed[(ckpt_path, opp, z, seed)]["return"]) for z in zs_sorted
            ]
            wins = [int(indexed[(ckpt_path, opp, z, seed)]["won"]) for z in zs_sorted]
            pc_list.append(float(max(rets) - min(rets)))
            pw_list.append(float(max(wins) - min(wins)))
            best_local = max(range(len(rets)), key=lambda i: rets[i])
            hist[zs_sorted[best_local]] += 1
        paired_R_contrast[(ckpt_path, opp)] = pc_list
        paired_W_contrast[(ckpt_path, opp)] = pw_list
        best_z_hist[(ckpt_path, opp)] = hist

    # ------------------------------------------------------------------
    # Step 4: per (ckpt, opp) scalar metrics
    # ------------------------------------------------------------------
    metrics_per_opp: dict[tuple[str, str], dict[str, float]] = {}
    for (ckpt_path, opp), zs in z_set_by_opp.items():
        zs_sorted = sorted(zs)
        # Unpaired contrasts: max_z mean_seed - min_z mean_seed
        unpaired_R_by_z: dict[int, float] = {}
        unpaired_WR_by_z: dict[int, float] = {}
        for z in zs_sorted:
            seeds_for_z = [
                s for s in seeds_by_opp[(ckpt_path, opp)]
                if (ckpt_path, opp, z, s) in indexed
            ]
            if not seeds_for_z:
                continue
            unpaired_R_by_z[z] = statistics.fmean(
                float(indexed[(ckpt_path, opp, z, s)]["return"]) for s in seeds_for_z
            )
            unpaired_WR_by_z[z] = statistics.fmean(
                int(indexed[(ckpt_path, opp, z, s)]["won"]) for s in seeds_for_z
            )
        unpaired_R = (
            float(max(unpaired_R_by_z.values()) - min(unpaired_R_by_z.values()))
            if len(unpaired_R_by_z) >= 2
            else 0.0
        )
        unpaired_W = (
            float(max(unpaired_WR_by_z.values()) - min(unpaired_WR_by_z.values()))
            if len(unpaired_WR_by_z) >= 2
            else 0.0
        )

        pc_list = paired_R_contrast.get((ckpt_path, opp), [])
        pw_list = paired_W_contrast.get((ckpt_path, opp), [])
        paired_R = float(statistics.fmean(pc_list)) if pc_list else 0.0
        paired_R_std = float(statistics.pstdev(pc_list)) if len(pc_list) >= 2 else 0.0
        paired_W = float(statistics.fmean(pw_list)) if pw_list else 0.0

        hist = best_z_hist.get((ckpt_path, opp), {})
        K_eff = max(len(zs_sorted), 1)
        h_nats = _shannon_nats(hist)
        h_norm = float(h_nats / math.log(K_eff)) if K_eff >= 2 else 0.0

        metrics_per_opp[(ckpt_path, opp)] = {
            "contrast_within_opponent_paired": paired_R,
            "contrast_within_opponent_paired_std": paired_R_std,
            "contrast_within_opponent_unpaired": unpaired_R,
            "win_rate_contrast_paired": paired_W,
            "win_rate_contrast_unpaired": unpaired_W,
            "best_z_entropy_nats": h_nats,
            "best_z_entropy_norm": h_norm,
        }

    # ------------------------------------------------------------------
    # Step 5: per-checkpoint rollups
    # ------------------------------------------------------------------
    ckpt_rollup: dict[str, dict[str, float]] = {}
    paired_by_ckpt: dict[str, list[float]] = {}
    unpaired_by_ckpt: dict[str, list[float]] = {}
    wpc_by_ckpt: dict[str, list[float]] = {}
    hent_by_ckpt: dict[str, list[float]] = {}
    for (ckpt_path, _opp), m in metrics_per_opp.items():
        paired_by_ckpt.setdefault(ckpt_path, []).append(
            float(m["contrast_within_opponent_paired"])
        )
        unpaired_by_ckpt.setdefault(ckpt_path, []).append(
            float(m["contrast_within_opponent_unpaired"])
        )
        wpc_by_ckpt.setdefault(ckpt_path, []).append(
            float(m["win_rate_contrast_paired"])
        )
        hent_by_ckpt.setdefault(ckpt_path, []).append(
            float(m["best_z_entropy_nats"])
        )
    for ckpt_path in paired_by_ckpt:
        pv = paired_by_ckpt[ckpt_path]
        uv = unpaired_by_ckpt.get(ckpt_path, [])
        wv = wpc_by_ckpt.get(ckpt_path, [])
        hv = hent_by_ckpt.get(ckpt_path, [])
        ckpt_rollup[ckpt_path] = {
            "return_contrast_overall_paired": (
                float(statistics.fmean(pv)) if pv else 0.0
            ),
            "return_contrast_worst_paired": float(min(pv)) if pv else 0.0,
            "return_contrast_overall_unpaired": (
                float(statistics.fmean(uv)) if uv else 0.0
            ),
            "return_contrast_worst_unpaired": float(min(uv)) if uv else 0.0,
            "win_rate_contrast_overall_paired": (
                float(statistics.fmean(wv)) if wv else 0.0
            ),
            "best_z_entropy_overall": float(statistics.fmean(hv)) if hv else 0.0,
        }

    # ------------------------------------------------------------------
    # Step 6: emit one row per (ckpt, opp, z)
    # ------------------------------------------------------------------
    out: list[dict[str, Any]] = []
    for (ckpt_path, opp), zs in z_set_by_opp.items():
        zs_sorted = sorted(zs)
        opp_metrics = metrics_per_opp.get((ckpt_path, opp), {})
        ckpt_metrics = ckpt_rollup.get(ckpt_path, {})
        hist = best_z_hist.get((ckpt_path, opp), {})
        n_complete = len(complete_seeds.get((ckpt_path, opp), []))
        steps, kind = ckpt_meta.get(ckpt_path, (-1, "ckpt"))
        for z in zs_sorted:
            seeds_for_z = [
                s for s in seeds_by_opp[(ckpt_path, opp)]
                if (ckpt_path, opp, z, s) in indexed
            ]
            cells = [indexed[(ckpt_path, opp, z, s)] for s in seeds_for_z]
            rets = [float(c["return"]) for c in cells]
            wins = [int(c["won"]) for c in cells]
            bmrs = [int(c["bmr"]) for c in cells]
            blues = [int(c["blue_score"]) for c in cells]
            reds = [int(c["red_score"]) for c in cells]
            lens = [int(c["episode_length"]) for c in cells]
            mean_len = float(statistics.fmean(lens)) if lens else 0.0
            mean_blue = float(statistics.fmean(blues)) if blues else 0.0
            mean_red = float(statistics.fmean(reds)) if reds else 0.0
            scoring_rate = float(mean_blue / mean_len) if mean_len > 0 else 0.0
            best_z_count = int(hist.get(z, 0))
            best_z_frac = (
                float(best_z_count) / float(n_complete) if n_complete > 0 else 0.0
            )
            row = {
                "checkpoint_path": ckpt_path,
                "checkpoint_steps": int(steps),
                "checkpoint_kind": str(kind),
                "opponent": opp,
                "z": int(z),
                "n_seeds": len(rets),
                "n_complete_seeds": n_complete,
                "mean_return": float(statistics.fmean(rets)) if rets else 0.0,
                "std_return": float(statistics.pstdev(rets)) if len(rets) >= 2 else 0.0,
                "min_return": float(min(rets)) if rets else 0.0,
                "max_return": float(max(rets)) if rets else 0.0,
                "win_rate": float(statistics.fmean(wins)) if wins else 0.0,
                "mean_blue_minus_red": float(statistics.fmean(bmrs)) if bmrs else 0.0,
                "mean_blue_score": mean_blue,
                "mean_red_score": mean_red,
                "mean_episode_length": mean_len,
                "scoring_rate": scoring_rate,
                "best_z_count": best_z_count,
                "best_z_frac": best_z_frac,
                **opp_metrics,
                **ckpt_metrics,
            }
            out.append(row)
    # Stable sort: by checkpoint_steps then path then opp then z.
    out.sort(
        key=lambda r: (
            int(r.get("checkpoint_steps", -1) or -1),
            str(r.get("checkpoint_path", "")),
            str(r.get("opponent", "")),
            int(r.get("z", 0)),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _verdict(contrast: float, *, success_threshold: float, failure_threshold: float) -> str:
    if contrast < failure_threshold:
        return "FAIL (env strategically flat - move to Environment v2)"
    if contrast >= success_threshold:
        return "PASS (z carries strategic consequence - proceed to v4i2)"
    return "INCONCLUSIVE (between failure and success thresholds)"


def write_report(
    *,
    report_path: Path,
    summary_rows: list[dict[str, Any]],
    run_tag: str,
    opponents: list[str],
    n_seeds: int,
    success_threshold: float,
    failure_threshold: float,
) -> None:
    """Write a per-run markdown rollup focused on the latest checkpoint."""
    if not summary_rows:
        report_path.write_text(
            f"# {run_tag} - q_probe report\n\nNo data yet.\n", encoding="utf-8"
        )
        return

    # Find the latest checkpoint by steps (with -1 = final treated as max).
    by_ckpt: dict[str, list[dict[str, Any]]] = {}
    for r in summary_rows:
        by_ckpt.setdefault(str(r["checkpoint_path"]), []).append(r)

    def _sort_key(rows: list[dict[str, Any]]) -> tuple[int, int, str]:
        steps_val = rows[0].get("checkpoint_steps", -1)
        try:
            steps_int = int(steps_val)
        except (TypeError, ValueError):
            steps_int = -1
        # ``final`` > ``interrupt`` > ``ckpt`` at the same step count, so a
        # "final_*.zip" beats a "ckpt_*_<last_step>.zip" even when steps tie.
        kind = str(rows[0].get("checkpoint_kind", "ckpt"))
        kind_rank = {"final": 2, "interrupt": 1, "ckpt": 0}.get(kind, 0)
        return (steps_int, kind_rank, str(rows[0]["checkpoint_path"]))

    latest_path = max(by_ckpt.keys(), key=lambda p: _sort_key(by_ckpt[p]))
    latest_rows = by_ckpt[latest_path]

    by_opp_z: dict[str, dict[int, dict[str, Any]]] = {}
    for r in latest_rows:
        by_opp_z.setdefault(str(r["opponent"]), {})[int(r["z"])] = r

    paired_overall = float(
        latest_rows[0].get("return_contrast_overall_paired", 0.0) or 0.0
    )
    paired_worst = float(
        latest_rows[0].get("return_contrast_worst_paired", 0.0) or 0.0
    )
    unpaired_overall = float(
        latest_rows[0].get("return_contrast_overall_unpaired", 0.0) or 0.0
    )
    unpaired_worst = float(
        latest_rows[0].get("return_contrast_worst_unpaired", 0.0) or 0.0
    )
    wr_contrast_overall = float(
        latest_rows[0].get("win_rate_contrast_overall_paired", 0.0) or 0.0
    )
    best_z_entropy_overall = float(
        latest_rows[0].get("best_z_entropy_overall", 0.0) or 0.0
    )

    # Verdicts read off the PAIRED metric (the sharper v4i1 primary).
    verdict_overall = _verdict(
        paired_overall,
        success_threshold=success_threshold,
        failure_threshold=failure_threshold,
    )
    verdict_worst = _verdict(
        paired_worst,
        success_threshold=success_threshold,
        failure_threshold=failure_threshold,
    )

    all_zs = sorted({int(z) for rows in by_opp_z.values() for z in rows})
    K_eff = max(len(all_zs), 1)
    h_uniform = math.log(K_eff) if K_eff >= 2 else 0.0

    lines: list[str] = []
    lines.append(f"# {run_tag} - q_probe report")
    lines.append("")
    lines.append(f"Latest checkpoint: `{Path(latest_path).name}`")
    steps_repr = latest_rows[0].get("checkpoint_steps", -1)
    lines.append(f"Steps: {steps_repr}")
    lines.append(
        f"Probe config: opponents={opponents}  n_seeds={n_seeds}  K_eff={K_eff}"
    )
    lines.append("")
    lines.append("## Primary metric: paired return_contrast")
    lines.append("")
    lines.append(
        "For each `(opponent, probe_seed)` we force every `z` from the same "
        "initial state and compute `max_z(R) - min_z(R)`. The headline number "
        "averages this paired contrast across seeds. This preserves the "
        "\"same starting world\" comparison that motivates v4i1, and is "
        "strictly sharper than the unpaired `max_z(mean_seed R) - "
        "min_z(mean_seed R)` (also reported below as a diagnostic)."
    )
    lines.append("")
    lines.append(
        f"Thresholds: failure < {failure_threshold:.3f}, "
        f"success >= {success_threshold:.3f}."
    )
    lines.append("")
    lines.append(
        f"- **return_contrast_overall_paired = {paired_overall:+.4f}**  --  "
        f"{verdict_overall}"
    )
    lines.append(
        f"- **return_contrast_worst_paired   = {paired_worst:+.4f}**  --  "
        f"{verdict_worst}"
    )
    lines.append("")
    lines.append("Diagnostic (unpaired):")
    lines.append(
        f"- return_contrast_overall_unpaired = {unpaired_overall:+.4f}"
    )
    lines.append(
        f"- return_contrast_worst_unpaired   = {unpaired_worst:+.4f}"
    )
    lines.append("")
    lines.append("## Per-opponent paired return contrast")
    lines.append("")
    header = (
        ["opponent"]
        + [f"z={z}" for z in all_zs]
        + ["paired", "paired_std", "unpaired", "verdict"]
    )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for opp in sorted(by_opp_z):
        rows_for_opp = by_opp_z[opp]
        means = [
            f"{float(rows_for_opp[z]['mean_return']):+.3f}"
            for z in all_zs
            if z in rows_for_opp
        ]
        any_row = next(iter(rows_for_opp.values()))
        paired = float(any_row.get("contrast_within_opponent_paired", 0.0))
        paired_std = float(any_row.get("contrast_within_opponent_paired_std", 0.0))
        unpaired = float(any_row.get("contrast_within_opponent_unpaired", 0.0))
        v = _verdict(
            paired,
            success_threshold=success_threshold,
            failure_threshold=failure_threshold,
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    opp,
                    *means,
                    f"{paired:+.4f}",
                    f"{paired_std:.4f}",
                    f"{unpaired:+.4f}",
                    v,
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Per-opponent win rate by forced z")
    lines.append("")
    wr_header = (
        ["opponent"]
        + [f"z={z}" for z in all_zs]
        + ["WR_contrast_paired", "WR_contrast_unpaired"]
    )
    lines.append("| " + " | ".join(wr_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(wr_header)) + " |")
    for opp in sorted(by_opp_z):
        rows_for_opp = by_opp_z[opp]
        wrs = [
            f"{float(rows_for_opp[z]['win_rate']):.3f}"
            for z in all_zs
            if z in rows_for_opp
        ]
        any_row = next(iter(rows_for_opp.values()))
        wp = float(any_row.get("win_rate_contrast_paired", 0.0))
        wu = float(any_row.get("win_rate_contrast_unpaired", 0.0))
        lines.append("| " + " | ".join([opp, *wrs, f"{wp:.4f}", f"{wu:.4f}"]) + " |")
    lines.append("")
    lines.append(
        f"win_rate_contrast_overall_paired = {wr_contrast_overall:+.4f}"
    )
    lines.append("")
    lines.append("## best_z routing across paired seeds")
    lines.append("")
    lines.append(
        "For each `(opponent, seed)` we record `argmax_z R` ("
        "the z that won the paired contest for that seed). Entropy is the "
        "Shannon entropy in nats of the per-opponent best-z histogram. "
        f"`log K_eff = {h_uniform:.4f}` (uniform max). High entropy means "
        "different seeds favor different z's; low entropy means one z dominates."
    )
    lines.append("")
    bz_header = (
        ["opponent"]
        + [f"best_frac_z{z}" for z in all_zs]
        + ["H(best_z) nats", "H/logK"]
    )
    lines.append("| " + " | ".join(bz_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(bz_header)) + " |")
    for opp in sorted(by_opp_z):
        rows_for_opp = by_opp_z[opp]
        fracs = [
            f"{float(rows_for_opp[z]['best_z_frac']):.3f}"
            for z in all_zs
            if z in rows_for_opp
        ]
        any_row = next(iter(rows_for_opp.values()))
        h = float(any_row.get("best_z_entropy_nats", 0.0))
        hn = float(any_row.get("best_z_entropy_norm", 0.0))
        lines.append(
            "| " + " | ".join([opp, *fracs, f"{h:.4f}", f"{hn:.3f}"]) + " |"
        )
    lines.append("")
    lines.append(
        f"best_z_entropy_overall = {best_z_entropy_overall:.4f} nats "
        f"(uniform max = log K_eff = {h_uniform:.4f})"
    )
    lines.append("")
    lines.append("## Behavior summary (episode-level)")
    lines.append("")
    lines.append(
        "Per `(opponent, z)`: mean episode length, mean blue/red score, "
        "scoring_rate (blue_score / episode_length). All episode-level only; "
        "no per-step behavior telemetry is recorded here."
    )
    lines.append("")
    beh_header = (
        ["opponent", "z", "mean_len", "mean_blue", "mean_red", "scoring_rate"]
    )
    lines.append("| " + " | ".join(beh_header) + " |")
    lines.append("| " + " | ".join(["---"] * len(beh_header)) + " |")
    for opp in sorted(by_opp_z):
        rows_for_opp = by_opp_z[opp]
        for z in all_zs:
            if z not in rows_for_opp:
                continue
            row = rows_for_opp[z]
            lines.append(
                "| "
                + " | ".join(
                    [
                        opp,
                        str(z),
                        f"{float(row['mean_episode_length']):.1f}",
                        f"{float(row['mean_blue_score']):.2f}",
                        f"{float(row['mean_red_score']):.2f}",
                        f"{float(row['scoring_rate']):.4f}",
                    ]
                )
                + " |"
            )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- The probe forces every z value via "
        "`policy.fixed_latent_strategy_id = z`. q_phi's own routing decisions "
        "are bypassed for this measurement."
    )
    lines.append(
        "- Within each `(opponent, probe_seed)` the env is reseeded so all z "
        "variants start from the same initial state and same opponent params. "
        "This matched-seed contract is what makes the paired contrast valid."
    )
    lines.append(
        "- Action sampling is stochastic but seeded; differences across z "
        "trace to the policy distribution shifting with z, not to RNG noise."
    )
    lines.append(
        "- Paired contrast is the v4i1 primary metric. Unpaired contrast is a "
        "diagnostic side-channel: if `paired > unpaired` by a lot, z is doing "
        "real per-state work that gets averaged out across seeds."
    )
    lines.append(
        "- These numbers do not change the in-trainer `metrics.csv` schema "
        "and do not affect training in any way."
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def _resolve_run_tag(
    *,
    arg_run_tag: str | None,
    arg_checkpoint: Path | None,
    arg_checkpoint_dir: Path | None,
) -> str:
    if arg_run_tag:
        return str(arg_run_tag).strip()
    if arg_checkpoint is not None:
        m = _CKPT_NUMERIC_RE.match(arg_checkpoint.name)
        if m is not None:
            return str(m.group(2))
    raise ValueError(
        "Could not infer --run-tag from the checkpoint filename. "
        "Pass --run-tag explicitly."
    )


def run_probe_once(
    *,
    entries: list[CheckpointEntry],
    opponents: list[str],
    n_seeds: int,
    base_seed: int,
    device: str,
    agents: int,
    max_steps: int,
    output_dir: Path,
    run_tag: str,
    success_threshold: float,
    failure_threshold: float,
    save_contexts: bool = False,
    save_arc_contexts: bool = False,
) -> None:
    """Run the probe over a batch of checkpoint entries, write CSVs + report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / f"{run_tag}_qprobe.csv"
    summary_csv = output_dir / f"{run_tag}_qprobe_summary.csv"
    report_md = output_dir / f"{run_tag}_qprobe_report.md"
    contexts_npz = output_dir / f"{run_tag}_qprobe_contexts.npz"
    arc_csv = output_dir / f"{run_tag}_qprobe_arcs.csv"
    arc_contexts_npz = output_dir / f"{run_tag}_qprobe_arc_contexts.npz"

    existing_keys = _read_existing_raw_keys(raw_csv)
    contexts_merged: dict[str, np.ndarray] = (
        _read_contexts_npz(contexts_npz) if save_contexts else {}
    )
    arc_keys_existing: set[tuple[str, str, int, int]] = (
        _read_existing_arc_keys(arc_csv) if save_arc_contexts else set()
    )
    arc_contexts_merged: dict[str, np.ndarray] = (
        _read_arc_contexts_npz(arc_contexts_npz) if save_arc_contexts else {}
    )

    for entry in entries:
        (
            new_rows,
            new_contexts,
            new_arc_rows,
            new_arc_contexts,
        ) = probe_checkpoint(
            entry=entry,
            opponents=opponents,
            n_seeds=n_seeds,
            base_seed=base_seed,
            device=device,
            agents=agents,
            max_steps=max_steps,
            skip_existing_keys=existing_keys,
            save_contexts=save_contexts,
            existing_context_keys=set(contexts_merged.keys()) if save_contexts else None,
            save_arc_contexts=save_arc_contexts,
            existing_arc_keys=arc_keys_existing if save_arc_contexts else None,
            existing_arc_context_keys=(
                set(arc_contexts_merged.keys()) if save_arc_contexts else None
            ),
        )
        if new_rows:
            _append_raw_rows(raw_csv, new_rows)
            for r in new_rows:
                existing_keys.add(
                    (
                        str(r["checkpoint_path"]),
                        int(r["z"]),
                        str(r["opponent"]).upper(),
                        int(r["probe_seed"]),
                    )
                )
        if save_contexts and new_contexts:
            contexts_merged.update(new_contexts)
            _write_contexts_npz(contexts_npz, contexts_merged)
            print(
                f"[q_probe] contexts: wrote {contexts_npz.name} "
                f"({len(contexts_merged)} total keys)"
            )
        if save_arc_contexts and new_arc_rows:
            _append_arc_rows(arc_csv, new_arc_rows)
            for r in new_arc_rows:
                arc_keys_existing.add(
                    (
                        str(r["checkpoint_path"]),
                        str(r["opponent"]).upper(),
                        int(r["probe_seed"]),
                        int(r["forced_z"]),
                    )
                )
        if save_arc_contexts and new_arc_contexts:
            arc_contexts_merged.update(new_arc_contexts)
            _write_arc_contexts_npz(arc_contexts_npz, arc_contexts_merged)
            print(
                f"[q_probe] arc-contexts: wrote {arc_contexts_npz.name} "
                f"({len(arc_contexts_merged)} total keys), arc-CSV rows now "
                f"in {arc_csv.name}"
            )

    all_raw = _read_raw_rows(raw_csv)
    summary_rows = aggregate_summary(all_raw)
    _rewrite_summary(summary_csv, summary_rows)
    write_report(
        report_path=report_md,
        summary_rows=summary_rows,
        run_tag=run_tag,
        opponents=opponents,
        n_seeds=n_seeds,
        success_threshold=success_threshold,
        failure_threshold=failure_threshold,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v4i1 offline Q probe / return_contrast over checkpoints."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", type=Path, help="Single checkpoint .zip to probe.")
    src.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing ckpt_*.zip / interrupt_*.zip / final_*.zip for a run.",
    )
    p.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help=(
            "Run tag without the ckpt/final prefix (e.g. "
            "v4i1_strategic_pressure_qprobe_OP5_OP6_OP7_2m_4v4). Required when "
            "using --checkpoint-dir; inferred from filename when using --checkpoint."
        ),
    )
    p.add_argument(
        "--opponents",
        nargs="+",
        default=["OP5", "OP6", "OP7"],
        help="Opponent labels to probe (default: OP5 OP6 OP7).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=8,
        help="Matched probe seeds per (opponent, z) (default 8).",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Probe seeds run as base_seed, base_seed+1, ..., base_seed+n_seeds-1.",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--agents", type=int, default=4)
    p.add_argument(
        "--max-steps",
        type=int,
        default=_DEFAULT_MAX_STEPS,
        help="Per-episode environment-step cap (safety net; episodes normally terminate well before this).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the qprobe CSVs and report (default: same as checkpoint-dir).",
    )
    p.add_argument("--watch", action="store_true", help="Poll mode: keep watching for new checkpoints.")
    p.add_argument("--watch-interval", type=float, default=60.0)
    p.add_argument("--success-threshold", type=float, default=0.10)
    p.add_argument("--failure-threshold", type=float, default=0.05)
    p.add_argument(
        "--save-contexts",
        action="store_true",
        help=(
            "Also write <run_tag>_qprobe_contexts.npz containing the first-step "
            "q_phi context vector (170-d) per (checkpoint_steps, opponent, "
            "probe_seed). One context per matched starting world, shared "
            "across z variants. Required input for v4i2 router distillation "
            "(see tools/router_distill_from_qprobe.py)."
        ),
    )
    p.add_argument(
        "--save-arc-contexts",
        action="store_true",
        help=(
            "v4i2b: also write <run_tag>_qprobe_arcs.csv and "
            "<run_tag>_qprobe_arc_contexts.npz containing one row per "
            "arc boundary (every latent_resample_every_n steps, default "
            "64) for EACH forced-z rollout. Each row carries "
            "approx_remaining_return_supervision = "
            "sum(step_rewards[arc_timestep:]) (the return of the forced-z "
            "policy from that arc onward) plus the q_phi context at that "
            "moment. NOT a counterfactual: this is logged contextual-bandit "
            "data, intended as the supervision input for the arc-mode of "
            "tools/router_distill_from_qprobe.py."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve mode.
    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"[q_probe] FATAL: checkpoint not found: {ckpt_path}")
            return 2
        run_tag = _resolve_run_tag(
            arg_run_tag=args.run_tag,
            arg_checkpoint=ckpt_path,
            arg_checkpoint_dir=None,
        )
        entry = _parse_checkpoint(ckpt_path, run_tag)
        if entry is None:
            # Single-shot mode tolerates a non-matching filename: just probe it.
            entry = CheckpointEntry(path=ckpt_path, steps=-1, kind="ckpt")
        directory = ckpt_path.parent
        entries = [entry]
        do_watch = False
    else:
        directory = Path(args.checkpoint_dir)
        if args.run_tag is None:
            print("[q_probe] FATAL: --run-tag is required when using --checkpoint-dir.")
            return 2
        run_tag = str(args.run_tag).strip()
        entries = discover_checkpoints(directory, run_tag)
        do_watch = bool(args.watch)
        if not entries and not do_watch:
            print(
                f"[q_probe] No checkpoints matching run_tag={run_tag!r} "
                f"found under {directory}. Nothing to do."
            )
            return 1

    output_dir = Path(args.output_dir) if args.output_dir else directory

    print(
        f"[q_probe] run_tag={run_tag}\n"
        f"[q_probe] opponents={list(args.opponents)} n_seeds={int(args.n_seeds)} "
        f"base_seed={int(args.base_seed)} device={args.device}\n"
        f"[q_probe] thresholds: failure<{args.failure_threshold:.3f} "
        f"pass>={args.success_threshold:.3f}\n"
        f"[q_probe] output_dir={output_dir}"
    )

    if entries:
        run_probe_once(
            entries=entries,
            opponents=list(args.opponents),
            n_seeds=int(args.n_seeds),
            base_seed=int(args.base_seed),
            device=str(args.device),
            agents=int(args.agents),
            max_steps=int(args.max_steps),
            output_dir=output_dir,
            run_tag=run_tag,
            success_threshold=float(args.success_threshold),
            failure_threshold=float(args.failure_threshold),
            save_contexts=bool(args.save_contexts),
            save_arc_contexts=bool(args.save_arc_contexts),
        )

    if not do_watch:
        return 0

    print(f"[q_probe] entering --watch loop (interval={args.watch_interval:.1f}s).")
    raw_csv = output_dir / f"{run_tag}_qprobe.csv"
    while True:
        try:
            time.sleep(max(1.0, float(args.watch_interval)))
        except KeyboardInterrupt:
            print("[q_probe] watch interrupted by user; exiting.")
            return 0
        all_entries = discover_checkpoints(directory, run_tag)
        existing_keys = _read_existing_raw_keys(raw_csv)
        # A checkpoint is "pending" if it has fewer raw rows than the full
        # (opponent x latent_k x n_seeds) matrix would produce. We peek at
        # latent_k via the checkpoint metadata to size the expected count.
        expected_seeds = int(args.n_seeds)
        opp_upper = [str(o).upper() for o in args.opponents]
        pending: list[CheckpointEntry] = []
        for entry in all_entries:
            present_for_entry = sum(
                1 for (p, _z, _opp, _s) in existing_keys if p == str(entry.path)
            )
            try:
                meta = read_custom_ppo_metadata(str(entry.path))
                latent_k_entry = int(meta.get("latent_k", 4))
                if not bool(meta.get("use_latent_strategy", False)):
                    continue
            except Exception:
                # If metadata read fails, optimistically enqueue and let
                # ``probe_checkpoint`` log the real error.
                pending.append(entry)
                continue
            expected_total = len(opp_upper) * latent_k_entry * expected_seeds
            if present_for_entry < expected_total:
                pending.append(entry)
        if not pending:
            continue
        print(f"[q_probe] watch: {len(pending)} checkpoint(s) need probing.")
        run_probe_once(
            entries=pending,
            opponents=list(args.opponents),
            n_seeds=int(args.n_seeds),
            base_seed=int(args.base_seed),
            device=str(args.device),
            agents=int(args.agents),
            max_steps=int(args.max_steps),
            output_dir=output_dir,
            run_tag=run_tag,
            success_threshold=float(args.success_threshold),
            failure_threshold=float(args.failure_threshold),
            save_contexts=bool(args.save_contexts),
            save_arc_contexts=bool(args.save_arc_contexts),
        )


if __name__ == "__main__":
    raise SystemExit(main())
