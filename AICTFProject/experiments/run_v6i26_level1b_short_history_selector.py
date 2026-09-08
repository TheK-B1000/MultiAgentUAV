#!/usr/bin/env python3
"""V6I26 Level 1b: short-legal-history selector test (protocol v2).

Level 1a (run_v6i26_usable_selector_eval.py) asked: does the router's legal
input AT EPISODE START (c0 = env.state()[0] right after reset) predict which
branch (z0 or z3) will be better for the WHOLE episode? Answer, confirmed
decisively across two collection rounds (160 then +320 selection units):
FAIL. Every selector in a bounded ladder tied or lost to the trivial
baseline.

Level 1b asks a separate, non-conflated question: does the router's legal
input after observing a SHORT SHARED PREFIX of real play (T ~ 10-20 decision
steps under a frozen control branch) predict which candidate branch (z0 or
z3) is better for the REMAINDER of the episode, once committed?

Concretely, this evaluates:
    shared prefix policy for T steps -> choose z0 or z3, hold to episode end
against baselines that use the SAME intervention (prefix-then-always-z0,
prefix-then-always-z3, prefix-then-legal-selector) -- NOT against raw
full-episode fixed z0/z3 (that would be a different, unlabeled comparison).

PROTOCOL v2 (v1 had a real, disclosed-then-fixed bug: reusing one model
object for both the prefix and the z0 continuation gave z0 a "warmed"
temporal context tracker while z3's separate model started cold at the
commit point -- a confound large enough to plausibly explain any apparent
z0 preference on its own, given Level 1a already found z0/z3 near-
indistinguishable). v2 fixes this with three separate model objects and an
explicit, symmetric context-priming step:

    prefix_model  = frozen z1 control branch (lives in the z0 checkpoint)
    candidate_z0  = fixed z0, SEPARATE model instance from prefix_model
    candidate_z3  = fixed z3, separate checkpoint entirely

Per prefix step: prefix_model.predict() supplies the actual action (this is
what steps the environment); candidate_z0 and candidate_z3 both have their
temporal context tracker (inference_policy.py's TemporalStateTracker) primed
with the SAME observed global_state via a direct tracker.update() call
(_prime_context below) -- bypassing the actor forward pass entirely, so
neither candidate ever "acts" or accumulates action-selection side effects
during the prefix, only context. This is symmetric by construction: both
candidates receive the identical global_state sequence.

reset_strategy() was read in full (inference_policy.py:235-267) and confirmed
to clear every per-episode inference field that exists on the wrapper
(_prev_z, _strategy_age, all _last_strategy_*, _selector_hidden,
_opportunity_counter/_occurred/_previous_opportunity_features, and the
temporal tracker via tracker.reset()) -- it is a complete per-episode reset
for this use case; no separate reset_episode_state() was needed.

VERIFICATION (three independent, all-fatal checks per unit, not just a
final-state comparison -- convergent trajectories are indistinguishable from
identical ones by final state alone):
  1. Prefix STATE-SEQUENCE hash and ACTION-SEQUENCE hash, accumulated step by
     step during the prefix, compared between the two legs of a unit.
  2. Tracker-state symmetry: candidate_z0's and candidate_z3's temporal
     tracker state (ema_short, ema_long, initialized -- literally the
     tracker's entire state; it is a pure EMA recursion with no separate
     history buffer or step counter to compare) must be bit-identical at the
     commit point, since both were primed with the identical sequence.
  3. Final c_T bit-identity between legs (np.array_equal).
Any failure raises immediately (MismatchedC0Error) and aborts the run --
these are bugs, not noise, for a matched-seed deterministic protocol.

SHORT EPISODES are not dropped and do not get a fabricated "forced z0"
fallback re-run. If the episode ends DURING the shared prefix, neither
candidate ever got a chance to influence anything -- prefix-then-z0 and
prefix-then-z3 are, for that unit, the literal same trajectory. The row is
written as a genuine tie: outcome_z0 = outcome_z3 = the actual result,
context_json = the state at actual termination, context_capture_step = the
number of steps actually observed (< prefix_steps). Both "always commit z0"
and "always commit z3" experience this identical shared-prefix outcome, so
including it is correct, not noise -- and the short/long split rate is
reported prominently as a scope statement on what population Level 1b tests.

SCHEMA (deliberately NOT reusing c0_json/Level 1a's exact column set, to
avoid disguising c_T as c0 or inviting an accidental cross-protocol merge):
  opponent, map, episode_index, episode_seed, context_json,
  context_capture_step, short_episode, prefix_steps, prefix_branch,
  prefix_checkpoint, candidate_z0_checkpoint, candidate_z3_checkpoint,
  protocol_version, prefix_state_hash, prefix_action_hash, outcome_z0,
  outcome_z3
run_v6i26_usable_selector_eval.py's `analyze` subcommand accepts
--feature-column context_json to reuse its bounded ladder / nested CV /
paired bootstrap machinery on this schema unchanged otherwise.

OPEN CAVEAT (deferred until a positive result needs trusting): z1 lives in
the z0 checkpoint. A "neutral" prefix drawn from z0's own model may still
land in a state distribution z0 finds more familiar than z3 does. If this
script reports a z0-favoring result, do not trust it before running the
mirror variant (prefix driven by a frozen control branch living in the z3
checkpoint instead) and checking for a sign flip -- a flip means the prefix
identity was measured, not branch selectability.

Read-only with respect to training: loads checkpoints for inference only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(line_buffering=True)

from experiments.forced_z_eval.runner import _make_env  # noqa: E402
from experiments.forced_z_eval.protocol import ForcedZProtocol  # noqa: E402
from experiments.run_v6i26_c0_conditional_oracle_audit import _seed_all, _load_fixed_z_model  # noqa: E402
from experiments.run_v6i26_usable_selector_eval import _contexts_from_mixture  # noqa: E402
from experiments.v6i26_lro_core import write_json  # noqa: E402

PROTOCOL_VERSION = "level1b_v2"
PREFIX_BRANCH = 1  # frozen control branch, deliberately not z0 or z3

_RAW_FIELDNAMES = [
    "opponent", "map", "episode_index", "episode_seed",
    "context_json", "context_capture_step", "short_episode",
    "prefix_steps", "prefix_branch", "prefix_checkpoint",
    "candidate_z0_checkpoint", "candidate_z3_checkpoint", "protocol_version",
    "prefix_state_hash", "prefix_action_hash", "outcome_z0", "outcome_z3",
]


class MismatchedC0Error(RuntimeError):
    pass


def _reset_and_capture(env, reset_seed: int) -> Any:
    _seed_all(reset_seed)
    if hasattr(env, "seed"):
        env.seed(reset_seed)
    return env.reset()


def _single_obs(env, obs: dict) -> dict:
    single = {
        k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
        for k, v in obs.items()
    }
    single["global_state"] = env.state()[0]  # hard requirement: see module docstring
    return single


def _predict_step(env, model, obs: dict):
    single = _single_obs(env, obs)
    act, _ = model.predict(single, deterministic=True)
    return act


def _prime_context(model, global_state_np: np.ndarray) -> None:
    """Update a model's OWN temporal tracker with an observed global_state,
    without running the actor forward pass or touching action-selection
    bookkeeping. Mirrors exactly the tracker.update() call predict() makes
    internally (inference_policy.py:419), so a candidate that never acts
    during the prefix still enters the commit point equally 'warmed'."""
    if not hasattr(model, "_get_temporal_tracker"):
        return
    gs = torch.as_tensor(global_state_np, dtype=torch.float32, device=model.device).unsqueeze(0)
    tracker = model._get_temporal_tracker(1)
    tracker.update(gs)


def _tracker_signature(model) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The temporal tracker's ENTIRE state -- (ema_short, ema_long,
    initialized). It is a pure EMA recursion with no separate history buffer
    or step counter, so this triple fully characterizes it."""
    tracker = getattr(model, "_temporal_tracker", None)
    if tracker is None:
        return None
    return (
        tracker.ema_short.detach().cpu().numpy().copy(),
        tracker.ema_long.detach().cpu().numpy().copy(),
        tracker.initialized.detach().cpu().numpy().copy(),
    )


def _assert_tracker_symmetry(model_a, model_b, *, context: str) -> None:
    sig_a, sig_b = _tracker_signature(model_a), _tracker_signature(model_b)
    if sig_a is None or sig_b is None:
        return
    for name, a, b in zip(("ema_short", "ema_long", "initialized"), sig_a, sig_b):
        if not np.array_equal(a, b):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
            raise MismatchedC0Error(
                f"Tracker state symmetry broken at {context} ({name}): candidate_z0 and candidate_z3 "
                f"trackers diverged despite identical prefix priming. max_abs_diff={diff:.6g}"
            )


def _run_unit_leg(
    env, prefix_model, candidate_z0, candidate_z3, *, reset_seed: int, prefix_steps: int, commit_branch: int,
) -> dict[str, Any]:
    for m in (prefix_model, candidate_z0, candidate_z3):
        if hasattr(m, "reset_strategy"):
            m.reset_strategy()
    prefix_model.fixed_latent_strategy_id = PREFIX_BRANCH
    candidate_z0.fixed_latent_strategy_id = 0
    candidate_z3.fixed_latent_strategy_id = 3

    obs = _reset_and_capture(env, reset_seed)
    state_hasher = hashlib.sha256()
    action_hasher = hashlib.sha256()

    steps = 0
    while steps < prefix_steps:
        global_state = np.asarray(env.state()[0], dtype=np.float64).reshape(-1)
        state_hasher.update(global_state.tobytes())

        act = _predict_step(env, prefix_model, obs)
        action_hasher.update(np.ascontiguousarray(act).tobytes())

        _prime_context(candidate_z0, global_state)
        _prime_context(candidate_z3, global_state)

        env.step_async(act)
        obs, rew, done, infos = env.step_wait()
        steps += 1
        if done.any():
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info)
            bs = int(ep_res.get("blue_score", 0))
            rs = int(ep_res.get("red_score", 0))
            return {
                "short": True, "context_capture_step": steps,
                "context": np.asarray(env.state()[0], dtype=np.float64).reshape(-1),
                "outcome": float(bs - rs),
                "prefix_state_hash": state_hasher.hexdigest(), "prefix_action_hash": action_hasher.hexdigest(),
            }

    context = np.asarray(env.state()[0], dtype=np.float64).reshape(-1)
    _assert_tracker_symmetry(candidate_z0, candidate_z3, context=f"seed={reset_seed} commit_branch={commit_branch}")

    commit_model = candidate_z0 if commit_branch == 0 else candidate_z3
    while True:
        act = _predict_step(env, commit_model, obs)
        env.step_async(act)
        obs, rew, done, infos = env.step_wait()
        if done.any():
            info = infos[0] if len(infos) > 0 else {}
            ep_res = info.get("episode_result", info)
            bs = int(ep_res.get("blue_score", 0))
            rs = int(ep_res.get("red_score", 0))
            return {
                "short": False, "context_capture_step": prefix_steps, "context": context,
                "outcome": float(bs - rs),
                "prefix_state_hash": state_hasher.hexdigest(), "prefix_action_hash": action_hasher.hexdigest(),
            }


def cmd_collect(args: argparse.Namespace) -> int:
    target = json.loads(Path(args.locked_target_json).read_text(encoding="utf-8"))
    contexts = _contexts_from_mixture(target)
    if args.max_contexts is not None:
        contexts = contexts[: int(args.max_contexts)]
        print(f"[pilot mode] restricting to first {len(contexts)} contexts (by mixture-weight order)", flush=True)
    print(f"Contexts ({len(contexts)}): {contexts}", flush=True)
    print(f"protocol={PROTOCOL_VERSION}  prefix_steps(T)={args.prefix_steps}  "
          f"episodes/context={args.episodes_per_context}  prefix_branch=z{PREFIX_BRANCH} "
          f"-> {len(contexts) * args.episodes_per_context} matched units planned", flush=True)

    out_path = Path(args.output)
    manifest_path = out_path.with_name(out_path.name + ".manifest.json")
    existing_keys: set[tuple[str, str, int]] = set()
    if out_path.is_file() and not args.overwrite:
        with out_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("protocol_version") != PROTOCOL_VERSION:
                    raise MismatchedC0Error(
                        f"{out_path} contains rows from protocol_version={row.get('protocol_version')!r}, "
                        f"not {PROTOCOL_VERSION!r} -- refusing to resume/append into a different protocol's file."
                    )
                existing_keys.add((row["opponent"], row["map"], int(row["episode_index"])))
        print(f"Resuming: {len(existing_keys)} units already in {out_path}", flush=True)

    mode = "a" if (out_path.is_file() and not args.overwrite) else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = out_path.open(mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=_RAW_FIELDNAMES)
    if mode == "w":
        writer.writeheader()

    write_json(manifest_path, {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_z0": str(args.checkpoint_z0), "checkpoint_z3": str(args.checkpoint_z3),
        "prefix_steps": int(args.prefix_steps), "prefix_branch": PREFIX_BRANCH,
        "locked_target_json": str(args.locked_target_json), "base_seed": int(args.base_seed),
        "note": "Fixes v1's same-object tracker-warmth confound: prefix_model, candidate_z0, candidate_z3 "
                "are three separate model instances; candidates are context-primed symmetrically during "
                "the prefix. Short episodes are included as ties (context_capture_step < prefix_steps), "
                "not dropped or given a fabricated forced-z0 fallback.",
    })

    t_start = time.time()
    n_written = 0
    n_short = 0
    for ctx_idx, (opponent, map_name) in enumerate(contexts):
        needed = [ep for ep in range(int(args.episodes_per_context)) if (opponent, map_name, ep) not in existing_keys]
        if not needed:
            print(f"  [{opponent}|{map_name}] all {args.episodes_per_context} units already collected, skipping", flush=True)
            continue

        cell_seed = int(args.base_seed) + 1000 * ctx_idx
        protocol = ForcedZProtocol(
            checkpoint=args.checkpoint_z0, opponents=(opponent,), maps=(map_name,),
            episodes_per_cell=1, base_seed=cell_seed, max_decision_steps=int(args.max_decision_steps),
            device=args.device,
        )
        env = _make_env(protocol, map_name, cell_seed)
        try:
            env.env_method("set_phase", opponent)
            env.env_method("set_next_opponent", "SCRIPTED", opponent)
            from rl.stress_schedule import STRESS_BY_PHASE  # hard requirement: see module docstring
            env.env_method("set_stress_schedule", STRESS_BY_PHASE)

            prefix_model = _load_fixed_z_model(args.checkpoint_z0, PREFIX_BRANCH, env, args.device)
            candidate_z0 = _load_fixed_z_model(args.checkpoint_z0, 0, env, args.device)
            candidate_z3 = _load_fixed_z_model(args.checkpoint_z3, 3, env, args.device)

            for ep_idx in needed:
                reset_seed = cell_seed + ep_idx
                leg_a = _run_unit_leg(
                    env, prefix_model, candidate_z0, candidate_z3,
                    reset_seed=reset_seed, prefix_steps=int(args.prefix_steps), commit_branch=0,
                )
                leg_b = _run_unit_leg(
                    env, prefix_model, candidate_z0, candidate_z3,
                    reset_seed=reset_seed, prefix_steps=int(args.prefix_steps), commit_branch=3,
                )

                if leg_a["short"] != leg_b["short"]:
                    raise MismatchedC0Error(
                        f"Non-deterministic prefix termination at {opponent}|{map_name} ep={ep_idx} "
                        f"seed={reset_seed}: leg_a short={leg_a['short']}, leg_b short={leg_b['short']}."
                    )
                if leg_a["prefix_state_hash"] != leg_b["prefix_state_hash"]:
                    raise MismatchedC0Error(
                        f"Prefix STATE-sequence hash mismatch at {opponent}|{map_name} ep={ep_idx} "
                        f"seed={reset_seed}: legs took different trajectories despite identical seed/policy."
                    )
                if leg_a["prefix_action_hash"] != leg_b["prefix_action_hash"]:
                    raise MismatchedC0Error(
                        f"Prefix ACTION-sequence hash mismatch at {opponent}|{map_name} ep={ep_idx} "
                        f"seed={reset_seed}: prefix_model chose different actions across legs."
                    )
                if not np.array_equal(leg_a["context"], leg_b["context"]):
                    raise MismatchedC0Error(
                        f"context (c_T or short-termination state) NOT bit-identical across legs at "
                        f"{opponent}|{map_name} ep={ep_idx} seed={reset_seed}: "
                        f"max_abs_diff={np.abs(leg_a['context'] - leg_b['context']).max():.6g}"
                    )
                if leg_a["short"] and leg_a["outcome"] != leg_b["outcome"]:
                    raise MismatchedC0Error(
                        f"Short-episode outcome mismatch at {opponent}|{map_name} ep={ep_idx} seed={reset_seed}: "
                        f"identical shared-prefix trajectories must yield identical outcomes."
                    )

                writer.writerow({
                    "opponent": opponent, "map": map_name, "episode_index": ep_idx, "episode_seed": reset_seed,
                    "context_json": json.dumps(leg_a["context"].tolist()),
                    "context_capture_step": leg_a["context_capture_step"], "short_episode": leg_a["short"],
                    "prefix_steps": int(args.prefix_steps), "prefix_branch": PREFIX_BRANCH,
                    "prefix_checkpoint": str(args.checkpoint_z0), "candidate_z0_checkpoint": str(args.checkpoint_z0),
                    "candidate_z3_checkpoint": str(args.checkpoint_z3), "protocol_version": PROTOCOL_VERSION,
                    "prefix_state_hash": leg_a["prefix_state_hash"], "prefix_action_hash": leg_a["prefix_action_hash"],
                    "outcome_z0": leg_a["outcome"], "outcome_z3": leg_b["outcome"],
                })
                n_written += 1
                if leg_a["short"]:
                    n_short += 1
            f.flush()
        finally:
            env.close()

        elapsed = time.time() - t_start
        done_contexts = ctx_idx + 1
        eta = (elapsed / done_contexts) * (len(contexts) - done_contexts) if done_contexts > 0 else 0.0
        short_rate = n_short / max(1, n_written)
        print(f"  [{opponent}|{map_name}] context done ({done_contexts}/{len(contexts)} contexts, "
              f"{n_written} units written this run, {n_short} short-episode ties (rate={short_rate:.1%}), "
              f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s)", flush=True)

    f.close()
    print(f"\nWrote {n_written} new rows -> {out_path}")
    if n_written > 0:
        print(f"Short-episode (prefix never completed) rate: {n_short / n_written:.1%} of units -- "
              f"these are included as ties (context_capture_step < prefix_steps), not dropped.")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-z0", required=True, help="Also supplies the frozen z1 prefix branch")
    p.add_argument("--checkpoint-z3", required=True)
    p.add_argument("--locked-target-json", required=True)
    p.add_argument("--prefix-steps", type=int, default=15, help="T: decision steps under z1 before committing")
    p.add_argument("--episodes-per-context", type=int, default=32)
    p.add_argument("--max-contexts", type=int, default=None,
                    help="Pilot mode: restrict to the first N contexts (mixture-weight order)")
    p.add_argument("--base-seed", type=int, default=70001)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--output", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_collect)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
