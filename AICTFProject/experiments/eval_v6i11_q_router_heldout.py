#!/usr/bin/env python3
"""Held-out prospective evaluator for the V6I11 contextual Q-router.

This is the DECISIVE behavioural gate for V6I11. A replay-fit verdict of
``SEPARATING`` is only a *candidate*: it says the online (context, z, return)
data is separable, not that argmax-Q routing actually earns more return on
fresh episodes. This script runs the prospective test.

Decisive gate
-------------
    Q-router argmax  >  valid cross-episode shuffled Q-router

Then (secondary):
    Q-router  >  uniform episode-persistent z
    Q-router  approaches or beats  fixed z2
Oracle (per-episode max over z) is the ceiling, not a promotion bar.

Matched-seed design
-------------------
For each ``(opponent, map, seed)`` held-out episode we:
  1. Read the LEGAL episode-start context (t=0 team geometry + opponent one-hot).
  2. Predict ``Q(context, z)`` for every latent z; ``q_z = argmax_z Q``.
  3. Run ALL FOUR forced-z rollouts ONCE on fresh, matched-seed envs.
Every condition is then derived from the SAME four paired returns, so the
comparisons are perfectly paired (no seed noise between conditions):
    q_router      = return[q_z]
    fixed_z2      = return[fixed_z]
    uniform       = return[uniform_z]        (seeded per episode)
    oracle        = max_z return[z]
    shuffled_q    = return[ perm(q_z) ]       (cross-episode, histogram-preserving)

Cross-episode shuffled control
------------------------------
Within each ``(opponent, map)`` cell we permute the chosen-z assignments across
episodes (preserving the chosen-z histogram) and re-score each episode with the
*reassigned* z. The permutation is required to ACTUALLY reassign episodes; if
every Q-router choice in a cell is identical there is nothing to permute and we
report ``cross_episode_gate_untestable = true`` for that cell rather than a
spurious zero-delta tie.

Fresh seeds
-----------
``--base-seed`` defaults to 30000, deliberately DISJOINT from Probe A (42),
the V6I9 diagnostic (4242), V6I10 experiments, and V6I11 replay training
(training seed 1). Do not reuse those seeds here.

Summer-faithfulness
-------------------
The frozen repertoire (shared actor + z-specific adapters) is never updated;
the Q-router is only READ. No hindsight/oracle labels enter the selection: the
episode-start context is legal information available before the episode runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.global_state import GLOBAL_STATE_DIM, build_global_state_batch  # noqa: E402
from rl.router.q_value_router import ContextualQRouter  # noqa: E402
from rl.custom_ppo.csv_writers import _OPPONENT_TAG_TO_ID  # noqa: E402

DEFAULT_ANCHOR = (
    "checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
)
DEFAULT_OPPONENTS = ("OP8", "OP9", "OP10")
DEFAULT_MAPS = ("map_b", "map_b_split_lane_v2")
# Fresh held-out base seed, disjoint from Probe A (42), v6i9 diag (4242),
# and v6i11 replay training (1).
DEFAULT_BASE_SEED = 30000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I11 held-out prospective Q-router evaluator")
    p.add_argument("--q-router", required=True, help="Path to q_router_final.pt")
    p.add_argument("--anchor", default=DEFAULT_ANCHOR, help="Frozen repertoire checkpoint")
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--episodes-per-cell", type=int, default=20)
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--device", default="cuda")
    p.add_argument("--latent-k", type=int, default=4)
    p.add_argument("--fixed-z", type=int, default=2)
    p.add_argument("--q-hidden", type=int, default=128)
    p.add_argument("--max-decision-steps", type=int, default=400)
    p.add_argument("--out-dir", default="artifacts/v6i11_q_router_heldout")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _cell_seed(base: int, opp_idx: int, map_idx: int) -> int:
    return int(base) + 1000 * int(opp_idx) + 100 * int(map_idx)


def _make_env(anchor: str, map_name: str, seed: int, device: str, max_decision_steps: int) -> Any:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(anchor)
    agents = int(meta.get("n_blue", 2))
    return GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=agents,
            max_red_agents=agents,
            map_layout=map_name,
            max_decision_steps=int(max_decision_steps),
            aquaticus_profile=True,
            rules_profile="OURS",
            device=device,
            seed=int(seed),
        )
    )


def _episode_start_context(
    q_router: ContextualQRouter,
    env: Any,
    raw_opp_id: int,
    device: str,
) -> torch.Tensor:
    """Legal t=0 context: team geometry (first ``state_dim`` dims) + opponent one-hot."""
    try:
        env.reset()
    except Exception:
        pass
    geom = build_global_state_batch(env.core).to(device).float()  # [1, 34]
    return q_router.build_context(geom, [int(raw_opp_id)], device=device)  # [1, ctx_dim]


def _episode_return(ep_dict: dict[str, Any]) -> float:
    for k in ("return", "episode_return", "blue_return"):
        if k in ep_dict and ep_dict[k] is not None:
            return float(ep_dict[k])
    return float("nan")


def _histogram_preserving_permutation(values: list[int], rng: np.random.Generator) -> tuple[list[int], bool]:
    """Return a permutation of ``values`` that actually reassigns at least one
    position when possible. Second element is ``reassigned`` (False iff every
    value is identical, i.e. nothing could be permuted)."""
    n = len(values)
    arr = np.asarray(values)
    if n <= 1 or np.all(arr == arr[0]):
        return list(values), False
    for _ in range(64):
        perm = rng.permutation(n)
        shuffled = [int(values[i]) for i in perm]
        if any(shuffled[i] != values[i] for i in range(n)):
            return shuffled, True
    # Fallback: rotate by one (guaranteed to move at least one distinct value).
    shuffled = values[1:] + values[:1]
    return shuffled, any(shuffled[i] != values[i] for i in range(n))


def _paired_bootstrap_ci(
    diff: np.ndarray, *, n_boot: int, rng: np.random.Generator, ci: float = 0.95
) -> dict[str, float]:
    diff = diff[~np.isnan(diff)]
    n = int(diff.size)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "ci_excludes_zero": False, "n": 0}
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boot[b] = float(np.mean(diff[rng.integers(0, n, n)]))
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1.0 - alpha))
    return {
        "mean": float(np.mean(diff)),
        "ci_low": lo,
        "ci_high": hi,
        "ci_excludes_zero": bool(lo > 0.0 or hi < 0.0),
        "n": n,
    }


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.glob("summary.json")) and not args.force:
        print(f"[heldout] {out_dir} already has results. Pass --force to overwrite.")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    q_path = Path(args.q_router)
    if not q_path.is_file():
        raise FileNotFoundError(f"Q-router checkpoint not found: {q_path}")
    anchor = Path(args.anchor)
    if not anchor.is_file():
        raise FileNotFoundError(f"Anchor checkpoint not found: {anchor}")

    opponents = [str(o).upper() for o in args.opponents]
    maps = list(args.maps)
    latent_k = int(args.latent_k)
    device = args.device

    # Opponent id -> row index using the canonical scheme (OP8->7, OP9->8, OP10->9).
    opp_id_to_idx = {int(_OPPONENT_TAG_TO_ID[o]): i for i, o in enumerate(opponents)}
    raw_id_for = {o: int(_OPPONENT_TAG_TO_ID[o]) for o in opponents}

    # Reconstruct + load the trained Q-router.
    q_router = ContextualQRouter(
        state_dim=GLOBAL_STATE_DIM,
        n_opponents=len(opponents),
        opponent_id_to_idx=opp_id_to_idx,
        latent_k=latent_k,
        hidden=int(args.q_hidden),
    ).to(device)
    q_router.load_state_dict(torch.load(q_path, map_location=device))
    q_router.eval()

    print("=" * 72)
    print(f"[heldout] q_router     = {q_path}")
    print(f"[heldout] anchor       = {anchor}")
    print(f"[heldout] opponents    = {opponents}")
    print(f"[heldout] maps         = {maps}")
    print(f"[heldout] episodes/cell= {args.episodes_per_cell}")
    print(f"[heldout] base_seed    = {args.base_seed} (fresh; disjoint from 42/4242/1)")
    print(f"[heldout] latent_k     = {latent_k}  fixed_z={args.fixed_z}")
    print("=" * 72)

    from plot.eval_rollout import run_eval_episodes
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.evaluation.router_ablation import model_parameter_sha256

    # Preload the frozen repertoire once; reuse across all forced-z rollouts.
    probe0 = _make_env(str(anchor), maps[0], args.base_seed, device, args.max_decision_steps)
    obs_space = probe0.observation_space
    act_space = probe0.action_space
    probe0.close()
    model = load_custom_ppo_policy(str(anchor), obs_space, act_space, device=device)
    frozen_hash_before = model_parameter_sha256(model)

    csv_rows: list[dict[str, Any]] = []
    # Per (opponent, map) cell accumulators of paired per-episode returns.
    cell_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for opp_idx, opp in enumerate(opponents):
        raw_opp = raw_id_for[opp]
        for map_idx, map_name in enumerate(maps):
            cseed = _cell_seed(args.base_seed, opp_idx, map_idx)
            uni_rng = np.random.default_rng(cseed + 777)
            for ep in range(int(args.episodes_per_cell)):
                seed = cseed + ep
                # 1. Legal episode-start context (opponent-independent geometry
                #    + opponent one-hot) on a throwaway probe env.
                probe = _make_env(str(anchor), map_name, seed, device, args.max_decision_steps)
                ctx = _episode_start_context(q_router, probe, raw_opp, device)
                probe.close()
                with torch.no_grad():
                    q_vals = q_router(ctx).squeeze(0).cpu().numpy()  # [K]
                q_z = int(np.argmax(q_vals))

                # 2. All four forced-z rollouts on fresh matched-seed envs.
                returns: list[float] = []
                for z in range(latent_k):
                    env_z = _make_env(str(anchor), map_name, seed, device, args.max_decision_steps)
                    try:
                        eps = run_eval_episodes(
                            str(anchor), env_z, 1, device, opp,
                            deterministic=True,
                            fixed_latent_id=z,
                            latent_resample_every_n=None,
                            latent_eval_mode="normal",
                            preloaded_model=model,
                        )
                    finally:
                        env_z.close()
                    returns.append(_episode_return(eps[0]) if eps else float("nan"))

                uniform_z = int(uni_rng.integers(0, latent_k))
                rec = {
                    "opponent": opp,
                    "map": map_name,
                    "seed": seed,
                    "episode": ep,
                    "q_z": q_z,
                    "uniform_z": uniform_z,
                    "returns": returns,
                    "q_values": [float(v) for v in q_vals],
                }
                cell_records[(opp, map_name)].append(rec)
                row = {
                    "opponent": opp, "map": map_name, "seed": seed, "episode": ep,
                    "q_z": q_z, "uniform_z": uniform_z, "fixed_z": int(args.fixed_z),
                    "q_router_return": returns[q_z],
                    "fixed_z2_return": returns[int(args.fixed_z)],
                    "uniform_return": returns[uniform_z],
                    "oracle_return": float(np.nanmax(returns)) if returns else float("nan"),
                    **{f"return_z{z}": returns[z] for z in range(latent_k)},
                    **{f"q_z{z}": float(q_vals[z]) for z in range(latent_k)},
                }
                csv_rows.append(row)
            print(f"[heldout] {opp} x {map_name}: {args.episodes_per_cell} episodes done "
                  f"(q_z choices={[r['q_z'] for r in cell_records[(opp, map_name)]]})")

    frozen_hash_after = model_parameter_sha256(model)
    frozen_ok = frozen_hash_before == frozen_hash_after

    # ---- Cross-episode shuffled control (histogram-preserving, per cell). ---- #
    shuffle_rng = np.random.default_rng(args.base_seed + 987654321)
    n_cells_reassigned = 0
    n_cells_total = 0
    for (opp, map_name), recs in cell_records.items():
        n_cells_total += 1
        q_choices = [r["q_z"] for r in recs]
        shuffled, reassigned = _histogram_preserving_permutation(q_choices, shuffle_rng)
        if reassigned:
            n_cells_reassigned += 1
        for r, sz in zip(recs, shuffled):
            r["shuffled_q_z"] = int(sz)
            r["cell_reassigned"] = bool(reassigned)

    cross_episode_gate_untestable = bool(n_cells_reassigned == 0)

    # ---- Aggregate paired returns. ---- #
    def _col(name: str) -> np.ndarray:
        vals: list[float] = []
        for recs in cell_records.values():
            for r in recs:
                if name == "q_router":
                    vals.append(r["returns"][r["q_z"]])
                elif name == "fixed_z2":
                    vals.append(r["returns"][int(args.fixed_z)])
                elif name == "uniform":
                    vals.append(r["returns"][r["uniform_z"]])
                elif name == "oracle":
                    vals.append(float(np.nanmax(r["returns"])))
                elif name == "shuffled_q":
                    vals.append(r["returns"][r.get("shuffled_q_z", r["q_z"])])
        return np.asarray(vals, dtype=np.float64)

    cond_means = {
        name: {
            "mean": float(np.nanmean(_col(name))),
            "sem": float(np.nanstd(_col(name), ddof=1) / np.sqrt(max(1, np.sum(~np.isnan(_col(name))))))
            if np.sum(~np.isnan(_col(name))) > 1 else float("nan"),
            "n": int(np.sum(~np.isnan(_col(name)))),
        }
        for name in ("q_router", "shuffled_q", "uniform", "fixed_z2", "oracle")
    }

    boot_rng = np.random.default_rng(args.base_seed + 13)
    q = _col("q_router")
    vs_shuffled = _paired_bootstrap_ci(q - _col("shuffled_q"), n_boot=args.n_boot, rng=boot_rng)
    vs_uniform = _paired_bootstrap_ci(q - _col("uniform"), n_boot=args.n_boot, rng=boot_rng)
    vs_fixed = _paired_bootstrap_ci(q - _col("fixed_z2"), n_boot=args.n_boot, rng=boot_rng)
    vs_oracle = _paired_bootstrap_ci(q - _col("oracle"), n_boot=args.n_boot, rng=boot_rng)

    # ---- Promotion gate. ---- #
    decisive_pass = bool((not cross_episode_gate_untestable) and vs_shuffled["ci_excludes_zero"] and vs_shuffled["mean"] > 0)
    beats_uniform = bool(vs_uniform["ci_excludes_zero"] and vs_uniform["mean"] > 0)
    # "approaches or beats fixed z2": Q-router not reliably WORSE than fixed_z2.
    approaches_fixed = bool(not (vs_fixed["ci_excludes_zero"] and vs_fixed["mean"] < 0))

    if cross_episode_gate_untestable:
        gate = "UNTESTABLE"
    elif decisive_pass and beats_uniform and approaches_fixed:
        gate = "PROMOTE"
    elif decisive_pass:
        gate = "PARTIAL"
    else:
        gate = "FAIL"
    if not frozen_ok:
        gate = "INVALID"

    # ---- Write artifacts. ---- #
    csv_path = out_dir / "episode_results.csv"
    if csv_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "q_router": str(q_path),
        "anchor": str(anchor),
        "opponents": opponents,
        "maps": maps,
        "episodes_per_cell": int(args.episodes_per_cell),
        "base_seed": int(args.base_seed),
        "seed_scheme": "cell_seed = base + 1000*opp_idx + 100*map_idx; episode_seed = cell_seed + ep",
        "fresh_seeds_note": "base 30000 is disjoint from Probe A (42), v6i9 diag (4242), v6i11 training (1)",
        "fixed_z": int(args.fixed_z),
        "latent_k": latent_k,
        "frozen_actor_ok": bool(frozen_ok),
        "condition_means": cond_means,
        "paired_vs_shuffled_q": vs_shuffled,
        "paired_vs_uniform": vs_uniform,
        "paired_vs_fixed_z2": vs_fixed,
        "paired_vs_oracle": vs_oracle,
        "cross_episode_gate_untestable": cross_episode_gate_untestable,
        "cells_reassigned": int(n_cells_reassigned),
        "cells_total": int(n_cells_total),
        "gate_criteria": {
            "decisive": "Q-router > valid cross-episode shuffled Q (paired CI excludes 0, mean > 0)",
            "then": ["Q-router > uniform (paired CI excludes 0)",
                     "Q-router approaches/beats fixed_z2 (not reliably worse)"],
            "note": "Oracle is the ceiling, not a promotion bar.",
        },
        "gate": gate,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Console report. ---- #
    print()
    print("=" * 72)
    print(f"[heldout] Frozen-actor check: {'PASS' if frozen_ok else 'FAIL'}")
    for name in ("q_router", "shuffled_q", "uniform", "fixed_z2", "oracle"):
        m = cond_means[name]
        print(f"  {name:12s}: mean={m['mean']:+.4f}  sem={m['sem']:+.4f}  n={m['n']}")
    print()
    print(f"  Q vs shuffled_q: dmean={vs_shuffled['mean']:+.4f} "
          f"CI=[{vs_shuffled['ci_low']:+.4f},{vs_shuffled['ci_high']:+.4f}] "
          f"excl0={vs_shuffled['ci_excludes_zero']}")
    print(f"  Q vs uniform   : dmean={vs_uniform['mean']:+.4f} "
          f"CI=[{vs_uniform['ci_low']:+.4f},{vs_uniform['ci_high']:+.4f}] "
          f"excl0={vs_uniform['ci_excludes_zero']}")
    print(f"  Q vs fixed_z2  : dmean={vs_fixed['mean']:+.4f} "
          f"CI=[{vs_fixed['ci_low']:+.4f},{vs_fixed['ci_high']:+.4f}] "
          f"excl0={vs_fixed['ci_excludes_zero']}")
    print(f"  cross_episode_gate_untestable={cross_episode_gate_untestable} "
          f"(cells reassigned {n_cells_reassigned}/{n_cells_total})")
    print(f"[heldout] GATE: {gate}")
    print("=" * 72)
    print(f"[heldout] summary -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
