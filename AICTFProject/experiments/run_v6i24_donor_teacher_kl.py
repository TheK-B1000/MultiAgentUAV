#!/usr/bin/env python3
"""Donor→teacher KL on emphasized cells (Path C / LRO basin diagnostic).

Tiny KL + flat niches ⇒ still in the generalist basin.
Large KL + flat niches ⇒ policies moved without strategic tradeoffs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i24_population_config import DEFAULT_MAPS, DEFAULT_OPPONENTS  # noqa: E402
from experiments.v6i26_lro_core import write_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Donor-to-teacher action KL diagnostic")
    p.add_argument(
        "--donor",
        default="artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip",
    )
    p.add_argument(
        "--teachers-dir",
        default="artifacts/v6i24_population_seed1/probe_05u",
    )
    p.add_argument("--output", default="artifacts/v6i24_population_seed1/donor_teacher_kl.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps-per-cell", type=int, default=64)
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--max-decision-steps", type=int, default=240)
    return p.parse_args()


def _kl_cat(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Mean KL(p||q) over categorical action heads in a MultiDiscrete stack."""
    # p_logits/q_logits: (N, action_dim) flattened MultiDiscrete logits
    # Use shared model._categoricals if available; else treat as single softmax.
    p = torch.softmax(p_logits.float(), dim=-1)
    q = torch.softmax(q_logits.float(), dim=-1)
    return (p * (torch.log(p.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(dim=-1)


def main() -> int:
    args = _parse_args()
    donor_path = Path(args.donor)
    teachers_dir = Path(args.teachers_dir)
    if not donor_path.is_file():
        print(f"ERROR: donor missing: {donor_path}")
        return 2
    from experiments.run_v6i24_population_eval_gates import (
        _collect_shared_history,
        _load_policies,
        _make_env,
        _obs_batch,
        find_member_checkpoints,
    )

    members = find_member_checkpoints(teachers_dir)
    if not members:
        # Also accept any member_*.zip flat listing
        zips = sorted(teachers_dir.glob("member_*.zip"))
        members = [(i, z.stem, z) for i, z in enumerate(zips)]
    if not members:
        print(f"ERROR: no teacher zips in {teachers_dir}")
        return 2

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    env0 = _make_env(donor_path, args.maps[0], int(args.seed), device, int(args.max_decision_steps))
    try:
        donor_pol = _load_policies(
            [(0, "donor", donor_path)],
            env0.observation_space,
            env0.action_space,
            device,
        )[0]["policy"]
        teachers = _load_policies(members, env0.observation_space, env0.action_space, device)
    finally:
        env0.close()

    rows = []
    for ci, (opp, mp) in enumerate((o, m) for o in args.opponents for m in args.maps):
        env = _make_env(donor_path, mp, int(args.seed) + ci, device, int(args.max_decision_steps))
        try:
            snaps = _collect_shared_history(
                donor_pol,
                env,
                opponent=opp,
                n_steps=int(args.steps_per_cell),
                seed=int(args.seed) + 1000 + ci,
            )
            obs_t = _obs_batch(snaps, torch.device(device))
            with torch.no_grad():
                z0 = torch.zeros((obs_t["grid"].shape[0],), dtype=torch.long, device=obs_t["grid"].device)
                donor_logits = donor_pol.model.policy_logits(obs_t, z_idx=z0)
                donor_logits = donor_pol.model._mask_logits(donor_logits, obs_t.get("mask"))
                for entry in teachers:
                    t_logits = entry["policy"].model.policy_logits(obs_t, z_idx=z0)
                    t_logits = entry["policy"].model._mask_logits(t_logits, obs_t.get("mask"))
                    kl = _kl_cat(donor_logits, t_logits).mean().item()
                    rows.append(
                        {
                            "context": f"{opp}|{mp}",
                            "teacher": entry["label"],
                            "mean_kl": float(kl),
                        }
                    )
                    print(f"  KL donor->{entry['label']} @ {opp}|{mp}: {kl:.6f}", flush=True)
        finally:
            env.close()

    by_teacher: dict[str, list[float]] = {}
    for r in rows:
        by_teacher.setdefault(r["teacher"], []).append(float(r["mean_kl"]))
    summary = {
        "donor": str(donor_path),
        "teachers_dir": str(teachers_dir),
        "per_cell": rows,
        "per_teacher_mean_kl": {k: float(np.mean(v)) for k, v in by_teacher.items()},
        "global_mean_kl": float(np.mean([r["mean_kl"] for r in rows])) if rows else float("nan"),
        "interpretation": (
            "tiny_kl_still_in_basin"
            if rows and float(np.mean([r["mean_kl"] for r in rows])) < 1e-3
            else "moved_check_niches"
        ),
    }
    write_json(Path(args.output), summary)
    print(f"global_mean_kl={summary['global_mean_kl']:.6f} → {summary['interpretation']}")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
