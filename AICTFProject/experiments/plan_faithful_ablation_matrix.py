#!/usr/bin/env python3
"""Generate Summer-plan-faithful latent ablation commands.

The matrix is deliberately narrow:

- latent + persistence + entropy
- latent without persistence
- latent without entropy
- collapsed latent, K=1
- no-latent baseline

It does not add supervised router labels, opponent-ID heads, Gumbel-Softmax,
VAE losses, or handcrafted strategy labels.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Ablation:
    name: str
    preset: str
    description: str


ABLATIONS: tuple[Ablation, ...] = (
    Ablation(
        "latent_persist_entropy",
        "plan_faithful_latent_persist_entropy",
        "K=4, interval=20, lambda_p=0.025, lambda_H=0.003.",
    ),
    Ablation(
        "latent_no_persistence",
        "plan_faithful_latent_no_persistence",
        "Same latent run with lambda_p=0.",
    ),
    Ablation(
        "latent_no_entropy",
        "plan_faithful_latent_no_entropy",
        "Same latent run with entropy term disabled.",
    ),
    Ablation(
        "latent_k1",
        "plan_faithful_latent_k1",
        "Collapsed latent control with K=1.",
    ),
    Ablation(
        "latent_option_a",
        "plan_faithful_latent_option_a",
        "Plan Option A (Fix D): episode-start z, lambda_p=0, lambda_H=0.001, no aux heads.",
    ),
    Ablation(
        "no_latent",
        "plan_faithful_no_latent",
        "Decentralized actor baseline with latent path disabled.",
    ),
)


def _quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _team_tag(agents: int) -> str:
    n = max(1, int(agents))
    return f"{n}v{n}"


def _rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    team = _team_tag(int(args.agents))
    checkpoint_dir = os.path.join(args.checkpoint_root, team)
    for seed in args.seeds:
        for ablation in ABLATIONS:
            run_tag = f"plan_faithful_{ablation.name}_seed{int(seed)}_{team}"
            train_parts = [
                args.python,
                "rl/train_ppo.py",
                "--preset",
                ablation.preset,
                "--agents",
                str(int(args.agents)),
                "--seed",
                str(int(seed)),
                "--total-steps",
                str(int(args.steps)),
                "--checkpoint-dir",
                checkpoint_dir,
                "--run-tag",
                run_tag,
                "--device",
                args.device,
                "--fresh-metrics-csv",
            ]
            if args.e3_step_telemetry:
                train_parts.append("--e3-step-telemetry")
            rows.append(
                {
                    "team": team,
                    "seed": str(int(seed)),
                    "ablation": ablation.name,
                    "preset": ablation.preset,
                    "description": ablation.description,
                    "run_tag": run_tag,
                    "train_command": _quote(train_parts),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plan-faithful ablation training commands.")
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-root", type=str, default=os.path.join("checkpoints", "plan_faithful"))
    parser.add_argument("--python", type=str, default=sys.executable or "python")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path for the command matrix.")
    parser.add_argument("--e3-step-telemetry", action="store_true", help="Write per-step q_phi/phase diagnostics CSVs.")
    parser.add_argument("--execute", action="store_true", help="Run each training command sequentially.")
    args = parser.parse_args()

    rows = _rows(args)
    for row in rows:
        print(f"\n[{row['team']} seed={row['seed']} {row['ablation']}] {row['description']}")
        print("TRAIN:", row["train_command"])

    if args.out:
        path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[plan_faithful_ablation_matrix] wrote {path}")

    if args.execute:
        for row in rows:
            command = row["train_command"]
            print(f"\n[plan_faithful_ablation_matrix] running: {command}", flush=True)
            subprocess.run(shlex.split(command), check=True)


if __name__ == "__main__":
    main()
