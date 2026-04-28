#!/usr/bin/env python3
"""Generate final-phase training and evaluation commands.

The matrix covers the professor-requested core comparisons:

- Summer latent default trained against fixed OP3
- curriculum baseline with no latent strategy and the OP1 -> OP2 -> OP3 schedule
- no-latent baseline from the Summer default fixed-OP3 setting
- train-map vs held-out eval-map generalization

The script prints commands by default and can also write them to CSV. It does
not run long jobs unless ``--execute`` is set.
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
class Variant:
    name: str
    mode: str
    fixed_opponent: str | None
    train_flags: tuple[str, ...]
    description: str


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "latent_default",
        "FIXED_OPPONENT",
        "OP3",
        (),
        "Summer latent team strategy default, K=4, episode-start z, fixed OP3.",
    ),
    Variant(
        "curriculum",
        "CURRICULUM",
        None,
        ("--no-latent-strategy",),
        "Curriculum baseline: no latent strategy, OP1->OP2->OP3.",
    ),
    Variant(
        "no_latent",
        "FIXED_OPPONENT",
        "OP3",
        ("--no-latent-strategy",),
        "No-latent PPO baseline under the Summer default fixed-OP3 setting.",
    ),
)


def _team_tag(agents: int) -> str:
    n = max(1, int(agents))
    return f"{n}v{n}"


def _quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _command_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for agents in args.agents:
        team = _team_tag(int(agents))
        checkpoint_dir = os.path.join(args.checkpoint_root, team)
        for seed in args.seeds:
            for variant in VARIANTS:
                opponent_tag = str(variant.fixed_opponent or "op123").lower()
                run_tag = f"phase6_{variant.name}_{opponent_tag}_seed{int(seed)}_{team}"
                checkpoint = os.path.join(checkpoint_dir, f"final_{run_tag}.zip")
                train_parts = [
                    args.python,
                    "rl/train_ppo.py",
                    "--mode",
                    variant.mode,
                    "--map-set",
                    "train",
                    "--agents",
                    str(int(agents)),
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
                    *variant.train_flags,
                ]
                if variant.fixed_opponent is not None:
                    train_parts.extend(["--fixed-opponent", variant.fixed_opponent])
                eval_parts = [
                    args.python,
                    "plot/eval_checkpoint.py",
                    "--checkpoint",
                    checkpoint,
                    "--label",
                    run_tag,
                    "--agents",
                    str(int(agents)),
                    "--opponents",
                    *args.eval_opponents,
                    "--map-sets",
                    *args.eval_map_sets,
                    "--episodes",
                    str(int(args.eval_episodes)),
                    "--device",
                    args.device,
                    "--seed",
                    str(int(seed)),
                ]
                rows.append(
                    {
                        "team": team,
                        "seed": str(int(seed)),
                        "variant": variant.name,
                        "description": variant.description,
                        "checkpoint": checkpoint,
                        "train_command": _quote(train_parts),
                        "eval_command": _quote(eval_parts),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final-phase experiment commands.")
    parser.add_argument("--agents", type=int, nargs="+", default=[2, 4, 6], help="Team sizes to include.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Training/eval seeds.")
    parser.add_argument("--steps", type=int, default=100_000, help="Training steps per run.")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-opponents", nargs="+", default=["OP3", "OP4"])
    parser.add_argument("--eval-map-sets", nargs="+", default=["train", "eval"], choices=["train", "eval"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-root", type=str, default=os.path.join("checkpoints", "phase6"))
    parser.add_argument("--python", type=str, default=sys.executable or "python")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV path for the command matrix.")
    parser.add_argument("--execute", action="store_true", help="Run commands sequentially. Long-running.")
    parser.add_argument("--eval-only", action="store_true", help="With --execute, skip training commands.")
    parser.add_argument("--train-only", action="store_true", help="With --execute, skip eval commands.")
    args = parser.parse_args()

    rows = _command_rows(args)
    for row in rows:
        print(f"\n[{row['team']} seed={row['seed']} {row['variant']}] {row['description']}")
        print("TRAIN:", row["train_command"])
        print("EVAL: ", row["eval_command"])

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[phase6_matrix] wrote {args.out}")

    if args.execute:
        for row in rows:
            commands: list[str] = []
            if not args.eval_only:
                commands.append(row["train_command"])
            if not args.train_only:
                commands.append(row["eval_command"])
            for command in commands:
                print(f"\n[phase6_matrix] running: {command}", flush=True)
                subprocess.run(shlex.split(command), check=True)


if __name__ == "__main__":
    main()
