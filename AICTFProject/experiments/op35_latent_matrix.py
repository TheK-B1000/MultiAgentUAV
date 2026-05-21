#!/usr/bin/env python3
"""OP3+OP5_RUSHER latent experiment matrix: preflight gates + training commands.

Run from ``AICTFProject`` (same convention as ``phase6_experiment_matrix.py``).

Three gates (run in order before a long OP35 latent job):

1. **OP5 difficulty (flat checkpoint)** — ``plot/eval_checkpoint.py`` vs OP3 and OP5_RUSHER.
   Ask: *does a strong flat policy beat OP5_RUSHER too easily?* If win rate vs OP5 is ~as high as vs OP3, tune OP5 before latent OP35 training.

2. **MI on existing latent E3 CSV** — ``plot/analyze_e3_latent_mi.py`` on ``*_e3_steps.csv``.
   If you have no E3 file yet, use the optional *collect* command (short resumed rollout with
   ``--e3-step-telemetry``) then re-run the analyzer.

3. **Training** — Use preset ``hypothesis_latent_opprand_optionb_lamp_coef05_op35`` only (Option B:
   resample every 20, λ_p=0.02, strategy PPO coef 0.5, pool OP3+OP5). Do **not** mix
   ``latent_a1_plan_faithful`` with manual ``--mode OPPONENT_POOL``; the preset already encodes
   the richer persistence machinery that matters under a switching opponent pool.

Example (known two-blade flat checkpoint; redirect to save the matrix)::

    python experiments/op35_latent_matrix.py \\
        --flat-checkpoint experiments/hypothesis_runs/20260509_103737/checkpoints/2v2/final_research_hypothesis_flat_opprand_seed42_2v2.zip \\
        > experiments/op35_latent_matrix_seed42_2v2.md

OP35-aligned flat (after ``hypothesis_flat_opprand_op35``) for the cleanest OP5 gate::

    python experiments/op35_latent_matrix.py \\
        --flat-checkpoint experiments/paper_runs/op35_seed42/checkpoints/2v2/final_paper_flat_op35_seed42_2v2.zip

With no ``--flat-checkpoint``, the default is the **intended** ``final_*.zip`` two-blade path; if your run
stopped early, pass a ``ckpt_*.zip`` instead. Unattended full pipeline (preflight + 1M train)::

    python experiments/op35_latent_matrix.py --execute --flat-checkpoint PATH_TO_FLAT.zip `
      2>&1 | Tee-Object -FilePath experiments/op35_matrix_run.log

Preflight only (no 1M training), still while you are away::

    python experiments/op35_latent_matrix.py --execute --skip-train-on-execute --flat-checkpoint PATH_TO_FLAT.zip
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

# Finished flat opponent-pool (OP1-OP3) checkpoint from the hypothesis two-blade run (seed 42, 2v2).
DEFAULT_FLAT_CHECKPOINT_TWO_BLADE = os.path.join(
    "experiments",
    "hypothesis_runs",
    "20260509_103737",
    "checkpoints",
    "2v2",
    "final_research_hypothesis_flat_opprand_seed42_2v2.zip",
)
# Use after training hypothesis_flat_opprand_op35 for OP5 calibration aligned with the latent OP35 pool.
OPTIONAL_FLAT_CHECKPOINT_OP35_ALIGNED = os.path.join(
    "experiments",
    "paper_runs",
    "op35_seed42",
    "checkpoints",
    "2v2",
    "final_paper_flat_op35_seed42_2v2.zip",
)


@dataclass(frozen=True)
class MatrixRow:
    phase: str
    step: str
    description: str
    argv: tuple[str, ...]

    @property
    def command(self) -> str:
        """Shell-style line for logs, CSV, and copy-paste (may use shlex quoting)."""
        return _quote(list(self.argv))


def _quote(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(p)) for p in parts)


def _split_cmd(command: str) -> list[str]:
    """Split a shell line (used in tests; ``--execute`` uses ``MatrixRow.argv`` instead)."""
    if os.name == "nt":
        return shlex.split(command, posix=False)
    return shlex.split(command)


def _resolved_flat_checkpoint(flat_arg: str) -> str:
    return str(flat_arg).strip() or DEFAULT_FLAT_CHECKPOINT_TWO_BLADE


def _rows_for_execute(rows: list[MatrixRow], *, skip_train_on_execute: bool) -> list[MatrixRow]:
    if not skip_train_on_execute:
        return rows
    return [r for r in rows if r.phase != "3_train"]


def _preflight_rows(
    *,
    python_exe: str,
    flat_ckpt: str,
    latent_ckpt: str,
    latent_e3_csv: str | None,
    agents: int,
    eval_episodes: int,
    device: str,
    eval_seed: int,
    collect_e3_steps: int,
    collect_run_tag: str,
) -> list[MatrixRow]:
    rows: list[MatrixRow] = []
    team = f"{agents}v{agents}"

    eval_both = [
        python_exe,
        "plot/eval_checkpoint.py",
        "--checkpoint",
        flat_ckpt,
        "--label",
        f"preflight_flat_op35_wr_{team}",
        "--agents",
        str(agents),
        "--opponents",
        "OP3",
        "OP5_RUSHER",
        "--map-sets",
        "train",
        "--episodes",
        str(eval_episodes),
        "--device",
        device,
        "--seed",
        str(eval_seed),
    ]
    rows.append(
        MatrixRow(
            "1_preflight",
            "eval_flat_vs_op3_op5",
            "Gate A: flat policy WR vs OP3 and vs OP5_RUSHER (train maps). Compare columns in csv/; "
            "if OP3 WR ~ OP5 WR, OP5 is too soft.",
            tuple(eval_both),
        )
    )

    if latent_e3_csv:
        mi_cmd = [
            python_exe,
            "plot/analyze_e3_latent_mi.py",
            latent_e3_csv,
        ]
        rows.append(
            MatrixRow(
                "2_preflight",
                "analyze_e3_mi",
                "Gate B: MI(z; phase), MI(z; opponent), MI(z; outcome) from existing E3 step CSV.",
                tuple(mi_cmd),
            )
        )
    else:
        collect = [
            python_exe,
            "rl/train_ppo.py",
            "--preset",
            "latent_a1_plan_faithful",
            "--agents",
            str(agents),
            "--total-steps",
            str(collect_e3_steps),
            "--load",
            latent_ckpt,
            "--run-tag",
            collect_run_tag,
            "--e3-step-telemetry",
            "--checkpoint-dir",
            os.path.dirname(os.path.abspath(latent_ckpt)) or "checkpoints",
            "--device",
            device,
        ]
        out_csv = os.path.join(
            os.path.dirname(os.path.abspath(latent_ckpt)) or ".",
            f"{collect_run_tag}_e3_steps.csv",
        )
        rows.append(
            MatrixRow(
                "2_preflight",
                "collect_e3_then_mi",
                "Gate B: no --latent-e3-csv passed. Short resumed rollout writes E3 CSV next to "
                f"checkpoint dir; expected path like {out_csv}. Then run analyze_e3_latent_mi.py on that file.",
                tuple(collect),
            )
        )

    return rows


def _train_rows(
    *,
    python_exe: str,
    seeds: list[int],
    agents: int,
    device: str,
    train_tag_prefix: str,
    e3_telemetry: bool,
) -> list[MatrixRow]:
    rows: list[MatrixRow] = []
    team = f"{agents}v{agents}"
    for seed in seeds:
        run_tag = f"{train_tag_prefix}_seed{seed}_{team}"
        parts = [
            python_exe,
            "rl/train_ppo.py",
            "--preset",
            "hypothesis_latent_opprand_optionb_lamp_coef05_op35",
            "--agents",
            str(agents),
            "--seed",
            str(seed),
            "--run-tag",
            run_tag,
            "--device",
            device,
        ]
        if e3_telemetry:
            parts.append("--e3-step-telemetry")
        rows.append(
            MatrixRow(
                "3_train",
                f"latent_op35_opb_seed{seed}",
                "Production: latent Option-B + OP3/OP5 pool (preset op35). Overrides default preset run_tag.",
                tuple(parts),
            )
        )
    return rows


def _print_preamble() -> None:
    print(
        """
================================================================================
OP3 + OP5_RUSHER latent matrix - read before --execute
================================================================================
(1) Flat eval: does a strong flat policy beat OP5_RUSHER too easily? If WR vs OP5 ~ WR vs OP3, tune OP5 first.
(2) E3 MI: run analyze_e3_latent_mi.py on an *_e3_steps.csv from a latent run; if MI(z;phase)
    is already > 0 on old data, you may not need new training to answer the phase question.
(3) Train: use ONLY --preset hypothesis_latent_opprand_optionb_lamp_coef05_op35 (not A1 + pool).
================================================================================
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--flat-checkpoint",
        type=str,
        default=DEFAULT_FLAT_CHECKPOINT_TWO_BLADE,
        help=(
            "Flat (no-latent) 2v2 .zip for gate 1. Default: two-blade "
            f"{DEFAULT_FLAT_CHECKPOINT_TWO_BLADE}. "
            f"For OP35-pool-aligned calibration use {OPTIONAL_FLAT_CHECKPOINT_OP35_ALIGNED} after that flat run exists."
        ),
    )
    parser.add_argument(
        "--latent-checkpoint",
        type=str,
        default="checkpoints/2v2/final_latent_a1_plan_faithful_1m_2v2.zip",
        help="Latent checkpoint for optional short E3 collection when --latent-e3-csv is omitted.",
    )
    parser.add_argument(
        "--latent-e3-csv",
        type=str,
        default="",
        help="Existing *_e3_steps.csv for analyze_e3_latent_mi.py. If empty, matrix emits a short collect command.",
    )
    parser.add_argument("--agents", type=int, default=2, choices=[2, 4, 6, 8])
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--collect-e3-steps", type=int, default=8192, help="Short rollout steps if collecting E3.")
    parser.add_argument(
        "--collect-run-tag",
        type=str,
        default="e3_mi_preflight_collect_2v2",
        help="Run tag for short E3 collection job.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python", type=str, default=sys.executable or "python")
    parser.add_argument(
        "--train-tag-prefix",
        type=str,
        default="hypothesis_latent_opb_op35",
        help="Prefix for --run-tag on training rows (suffix _seed{N}_{NvN} added).",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Omit gates 1-2; only print training commands.",
    )
    parser.add_argument(
        "--e3-on-train",
        action="store_true",
        help="Append --e3-step-telemetry to training commands (larger CSVs during 1M steps).",
    )
    parser.add_argument("--out", type=str, default=None, help="Write CSV with phase,step,description,command.")
    parser.add_argument("--execute", action="store_true", help="Run each command sequentially (long if training).")
    parser.add_argument(
        "--skip-train-on-execute",
        action="store_true",
        help="With --execute, run preflight rows only (skip 3_train / no 1M job).",
    )
    args = parser.parse_args()

    rows: list[MatrixRow] = []
    if not args.skip_preflight:
        flat_arg = str(args.flat_checkpoint).strip()
        if not flat_arg:
            print(
                "[op35_latent_matrix] WARNING: --flat-checkpoint empty; gate 1 uses default final_ path.\n"
                "  If training stopped early, pass a ckpt_*.zip explicitly.\n"
                f"  Default: {DEFAULT_FLAT_CHECKPOINT_TWO_BLADE}",
                file=sys.stderr,
            )
        rows.extend(
            _preflight_rows(
                python_exe=args.python,
                flat_ckpt=_resolved_flat_checkpoint(flat_arg),
                latent_ckpt=str(args.latent_checkpoint).strip(),
                latent_e3_csv=str(args.latent_e3_csv).strip() or None,
                agents=int(args.agents),
                eval_episodes=int(args.eval_episodes),
                device=str(args.device),
                eval_seed=int(args.eval_seed),
                collect_e3_steps=int(args.collect_e3_steps),
                collect_run_tag=str(args.collect_run_tag).strip(),
            )
        )
    rows.extend(
        _train_rows(
            python_exe=args.python,
            seeds=list(args.seeds),
            agents=int(args.agents),
            device=str(args.device),
            train_tag_prefix=str(args.train_tag_prefix).strip(),
            e3_telemetry=bool(args.e3_on_train),
        )
    )

    _print_preamble()
    for r in rows:
        print(f"\n[{r.phase} :: {r.step}]\n{r.description}\n{r.command}")

    if args.out:
        path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["phase", "step", "description", "command"])
            w.writeheader()
            for r in rows:
                w.writerow({"phase": r.phase, "step": r.step, "description": r.description, "command": r.command})
        print(f"\n[op35_latent_matrix] wrote {path}")

    if args.execute:
        flat_used = _resolved_flat_checkpoint(str(args.flat_checkpoint))
        if not args.skip_preflight and not os.path.isfile(os.path.abspath(flat_used)):
            raise SystemExit(
                "[op35_latent_matrix] --execute aborted: flat checkpoint not found:\n"
                f"  {os.path.abspath(flat_used)}\n"
                "Pass --flat-checkpoint to an existing .zip (final_ or ckpt_ from hypothesis_flat_opprand)."
            )
        exec_rows = _rows_for_execute(rows, skip_train_on_execute=bool(args.skip_train_on_execute))
        if args.skip_train_on_execute:
            print("\n[op35_latent_matrix] --skip-train-on-execute: will not run 3_train rows.", flush=True)
        for r in exec_rows:
            print(f"\n[op35_latent_matrix] executing:\n{r.command}", flush=True)
            subprocess.run(list(r.argv), check=True)


if __name__ == "__main__":
    main()
