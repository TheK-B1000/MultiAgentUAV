"""G0-V5 held-out evaluation + recurring-weakness discovery.

Evaluates ONLY the three preregistered 1,000,000-step checkpoints from the
G0-V5 long run, on evaluation seeds disjoint from training, from the
TASK_HEALTH panel, and from every earlier discovery/diagnostic set.

Reuses the frozen discovery machinery in ``run_g0_v2_evaluation`` unchanged --
same predicates, thresholds, opportunity matching, episode-clustered bootstrap
and leakage guarantees. Only the checkpoint source and the evaluation seed base
differ.

Run:  python experiments/run_g0_v5_evaluation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import experiments.run_g0_v2_evaluation as E  # noqa: E402

G0V5_SEEDS = (3_200_001, 3_200_002, 3_200_003)
# Disjoint from training (3200001-3), TASK_HEALTH panel (9300000-2),
# V6I9 discovery (9100000+) and the collapse diagnostic (9200000+).
G0V5_EVAL_SEED_BASE = 9_400_000


def _artifact_dir(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "g0_v5_long" / f"g0_v5_long_seed{seed}"


def _run_tag(seed: int) -> str:
    return f"g0_v5_long_seed{seed}"


def main() -> int:
    # Point the frozen harness at the G0-V5 long run.
    E.artifact_dir_for = _artifact_dir
    E.run_tag_for = _run_tag
    E.EVAL_SEED_BASE = G0V5_EVAL_SEED_BASE
    E.OUT_DIR = PROJECT_ROOT / "artifacts" / "g0_v5_evaluation"

    sys.argv = [
        "run_g0_v5_evaluation.py",
        "--episodes", "30",
        "--seeds", *[str(s) for s in G0V5_SEEDS],
    ]
    return E.main()


if __name__ == "__main__":
    raise SystemExit(main())
