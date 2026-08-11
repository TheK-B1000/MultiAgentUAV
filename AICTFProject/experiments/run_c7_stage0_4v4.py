"""C7 Stage 0 — train one 4v4 baseline seed under the frozen G0-V5 pipeline.

The G0-V5 policies cannot act at 4v4: their observation encoder is hard-bound to
two agents and raises "grid must have shape (B, 2, 7, 20, 20)". C7's treatment
arm therefore needs its own baseline, and there is no adapter that would avoid
confounding team size with out-of-distribution inference.

This deliberately does NOT reimplement training. It reuses run_g0_v5_long
wholesale -- the same reward, opponent pool, architecture, 1M-step budget,
health probes, validation panels and TASK_HEALTH/SYSTEM_HEALTH gates -- and
rebinds exactly three module globals:

    AGENTS            2 -> 4          the manipulation
    seed set          3200001..3  ->  3300001..3
    artifact paths    g0_v5_long/ ->  c7_stage0/

Everything else is inherited rather than copied, so the control and treatment
arms cannot drift apart through a transcription error.

FAIL-FAST: Stage 0 requires 3/3 competence. If a seed fails, the conjunction is
already false and the remaining seeds are not worth ~7 GPU-hours each. Run one
seed, check, then launch the next. This is short-circuit evaluation of a
conjunctive gate -- it cannot change the gate.

Run:  python experiments/run_c7_stage0_4v4.py --seed 3300001
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

C7_SEEDS = (3_300_001, 3_300_002, 3_300_003)
C7_AGENTS = 4


def _run_tag(seed: int) -> str:
    return f"c7_4v4_seed{int(seed)}"


def _artifact_dir(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "c7_stage0" / _run_tag(seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=list(C7_SEEDS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    import experiments.run_g0_v2_evaluation as E
    import experiments.run_g0_v5_long as G

    # TRAINING side.
    G.AGENTS = C7_AGENTS
    G.G0V5_SEEDS = C7_SEEDS
    G.ABLATION_SEEDS = C7_SEEDS
    G.run_tag_for = _run_tag
    G.artifact_dir_for = _artifact_dir

    # EVALUATION side. run_validation_panel calls run_g0_v2_evaluation.
    # run_eval_episode, which builds its env from that module's OWN AGENTS
    # global -- a separate binding from G.AGENTS. Patching only the training
    # side trains a 4-agent policy and then evaluates it in a 2v2 env, which
    # raises "grid must have shape (B, 4, 7, 20, 20), got (1, 2, 7, 20, 20)"
    # and yields TASK_HEALTH=ERROR at every panel. That is unmeasurable rather
    # than failing, so Stage 0's gate could never be evaluated.
    E.AGENTS = C7_AGENTS

    print("=" * 78)
    print(f"C7 STAGE 0 — 4v4 BASELINE  seed={args.seed}")
    print(f"  AGENTS {C7_AGENTS} (G0-V5 control trained at 2)")
    print(f"  artifacts -> {_artifact_dir(args.seed)}")
    print("  reward, opponent pool, architecture, budget and health gates: "
          "inherited unchanged from run_g0_v5_long")
    print("=" * 78, flush=True)

    sys.argv = ["run_g0_v5_long.py", "--seed", str(args.seed),
                "--threads", str(args.threads)]
    return G.main()


if __name__ == "__main__":
    raise SystemExit(main())
