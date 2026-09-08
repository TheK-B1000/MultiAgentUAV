"""SPPPO V1 -- the single 1M-step production run.

Protocol: artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json

    lambda_R    1.0        frozen by SPPPO_LAMBDA_SELECTION.json
    margin      0.04
    mode        FIXED_OPPONENT   (persistent assigned poles)
    block       10100001..10100032
    budget      1,000,000 steps, ONE run, terminal checkpoint only
    final eval  10300001..10300192, UNTOUCHED until this run's terminal exists

THE RUNNER IS PART OF THE TREATMENT, NOT MERELY ORCHESTRATION. Its job is to
guarantee the run is genuinely fresh. The most tempting accidental continuation
is the selected lambda = 1.0 development checkpoint: it shares the training
block, the lambda, and the architecture, so a resume path that "helpfully" finds
it would look entirely correct and silently invalidate the 1M claim. That path is
a hard refusal, with a positive-control test that points load_path directly at it.

Candidate checkpoints are SUPPOSED to exist -- they are provenance. Existence
alone never blocks a valid fresh run. What is forbidden is a candidate checkpoint
resolving as an INPUT, or the production output path overlapping the sweep tree.

Run:  python experiments/run_sppo_production.py            # contract only
      python experiments/run_sppo_production.py --launch   # spend the 1M budget
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
SPPO = SD / "sppo"
PROTOCOL = SPPO / "SPPPO_V1_PROTOCOL.json"
SELECTION = SPPO / "SPPPO_LAMBDA_SELECTION.json"
SWEEP_TREE = SPPO / "lambda_sweep"
OUT = SPPO / "production"
RUN_TAG = "sppo_v1_production_1M_seed10100001"
# Two DISTINCT provenance facts, deliberately separate files:
#   LAUNCH_CONTRACT  -- frozen BEFORE step 1; a prospective declaration
#   FIRST_INTERVAL   -- frozen only AFTER the first interval exists; runtime evidence
# Writing runtime evidence prospectively would imply it existed before the run.
LAUNCH_CONTRACT = SPPO / "SPPPO_PRODUCTION_LAUNCH_CONTRACT.json"
FIRST_INTERVAL = SPPO / "SPPPO_PRODUCTION_FIRST_INTERVAL_EVIDENCE.json"

PRODUCTION_STEPS = 1_000_000
FROZEN_LAMBDA = 1.0
FROZEN_MARGIN = 0.04
FROZEN_CADENCE = 1
TRAIN_SEED = 10_100_001
QPSI_SHA = "930051a725e55e4f14e05dfe178e5f1dc7bd8f3d7e3adeba01187958bb7417bf"

# Fields that could resolve to a checkpoint INPUT. exp2_teacher_checkpoints is
# deliberately excluded: the frozen SAPPO teachers are legitimate inputs and live
# outside the sweep tree.
CHECKPOINT_INPUT_FIELDS = (
    "load_path", "gate_config_fingerprint_checkpoint",
    "evaluation_only_checkpoint_family",
)


def _stable_hash(v) -> str:
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _under(path, root: Path) -> bool:
    """True if `path` resolves inside `root`."""
    if not path:
        return False
    try:
        Path(str(path)).resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def build_production_config():
    """Fresh production config, derived from the same base the sweep used."""
    from experiments.run_sppo_lambda_sweep import build_candidate
    from rl.config.ppo_config import TrainMode

    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    if sel.get("VERDICT") != "LAMBDA_SELECTED":
        raise RuntimeError(f"lambda selection did not succeed: {sel.get('VERDICT')}")
    lam = float(sel["SELECTED_LAMBDA_R"])
    if lam != FROZEN_LAMBDA:
        raise RuntimeError(f"selected lambda {lam} != frozen {FROZEN_LAMBDA}")

    cfg, _, parent_contract = build_candidate(lam)
    cfg.total_timesteps = PRODUCTION_STEPS
    cfg.run_tag = RUN_TAG
    art = OUT / RUN_TAG
    cfg.checkpoint_dir = str(art / "ckpts")          # OUTSIDE the sweep tree
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    # Exactly one terminal checkpoint: periodic writing is disabled so that
    # checkpoint selection is structurally impossible rather than merely
    # discouraged. This affects artifact WRITING only, never training.
    cfg.periodic_checkpoint_steps = 0
    cfg.load_path = None
    cfg.checkpoint_run_start_step = 0
    _ = TrainMode  # mode already set to FIXED_OPPONENT by build_candidate
    return cfg, parent_contract


def validate(cfg) -> dict:
    """Every launch refusal. Returns the validation surface when all pass."""
    checks: dict[str, object] = {}
    fail: list[str] = []

    def req(name, ok, detail):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            fail.append(f"{name}: {detail}")

    req("budget_exactly_1M", int(cfg.total_timesteps) == PRODUCTION_STEPS,
        f"total_timesteps={cfg.total_timesteps}")
    req("lambda_R_is_frozen_selection", float(cfg.sppo_lambda_rank) == FROZEN_LAMBDA,
        f"sppo_lambda_rank={cfg.sppo_lambda_rank}")
    req("margin_frozen", float(cfg.sppo_ranking_margin) == FROZEN_MARGIN,
        f"margin={cfg.sppo_ranking_margin}")
    req("ranking_cadence_frozen", int(cfg.sppo_ranking_cadence) == FROZEN_CADENCE,
        f"cadence={cfg.sppo_ranking_cadence}")
    req("mode_is_FIXED_OPPONENT", str(cfg.mode).upper() == "FIXED_OPPONENT",
        f"mode={cfg.mode}")
    req("opponent_randomize_off", not bool(cfg.opponent_randomize),
        f"opponent_randomize={cfg.opponent_randomize}")
    req("training_seed_is_the_training_block", int(cfg.seed) == TRAIN_SEED,
        f"seed={cfg.seed}")
    req("starts_at_timestep_zero", int(getattr(cfg, "checkpoint_run_start_step", 0)) == 0,
        f"checkpoint_run_start_step={getattr(cfg, 'checkpoint_run_start_step', 0)}")

    # --- fresh start: no resume/continue path of any kind -------------------
    for f in CHECKPOINT_INPUT_FIELDS:
        v = getattr(cfg, f, None)
        req(f"no_checkpoint_input__{f}", not v, f"{f}={v!r}")
    req("load_weights_only_off", not bool(getattr(cfg, "load_weights_only", False)),
        f"load_weights_only={getattr(cfg, 'load_weights_only', False)}")

    # --- candidate checkpoints may EXIST but never be INPUTS ----------------
    resolved_inputs = {f: getattr(cfg, f, None) for f in CHECKPOINT_INPUT_FIELDS}
    offenders = [f"{f}={v}" for f, v in resolved_inputs.items() if _under(v, SWEEP_TREE)]
    req("no_input_resolves_into_the_sweep_tree", not offenders, str(offenders))

    # --- production output must not overlap the sweep tree ------------------
    for f in ("checkpoint_dir", "metrics_csv_path", "episode_csv_path"):
        req(f"output_outside_sweep_tree__{f}", not _under(getattr(cfg, f, None), SWEEP_TREE),
            f"{f}={getattr(cfg, f, None)}")

    # --- frozen scorer ------------------------------------------------------
    from experiments.phase0_scorer_common import sha256_file
    qp = ROOT / str(cfg.sppo_qpsi_path)
    sha = sha256_file(qp) if qp.is_file() else None
    req("qpsi_sha_matches_frozen", sha == QPSI_SHA, f"sha={None if sha is None else sha[:16]}")

    # --- inherited teacher treatment ---------------------------------------
    req("teacher_lambda_unchanged", float(cfg.exp2_teacher_lambda) == 0.10,
        f"exp2_teacher_lambda={cfg.exp2_teacher_lambda}")
    req("teacher_cadence_unchanged", int(cfg.exp2_teacher_cadence) == 4,
        f"exp2_teacher_cadence={cfg.exp2_teacher_cadence}")
    req("n_envs_is_32", int(cfg.n_envs) == 32, f"n_envs={cfg.n_envs}")

    # --- exactly one terminal checkpoint -----------------------------------
    req("periodic_checkpointing_disabled",
        int(getattr(cfg, "periodic_checkpoint_steps", 0)) == 0,
        f"periodic_checkpoint_steps={getattr(cfg, 'periodic_checkpoint_steps', 0)}")

    # --- one-attempt ---------------------------------------------------------
    run_dir = OUT / RUN_TAG
    req("production_dir_is_empty", not (run_dir.exists() and any(run_dir.iterdir())),
        str(run_dir))

    if fail:
        raise RuntimeError("PRODUCTION LAUNCH REFUSED:\n  " + "\n  ".join(fail))
    return checks


def launch_contract(cfg, checks) -> dict:
    return {
        "record": "SPPPO V1 production launch contract",
        "run_tag": RUN_TAG,
        "lambda_R": float(cfg.sppo_lambda_rank),
        "lambda_R_provenance": "SPPPO_LAMBDA_SELECTION.json, boundary caveat recorded",
        "margin": float(cfg.sppo_ranking_margin),
        "ranking_cadence": int(cfg.sppo_ranking_cadence),
        "mode": str(cfg.mode),
        "training_block": "10100001..10100032",
        "total_timesteps": int(cfg.total_timesteps),
        "fresh_initialisation": True,
        "resumed_from": None,
        "candidate_checkpoints_reused": False,
        "development_steps_counted_toward_budget": 0,
        "terminal_checkpoint_only": True,
        "final_evaluation_block": "10300001..10300192 (UNTOUCHED until terminal exists)",
        "qpsi_sha256": QPSI_SHA,
        "resolved_config_sha256": _stable_hash(dataclasses.asdict(cfg)),
        "validation": checks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true")
    a = ap.parse_args()

    cfg, parent_contract = build_production_config()
    checks = validate(cfg)
    contract = launch_contract(cfg, checks)
    print(json.dumps({k: v for k, v in contract.items() if k != "validation"},
                     indent=2, default=str))
    print(f"\nvalidation surface: {len(checks)} checks, all PASS")
    for k in sorted(checks):
        print(f"  PASS  {k}")
    if not a.launch:
        print("\nCONTRACT ONLY. No environment constructed, no training step spent.")
        return 0

    from experiments.run_sppo_lambda_sweep import configure_sppo_live_environment
    from rl.training.orchestrator import orchestrate_training_run
    (OUT / RUN_TAG).mkdir(parents=True, exist_ok=True)
    contract["frozen_when"] = "BEFORE step 1 -- prospective declaration, not runtime evidence"
    contract["first_interval_evidence"] = (
        f"NOT YET COLLECTED. Captured separately in {FIRST_INTERVAL.name} only after "
        "the first reporting interval exists.")
    LAUNCH_CONTRACT.write_text(json.dumps(contract, indent=2, default=str), encoding="utf-8")
    print(f"\npre-launch contract frozen -> {LAUNCH_CONTRACT}")
    print(f"first-interval evidence deferred -> {FIRST_INTERVAL.name} (after interval 1)\n")
    orchestrate_training_run(
        cfg, pre_rollout_env_setup=partial(configure_sppo_live_environment,
                                           contract=parent_contract))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
