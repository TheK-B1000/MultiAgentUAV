#!/bin/bash
# 6v6 pipeline driver: specialists -> distillation collection -> Rung-1 training ->
# sealed crossover eval -> deployment robustness. See RUNBOOK_6V6.md for the full runbook.
#
# Usage:
#   bash experiments/run_6v6_pipeline.sh                # run all stages in order, resuming
#   bash experiments/run_6v6_pipeline.sh --stage 3       # run ONLY stage 3
#   bash experiments/run_6v6_pipeline.sh --from-stage 3  # run stage 3 onward
#   SKIP_GPU_CHECK=1 bash experiments/run_6v6_pipeline.sh   # skip the "is the GPU free" check
#                                                             (fine on a dedicated machine)
#
# Design (per the fail-safe requirements this driver implements):
#   - set -e: any command failing stops the whole script immediately, with the failing
#     command's own error message intact (never swallowed).
#   - Each stage is RESUMABLE: before running, it checks whether its own terminal artifact
#     already exists and, if so, prints "already complete, skipping" and moves on. A completed
#     scientific output is NEVER overwritten -- the underlying Python scripts also refuse this
#     independently (defense in depth), but the driver skips proactively so re-running the
#     whole pipeline after an interruption does the right thing without manual bookkeeping.
#   - Every expensive stage is dry-run/preflighted FIRST; the driver aborts before spending
#     real compute if that fails.
#   - Every stage's success is verified by checking its ACTUAL output file(s) exist on disk
#     afterward, not by trusting the launched process's exit code alone.
#   - Every stage prints which stage is running and exactly which files it is reading from /
#     writing to, before doing anything.
set -e
cd "$(dirname "$0")/.."
PY="./.venv/Scripts/python.exe"
SD="artifacts/strategic_demand/sppo"

log()  { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1"; }
banner() { echo; echo "=================================================================="; echo "  $1"; echo "=================================================================="; }
require_file() {
  if [ ! -f "$1" ]; then
    echo "FAIL-CLOSED: required file missing: $1"
    echo "  $2"
    exit 1
  fi
}

# ---------------------------------------------------------------- stage 0: environment
stage0_verify_environment() {
  banner "STAGE 0: environment verification"
  require_file "$PY" "the project venv is missing or not at .venv -- create it before running this pipeline (see RUNBOOK_6V6.md)."
  log "python: $($PY --version 2>&1)"

  log "checking CUDA availability..."
  $PY -c "
import torch
ok = torch.cuda.is_available()
print(f'  torch {torch.__version__}  cuda_available={ok}')
if ok:
    print(f'  device: {torch.cuda.get_device_name(0)}')
else:
    raise SystemExit('FAIL-CLOSED: CUDA is not available on this machine. This pipeline trains real PPO policies and needs a GPU.')
"

  log "checking required frozen specs exist..."
  for f in \
    "$SD/STRATEGIC_DEMAND_6v6_GUARD_DISTRIBUTED_V2_CERTIFICATION.json" \
    "$SD/SCALE_6V6_PRODUCTION_SPECIALIST_SEEDS.json" \
    "$SD/TEACHER_DISTILLATION_6V6_SPEC.json" \
    "$SD/TEACHER_DISTILLATION_6V6_SEED_RETIREMENT_AMENDMENT.json" \
    "$SD/RUNG1_CONSTRUCTION_6V6_AMENDMENT.json" \
    "$SD/RUNG1_6V6_CROSSOVER_EVAL_SPEC.json" \
    "$SD/6V6_PIPELINE_SEED_ALLOCATION.json" \
    "$SD/DEPLOYMENT_ROBUSTNESS_SPEC.json" \
    "$SD/DEPLOYMENT_ONLY_GUARANTEE_CHECK.json"; do
    require_file "$f" "this frozen spec is a precondition for the pipeline; it should have shipped with the repo at the pinned commit."
  done
  log "STAGE 0 PASS -- environment and preconditions verified."
}

# ---------------------------------------------------------------- stage 1: specialists
stage1_specialists() {
  banner "STAGE 1: 6v6 specialist training (pi_A, pi_B)"
  local PI_A_SEED=7610001 PI_B_SEED=7620001
  local PI_A_CKPT="artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip"
  local PI_B_CKPT="artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip"

  if [ -f "$PI_A_CKPT" ] && [ -f "$PI_B_CKPT" ]; then
    log "STAGE 1 already complete -- skipping. ($PI_A_CKPT, $PI_B_CKPT)"
    return 0
  fi

  if [ -z "$SKIP_GPU_CHECK" ]; then
    local BUSY
    BUSY=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$BUSY" ] && [ "$BUSY" -gt 20 ]; then
      echo "REFUSING: GPU utilization is ${BUSY}% -- another job appears to be running."
      echo "  Wait for it, or re-run with SKIP_GPU_CHECK=1 if this machine is dedicated to this pipeline."
      exit 1
    fi
  fi

  if [ ! -f "$PI_A_CKPT" ]; then
    log "dry-run pi_A_6v6 (seed $PI_A_SEED)"
    $PY experiments/train_specialist_scale.py --team-size 6 --policy A --seed $PI_A_SEED --device cuda --dry-run
    log "launching pi_A_6v6 for real -- writes to artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ (~1M steps, ~5h)"
    $PY -u experiments/train_specialist_scale.py --team-size 6 --policy A --seed $PI_A_SEED --device cuda
    if [ ! -f "$PI_A_CKPT" ]; then
      echo "FAIL-SAFE TRIGGERED: pi_A_6v6 training returned but $PI_A_CKPT does not exist. Stopping before pi_B."
      exit 1
    fi
    log "pi_A_6v6 terminal checkpoint verified: $PI_A_CKPT"
  else
    log "pi_A_6v6 already complete -- skipping."
  fi

  log "dry-run pi_B_6v6 (seed $PI_B_SEED)"
  $PY experiments/train_specialist_scale.py --team-size 6 --policy B --seed $PI_B_SEED --device cuda --dry-run
  log "launching pi_B_6v6 for real -- writes to artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ (~1M steps, ~5h)"
  $PY -u experiments/train_specialist_scale.py --team-size 6 --policy B --seed $PI_B_SEED --device cuda
  if [ ! -f "$PI_B_CKPT" ]; then
    echo "FAIL-SAFE TRIGGERED: pi_B_6v6 training returned but $PI_B_CKPT does not exist."
    exit 1
  fi
  log "pi_B_6v6 terminal checkpoint verified: $PI_B_CKPT"
  log "STAGE 1 COMPLETE."
  sha256sum "$PI_A_CKPT" "$PI_B_CKPT"
}

# ---------------------------------------------------------------- stage 2: collection
stage2_collection() {
  banner "STAGE 2: teacher-distillation state collection (6v6)"
  local PI_A_CKPT="artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip"
  local PI_B_CKPT="artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip"
  local MANIFEST="$SD/TEACHER_DISTILLATION_6V6_DATASET.json"

  if [ -f "$MANIFEST" ]; then
    log "STAGE 2 already complete -- skipping. ($MANIFEST)"
    return 0
  fi
  require_file "$PI_A_CKPT" "run stage 1 first."
  require_file "$PI_B_CKPT" "run stage 1 first."

  log "check-only: does team size 6 reach the real construction path?"
  $PY experiments/collect_distillation_states_scale.py --team-size 6 --check-only --device cpu

  log "collecting the real dataset (96 episodes/pole, seeds from TEACHER_DISTILLATION_6V6_SPEC.json) -- writes $MANIFEST, ~minutes to low hours on CPU"
  $PY -u experiments/collect_distillation_states_scale.py --team-size 6 --device cpu \
    --pi-a-path "$PI_A_CKPT" --pi-b-path "$PI_B_CKPT"

  require_file "$MANIFEST" "collection returned but did not write its manifest -- do not proceed to stage 3."
  log "STAGE 2 COMPLETE. -> $MANIFEST"
}

# ---------------------------------------------------------------- stage 3: rung-1 training
stage3_rung1_training() {
  banner "STAGE 3: Rung-1 latent-policy training (6v6)"
  local FROZEN="$SD/RUNG1_6V6_STUDENT_FROZEN.json"
  local PREFLIGHT="$SD/RUNG1_6V6_PREFLIGHT.json"

  if [ -f "$FROZEN" ]; then
    log "STAGE 3 already complete -- skipping. ($FROZEN)"
    return 0
  fi
  require_file "$SD/TEACHER_DISTILLATION_6V6_DATASET.json" "run stage 2 first."

  log "preflight (8 mechanical checks; trains nothing) -- writes $PREFLIGHT"
  $PY experiments/run_ladder_rung1_distillation_scale.py --team-size 6 --preflight --device cpu

  log "checking preflight VERDICT..."
  $PY -c "
import json
d = json.load(open('$PREFLIGHT'))
if d.get('VERDICT') != 'PASS':
    raise SystemExit(f\"FAIL-CLOSED: Rung-1-at-6v6 preflight did not pass 8/8: {d.get('passed')}\")
print('  preflight PASS:', d.get('passed'))
"

  log "launching the real 20-epoch distillation (device=cuda, minutes not hours -- supervised, not PPO) -- writes $FROZEN"
  $PY -u experiments/run_ladder_rung1_distillation_scale.py --team-size 6 --device cuda

  require_file "$FROZEN" "training returned but did not write its frozen record -- do not proceed to stage 4."
  $PY -c "
import json
d = json.load(open('$FROZEN'))
if d.get('status') != 'FROZEN_STUDENT':
    raise SystemExit(f\"FAIL-CLOSED: Rung-1-at-6v6 fit check failed (status={d.get('status')}) -- do not proceed to stage 4.\")
print('  fit check PASS')
"
  log "STAGE 3 COMPLETE. -> $FROZEN"
}

# ---------------------------------------------------------------- stage 4: crossover eval
stage4_crossover_eval() {
  banner "STAGE 4: sealed 6v6 crossover evaluation (z0/z1 x Pole A/B)"
  local FROZEN="$SD/RUNG1_6V6_STUDENT_FROZEN.json"
  local OUT="$SD/RUNG1_6V6_CROSSOVER_EVAL_RESULT.json"
  local PREAUDIT="$SD/RUNG1_6V6_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

  if [ -f "$OUT" ]; then
    log "STAGE 4 already complete -- skipping. ($OUT)"
    return 0
  fi
  if [ -f "$PREAUDIT" ]; then
    echo "STOPPING: a tie/reversal integrity flag from a previous run exists at $PREAUDIT."
    echo "  This requires human review before any further stage 4 attempt (it is one-shot)."
    exit 1
  fi
  require_file "$FROZEN" "run stage 3 first."

  local CKPT_PATH
  CKPT_PATH=$($PY -c "import json; print(json.load(open('$FROZEN'))['TERMINAL_CHECKPOINT']['path'])")
  require_file "$CKPT_PATH" "named in $FROZEN but missing on disk."

  log "dry-run against $CKPT_PATH"
  $PY experiments/eval_rung1_crossover_scaled.py --team-size 6 --rung 1 \
    --checkpoint "$CKPT_PATH" --spec "$SD/RUNG1_6V6_CROSSOVER_EVAL_SPEC.json" \
    --seed-base 13640001 --n-seeds 128 --label RUNG1_6V6 --device cuda --dry-run

  log "launching the real sealed eval (512 episodes, sealed block 13640001-13640128) -- writes $OUT"
  $PY -u experiments/eval_rung1_crossover_scaled.py --team-size 6 --rung 1 \
    --checkpoint "$CKPT_PATH" --spec "$SD/RUNG1_6V6_CROSSOVER_EVAL_SPEC.json" \
    --seed-base 13640001 --n-seeds 128 --label RUNG1_6V6 --device cuda

  if [ -f "$PREAUDIT" ]; then
    echo "STOPPING: stage 4 flagged a tie/reversal integrity check -- see $PREAUDIT. Human review required."
    exit 1
  fi
  require_file "$OUT" "eval returned but did not write its result -- check for a PREAUDIT flag above."
  log "STAGE 4 COMPLETE. -> $OUT"
}

# ---------------------------------------------------------------- stage 5: robustness
stage5_robustness() {
  banner "STAGE 5: deployment robustness (localization / motion / control delay)"
  local FROZEN="$SD/RUNG1_6V6_STUDENT_FROZEN.json"
  local OUT_DIR="$SD/robustness_eval_rows"
  require_file "$FROZEN" "run stage 3 (and pass stage 4) first."
  local CKPT_PATH
  CKPT_PATH=$($PY -c "import json; print(json.load(open('$FROZEN'))['TERMINAL_CHECKPOINT']['path'])")
  require_file "$CKPT_PATH" "named in $FROZEN but missing on disk."

  local ALL_DONE=1
  for pole in A B; do
    for z in 0 1; do
      for fam_sev in "nominal:nominal" "localization_noise:medium" "motion_error:medium" "control_delay:medium"; do
        fam="${fam_sev%%:*}"; sev="${fam_sev##*:}"
        f="$OUT_DIR/rung1_6v6__6v6__pole${pole}__z${z}__${fam}__${sev}.csv"
        [ -f "$f" ] || ALL_DONE=0
      done
    done
  done
  if [ "$ALL_DONE" = "1" ]; then
    log "STAGE 5 already complete -- skipping (all 16 cells present in $OUT_DIR)."
    return 0
  fi

  for pole in A B; do
    for z in 0 1; do
      local target_check="$OUT_DIR/rung1_6v6__6v6__pole${pole}__z${z}__control_delay__medium.csv"
      if [ -f "$target_check" ]; then
        log "pole=$pole z=$z already complete -- skipping."
        continue
      fi
      log "running pole=$pole z=$z (nominal + 3 mid-severity disturbances, 128 seeds each = 512 episodes)"
      $PY -u experiments/eval_deployment_robustness.py \
        --checkpoint "$CKPT_PATH" --checkpoint-id rung1_6v6 --team-label 6v6 --rung 1 --team-size 6 \
        --pole "$pole" --z "$z" --seeds-start 13660001 --n-seeds 128 --severities medium --device cuda
    done
  done
  log "STAGE 5 COMPLETE. -> $OUT_DIR (16 CSVs; compute Delta_A/Delta_B per condition per RUNBOOK_6V6.md)"
}

# ---------------------------------------------------------------- main
STAGE_ONLY=""
FROM_STAGE=1
while [ $# -gt 0 ]; do
  case "$1" in
    --stage) STAGE_ONLY="$2"; shift 2 ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1"; exit 1 ;;
  esac
done

stage0_verify_environment

run_stage() {
  case "$1" in
    1) stage1_specialists ;;
    2) stage2_collection ;;
    3) stage3_rung1_training ;;
    4) stage4_crossover_eval ;;
    5) stage5_robustness ;;
    *) echo "unknown stage: $1"; exit 1 ;;
  esac
}

if [ -n "$STAGE_ONLY" ]; then
  run_stage "$STAGE_ONLY"
else
  for s in 1 2 3 4 5; do
    if [ "$s" -ge "$FROM_STAGE" ]; then
      run_stage "$s"
    fi
  done
fi

banner "6v6 PIPELINE: ALL REQUESTED STAGES COMPLETE"
bash experiments/export_6v6_results.sh
