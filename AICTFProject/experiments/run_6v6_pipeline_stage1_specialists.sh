#!/bin/bash
# Stage 1 of the 6v6 pipeline: train pi_A_6v6 then pi_B_6v6, sequentially, with fail-safes.
#
# Fail-safe design:
#   - set -e: any command failing (non-zero exit) stops the whole script immediately.
#   - Each stage's dry-run is checked BEFORE the real run, so a config/pole problem is caught
#     before spending any GPU time.
#   - Each stage's success is verified by checking its OWN terminal checkpoint file exists on
#     disk afterward, not just by trusting the process exit code (matches this project's
#     standing discipline: verify the artifact, not just the return code).
#   - Optional GPU-free check: refuses to start if another python/CUDA job is already running,
#     unless SKIP_GPU_CHECK=1 is set (e.g. if you deliberately want this on its own machine).
#
# Run:  bash experiments/run_6v6_pipeline_stage1_specialists.sh
set -e
cd "$(dirname "$0")/.."

PI_A_SEED=7610001
PI_B_SEED=7620001
PI_A_CKPT="artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip"
PI_B_CKPT="artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip"
PY="./.venv/Scripts/python.exe"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $1"; }

if [ -z "$SKIP_GPU_CHECK" ]; then
  BUSY=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$BUSY" ] && [ "$BUSY" -gt 20 ]; then
    echo "REFUSING: GPU utilization is ${BUSY}% -- another job appears to be running."
    echo "  Wait for it to finish, or re-run with SKIP_GPU_CHECK=1 if this machine is dedicated to this pipeline."
    exit 1
  fi
fi

if [ -f "$PI_A_CKPT" ] || [ -f "$PI_B_CKPT" ]; then
  echo "REFUSING: a terminal checkpoint already exists -- this stage is one-shot per seed."
  echo "  pi_A: $PI_A_CKPT"
  echo "  pi_B: $PI_B_CKPT"
  exit 1
fi

log "dry-run pi_A_6v6 (seed $PI_A_SEED)"
$PY experiments/train_specialist_scale.py --team-size 6 --policy A --seed $PI_A_SEED --device cuda --dry-run

log "launching pi_A_6v6 for real (seed $PI_A_SEED, ~1M steps, ~5h expected)"
$PY -u experiments/train_specialist_scale.py --team-size 6 --policy A --seed $PI_A_SEED --device cuda

if [ ! -f "$PI_A_CKPT" ]; then
  echo "FAIL-SAFE TRIGGERED: pi_A_6v6 training returned but $PI_A_CKPT does not exist."
  echo "  Stopping before pi_B -- do not proceed on an unverified pi_A checkpoint."
  exit 1
fi
log "pi_A_6v6 terminal checkpoint verified: $PI_A_CKPT"

log "dry-run pi_B_6v6 (seed $PI_B_SEED)"
$PY experiments/train_specialist_scale.py --team-size 6 --policy B --seed $PI_B_SEED --device cuda --dry-run

log "launching pi_B_6v6 for real (seed $PI_B_SEED, ~1M steps, ~5h expected)"
$PY -u experiments/train_specialist_scale.py --team-size 6 --policy B --seed $PI_B_SEED --device cuda

if [ ! -f "$PI_B_CKPT" ]; then
  echo "FAIL-SAFE TRIGGERED: pi_B_6v6 training returned but $PI_B_CKPT does not exist."
  exit 1
fi
log "pi_B_6v6 terminal checkpoint verified: $PI_B_CKPT"

log "STAGE 1 COMPLETE. Both terminal checkpoints exist. Freeze their sha256 next, then stage 2 (distillation collection) can begin once it exists."
sha256sum "$PI_A_CKPT" "$PI_B_CKPT"
