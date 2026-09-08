#!/bin/bash
# Collects everything from a completed (or partially completed) 6v6 pipeline run into ONE
# folder, artifacts/6v6_results/, so it is obvious what to zip and send when the pipeline
# finishes on a school PC. COPIES only -- never touches or moves the originals, so nothing
# about resuming the pipeline (section 8 of RUNBOOK_6V6.md) is affected by running this.
#
# Called automatically by run_6v6_pipeline.sh after EVERY stage (not just at the end), so
# artifacts/6v6_results/ is always current -- checking it mid-run shows real progress, not
# just whatever existed at the last full completion.
#
# Run standalone any time too: it just copies whatever exists so far and skips anything not
# produced yet, printing which stages are/aren't present.
#
# Usage: bash experiments/export_6v6_results.sh
set -e
cd "$(dirname "$0")/.."
SD="artifacts/strategic_demand/sppo"
OUT="artifacts/6v6_results"
mkdir -p "$OUT/specialists" "$OUT/distillation" "$OUT/crossover_eval" "$OUT/robustness" "$OUT/specs"

log() { echo "[export] $1"; }
copy_if_exists() {
  if [ -e "$1" ]; then cp -r "$1" "$2"; log "included: $1"; else log "not yet produced, skipped: $1"; fi
}

log "=== stage 1: specialists ==="
copy_if_exists "artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip" "$OUT/specialists/"
copy_if_exists "artifacts/scale_6v6_specialists/pi_A_specialist_6v6/training_manifest.json" "$OUT/specialists/pi_A_training_manifest.json"
copy_if_exists "artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip" "$OUT/specialists/"
copy_if_exists "artifacts/scale_6v6_specialists/pi_B_specialist_6v6/training_manifest.json" "$OUT/specialists/pi_B_training_manifest.json"

log "=== stage 2: distillation dataset manifest (NOT the raw state shards -- large, "
log "    intermediate-only; only needed to rerun stage 3 without redoing stage 2) ==="
copy_if_exists "$SD/TEACHER_DISTILLATION_6V6_DATASET.json" "$OUT/distillation/"

log "=== stage 3: Rung-1 latent policy ==="
copy_if_exists "$SD/RUNG1_6V6_PREFLIGHT.json" "$OUT/distillation/"
copy_if_exists "$SD/RUNG1_6V6_STUDENT_FROZEN.json" "$OUT/distillation/"
if [ -f "$SD/RUNG1_6V6_STUDENT_FROZEN.json" ]; then
  CKPT=$(./.venv/Scripts/python.exe -c "import json; print(json.load(open('$SD/RUNG1_6V6_STUDENT_FROZEN.json'))['TERMINAL_CHECKPOINT']['path'])" 2>/dev/null || true)
  copy_if_exists "$CKPT" "$OUT/distillation/"
fi
copy_if_exists "$SD/sharing_ladder_6v6/rung1_metrics.csv" "$OUT/distillation/"

log "=== stage 4: sealed crossover evaluation (THE main scientific result) ==="
copy_if_exists "$SD/RUNG1_6V6_CROSSOVER_EVAL_RESULT.json" "$OUT/crossover_eval/"
copy_if_exists "$SD/RUNG1_6V6_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json" "$OUT/crossover_eval/"
copy_if_exists "$SD/rung1_6v6_crossover_eval_rows.csv" "$OUT/crossover_eval/"

log "=== stage 5: deployment robustness (16 CSVs if complete) ==="
if [ -d "$SD/robustness_eval_rows" ]; then
  for f in "$SD"/robustness_eval_rows/rung1_6v6__*.csv; do
    [ -e "$f" ] && cp "$f" "$OUT/robustness/" && log "included: $f"
  done
fi

log "=== reference specs (so the results are interpretable without the full repo) ==="
for f in RUNG1_CONSTRUCTION_6V6_AMENDMENT.json 6V6_PIPELINE_SEED_ALLOCATION.json \
         RUNG1_6V6_CROSSOVER_EVAL_SPEC.json TEACHER_DISTILLATION_6V6_SEED_RETIREMENT_AMENDMENT.json \
         TEACHER_DISTILLATION_6V6_SPEC.json SCALE_6V6_PRODUCTION_SPECIALIST_SEEDS.json \
         STRATEGIC_DEMAND_6v6_GUARD_DISTRIBUTED_V2_CERTIFICATION.json; do
  copy_if_exists "$SD/$f" "$OUT/specs/"
done

echo
echo "=================================================================="
echo "  Everything produced so far is now in: $OUT"
echo "  Zip that one folder and send it -- that is the complete package."
echo "=================================================================="
du -sh "$OUT" 2>/dev/null || true
