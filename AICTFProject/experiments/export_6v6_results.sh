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
# RELATIONSHIP TO export_run_bundle.py (2026-09-12): that Python tool is the GENERAL
# Rule 8 exporter and works at any scale (2v2/4v4/6v6). This script stays because it
# also knows the 6v6 PIPELINE's stage layout -- distillation, crossover eval,
# robustness -- which the general tool does not model, and because run_6v6_pipeline.sh
# and RUNBOOK_6V6.md call it by name. The manifests are compatible: the general tool's
#   python experiments/export_run_bundle.py verify artifacts/6v6_results
# reads the MANIFEST.json written below. Use that to verify a transfer ON THE
# RECEIVING MACHINE, which is where the original loss should have been caught.
#
# Usage:
#   bash experiments/export_6v6_results.sh            # permissive: copy whatever exists
#   bash experiments/export_6v6_results.sh --strict   # VALIDATE: fail loudly if incomplete
#
# --strict is what you run BEFORE declaring a run "transferred successfully". It checks the
# bundle against an expected manifest and exits non-zero naming every missing artifact.
# Added 2026-09-12 after the original exporter silently dropped metrics.csv, episode_rows.csv
# and every intermediate checkpoint -- a loss that was unrecoverable once the source machine
# was gone. A transfer is not complete until --strict passes.
set -e
cd "$(dirname "$0")/.."
SD="artifacts/strategic_demand/sppo"
OUT="artifacts/6v6_results"
STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1
mkdir -p "$OUT/specialists" "$OUT/distillation" "$OUT/crossover_eval" "$OUT/robustness" "$OUT/specs"

log() { echo "[export] $1"; }
copy_if_exists() {
  if [ -e "$1" ]; then cp -r "$1" "$2"; log "included: $1"; else log "not yet produced, skipped: $1"; fi
}

log "=== stage 1: specialists ==="
# NOTE (2026-09-12): the original version of this script exported ONLY the terminal
# checkpoint + training manifest. That lost the training CURVES (metrics.csv /
# episode_rows.csv) and every INTERMEDIATE checkpoint, which are exactly what a
# later training-dynamics diagnosis needs -- and they could not be recovered
# afterwards. Curves and intermediate checkpoints are now exported too.
for P in A B; do
  D="artifacts/scale_6v6_specialists/pi_${P}_specialist_6v6"
  copy_if_exists "$D/ckpts/final_pi_${P}_specialist_6v6.zip" "$OUT/specialists/"
  copy_if_exists "$D/training_manifest.json" "$OUT/specialists/pi_${P}_training_manifest.json"
  # training dynamics -- small text files, decisive for later diagnosis
  copy_if_exists "$D/metrics.csv"        "$OUT/specialists/pi_${P}_metrics.csv"
  copy_if_exists "$D/episode_rows.csv"   "$OUT/specialists/pi_${P}_episode_rows.csv"
  copy_if_exists "$D/result_summary.json" "$OUT/specialists/pi_${P}_result_summary.json"
  copy_if_exists "$D/evaluation_manifest.json" "$OUT/specialists/pi_${P}_evaluation_manifest.json"
  copy_if_exists "$D/run_manifest.json"  "$OUT/specialists/pi_${P}_run_manifest.json"
  # intermediate checkpoints -- required for any checkpoint-over-time trajectory
  mkdir -p "$OUT/specialists/pi_${P}_ckpts"
  for C in "$D"/ckpts/ckpt_pi_${P}_specialist_6v6_*.zip; do
    [ -e "$C" ] && cp "$C" "$OUT/specialists/pi_${P}_ckpts/" && log "included: $C"
  done
done

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


# ---------------------------------------------------------------------------
# MANIFEST + COMPLETENESS VALIDATION
# ---------------------------------------------------------------------------
log "=== writing MANIFEST.json (inventory + checkpoint hashes) ==="
./.venv/Scripts/python.exe - "$OUT" "$STRICT" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

out, strict = Path(sys.argv[1]), sys.argv[2] == "1"

REQUIRED = {
  "specialists/final_pi_A_specialist_6v6.zip":      "stage1 terminal checkpoint pi_A",
  "specialists/final_pi_B_specialist_6v6.zip":      "stage1 terminal checkpoint pi_B",
  "specialists/pi_A_metrics.csv":                   "stage1 training curve pi_A",
  "specialists/pi_B_metrics.csv":                   "stage1 training curve pi_B",
  "specialists/pi_A_episode_rows.csv":              "stage1 episode rows pi_A",
  "specialists/pi_B_episode_rows.csv":              "stage1 episode rows pi_B",
  "specialists/pi_A_training_manifest.json":        "stage1 training manifest pi_A",
  "specialists/pi_B_training_manifest.json":        "stage1 training manifest pi_B",
  "specialists/pi_A_result_summary.json":           "stage1 result summary pi_A",
  "specialists/pi_B_result_summary.json":           "stage1 result summary pi_B",
  "specialists/pi_A_evaluation_manifest.json":      "stage1 evaluation manifest pi_A",
  "specialists/pi_B_evaluation_manifest.json":      "stage1 evaluation manifest pi_B",
  "distillation/TEACHER_DISTILLATION_6V6_DATASET.json": "stage2 dataset record",
  "distillation/RUNG1_6V6_STUDENT_FROZEN.json":     "stage3 frozen student",
  "distillation/RUNG1_6V6_PREFLIGHT.json":          "stage3 preflight",
  "distillation/final_rung1_6v6.pt":                "stage3 Rung-1 checkpoint",
  "crossover_eval/RUNG1_6V6_CROSSOVER_EVAL_RESULT.json": "stage4 sealed result",
  "crossover_eval/rung1_6v6_crossover_eval_rows.csv":    "stage4 raw rows",
}
REQUIRED_DIRS = {
  "specialists/pi_A_ckpts": "stage1 intermediate checkpoints pi_A",
  "specialists/pi_B_ckpts": "stage1 intermediate checkpoints pi_B",
}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()

inventory, missing = {}, []
for rel, desc in REQUIRED.items():
    p = out / rel
    if p.is_file():
        e = {"present": True, "bytes": p.stat().st_size, "desc": desc}
        if p.suffix in (".zip", ".pt"):
            e["sha256"] = sha(p)
        inventory[rel] = e
    else:
        inventory[rel] = {"present": False, "desc": desc}
        missing.append(f"{rel}  ({desc})")

for rel, desc in REQUIRED_DIRS.items():
    d = out / rel
    files = sorted(x.name for x in d.glob("*.zip")) if d.is_dir() else []
    inventory[rel] = {"present": bool(files), "n_files": len(files),
                      "files": files, "desc": desc}
    if not files:
        missing.append(f"{rel}/  ({desc}) -- NO intermediate checkpoints")

(out / "MANIFEST.json").write_text(json.dumps({
    "record": "6v6 export bundle manifest",
    "rule": "a transfer is NOT complete until every required artifact is present. "
            "Checkpoint hashes are recorded so the receiving machine can verify integrity.",
    "complete": not missing,
    "n_missing": len(missing),
    "inventory": inventory,
}, indent=2), encoding="utf-8")

print(f"[export] manifest written: {out/'MANIFEST.json'}")
if missing:
    print(f"[export] INCOMPLETE -- {len(missing)} required artifact(s) missing:")
    for m in missing:
        print(f"[export]    MISSING: {m}")
    if strict:
        print("[export] FAILING: --strict was requested and the bundle is incomplete.")
        print("[export] Do NOT treat this run as transferred. Recover the missing files "
              "from the source machine BEFORE it is wiped.")
        sys.exit(1)
    print("[export] (permissive mode: run with --strict before declaring a transfer complete)")
else:
    print("[export] COMPLETE -- every required artifact present, hashes recorded.")
PYEOF
