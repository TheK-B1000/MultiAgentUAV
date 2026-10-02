# DUAL_BRANCH_2V2_OWN_TOP50 + OWN50 Stage-4 re-eval chain.
#
# After POSTHOC_MATCHED128_2V2_DUAL_BRANCH seals (MATCHED128_DONE.txt):
#   1) Select dual-branch own top-50 (historical rule on matched-128 rows; no new episodes).
#   2) Evaluation-only re-score of already-frozen Stage-4 students on those seeds
#      (OWN50_* labels). Does NOT touch sealed TOP50_* historical-seed Stage-4 results.
#
# Salvage lock: STAGE4_2V2_SALVAGE_AND_OWN50_REEVAL_V1.json
# Log: own_top50.log next to this file.
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$here = 'artifacts\strategic_demand\sppo\dual_branch_v1\matched128_2v2'
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 "$here\own_top50.log" }

Log "waiting for MATCHED128_DONE.txt"
while (-not (Test-Path "$here\MATCHED128_DONE.txt")) { Start-Sleep -Seconds 300 }

if (-not (Test-Path "$here\OWN_TOP50_DONE.txt")) {
    $rows = 'artifacts/strategic_demand/sppo/posthoc_matched128_2v2_dual_branch_specialist_crossover_eval_rows.csv'
    $p = Start-Process -FilePath '.venv\Scripts\python.exe' -NoNewWindow -Wait -PassThru `
         -ArgumentList "experiments/select_own_top50.py --scale 2 --rows $rows --label DUAL_BRANCH_2V2_OWN_TOP50 --out $here" `
         -RedirectStandardOutput "$here\own_top50.out" -RedirectStandardError "$here\own_top50.err"
    if ($p.ExitCode -ne 0) { Log "STOP: own-top50 selection failed; see own_top50.err"; exit 1 }
    Log "written: DUAL_BRANCH_2V2_OWN_TOP50.md / .json / _seed_ids.json"
    "DONE" | Set-Content "$here\OWN_TOP50_DONE.txt"
} else {
    Log "OWN_TOP50_DONE.txt already present; skipping selection"
}

if (Test-Path "$here\OWN50_STAGE4_REEVAL_DONE.txt") {
    Log "OWN50_STAGE4_REEVAL_DONE.txt already present; nothing more to do"
    exit 0
}

# Historical Stage-4 eval may still be running (diagnostic). Re-eval needs the sealed
# student pins in STANDARDIZED_2V2_STAGE4_SHARING_EVAL_SPEC.json — already frozen before
# the live TOP50 eval started. Safe to run as soon as own-top50 exists; GPU contention
# with the live diagnostic is acceptable (do not restart either job).
Log "starting evaluation-only Stage-4 re-score on own-top50 (OWN50_* labels)"
$p2 = Start-Process -FilePath '.venv\Scripts\python.exe' -NoNewWindow -Wait -PassThru `
     -ArgumentList "experiments/reeval_stage4_on_own_top50.py --team-size 2 --device cuda" `
     -RedirectStandardOutput "$here\own50_stage4_reeval_chain.out" `
     -RedirectStandardError "$here\own50_stage4_reeval_chain.err"
if ($p2.ExitCode -ne 0) {
    Log "STOP: OWN50 Stage-4 re-eval failed; see own50_stage4_reeval_chain.err / own50_stage4_reeval_*.err"
    exit 1
}
Log "OWN50 Stage-4 re-eval sealed (historical TOP50_* diagnostic preserved)"
exit 0
