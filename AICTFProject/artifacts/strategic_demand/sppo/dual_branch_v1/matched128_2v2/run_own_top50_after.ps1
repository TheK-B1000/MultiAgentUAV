# DUAL_BRANCH_2V2_OWN_TOP50 + OWN50 Stage-4 re-eval chain.
#
# After POSTHOC_MATCHED128_2V2_DUAL_BRANCH seals (MATCHED128_DONE.txt):
#   1) Select dual-branch own top-50 (historical rule on matched-128 rows; no new episodes).
#   2) Wait for historical-seed Stage-4 Phase-5 evals to finish (Share-Encoder /
#      Fully Shared+z+r / Role-only RESULT files) so GPU is free and the diagnostic
#      package can complete first.
#   3) Evaluation-only re-score of already-frozen Stage-4 students on own-top50
#      (OWN50_* labels). Does NOT touch sealed TOP50_* historical-seed results.
#   4) Write OWN50_STAGE4_REEVAL_DONE.txt and refresh the dual-branch zip if present.
#
# Salvage lock: STAGE4_2V2_SALVAGE_AND_OWN50_REEVAL_V1.json
# Log: own_top50.log next to this file.
$ErrorActionPreference = 'Continue'
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$here = 'artifacts\strategic_demand\sppo\dual_branch_v1\matched128_2v2'
$sd = 'artifacts\strategic_demand\sppo'
$py = '.venv\Scripts\python.exe'
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 "$here\own_top50.log" }

Log "OWN50 chain start (wait: MATCHED128_DONE -> select top50 -> Phase5 historical Stage4 seals -> re-eval)"

Log "waiting for MATCHED128_DONE.txt"
while (-not (Test-Path "$here\MATCHED128_DONE.txt")) { Start-Sleep -Seconds 300 }

if (-not (Test-Path "$here\OWN_TOP50_DONE.txt")) {
    $rows = 'artifacts/strategic_demand/sppo/posthoc_matched128_2v2_dual_branch_specialist_crossover_eval_rows.csv'
    $p = Start-Process -FilePath $py -NoNewWindow -Wait -PassThru `
         -ArgumentList "experiments/select_own_top50.py --scale 2 --rows $rows --label DUAL_BRANCH_2V2_OWN_TOP50 --out $here" `
         -RedirectStandardOutput "$here\own_top50.out" -RedirectStandardError "$here\own_top50.err"
    if ($p.ExitCode -ne 0) { Log "STOP: own-top50 selection failed; see own_top50.err"; exit 1 }
    Log "written: DUAL_BRANCH_2V2_OWN_TOP50.md / .json / _seed_ids.json"
    "DONE $(Get-Date -Format o)" | Set-Content "$here\OWN_TOP50_DONE.txt"
} else {
    Log "OWN_TOP50_DONE.txt already present; skipping selection"
}

# Own-top50 selection is CPU-only and can finish while historical Phase-5 still runs.
# Do NOT start OWN50 GPU re-eval until the three historical Stage-4 RESULT files exist
# (and preferably Phase-6 package), so the live diagnostic finishes cleanly first.
$hist = @(
    "$sd\TOP50_2V2_STAGE4_SHARE_ENCODER_CROSSOVER_EVAL_RESULT.json",
    "$sd\TOP50_2V2_STAGE4_FULLY_SHARED_ZR_CROSSOVER_EVAL_RESULT.json",
    "$sd\TOP50_2V2_STAGE4_ROLE_ONLY_CROSSOVER_EVAL_RESULT.json"
)
Log "waiting for historical Stage-4 Phase-5 RESULT seals (Share-Encoder / Fully Shared+z+r / Role-only)"
while ($true) {
    $missing = @($hist | Where-Object { -not (Test-Path $_) })
    if ($missing.Count -eq 0) { break }
    Log ("still waiting historical Stage4: " + (($missing | ForEach-Object { Split-Path $_ -Leaf }) -join ', '))
    Start-Sleep -Seconds 300
}
Log "historical Stage-4 Phase-5 seals present"

# Prefer Phase-6 package done so zip exists before we refresh it; do not block forever.
$pkg = '2v2\manifests\phase6_package.json'
if (-not (Test-Path $pkg)) { $pkg = '2v2\dual_branch_2v2_STATE.json' }
$deadline = (Get-Date).AddHours(6)
while (-not (Test-Path '2v2\manifests\phase6_package.json') -and (Get-Date) -lt $deadline) {
    Log "waiting for Phase-6 package (optional; will proceed after 6h max)"
    Start-Sleep -Seconds 300
}

if (Test-Path "$here\OWN50_STAGE4_REEVAL_DONE.txt") {
    Log "OWN50_STAGE4_REEVAL_DONE.txt already present; nothing more to do"
    exit 0
}

Log "starting evaluation-only Stage-4 re-score on own-top50 (OWN50_* labels)"
$p2 = Start-Process -FilePath $py -NoNewWindow -Wait -PassThru `
     -ArgumentList "experiments/reeval_stage4_on_own_top50.py --team-size 2 --device cuda" `
     -RedirectStandardOutput "$here\own50_stage4_reeval_chain.out" `
     -RedirectStandardError "$here\own50_stage4_reeval_chain.err"
if ($p2.ExitCode -ne 0) {
    Log "STOP: OWN50 Stage-4 re-eval failed; see own50_stage4_reeval_chain.err / own50_stage4_reeval_*.err"
    exit 1
}
if (-not (Test-Path "$here\OWN50_STAGE4_REEVAL_DONE.txt")) {
    Log "STOP: re-eval exited 0 but OWN50_STAGE4_REEVAL_DONE.txt missing"
    exit 1
}
Log "OWN50 Stage-4 re-eval sealed (historical TOP50_* diagnostic preserved)"

# Copy OWN50 results into the dual-branch bundle folder if present, then refresh zip.
$bundleDir = '2v2\dual_branch_2v2_bundle'
$zip = '2v2\dual_branch_2v2_results.zip'
if (Test-Path $bundleDir) {
    $dst = Join-Path $bundleDir 'own50_stage4'
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Copy-Item "$here\DUAL_BRANCH_2V2_OWN_TOP50*" $dst -Force -ErrorAction SilentlyContinue
    Copy-Item "$sd\OWN50_2V2_STAGE4_*" $dst -Force -ErrorAction SilentlyContinue
    Copy-Item "$sd\own50_2v2_stage4_*" $dst -Force -ErrorAction SilentlyContinue
    Copy-Item "$sd\STANDARDIZED_2V2_STAGE4_OWN50*" $dst -Force -ErrorAction SilentlyContinue
    "OWN50 Stage-4 re-eval packaged $(Get-Date -Format o)`nHistorical TOP50_* Stage-4 remains diagnostic only.`n" |
        Set-Content (Join-Path $dst 'README.txt')
    if (Test-Path $zip) { Remove-Item $zip -Force }
    Compress-Archive -Path (Join-Path $bundleDir '*') -DestinationPath $zip -Force
    Log "refreshed $zip with own50_stage4/"
} else {
    Log "bundle dir missing; OWN50 results sealed under $sd (OWN50_2V2_STAGE4_*)"
}

"ALL_DONE $(Get-Date -Format o)" | Set-Content "$here\2V2_CLEAN_EVAL_PATH_DONE.txt"
Log "DONE -- historical Phase5+6 + matched-128 + own-top50 + OWN50 Stage4 re-eval"
exit 0
