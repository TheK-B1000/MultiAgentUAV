# Wait for 2v2 FROZEN_DATASET, then smoke + check-only + launch 4v4 V2, then cross-scale audit.
# Detached companion to suite distillation recovery at commit 342cbc46.
$ErrorActionPreference = "Continue"
$Root = "K:\MultiAgentUAV\AICTFProject"
$SD = Join-Path $Root "artifacts\strategic_demand\sppo"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Watch = Join-Path $SD "suite_distillation_chain_2v2_then_4v4v2_watch.log"
$Man2 = Join-Path $SD "SUITE_DISTILLATION_2V2_DATASET.json"
$Man4 = Join-Path $SD "SUITE_DISTILLATION_4V4_V2_DATASET.json"
$LaunchPs1 = Join-Path $SD "run_suite_dataset_collection_detached.ps1"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONPATH = $Root
Set-Location $Root
function Log([string]$m) { Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $m" }

Log "chain start (commit $(git -C K:\MultiAgentUAV log -1 --format=%h)); waiting on 2v2 FROZEN_DATASET"
while ($true) {
  if (Test-Path -LiteralPath $Man2) {
    try {
      $st = (Get-Content -Raw -LiteralPath $Man2 | ConvertFrom-Json).status
      if ($st -eq "FROZEN_DATASET") { Log "2v2 sealed: $st"; break }
      Log "2v2 manifest present status=$st; still waiting"
    } catch { Log "2v2 manifest unreadable: $_" }
  }
  Start-Sleep -Seconds 60
}

Log "4v4 V2 smoke start"
& $Py experiments/collect_suite_distillation_states.py --team-size 4 --dataset-tag V2 --device cuda --smoke *>> (Join-Path $SD "suite_distillation_4v4_v2_smoke.log")
$rc = $LASTEXITCODE
Log "4v4 V2 smoke exited $rc"
if ($rc -ne 0) { Log "STOP: smoke failed"; exit 5 }

Log "4v4 V2 check-only"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $LaunchPs1 -TeamSize 4 -DatasetTag V2
if ($LASTEXITCODE -ne 0) { Log "STOP: 4v4 V2 check-only failed"; exit 3 }

Log "4v4 V2 LAUNCH"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $LaunchPs1 -TeamSize 4 -DatasetTag V2 -Launch
$rc = $LASTEXITCODE
Log "4v4 V2 launch wrapper exited $rc"
if (-not (Test-Path -LiteralPath $Man4)) { Log "STOP: no 4v4 V2 manifest"; exit 4 }
$st4 = (Get-Content -Raw -LiteralPath $Man4 | ConvertFrom-Json).status
Log "4v4 V2 manifest status $st4"
if ($st4 -ne "FROZEN_DATASET") { exit 4 }

Log "cross-scale audit"
& $Py experiments/audit_suite_datasets_cross_scale.py --write *>> (Join-Path $SD "suite_datasets_cross_scale_audit.log")
$rc = $LASTEXITCODE
Log "audit exited $rc"
Log "chain DONE"
exit $rc
