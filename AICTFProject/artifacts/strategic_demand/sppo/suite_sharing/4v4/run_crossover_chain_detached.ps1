# Detached 4v4 suite crossover chain: Fully Shared then Share-Encoder.
# Survives Cursor shell aborts (Start-Process, no job control).
$ErrorActionPreference = "Continue"
$Root = "k:\MultiAgentUAV\AICTFProject"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Eval = Join-Path $Root "experiments\eval_suite_sharing_crossover_4v4.py"
$SpDir = Join-Path $Root "artifacts\strategic_demand\sppo"
$Suite = Join-Path $SpDir "suite_sharing\4v4"
$WatchLog = Join-Path $Suite "crossover_chain_watch.log"
$FsLog = Join-Path $Suite "fully_shared_z\crossover_eval.log"
$FsErr = Join-Path $Suite "fully_shared_z\crossover_eval.log.err"
$SeLog = Join-Path $Suite "share_encoder\crossover_eval.log"
$SeErr = Join-Path $Suite "share_encoder\crossover_eval.log.err"
$FsResult = Join-Path $SpDir "SUITE_FULLY_SHARED_Z_4V4_EXPLORATORY_CROSSOVER_EVAL_RESULT.json"
$FsFlag = Join-Path $SpDir "SUITE_FULLY_SHARED_Z_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"
$SeResult = Join-Path $SpDir "SUITE_SHARE_ENCODER_4V4_EXPLORATORY_CROSSOVER_EVAL_RESULT.json"
$SeFlag = Join-Path $SpDir "SUITE_SHARE_ENCODER_4V4_EXPLORATORY_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

function Log([string]$msg) {
  $line = "$(Get-Date -Format o) $msg"
  Add-Content -LiteralPath $WatchLog -Value $line
}

New-Item -ItemType Directory -Force -Path (Split-Path $FsLog), (Split-Path $SeLog) | Out-Null
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = $Root
Set-Location $Root

function EvalAlive([string]$arm) {
  $pat = "*eval_suite_sharing_crossover_4v4.py*$arm*"
  return @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like $pat }).Count -gt 0
}

# --- Fully Shared ---
if ((Test-Path -LiteralPath $FsResult) -or (Test-Path -LiteralPath $FsFlag)) {
  Log "FS already sealed; skipping"
} elseif (EvalAlive "fully_shared") {
  Log "FS already running; waiting"
} else {
  Log "launching Fully Shared"
  "=== DETACHED START $(Get-Date -Format o) ===" | Add-Content -LiteralPath $FsLog
  $fs = Start-Process -FilePath $Py `
    -ArgumentList @("-u", $Eval, "--arm", "fully_shared", "--device", "cuda") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $FsLog `
    -RedirectStandardError $FsErr `
    -PassThru -WindowStyle Hidden
  Log "FS pid=$($fs.Id)"
}

while (-not ((Test-Path -LiteralPath $FsResult) -or (Test-Path -LiteralPath $FsFlag))) {
  if (-not (EvalAlive "fully_shared")) {
    # brief grace for RESULT write
    Start-Sleep -Seconds 10
    if (-not ((Test-Path -LiteralPath $FsResult) -or (Test-Path -LiteralPath $FsFlag))) {
      Log "FS died without RESULT/FLAG — abort"
      exit 2
    }
    break
  }
  Start-Sleep -Seconds 120
}
Log "FS done RESULT=$((Test-Path -LiteralPath $FsResult)) FLAG=$((Test-Path -LiteralPath $FsFlag))"

# --- Share-Encoder ---
if ((Test-Path -LiteralPath $SeResult) -or (Test-Path -LiteralPath $SeFlag)) {
  Log "SE already sealed; done"
  exit 0
}
if (EvalAlive "share_encoder") {
  Log "SE already running; waiting"
} else {
  Log "launching Share-Encoder"
  "=== DETACHED START $(Get-Date -Format o) ===" | Add-Content -LiteralPath $SeLog
  $se = Start-Process -FilePath $Py `
    -ArgumentList @("-u", $Eval, "--arm", "share_encoder", "--device", "cuda") `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $SeLog `
    -RedirectStandardError $SeErr `
    -PassThru -WindowStyle Hidden
  Log "SE pid=$($se.Id)"
}

while (-not ((Test-Path -LiteralPath $SeResult) -or (Test-Path -LiteralPath $SeFlag))) {
  if (-not (EvalAlive "share_encoder")) {
    Start-Sleep -Seconds 10
    if (-not ((Test-Path -LiteralPath $SeResult) -or (Test-Path -LiteralPath $SeFlag))) {
      Log "SE died without RESULT/FLAG — abort"
      exit 2
    }
    break
  }
  Start-Sleep -Seconds 120
}
Log "SE done RESULT=$((Test-Path -LiteralPath $SeResult)) FLAG=$((Test-Path -LiteralPath $SeFlag))"
exit 0
