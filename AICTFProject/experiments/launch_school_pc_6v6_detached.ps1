# Detached overnight launcher for SCHOOL_PC_6V6_LOCKED_PIPELINE.
# Run once from AICTFProject, then you can close the interactive terminal.
#
#   powershell -ExecutionPolicy Bypass -File experiments/launch_school_pc_6v6_detached.ps1
#
# Optional: -SkipRepair / -SkipSplit / -SkipPreflight (resume only)
#
# Watch while away (any machine with the repo share):
#   Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_OVERALL.err -Wait -Tail 5
#   Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_OVERALL_PROGRESS.json
#   Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_repair_A.err -Wait -Tail 3

param(
  [switch]$SkipRepair,
  [switch]$SkipSplit,
  [switch]$SkipPreflight
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "experiments\run_school_pc_6v6_locked_pipeline.py"))) {
  $Root = (Get-Location).Path
}
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "missing venv: $Py" }

$LogDir = Join-Path $Root "artifacts\strategic_demand\sppo"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$argList = @("experiments/run_school_pc_6v6_locked_pipeline.py")
if ($SkipRepair) { $argList += "--skip-repair" }
if ($SkipSplit) { $argList += "--skip-split" }
if ($SkipPreflight) { $argList += "--skip-preflight" }

$out = Join-Path $LogDir "school_pc_6v6_OVERALL.log"
$err = Join-Path $LogDir "school_pc_6v6_OVERALL.err"
$pidFile = Join-Path $LogDir "school_pc_6v6_OVERALL.pid"

$env:PYTHONUNBUFFERED = "1"
$p = Start-Process -FilePath $Py -ArgumentList $argList -WorkingDirectory $Root `
  -RedirectStandardOutput $out -RedirectStandardError $err `
  -PassThru -WindowStyle Hidden

Set-Content -Path $pidFile -Value $p.Id -Encoding ascii
Write-Host "DETACHED school_pc_6v6 pid=$($p.Id)"
Write-Host "overall bar:  Get-Content '$err' -Wait -Tail 5"
Write-Host "handoff dir:  $(Join-Path $Root '6v6')  (models+results — zip with experiments/pack_6v6_handoff.ps1)"
Write-Host "stage A bar:  Get-Content '$((Join-Path $LogDir 'school_pc_6v6_repair_A.err'))' -Wait -Tail 3"
Write-Host "heartbeat:    $((Join-Path $LogDir 'school_pc_6v6_OVERALL_PROGRESS.json'))"
Write-Host "pid file:     $pidFile"
Write-Host "You can close this window; the job keeps running."
