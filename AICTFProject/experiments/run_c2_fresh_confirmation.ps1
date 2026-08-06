<#
.SYNOPSIS
    Launch C2 fresh confirmation with exit-code + fault capture.

.DESCRIPTION
    Process-lifecycle wrapper for experiments/run_c2_fresh_confirmation.py.

    Why this exists:
      Prior confirmation attempts created a lock, reached ~1.7 GB RSS, then
      vanished before writing scientific artifacts, leaving a stale lock and
      almost no stdout. That failure window is after lock acquire and before
      the first confirmation episode — not a C2 statistic problem.

    This wrapper:
      - forces unbuffered Python (-u) and PYTHONFAULTHANDLER=1
      - tees all stdout/stderr to confirmation_full.log
      - appends EXIT_CODE=... so Windows 0xC000... native deaths are visible
      - refuses to start a fourth blind relaunch if a live lock is held

    Also watch these runner-written files (independent of terminal capture):
      - C2_CONFIRMATION_PROGRESS.log   (every progress line)
      - C2_CONFIRMATION_PROGRESS.json  (phase / episodes / ETA heartbeat)
      - C2_CONFIRMATION_MANIFEST.json  (startup_phase)

    Do NOT use this to immediately relaunch after another silent death.
    Inspect confirmation_full.log + C2_CONFIRMATION_MANIFEST.json startup_phase
    first.
#>
[CmdletBinding()]
param(
    [int]$Episodes = 30,
    [string]$Device = "",
    [string]$Python = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectRoot "experiments\run_c2_fresh_confirmation.py"))) {
    # Script lives under AICTFProject/experiments → project root is parent.
    $ProjectRoot = $PSScriptRoot | Split-Path -Parent
}

Set-Location $ProjectRoot

$OutDir = Join-Path $ProjectRoot "artifacts\c2_confirmation"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$LockPath = Join-Path $OutDir "C2_CONFIRMATION_RUNNING.lock"
$LogPath = Join-Path $OutDir "confirmation_full.log"
$ProgressLog = Join-Path $OutDir "C2_CONFIRMATION_PROGRESS.log"
$ProgressJson = Join-Path $OutDir "C2_CONFIRMATION_PROGRESS.json"
$ManifestPath = Join-Path $OutDir "C2_CONFIRMATION_MANIFEST.json"
$Runner = Join-Path $ProjectRoot "experiments\run_c2_fresh_confirmation.py"

if (-not $Python) {
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $Python = $VenvPython
    } else {
        $Python = "python"
    }
}

if ((Test-Path $LockPath) -and -not $Force) {
    $lockText = Get-Content -Raw $LockPath
    Write-Host "REFUSING LAUNCH: lock already present at $LockPath"
    Write-Host $lockText
    Write-Host "Inspect startup_phase in $ManifestPath and $LogPath / $ProgressLog before relaunching."
    Write-Host "Pass -Force only after confirming the prior PID is dead and this is an instrumented debug relaunch."
    exit 2
}

$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$ts = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"===== C2 confirmation wrapper start $ts =====" | Tee-Object -FilePath $LogPath -Append
"python=$Python" | Tee-Object -FilePath $LogPath -Append
"runner=$Runner" | Tee-Object -FilePath $LogPath -Append
"episodes=$Episodes device=$Device" | Tee-Object -FilePath $LogPath -Append
"watch_progress_log=$ProgressLog" | Tee-Object -FilePath $LogPath -Append
"watch_progress_json=$ProgressJson" | Tee-Object -FilePath $LogPath -Append
"watch_manifest=$ManifestPath" | Tee-Object -FilePath $LogPath -Append
Write-Host "Progress files:"
Write-Host "  $ProgressLog"
Write-Host "  $ProgressJson"
Write-Host "  $ManifestPath"
Write-Host "  Live tqdm bars print to stderr (also teed into the full log)."
Write-Host "Full tee log: $LogPath"

$argList = @("-u", $Runner, "--episodes", "$Episodes")
if ($Device) {
    $argList += @("--device", $Device)
}

& $Python @argList *>&1 | Tee-Object -FilePath $LogPath -Append
$code = $LASTEXITCODE
if ($null -eq $code) { $code = -999 }

"EXIT_CODE=$code" | Tee-Object -FilePath $LogPath -Append
"===== C2 confirmation wrapper end $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK') =====" | Tee-Object -FilePath $LogPath -Append

Write-Host "EXIT_CODE=$code  log=$LogPath"
exit $code
