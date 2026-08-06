<#
.SYNOPSIS
    Launch C3 discovery with unbuffered output, tee, and progress-file paths.

.DESCRIPTION
    Process-lifecycle wrapper for experiments/run_c3_decision_proximal_discovery.py.

    Forces python -u + PYTHONUNBUFFERED, tees stdout/stderr to
    artifacts/c3_discovery/discovery_full.log, and prints the durable progress
    paths (C3_DISCOVERY_PROGRESS.log / .json) written by the runner.

    The runner itself hard-blocks until C3_EXECUTION_AUTHORIZATION.json exists.
    This wrapper does not bypass that guard.
#>
[CmdletBinding()]
param(
    [int]$Episodes = 30,
    [int]$Stage = 3,
    [string]$Device = "",
    [string]$Python = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectRoot "experiments\run_c3_decision_proximal_discovery.py"))) {
    $ProjectRoot = $PSScriptRoot | Split-Path -Parent
}

Set-Location $ProjectRoot

$OutDir = Join-Path $ProjectRoot "artifacts\c3_discovery"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$AuthPath = Join-Path $OutDir "C3_EXECUTION_AUTHORIZATION.json"
$LogPath = Join-Path $OutDir "discovery_full.log"
$ProgressLog = Join-Path $OutDir "C3_DISCOVERY_PROGRESS.log"
$ProgressJson = Join-Path $OutDir "C3_DISCOVERY_PROGRESS.json"
$Runner = Join-Path $ProjectRoot "experiments\run_c3_decision_proximal_discovery.py"

if (-not $Python) {
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $Python = $VenvPython
    } else {
        $Python = "python"
    }
}

if (-not (Test-Path $AuthPath) -and -not $Force) {
    Write-Host "REFUSING LAUNCH: C3 is DRAFT / NOT AUTHORIZED."
    Write-Host "Missing: $AuthPath"
    Write-Host "Close the C3 freeze checklist, freeze contracts, write the authorization artifact, then relaunch."
    Write-Host "Pass -Force only to exercise the runner's own SystemExit guard (still will not run science)."
    exit 2
}

$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

$ts = Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"
"===== C3 discovery wrapper start $ts =====" | Tee-Object -FilePath $LogPath -Append
"python=$Python" | Tee-Object -FilePath $LogPath -Append
"runner=$Runner" | Tee-Object -FilePath $LogPath -Append
"episodes=$Episodes stage=$Stage device=$Device" | Tee-Object -FilePath $LogPath -Append
"watch_progress_log=$ProgressLog" | Tee-Object -FilePath $LogPath -Append
"watch_progress_json=$ProgressJson" | Tee-Object -FilePath $LogPath -Append
Write-Host "Progress files (created once the runner passes the auth guard):"
Write-Host "  $ProgressLog"
Write-Host "  $ProgressJson"
Write-Host "Full tee log: $LogPath"

$argList = @("-u", $Runner, "--episodes", "$Episodes", "--stage", "$Stage")
if ($Device) {
    $argList += @("--device", $Device)
}

& $Python @argList *>&1 | Tee-Object -FilePath $LogPath -Append
$code = $LASTEXITCODE
if ($null -eq $code) { $code = -999 }

"EXIT_CODE=$code" | Tee-Object -FilePath $LogPath -Append
"===== C3 discovery wrapper end $(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK') =====" | Tee-Object -FilePath $LogPath -Append

Write-Host "EXIT_CODE=$code  log=$LogPath"
exit $code
