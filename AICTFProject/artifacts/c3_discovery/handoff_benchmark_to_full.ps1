# C3: wait for 15-ep benchmark (pid 41088) → archive bench artifacts → full scan.
# Exit 0 from benchmark required. Full run uses Stage-3 resume (do NOT pass --no-resume-stage3).

$ErrorActionPreference = "Stop"
Set-Location K:\MultiAgentUAV\AICTFProject
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONFAULTHANDLER = "1"

$BenchPid = 41088
$LogPath = "artifacts\c3_discovery\handoff_benchmark_to_full.log"
$FullLog = "artifacts\c3_discovery\discovery_full_relaunch.log"

function Write-Handoff([string]$msg, [string]$color = "White") {
    $line = "$(Get-Date -Format 'yyyy-MM-ddTHH:mm:ssK') $msg"
    Write-Host $line -ForegroundColor $color
    Add-Content -Path $LogPath -Value $line
}

New-Item -ItemType Directory -Force -Path "artifacts\c3_discovery" | Out-Null
Write-Handoff "Waiting for C3 benchmark PID $BenchPid..." "Cyan"

$proc = Get-Process -Id $BenchPid -ErrorAction SilentlyContinue
if ($null -eq $proc) {
    Write-Handoff "PID $BenchPid already gone before waiter attached. Refusing auto-launch (unknown exit code)." "Red"
    exit 2
}

$proc.WaitForExit()
# ExitCode can be $null for processes not started by this session; treat null as
# unknown and fall back to the Cursor/terminal footer or assume success only if
# C3_BENCHMARK_REPORT.json / COMPLETE progress exists.
$exitCode = $proc.ExitCode
if ($null -eq $exitCode) {
    $progress = Get-Content "artifacts\c3_discovery\C3_DISCOVERY_PROGRESS.json" -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
    if ($progress.phase -eq "COMPLETE" -and (Test-Path "artifacts\c3_discovery\C3_BENCHMARK_REPORT.json")) {
        Write-Handoff "ExitCode unavailable from OS handle; treating COMPLETE+benchmark report as success." "Yellow"
        $exitCode = 0
    } else {
        Write-Handoff "ExitCode unavailable and COMPLETE benchmark artifacts missing. Refusing launch." "Red"
        exit 2
    }
}
Write-Handoff "Benchmark finished. Exit code: $exitCode"

if ($exitCode -ne 0) {
    Write-Handoff "BENCHMARK FAILED. Full C3 will NOT launch." "Red"
    exit $exitCode
}

Write-Handoff "Benchmark passed. Archiving 15-ep resume artifacts..." "Green"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$benchArchive = "artifacts\c3_discovery\benchmark_archive_$stamp"
New-Item -ItemType Directory -Force -Path $benchArchive | Out-Null

$benchmarkArtifacts = @(
    "artifacts\c3_discovery\C3_STAGE1_ANCHORS.jsonl",
    "artifacts\c3_discovery\C3_STAGE1_MANIFEST.json",
    "artifacts\c3_discovery\C3_STAGE3_ANCHOR_RESULTS.jsonl",
    "artifacts\c3_discovery\C3_BENCHMARK_REPORT.json",
    "artifacts\c3_discovery\C3_DISCOVERY.json",
    "artifacts\c3_discovery\C3_PRESSURE_ANCHORS.csv",
    "artifacts\c3_discovery\C3_QUALIFIED_COMMITMENT_FORKS.json",
    "artifacts\c3_discovery\C3_NO_QUALIFIED_STRATEGIC_FORK.json",
    "artifacts\c3_discovery\benchmark_15ep.log"
)

foreach ($f in $benchmarkArtifacts) {
    if (Test-Path $f) {
        Move-Item $f $benchArchive -Force
        Write-Handoff "archived $f -> $benchArchive"
    }
}

Write-Handoff "Launching FULL C3 scan (3 policies x OP6-OP12 x 30 eps, resume ON)..." "Cyan"

& .\.venv\Scripts\python.exe -u `
    experiments/run_c3_decision_proximal_discovery.py `
    --seeds 3200001 3200002 3200003 `
    --opponents OP6 OP7 OP8 OP9 OP10 OP11 OP12 `
    --episodes 30 `
    --stage 3 `
    --device cuda `
    2>&1 | Tee-Object -FilePath $FullLog

$fullExit = $LASTEXITCODE
Write-Handoff ""
Write-Handoff "FULL C3 EXIT CODE: $fullExit"

if ($fullExit -eq 0) {
    Write-Handoff "FULL C3 FINISHED CLEANLY" "Green"
} else {
    Write-Handoff "FULL C3 FAILED. Inspect $FullLog" "Red"
}

exit $fullExit
