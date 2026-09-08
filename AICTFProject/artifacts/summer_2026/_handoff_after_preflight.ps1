# Watch for D3 preflight completion, then hand off to D3×3 WITHOUT a dual-supervisor race.
#
# Safe sequence (no second supervisor while ownership is valid):
#   preflight RESULT appears
#        ↓
#   stop old supervisor IMMEDIATELY (before in-memory FP_SMOKE can launch)
#        ↓
#   wait until supervisor PID dead + owned children gone + lock released
#        ↓
#   write state = DIVERSITY_TRAIN
#        ↓
#   acquire lock / relaunch single supervisor
$ErrorActionPreference = "Continue"
$root = "K:\MultiAgentUAV\AICTFProject"
$pre = Join-Path $root "artifacts\vgc_diversity\D3_POOL_PREFLIGHT_RESULT.json"
$state = Join-Path $root "artifacts\summer_2026\state.json"
$lock = Join-Path $root "artifacts\summer_2026\supervisor.lock"
$launch = Join-Path $root "artifacts\summer_2026\supervisor_launch.json"
$log = Join-Path $root "artifacts\summer_2026\logs\supervisor_handoff.log"
$py = Join-Path $root ".venv\Scripts\python.exe"

function Write-Log($m) {
  $line = "$(Get-Date -Format o)  $m"
  Add-Content -Path $log -Value $line -Encoding utf8
  Write-Host $line
}

function Get-OwnedPids {
  $pids = [System.Collections.Generic.List[int]]::new()
  if (Test-Path $launch) {
    try { $pids.Add([int]((Get-Content $launch -Raw | ConvertFrom-Json).pid)) } catch {}
  }
  if (Test-Path $lock) {
    try { $pids.Add([int]((Get-Content $lock -Raw).Trim())) } catch {}
  }
  return @($pids | Where-Object { $_ -gt 0 } | Select-Object -Unique)
}

function Get-DescendantPids([int]$RootPid) {
  $out = [System.Collections.Generic.List[int]]::new()
  $queue = [System.Collections.Generic.Queue[int]]::new()
  $queue.Enqueue($RootPid)
  while ($queue.Count -gt 0) {
    $cur = $queue.Dequeue()
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$cur" -EA SilentlyContinue | ForEach-Object {
      $cid = [int]$_.ProcessId
      if (-not $out.Contains($cid)) {
        $out.Add($cid)
        $queue.Enqueue($cid)
      }
    }
  }
  return @($out)
}

function Stop-OwnedTree {
  $roots = Get-OwnedPids
  $all = [System.Collections.Generic.List[int]]::new()
  foreach ($r in $roots) {
    if (-not $all.Contains($r)) { $all.Add($r) }
    foreach ($c in (Get-DescendantPids $r)) {
      if (-not $all.Contains($c)) { $all.Add($c) }
    }
  }
  # Also stop any known FP smoke / preflight orphans under AICTFProject if still
  # parented to a dead supervisor (best-effort; do not kill unrelated GPU jobs).
  foreach ($pid in ($all | Sort-Object -Descending)) {
    $proc = Get-Process -Id $pid -EA SilentlyContinue
    if ($proc) {
      Write-Log "stopping pid=$pid name=$($proc.ProcessName)"
      Stop-Process -Id $pid -Force -EA SilentlyContinue
    }
  }
  return @($all)
}

function Wait-CleanBoundary($priorPids, [int]$timeoutSec = 180) {
  $deadline = (Get-Date).AddSeconds($timeoutSec)
  while ((Get-Date) -lt $deadline) {
    $alive = @()
    foreach ($pid in $priorPids) {
      if (Get-Process -Id $pid -EA SilentlyContinue) { $alive += $pid }
    }
    $lockHeld = $false
    if (Test-Path $lock) {
      try {
        $lp = [int]((Get-Content $lock -Raw).Trim())
        if (Get-Process -Id $lp -EA SilentlyContinue) { $lockHeld = $true }
      } catch { $lockHeld = $true }
    }
    if ($alive.Count -eq 0 -and -not $lockHeld) {
      Remove-Item $lock -Force -EA SilentlyContinue
      Write-Log "clean stop boundary: no owned PIDs, lock free"
      return $true
    }
    Write-Log "waiting for stop boundary alive=[$($alive -join ',')] lockHeld=$lockHeld"
    Start-Sleep 2
  }
  Write-Log "TIMEOUT waiting for clean stop boundary"
  return $false
}

Write-Log "handoff watcher started; waiting for D3_POOL_PREFLIGHT_RESULT.json"
while (-not (Test-Path $pre)) { Start-Sleep 10 }

# RESULT on disk means the preflight subprocess finished writing. Stop the OLD
# supervisor immediately so its in-memory control flow cannot start FP_SMOKE.
Write-Log "preflight result present — stopping old supervisor BEFORE any FP launch window"
$owned = Stop-OwnedTree
Start-Sleep 2
# Second pass in case children outlived the parent briefly
$owned2 = Stop-OwnedTree
$allOwned = @($owned + $owned2 | Select-Object -Unique)

if (-not (Wait-CleanBoundary $allOwned)) {
  Write-Log "ABORT: refusing to relaunch while ownership unclean"
  exit 2
}

$verdict = (Get-Content $pre -Raw | ConvertFrom-Json).verdict
Write-Log "verdict=$verdict"

& $py -u (Join-Path $root "experiments\analyze_d3_preflight_receipt.py") `
  | Tee-Object -FilePath (Join-Path $root "artifacts\summer_2026\logs\four_questions.log")

if (-not $verdict.EndsWith("PASS")) {
  Write-Log "PREFLIGHT FAIL — not launching D3"
  $fail = @{
    state = "STOPPED_ERROR"
    history = @(@{ from = "PREFLIGHT"; to = "STOPPED_ERROR"; utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"); reason = "D3_POOL_PREFLIGHT_FAIL" })
    gates = @{ MIXED_SAMPLING_PASS = $verdict; FP_SMOKE = "INCOMPLETE_PROBE_ONLY_NOT_PASS" }
  }
  $tmp = Join-Path $root "artifacts\summer_2026\state.tmp"
  ($fail | ConvertTo-Json -Depth 6) | Set-Content $tmp -Encoding utf8
  Move-Item -Force $tmp $state
  exit 1
}

# Acquire lock BEFORE writing resume state / launching (single-owner handoff).
$myPid = $PID
Set-Content -Path $lock -Value $myPid -Encoding utf8
Write-Log "handoff watcher acquired lock pid=$myPid"

$board = Get-Content (Join-Path $root "artifacts\vgc_fp\FP_PROBE_2026-08-13.json") -Raw | ConvertFrom-Json
$new = @{
  state = "DIVERSITY_TRAIN"
  history = @(
    @{ from = "PREFLIGHT"; to = "DIVERSITY_TRAIN"; utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ");
       reason = "clean handoff: D3 critical path; FP full smoke deferred (probe incomplete, not PASS)" }
  )
  gates = @{
    MIXED_SAMPLING_PASS = $verdict
    FP_SNAPSHOT_FORMAT = $board.status_board.FP_SNAPSHOT_FORMAT
    PPO_AS_OPPONENT_SEAM = $board.status_board.PPO_AS_OPPONENT_SEAM
    FP_FULL_SMOKE = $board.status_board.FP_FULL_SMOKE
    FP_ABLATION_REPORT = $board.status_board.FP_ABLATION_REPORT
    FP_SCIENTIFIC_GATE = $board.status_board.FP_SCIENTIFIC_GATE
    FP_SMOKE = "INCOMPLETE_PROBE_ONLY_NOT_PASS"
    Q3_OPPONENT_BEFORE_FIRST_ROLLOUT = "CODE_INVARIANT_PASS"
  }
}
$tmp = Join-Path $root "artifacts\summer_2026\state.tmp"
($new | ConvertTo-Json -Depth 6) | Set-Content $tmp -Encoding utf8
Move-Item -Force $tmp $state
Write-Log "state forced to DIVERSITY_TRAIN under handoff lock"

# Release handoff lock so the new supervisor can take ownership via its Lock().
Remove-Item $lock -Force -EA SilentlyContinue
Write-Log "handoff lock released; launching single supervisor"

Add-Content (Join-Path $root "artifacts\summer_2026\logs\supervisor.log") `
  "`n==== relaunch after clean preflight handoff $(Get-Date -Format o) ====`n"

$p = Start-Process -FilePath $py `
  -ArgumentList "-u","scripts\run_summer_2026.py" `
  -WorkingDirectory $root `
  -RedirectStandardOutput (Join-Path $root "artifacts\summer_2026\logs\supervisor_relaunch.out.log") `
  -RedirectStandardError (Join-Path $root "artifacts\summer_2026\logs\supervisor_relaunch.err.log") `
  -PassThru -WindowStyle Hidden

@{ pid = $p.Id; launched_utc = (Get-Date).ToUniversalTime().ToString("o");
   cmd = "python -u scripts/run_summer_2026.py";
   note = "post-preflight D3 path after clean stop boundary" } |
  ConvertTo-Json | Set-Content $launch -Encoding utf8

# Confirm new supervisor acquired its own lock and no dual owners
Start-Sleep 3
$lockPid = $null
if (Test-Path $lock) { try { $lockPid = [int]((Get-Content $lock -Raw).Trim()) } catch {} }
if ($lockPid -ne $p.Id) {
  Write-Log "WARNING: expected lock pid=$($p.Id) got=$lockPid"
} else {
  Write-Log "new supervisor owns lock pid=$($p.Id)"
}
Write-Log "relaunch complete"
exit 0
