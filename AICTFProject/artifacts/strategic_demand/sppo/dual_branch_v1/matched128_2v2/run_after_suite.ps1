# POSTHOC_MATCHED128_2V2_DUAL_BRANCH: post-hoc matched evaluation on the historical 128-seed 2v2 block.
# User 2026-10-02: start as soon as possible (alongside the 2v2 dual-branch suite) and have 4v4 wait.
# 1. hold the 4v4 dual-branch suite (suspend its idle driver; hold_4v4.py)
# 2. dry-run -> 512 episodes (crash-safe, --resume) -> readout (win rate + score margin, paired vs old Ours)
# 3. release 4v4 -- on success AND on every stop, so 4v4 is never left frozen
# Rerunning this script continues where it stopped. Log: matched128_chain.log next to this file.
$ErrorActionPreference = 'Continue'
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$here = 'artifacts\strategic_demand\sppo\dual_branch_v1\matched128_2v2'
$log = "$here\matched128_chain.log"
$py = '.venv\Scripts\python.exe'
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
function Release4v4 { $o = & $py "$here\hold_4v4.py" resume 2>&1; Log "4v4 hold released: $o" }
function Stop($m) { Log "STOP: $m"; Release4v4; exit 1 }

$specPath = 'artifacts\strategic_demand\sppo\DUAL_BRANCH_2V2_POSTHOC_MATCHED128_SPEC.json'
$spec = Get-Content $specPath -Raw -Encoding utf8 | ConvertFrom-Json
$evalArgs = ($spec.LAUNCH.eval -replace '^\S*python\.exe\s+', '')
$readArgs = ($spec.LAUNCH.readout -replace '^\S*python\.exe\s+', '')
$result = 'artifacts\strategic_demand\sppo\POSTHOC_MATCHED128_2V2_DUAL_BRANCH_SPECIALIST_CROSSOVER_EVAL_RESULT.json'
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'

$o = & $py "$here\hold_4v4.py" suspend 2>&1
Log "4v4 held: $o"

if (-not (Test-Path $result)) {
    $dry = $evalArgs -replace '\s--resume$', ''
    $d = Start-Process -FilePath $py -ArgumentList "$dry --dry-run" -NoNewWindow -Wait -PassThru `
         -RedirectStandardOutput "$here\dryrun.log" -RedirectStandardError "$here\dryrun.log.err"
    if ($d.ExitCode -ne 0) { Stop "dry-run failed (exit $($d.ExitCode)); see dryrun.log*" }
    Log "dry-run PASS; launching 512 episodes"
    $p = Start-Process -FilePath $py -ArgumentList $evalArgs -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput "$here\eval.log" -RedirectStandardError "$here\eval.log.err"
    Log "eval pid=$($p.Id)"
    while (Get-Process -Id $p.Id -EA SilentlyContinue) { Start-Sleep -Seconds 120 }
    if (-not (Test-Path $result)) { Stop "eval exited without a sealed result; rerun this script to resume" }
}
Log "SEALED $result"
$r = Start-Process -FilePath $py -ArgumentList $readArgs -NoNewWindow -Wait -PassThru `
     -RedirectStandardOutput "$here\readout.log" -RedirectStandardError "$here\readout.log.err"
if ($r.ExitCode -ne 0) { Stop "readout failed; see readout.log.err" }
Log "readout written: MATCHED128_2V2_READOUT.md / .json"
Release4v4
"DONE" | Set-Content "$here\MATCHED128_DONE.txt"
