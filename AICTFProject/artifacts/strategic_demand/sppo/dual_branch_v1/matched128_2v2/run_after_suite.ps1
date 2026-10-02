# POSTHOC_MATCHED128_2V2_DUAL_BRANCH: post-hoc matched evaluation on the historical 128-seed 2v2 block.
# Starts only after the active 2v2 dual-branch suite (2v2\run_dual_branch_2v2.py) has exited -- never alongside it.
# Then: dry-run -> 512 episodes (crash-safe, --resume) -> readout (win rate + score margin, paired vs old Ours).
# Rerunning this script continues where it stopped. Log: matched128_chain.log next to this file.
$ErrorActionPreference = 'Continue'
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$here = 'artifacts\strategic_demand\sppo\dual_branch_v1\matched128_2v2'
$log = "$here\matched128_chain.log"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
function SuiteRunning { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'run_dual_branch_2v2\.py' }).Count -gt 0 }

$specPath = 'artifacts\strategic_demand\sppo\DUAL_BRANCH_2V2_POSTHOC_MATCHED128_SPEC.json'
$spec = Get-Content $specPath -Raw -Encoding utf8 | ConvertFrom-Json
$evalArgs = ($spec.LAUNCH.eval -replace '^\S*python\.exe\s+', '')
$readArgs = ($spec.LAUNCH.readout -replace '^\S*python\.exe\s+', '')
$result = 'artifacts\strategic_demand\sppo\POSTHOC_MATCHED128_2V2_DUAL_BRANCH_SPECIALIST_CROSSOVER_EVAL_RESULT.json'
$py = '.venv\Scripts\python.exe'
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'

Log "waiting for 2v2\run_dual_branch_2v2.py to exit"
while (SuiteRunning) { Start-Sleep -Seconds 300 }
Log "2v2 dual-branch suite not running"

if (-not (Test-Path $result)) {
    $dry = $evalArgs -replace '\s--resume$', ''
    $d = Start-Process -FilePath $py -ArgumentList "$dry --dry-run" -NoNewWindow -Wait -PassThru `
         -RedirectStandardOutput "$here\dryrun.log" -RedirectStandardError "$here\dryrun.log.err"
    if ($d.ExitCode -ne 0) { Log "STOP: dry-run failed (exit $($d.ExitCode)); see dryrun.log*"; exit 1 }
    Log "dry-run PASS; launching 512 episodes"
    $p = Start-Process -FilePath $py -ArgumentList $evalArgs -WindowStyle Hidden -PassThru `
         -RedirectStandardOutput "$here\eval.log" -RedirectStandardError "$here\eval.log.err"
    Log "eval pid=$($p.Id)"
    while (Get-Process -Id $p.Id -EA SilentlyContinue) { Start-Sleep -Seconds 120 }
    if (-not (Test-Path $result)) { Log "STOP: eval exited without a sealed result; rerun this script to resume"; exit 1 }
}
Log "SEALED $result"
$r = Start-Process -FilePath $py -ArgumentList $readArgs -NoNewWindow -Wait -PassThru `
     -RedirectStandardOutput "$here\readout.log" -RedirectStandardError "$here\readout.log.err"
if ($r.ExitCode -ne 0) { Log "STOP: readout failed; see readout.log.err"; exit 1 }
Log "readout written: MATCHED128_2V2_READOUT.md / .json"
"DONE" | Set-Content "$here\MATCHED128_DONE.txt"
