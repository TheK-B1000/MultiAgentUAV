# DUAL_BRANCH_2V2_OWN_TOP50: after POSTHOC_MATCHED128_2V2_DUAL_BRANCH seals (MATCHED128_DONE.txt), apply the
# HISTORICAL top-50 rule (experiments/select_own_top50.py, verified to reproduce the old list) to the
# dual-branch 128-seed rows and read the crossover on them. No new episodes. Best-case capability, not an
# unbiased estimate. Log: own_top50.log next to this file.
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$here = 'artifacts\strategic_demand\sppo\dual_branch_v1\matched128_2v2'
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 "$here\own_top50.log" }
Log "waiting for MATCHED128_DONE.txt"
while (-not (Test-Path "$here\MATCHED128_DONE.txt")) { Start-Sleep -Seconds 300 }
$rows = 'artifacts/strategic_demand/sppo/posthoc_matched128_2v2_dual_branch_specialist_crossover_eval_rows.csv'
$p = Start-Process -FilePath '.venv\Scripts\python.exe' -NoNewWindow -Wait -PassThru `
     -ArgumentList "experiments/select_own_top50.py --rows $rows --label DUAL_BRANCH_2V2_OWN_TOP50 --out $here" `
     -RedirectStandardOutput "$here\own_top50.out" -RedirectStandardError "$here\own_top50.err"
if ($p.ExitCode -ne 0) { Log "STOP: own-top50 selection failed; see own_top50.err"; exit 1 }
Log "written: DUAL_BRANCH_2V2_OWN_TOP50.md / .json / _seed_ids.json"
"DONE" | Set-Content "$here\OWN_TOP50_DONE.txt"
