# Relaunch the 4v4 strategy-conditioned-k runner after the dev k=1 evaluation (pid 44068) has exited and sealed.
# The runner then applies AMENDMENT_1 (shared k), seals the selection, runs the fresh confirmation, writes the readout, stops.
$proj = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $proj
$log = "$proj\strategy_conditioned_k\4v4\sck.log"
function Log($m) { "$((Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')) $m" | Add-Content -Encoding utf8 $log }
$res = "$proj\artifacts\strategic_demand\sppo\SCK_4V4_DEV_K1_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
Log "waiter: runner stopped for AMENDMENT_1; waiting for dev k=1 eval (pid 44068) to exit and seal"
while (Get-Process -Id 44068 -EA SilentlyContinue) { Start-Sleep -Seconds 30 }
if (-not (Test-Path $res)) { Log "STOP: dev k=1 eval exited without a sealed result; rerun the runner by hand (it resumes)"; exit 1 }
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'; $env:PYTHONIOENCODING = 'utf-8'
Log "waiter: dev k=1 sealed; relaunching runner (AMENDMENT_1 shared-k selection)"
Start-Process -FilePath "$proj\.venv\Scripts\python.exe" -ArgumentList 'experiments\strategy_conditioned_k.py','--team-size','4','--run' `
    -WorkingDirectory $proj -WindowStyle Hidden -RedirectStandardOutput "$proj\strategy_conditioned_k\4v4\run2.stdout" `
    -RedirectStandardError "$proj\strategy_conditioned_k\4v4\run2.stderr"
