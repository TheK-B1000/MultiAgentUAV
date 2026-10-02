# 2v2 DUAL_BRANCH_ROLE_COMPOSITE_V1 full suite. Same stages as 6v6. k=ceil(2/3)=1.
#   powershell -ExecutionPolicy Bypass -File <repo>\AICTFProject\2v2\run_dual_branch_2v2.ps1
# If the sequential 2v2 chain is already training, the python orchestrator waits
# for SYM_DUAL_BRANCH_2V2_TRAIN_DONE.txt and does not start a second A/B run.
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$py = Join-Path $proj '.venv\Scripts\python.exe'

& $py '2v2\run_dual_branch_2v2.py' --check
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nNot started: fix the problems above."
  exit 1
}

$p = Start-Process -FilePath $py -ArgumentList '2v2\run_dual_branch_2v2.py' -WorkingDirectory $proj -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput '2v2\dual_branch_2v2.stdout' -RedirectStandardError '2v2\dual_branch_2v2.stderr'
Write-Host "Started 2v2 full suite watcher (pid $($p.Id))."
Write-Host "  waits for the live Phase 1 chain, then export -> top-50 diagnostic -> Stage 4 -> zip"
Write-Host "  Get-Content $proj\2v2\dual_branch_2v2.log -Wait -Tail 20"
Write-Host "Zip when finished: $proj\2v2\dual_branch_2v2_full_suite.zip"
Write-Host "If the PC restarts, run this same command again -- it continues where it stopped."
