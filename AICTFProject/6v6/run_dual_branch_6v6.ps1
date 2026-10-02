# 6v6 DUAL_BRANCH_ROLE_COMPOSITE_V1 (school PC). Detached; safe to close the window.
#   powershell -ExecutionPolicy Bypass -File <repo>\AICTFProject\6v6\run_dual_branch_6v6.ps1
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$py = Join-Path $proj '.venv\Scripts\python.exe'

& $py '6v6\run_dual_branch_6v6.py' --check
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nNot started: fix the problems above (usually: git pull + foundation zips)."
  exit 1
}

$p = Start-Process -FilePath $py -ArgumentList '6v6\run_dual_branch_6v6.py' -WorkingDirectory $proj -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput '6v6\dual_branch_6v6.stdout' -RedirectStandardError '6v6\dual_branch_6v6.stderr'
Write-Host "Started dual-branch 6v6 suite (pid $($p.Id))."
Write-Host "  smoke A/B -> 200k A -> 200k B -> export ATTACK branches -> top-50 crossover -> zip"
Write-Host "  Get-Content $proj\6v6\dual_branch_6v6.log -Wait -Tail 20"
Write-Host "Zip when finished: $proj\6v6\dual_branch_6v6_results.zip"
Write-Host "If the PC restarts, run this same command again -- it continues where it stopped."
Write-Host "Do not run run_symmetric_6v6.ps1 (defender-only ablation)."
