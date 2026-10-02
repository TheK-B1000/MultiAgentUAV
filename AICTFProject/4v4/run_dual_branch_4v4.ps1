# 4v4 DUAL_BRANCH_ROLE_COMPOSITE_V1 full suite. Same stages as 2v2. k=ceil(4/3)=2.
# Starts only after 2v2/manifests/phase6_package.json exists.
#   powershell -ExecutionPolicy Bypass -File <repo>\AICTFProject\4v4\run_dual_branch_4v4.ps1
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$py = Join-Path $proj '.venv\Scripts\python.exe'

& $py '4v4\run_dual_branch_4v4.py' --check
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nNot started: fix the problems above."
  exit 1
}

$p = Start-Process -FilePath $py -ArgumentList '4v4\run_dual_branch_4v4.py' -WorkingDirectory $proj -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput '4v4\dual_branch_4v4.stdout' -RedirectStandardError '4v4\dual_branch_4v4.stderr'
Write-Host "Started 4v4 suite waiter (pid $($p.Id))."
Write-Host "  idle until 2v2/manifests/phase6_package.json, then the same pipeline with k=2"
Write-Host "  Get-Content $proj\4v4\dual_branch_4v4.log -Wait -Tail 20"
Write-Host "Zip when finished: $proj\4v4\dual_branch_4v4_full_suite.zip"
