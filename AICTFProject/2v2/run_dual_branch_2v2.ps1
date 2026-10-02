# 2v2 DUAL_BRANCH_ROLE_COMPOSITE_V1 full suite. Same stages as 6v6. k=ceil(2/3)=1.
#   powershell -ExecutionPolicy Bypass -File <repo>\AICTFProject\2v2\run_dual_branch_2v2.ps1
#   powershell -ExecutionPolicy Bypass -File ...\run_dual_branch_2v2.ps1 -AllowHistoricalTop50
# Historical top-50 is provenance only and skipped by default.
# Matched-128 + own top-50 are the Stage-3 path (dual_branch_v1/matched128_2v2/).
param(
  [switch]$AllowHistoricalTop50
)
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$py = Join-Path $proj '.venv\Scripts\python.exe'
$extra = @()
if ($AllowHistoricalTop50) { $extra += '--allow-historical-top50' }

& $py '2v2\run_dual_branch_2v2.py' --check @extra
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nNot started: fix the problems above."
  exit 1
}

$p = Start-Process -FilePath $py -ArgumentList (@('2v2\run_dual_branch_2v2.py') + $extra) -WorkingDirectory $proj -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput '2v2\dual_branch_2v2.stdout' -RedirectStandardError '2v2\dual_branch_2v2.stderr'
Write-Host "Started 2v2 full suite watcher (pid $($p.Id))."
Write-Host "  Stage-3: matched-128 (PRIMARY) + own top-50; historical top-50 skipped unless -AllowHistoricalTop50"
Write-Host "  Get-Content $proj\2v2\dual_branch_2v2.log -Wait -Tail 20"
Write-Host "Zip when finished: $proj\2v2\dual_branch_2v2_full_suite.zip"
Write-Host "If the PC restarts, run this same command again -- it continues where it stopped."
