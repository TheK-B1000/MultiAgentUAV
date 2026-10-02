# 6v6 DUAL_BRANCH + STAGE4 frozen pipeline (school PC). Detached; safe to close the window.
#   powershell -ExecutionPolicy Bypass -File <repo>\AICTFProject\6v6\run_dual_branch_6v6.ps1
#   powershell -ExecutionPolicy Bypass -File ...\run_dual_branch_6v6.ps1 -AllowHistoricalTop50
# Historical top-50 is provenance only and skipped by default.
param(
  [switch]$AllowHistoricalTop50
)
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$py = Join-Path $proj '.venv\Scripts\python.exe'
$extra = @()
if ($AllowHistoricalTop50) { $extra += '--allow-historical-top50' }

& $py 'experiments\prepare_stage4_baselines.py' --reserve
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nNot started: Stage4 seed reserve failed."
  exit 1
}

& $py '6v6\run_dual_branch_6v6.py' --check @extra
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nNot started: fix the problems above (usually: git pull + foundation zips)."
  exit 1
}

$p = Start-Process -FilePath $py -ArgumentList (@('6v6\run_dual_branch_6v6.py') + $extra) -WorkingDirectory $proj -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput '6v6\dual_branch_6v6.stdout' -RedirectStandardError '6v6\dual_branch_6v6.stderr'
Write-Host "Started dual-branch + Stage4 6v6 pipeline (pid $($p.Id))."
Write-Host "  Phase1: Ours-Teachers smoke -> 200k A/B -> export -> TECHNICAL SEAL"
Write-Host "  Stage-3: matched-128 (PRIMARY) -> own top-50; historical top-50 skipped unless -AllowHistoricalTop50"
Write-Host "  Stage4: dataset + Share-Encoder / Ours-Shared z+r / Role-only + evals"
Write-Host "  Get-Content $proj\6v6\dual_branch_OVERALL.log.err -Wait -Tail 5"
Write-Host "When finished: zip 6v6\FOR_PROFESSOR in File Explorer (see START_HERE.txt)."
Write-Host "If the PC restarts, run this same command again -- it resumes from STATE.json."
Write-Host "Do not run run_symmetric_6v6.ps1 (defender-only ablation)."
