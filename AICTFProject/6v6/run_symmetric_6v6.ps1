# Start the 6v6 symmetric-role diagnostic detached (safe to close this window or log off the console).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File <repo>\AICTFProject\6v6\run_symmetric_6v6.ps1
# HARD STOP: defender-only (frozen ATTACK) pipeline is exploratory ablation under
# artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json.
# Pass -AllowExploratoryAblation to run anyway.
if (-not ($args -contains '-AllowExploratoryAblation')) {
  Write-Host 'REFUSING: 6v6/run_symmetric_6v6.ps1 is the defender-only symmetric diagnostic.'
  Write-Host 'Main candidate is DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json (dual-branch ATTACK+DEFEND).'
  Write-Host 'Re-run with -AllowExploratoryAblation only for intentional ablation.'
  exit 2
}
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$py = Join-Path $proj '.venv\Scripts\python.exe'

& $py '6v6\run_symmetric_6v6.py' --check
if ($LASTEXITCODE -ne 0) { Write-Host "`nNot started: fix the problems above (usually: git pull)."; exit 1 }

$p = Start-Process -FilePath $py -ArgumentList '6v6\run_symmetric_6v6.py','--allow-exploratory-ablation' -WorkingDirectory $proj -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput '6v6\symmetric_6v6.stdout' -RedirectStandardError '6v6\symmetric_6v6.stderr'
Write-Host "Started (pid $($p.Id)). Progress:"
Write-Host "  Get-Content $proj\6v6\symmetric_6v6.log -Wait -Tail 20"
Write-Host "Phase 1 (core) zip, ready first:      $proj\6v6\symmetric_results.zip"
Write-Host "Phase 2 (baselines) zip, at the end: $proj\6v6\symmetric_baselines.zip"
Write-Host "If the PC restarts, run this same command again -- it continues where it stopped."
