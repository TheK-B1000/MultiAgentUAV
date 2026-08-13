# Follow Summer 2026 supervisor (read-only).
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not (Test-Path (Join-Path $root "scripts\run_summer_2026.py"))) {
  $root = "K:\MultiAgentUAV\AICTFProject"
}
$state = Join-Path $root "artifacts\summer_2026\state.json"
$log = Join-Path $root "artifacts\summer_2026\logs\supervisor.log"
$launch = Join-Path $root "artifacts\summer_2026\supervisor_launch.json"
Write-Host "=== summer_2026 follow ===" -ForegroundColor Cyan
if (Test-Path $launch) { Get-Content $launch }
if (Test-Path $state) {
  Write-Host "`nstate.json:" -ForegroundColor Yellow
  Get-Content $state
} else { Write-Host "no state.json yet" }
Write-Host "`nsupervisor.log (tail):" -ForegroundColor Yellow
if (Test-Path $log) { Get-Content $log -Tail 30 } else { Write-Host "(missing)" }
$pre = Join-Path $root "artifacts\vgc_diversity\D3_POOL_PREFLIGHT_RESULT.json"
$fp = Join-Path $root "artifacts\vgc_fp\FP_SMOKE_RESULT.json"
Write-Host "`ngates:" -ForegroundColor Yellow
Write-Host "  D3_POOL_PREFLIGHT: $(if (Test-Path $pre) { (Get-Content $pre | ConvertFrom-Json).verdict } else { 'NOT_YET' })"
Write-Host "  FP_SMOKE:          $(if (Test-Path $fp) { (Get-Content $fp | ConvertFrom-Json).verdict } else { 'NOT_YET' })"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
