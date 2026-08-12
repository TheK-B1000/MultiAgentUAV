Write-Host ""
Get-Content -Raw "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage2\STAGE2_STATUS_BOARD.txt"
Write-Host ""
Write-Host "=== C7 Stage 2 4v4 resume LIVE — Ctrl+C stops follow only ===" -ForegroundColor Cyan
$log = "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage2_4v4_resume.log"
$shards = "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage2\shards_4v4"
$n = @(Get-ChildItem $shards -Filter "states_*.json.manifest.json" -EA SilentlyContinue).Count
Write-Host ("snapshot: complete_manifests={0}/21" -f $n) -ForegroundColor DarkYellow
if (Test-Path $log) {
  Get-Content $log -ErrorAction SilentlyContinue
  Get-Content $log -Wait -Tail 0
} else {
  Write-Host "resume log missing; following c7_stage2_4v4.log"
  Get-Content "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage2_4v4.log" -Wait -Tail 30
}
