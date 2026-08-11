Write-Host ""
Get-Content -Raw "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage0\SEED2_STATUS_BOARD.txt"
Write-Host ""
Write-Host "=== C7 Stage 0 seed 3300002 LIVE ===" -ForegroundColor Cyan
Get-Content "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage0\seed3300002.log" -Tail 50 -ErrorAction SilentlyContinue
Get-Content "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage0\seed3300002.log" -Wait -Tail 0
