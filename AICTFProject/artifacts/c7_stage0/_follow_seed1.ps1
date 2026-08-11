$ErrorActionPreference = "Continue"
Write-Host ""
Get-Content -Raw "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage0\SEED1_STATUS_BOARD.txt"
Write-Host ""
Write-Host "=== C7 Stage 0 seed 3300001 LIVE — Ctrl+C stops follow only ===" -ForegroundColor Cyan
# Show current log immediately, then follow
Get-Content "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage0\seed3300001.log" -Tail 40 -ErrorAction SilentlyContinue
Get-Content "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage0\seed3300001.log" -Wait -Tail 0
