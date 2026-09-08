Write-Host ""
Get-Content -Raw "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage2\STAGE2_STATUS_BOARD.txt"
Write-Host ""
Write-Host "=== C7 Stage 2 resume log (complete) ===" -ForegroundColor Cyan
Get-Content "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage2_4v4_resume.log" -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "=== VERDICT ===" -ForegroundColor Green
$j = Get-Content "K:\MultiAgentUAV\AICTFProject\artifacts\c7_stage2\C7_STAGE2_VERDICT.json" -Raw | ConvertFrom-Json
Write-Host ("verdict: {0}" -f $j.verdict)
Write-Host ("claim:   {0}" -f $j.the_permitted_claim)
Write-Host ("action:  {0}" -f $j.pre_committed_action_taken.action)
Write-Host ("next:    {0}" -f $j.pre_committed_action_taken.next_main_line)
