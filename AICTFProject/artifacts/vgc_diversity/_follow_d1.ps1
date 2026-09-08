Write-Host ""
Get-Content -Raw "K:\MultiAgentUAV\AICTFProject\artifacts\vgc_diversity\D1_STATUS_BOARD.txt"
Write-Host ""
function Show-D1 {
  Write-Host ("---- {0} ----" -f (Get-Date -Format "HH:mm:ss")) -ForegroundColor Cyan
  $alive = @(Get-CimInstance Win32_Process -EA SilentlyContinue | Where-Object {
    $_.CommandLine -match "run_vgc_diversity.py --condition D1" -and $_.CommandLine -match "Python312"
  })
  Write-Host ("trainers: {0}" -f $alive.Count) -ForegroundColor DarkYellow
  foreach ($s in 3600001,3600002,3600003) {
    $m = "K:\MultiAgentUAV\AICTFProject\artifacts\vgc_diversity\vgc_d1_seed$s\metrics.csv"
    $log = "K:\MultiAgentUAV\AICTFProject\artifacts\vgc_diversity\d1_seed$s.log"
    if (Test-Path $m) {
      $r = (Import-Csv $m)[-1]
      $ts = [int]$r.timesteps
      $pct = [math]::Round(100.0 * $ts / 1000000.0, 1)
      Write-Host ("  seed {0}: ts={1:N0}/1M ({2}%) wr={3:N3} ep={4} update={5}" -f $s, $ts, $pct, [double]$r.win_rate, $r.episodes_completed, $r.update)
    } elseif (Test-Path $log) {
      $tail = Get-Content $log -Tail 2
      Write-Host ("  seed {0}: (no metrics yet) {1}" -f $s, ($tail -join " | "))
    } else {
      Write-Host ("  seed {0}: missing" -f $s) -ForegroundColor Red
    }
  }
  Write-Host ""
  Write-Host "=== latest log lines ===" -ForegroundColor DarkGray
  foreach ($s in 3600001,3600002,3600003) {
    $log = "K:\MultiAgentUAV\AICTFProject\artifacts\vgc_diversity\d1_seed$s.log"
    if (Test-Path $log) {
      Write-Host ("[d1_seed{0}.log]" -f $s) -ForegroundColor Green
      Get-Content $log -Tail 3
    }
  }
}
Show-D1
Write-Host ""
Write-Host "Refreshing every 30s... Ctrl+C to stop." -ForegroundColor DarkCyan
while ($true) {
  Start-Sleep -Seconds 30
  Write-Host ""
  Show-D1
}
