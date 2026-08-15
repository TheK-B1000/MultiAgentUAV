$root = "K:\MultiAgentUAV\AICTFProject\artifacts\vgc_specialists"
Write-Host "S_OP7 / S_OP8 1M confirmatory training" -ForegroundColor Cyan
while ($true) {
  Clear-Host
  Write-Host "SPECIALIST PILOT — 1M confirmatory" -ForegroundColor Cyan
  Write-Host (Get-Date)
  Write-Host ""
  foreach ($tag in @("vgc_s_op7_seed3900007","vgc_s_op8_seed3900008")) {
    $m = Join-Path $root "$tag\metrics.csv"
    $final = Test-Path (Join-Path $root "$tag\ckpts\final_$tag.zip")
    if (Test-Path $m) {
      $rows = Import-Csv $m
      $r = $rows[-1]
      $pct = [math]::Round(100.0 * [double]$r.timesteps / 1000000.0, 1)
      $bar = ("#" * [int]($pct/2)) + ("-" * (50 - [int]($pct/2)))
      Write-Host "$tag"
      Write-Host ("  [{0}] {1}%  ts={2}  eps={3}  wr={4}  final={5}" -f $bar,$pct,$r.timesteps,$r.episodes_completed,[math]::Round([double]$r.win_rate,3),$final)
    } else { Write-Host "$tag waiting..." }
    Write-Host ""
  }
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
  $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'run_vgc_specialist' -and $_.CommandLine -notmatch 'smoke' }
  if (-not $alive -and (Test-Path "$root\vgc_s_op7_seed3900007\ckpts\final_vgc_s_op7_seed3900007.zip") -and (Test-Path "$root\vgc_s_op8_seed3900008\ckpts\final_vgc_s_op8_seed3900008.zip")) {
    Write-Host "BOTH COMPLETE" -ForegroundColor Green; break
  }
  Start-Sleep 15
}
