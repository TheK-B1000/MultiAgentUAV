$art = "K:\MultiAgentUAV\AICTFProject\artifacts\vgc_specialists\vgc_s_op7_smoke_seed3900007"
Write-Host "S_OP7 smoke (4096 steps) — live" -ForegroundColor Cyan
Write-Host $art
while ($true) {
  Clear-Host
  Write-Host "S_OP7 smoke --smoke-steps 4096" -ForegroundColor Cyan
  Write-Host (Get-Date)
  if (Test-Path "$art\metrics.csv") {
    $m = Import-Csv "$art\metrics.csv"
    $r = $m[-1]
    Write-Host "updates=$($m.Count)  timesteps=$($r.timesteps)  eps=$($r.episodes_completed)  wr=$([math]::Round([double]$r.win_rate,3))  kl=$([math]::Round([double]$r.approx_kl,5))"
  } else { "waiting for metrics..." }
  if (Test-Path "$art\ablation_report.json") { Write-Host "ABLATION REPORT PRESENT — smoke finished" -ForegroundColor Green; break }
  $alive = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -EA SilentlyContinue | Where-Object { $_.CommandLine -match 'S_OP7.*4096|smoke-steps 4096' }
  if (-not $alive) { Write-Host "process gone" -ForegroundColor Yellow; break }
  Start-Sleep 5
}
