# Dual-lane checkpoint pulse: 2v2 robustness sealed + B3 gate + pi_A3 progress.
# Usage (from AICTFProject):  powershell -File tools/status_b3_checkpoint.ps1
$ErrorActionPreference = 'SilentlyContinue'
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host ''
Write-Host '========== CHECKPOINT STATUS ==========' -ForegroundColor Cyan
Write-Host ''
Write-Host '--- 2v2 robustness ---' -ForegroundColor Yellow
$low = @(Get-ChildItem artifacts/strategic_demand/sppo/robustness_eval_rows -Filter 'rung1_2v2*__low.csv')
Write-Host ("  low CSVs: {0}/12" -f $low.Count)
foreach ($f in @(
  'ROBUSTNESS_2V2_RUNG1_RESULT.json',
  'ROBUSTNESS_2V2_HIGH_TIER_RESULT.json',
  'ROBUSTNESS_2V2_DOSE_RESPONSE_LOW_TIER_RESULT.json'
)) {
  $p = "artifacts/strategic_demand/sppo/$f"
  if (Test-Path $p) {
    Write-Host ("  sealed: {0}  ({1})" -f $f, (Get-Item $p).LastWriteTime.ToString('yyyy-MM-dd HH:mm'))
  } else {
    Write-Host ("  MISSING: {0}" -f $f) -ForegroundColor Red
  }
}

Write-Host ''
Write-Host '--- 4v4 B3 gate ---' -ForegroundColor Yellow
$cert = 'artifacts/strategic_demand/sppo/STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json'
if (Test-Path $cert) {
  $j = Get-Content $cert -Raw | ConvertFrom-Json
  Write-Host ("  verdict: {0}" -f $j.VERDICT)
  Write-Host ("  delta_A: {0:+0.0000} [{1:+0.0000}, {2:+0.0000}]" -f $j.PRIMARY.delta_A.mean, $j.PRIMARY.delta_A.lcb95, $j.PRIMARY.delta_A.ucb95)
  Write-Host ("  delta_B: {0:+0.0000} [{1:+0.0000}, {2:+0.0000}]" -f $j.PRIMARY.delta_B.mean, $j.PRIMARY.delta_B.lcb95, $j.PRIMARY.delta_B.ucb95)
} else {
  Write-Host '  cert MISSING' -ForegroundColor Red
}

Write-Host ''
Write-Host '--- pi_A3 training ---' -ForegroundColor Yellow
$m = 'artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3/metrics.csv'
$proc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -match 'train_specialist_scale' -and $_.CommandLine -match 'Python312' -and $_.CommandLine -match '_b3' } |
  Select-Object -First 1
if ($proc) {
  $g = Get-Process -Id $proc.ProcessId
  Write-Host ("  alive pid={0}  wall={1:N1}h  mem={2}MB" -f $proc.ProcessId, ((Get-Date) - $g.StartTime).TotalHours, [math]::Round($g.WS / 1MB))
  Write-Host '  cmd: policy A  seed 16500001  suffix _b3'
} else {
  Write-Host '  NOT RUNNING' -ForegroundColor Red
}
if (Test-Path $m) {
  $lines = Get-Content $m
  $h = $lines[0].Split(',')
  $last = $lines[-1].Split(',')
  $idx = @{}
  for ($i = 0; $i -lt $h.Count; $i++) { $idx[$h[$i]] = $i }
  $ts = [double]$last[$idx['timesteps']]
  $wr = [double]$last[$idx['win_rate']]
  $r50 = [double]$last[$idx['rolling_win_rate_50ep']]
  $r200 = [double]$last[$idx['rolling_win_rate_200ep']]
  $ep = $last[$idx['episodes_completed']]
  Write-Host ("  timesteps: {0:N0} / 1,000,000  ({1:P0})" -f $ts, ($ts / 1e6))
  Write-Host ("  episodes:  {0}" -f $ep)
  Write-Host ("  win_rate:  {0:N3}   rolling_50={1:N3}   rolling_200={2:N3}" -f $wr, $r50, $r200)
  Write-Host ("  metrics mtime: {0}" -f (Get-Item $m).LastWriteTime.ToString('HH:mm:ss'))
}

Write-Host ''
Write-Host '--- locked next ---' -ForegroundColor Yellow
Write-Host '  pi_A3 finish -> pi_B3 train -> sealed crossover'
Write-Host '  PASS only if LCB95(delta_A)>0 AND LCB95(delta_B)>0'
Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
Write-Host ''
