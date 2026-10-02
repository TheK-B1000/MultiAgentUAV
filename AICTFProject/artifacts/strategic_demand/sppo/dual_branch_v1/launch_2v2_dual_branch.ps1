# DUAL_BRANCH_ROLE_COMPOSITE_V1 — 2v2 joint ATTACK+DEFEND training.
# Smoke A then B (5k), then full 200k A then B. Detached logs under dual_branch_v1/.
$ErrorActionPreference = 'Continue'
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$ld = 'artifacts\strategic_demand\sppo\dual_branch_v1'
New-Item -ItemType Directory -Force -Path $ld | Out-Null
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'

$spec = 'artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json'
$A = 'artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip'
$B = 'artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip'
$shaA = '858805dde3588a686868ae15cfcc4b4c2d63f30d23222dfe6ae421ab7a563b7f'
$shaB = '9ee024ad6356ea79ee765c7d55bba184a510a73fe8a4738a1f7203630636fd0d'
$py = '.venv\Scripts\python.exe'
$teacher = @(
  '--defend-teacher-lambda','0.1','--defend-teacher-lambda-end','0.0',
  '--defend-teacher-decay-start-step','50000','--defend-teacher-decay-end-step','150000',
  '--defend-teacher-cadence','4'
)

function Dual-Common([string]$pol, [string]$ck, [string]$sha) {
  return @(
    'experiments/train_specialist_scale.py','--team-size','2','--policy',$pol,
    '--device','cuda','--entity-repair-enabled','--entity-hidden-dim','32',
    '--role-conditioning-enabled','--role-hold-ticks','8','--role-fixed-for-episode',
    '--role-k-defend','1',
    '--split-attack-defend-enabled',
    '--split-attack-defend-frozen-ckpt',$ck,'--split-attack-defend-frozen-ckpt-sha256',$sha,
    '--load-path',$ck,
    '--dual-branch-role-composite-enabled','--dual-branch-spec',$spec,
    '--run-label-suffix','_dual_branch_v1'
  ) + $teacher
}

function Run-Smoke([string]$pol, [string]$ck, [string]$sha, [int]$seed) {
  Write-Host "=== 2v2 dual-branch smoke $pol seed=$seed ==="
  $argsList = (Dual-Common $pol $ck $sha) + @('--seed', "$seed", '--smoke', '--total-timesteps', '5000')
  $out = "$ld\smoke_2v2_$pol.log"
  $err = "$ld\smoke_2v2_$pol.log.err"
  $p = Start-Process -FilePath $py -ArgumentList $argsList -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput $out -RedirectStandardError $err
  if ($p.ExitCode -ne 0) {
    Write-Host "STOP: smoke $pol failed exit $($p.ExitCode); see $err"
    exit $p.ExitCode
  }
  Write-Host "smoke $pol PASS"
}

function Launch-Full([string]$pol, [string]$ck, [string]$sha, [int]$seed, [string]$eid) {
  $final = "artifacts\scale_2v2_specialists\pi_${pol}_specialist_2v2_dual_branch_v1\ckpts\final_pi_${pol}_specialist_2v2_dual_branch_v1.zip"
  if (Test-Path $final) {
    Write-Host "ALREADY BUILT: $final"
    return $null
  }
  Write-Host "=== 2v2 dual-branch 200k $pol seed=$seed ==="
  $argsList = (Dual-Common $pol $ck $sha) + @(
    '--seed', "$seed",
    '--experiment-id', $eid,
    '--total-timesteps', '200000'
  )
  $out = "$ld\train_2v2_$pol.log"
  $err = "$ld\train_2v2_$pol.log.err"
  $p = Start-Process -FilePath $py -ArgumentList $argsList -WindowStyle Hidden -PassThru `
       -RedirectStandardOutput $out -RedirectStandardError $err
  Write-Host "launched $pol pid=$($p.Id); watch: Get-Content $err -Tail 5 -Wait"
  return $p.Id
}

if (-not ($args -contains '-SkipSmoke')) {
  Run-Smoke 'A' $A $shaA 99903001
  Run-Smoke 'B' $B $shaB 99903002
} else {
  Write-Host 'Skipping smoke (-SkipSmoke)'
}

$pidA = Launch-Full 'A' $A $shaA 26900001 'DUAL_BRANCH_ROLE_COMPOSITE_V1_2V2_A_TRAIN'
$pidB = Launch-Full 'B' $B $shaB 26900002 'DUAL_BRANCH_ROLE_COMPOSITE_V1_2V2_B_TRAIN'
@{
  record_id = 'DUAL_BRANCH_V1_2V2_LAUNCH'
  utc = (Get-Date).ToUniversalTime().ToString('o')
  smoke_skipped = [bool]($args -contains '-SkipSmoke')
  pid_A = $pidA
  pid_B = $pidB
  seeds = @{ A = 26900001; B = 26900002 }
  spec = $spec
} | ConvertTo-Json | Set-Content -Encoding utf8 "$ld\TRAINING_LAUNCH_2V2.json"
Write-Host "wrote $ld\TRAINING_LAUNCH_2V2.json"
