# DUAL_BRANCH_ROLE_COMPOSITE_V1 — 2v2: smoke A/B then SEQUENTIAL 200k A then 200k B.
# One GPU: never train A and B in parallel. Logs under dual_branch_v1/.
# Do not steer on intermediate win rates; finish both 200k unless implementation fails.
$ErrorActionPreference = 'Continue'
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$ld = 'artifacts\strategic_demand\sppo\dual_branch_v1'
New-Item -ItemType Directory -Force -Path $ld | Out-Null
$log = Join-Path $ld 'chain_2v2.log'
function Log($m) { "$(Get-Date -Format s) $m" | Tee-Object -FilePath $log -Append }

$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$env:PYTHONUNBUFFERED = '1'

$spec = 'artifacts/strategic_demand/sppo/DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json'
$A = 'artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip'
$B = 'artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip'
$shaA = '858805dde3588a686868ae15cfcc4b4c2d63f30d23222dfe6ae421ab7a563b7f'
$shaB = '9ee024ad6356ea79ee765c7d55bba184a510a73fe8a4738a1f7203630636fd0d'
$py = Join-Path $root '.venv\Scripts\python.exe'
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

function FinalPath([string]$pol) {
  return "artifacts\scale_2v2_specialists\pi_${pol}_specialist_2v2_dual_branch_v1\ckpts\final_pi_${pol}_specialist_2v2_dual_branch_v1.zip"
}

function Run-Smoke([string]$pol, [string]$ck, [string]$sha, [int]$seed) {
  Log "=== smoke $pol seed=$seed ==="
  $argsList = (Dual-Common $pol $ck $sha) + @('--seed', "$seed", '--smoke', '--total-timesteps', '5000')
  $p = Start-Process -FilePath $py -ArgumentList $argsList -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$ld\smoke_2v2_$pol.log" -RedirectStandardError "$ld\smoke_2v2_$pol.log.err"
  if ($p.ExitCode -ne 0) {
    Log "STOP: smoke $pol failed exit $($p.ExitCode)"
    exit $p.ExitCode
  }
  Log "smoke $pol PASS"
}

function Run-Full([string]$pol, [string]$ck, [string]$sha, [int]$seed, [string]$eid) {
  $final = FinalPath $pol
  if (Test-Path $final) {
    Log "ALREADY BUILT: $final sha=$((Get-FileHash $final).Hash.ToLower())"
    return
  }
  Log "=== 200k $pol seed=$seed eid=$eid ==="
  $argsList = (Dual-Common $pol $ck $sha) + @(
    '--seed', "$seed",
    '--experiment-id', $eid,
    '--total-timesteps', '200000'
  )
  $p = Start-Process -FilePath $py -ArgumentList $argsList -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$ld\train_2v2_$pol.log" -RedirectStandardError "$ld\train_2v2_$pol.log.err"
  if ($p.ExitCode -ne 0 -or -not (Test-Path $final)) {
    Log "STOP: 200k $pol ended without $final (exit $($p.ExitCode))"
    exit 1
  }
  Log "DONE $pol sha=$((Get-FileHash $final).Hash.ToLower())"
}

# Clear prior stop marker for this authorized relaunch
if (Test-Path "$ld\STOP_2V2.json") {
  Rename-Item "$ld\STOP_2V2.json" "STOP_2V2_superseded_$(Get-Date -Format yyyyMMdd_HHmmss).json" -Force
}

Log "IMPLEMENTATION_GATE: re-verified tests green before this chain (30 passed)"
if (-not ($args -contains '-SkipSmoke')) {
  Run-Smoke 'A' $A $shaA 99903001
  Run-Smoke 'B' $B $shaB 99903002
} else {
  Log 'Skipping smoke (-SkipSmoke)'
}

Run-Full 'A' $A $shaA 26900001 'DUAL_BRANCH_ROLE_COMPOSITE_V1_2V2_A_TRAIN'
Run-Full 'B' $B $shaB 26900002 'DUAL_BRANCH_ROLE_COMPOSITE_V1_2V2_B_TRAIN'

@{
  record_id = 'DUAL_BRANCH_V1_2V2_TRAINING_SEALED'
  utc = (Get-Date).ToUniversalTime().ToString('o')
  A = @{ path = (FinalPath 'A'); sha256 = (Get-FileHash (FinalPath 'A')).Hash.ToLower() }
  B = @{ path = (FinalPath 'B'); sha256 = (Get-FileHash (FinalPath 'B')).Hash.ToLower() }
  next = 'behavior checks then fresh diagnostic crossover (not asymmetric top-50)'
} | ConvertTo-Json -Depth 4 | Set-Content -Encoding utf8 "$ld\TRAINING_SEALED_2V2.json"
'DONE' | Set-Content "$ld\SYM_DUAL_BRANCH_2V2_TRAIN_DONE.txt"
Log "BOTH 2v2 dual-branch 200k sealed. See TRAINING_SEALED_2V2.json"
