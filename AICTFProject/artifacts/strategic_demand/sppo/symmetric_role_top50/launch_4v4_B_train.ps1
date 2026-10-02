# 4v4 defender-only pi_DB (FROZEN ATTACK + trainable DEFEND).
# Main candidate is now DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json.
# This script is exploratory ablation only. Refuse unless -AllowExploratoryAblation.
$ErrorActionPreference = 'Continue'
if (-not ($args -contains '-AllowExploratoryAblation')) {
  Write-Host 'REFUSING: defender-only path. Main candidate is DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json.'
  Write-Host 'See PRIOR_ROLE_WORK_RECLASSIFIED_AS_EXPLORATORY_ABLATION.json'
  Write-Host 'Re-run with -AllowExploratoryAblation only for the intentional ablation.'
  exit 2
}
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$ld = 'artifacts\strategic_demand\sppo\symmetric_role_top50'
$final = 'artifacts\scale_4v4_specialists\pi_B_specialist_4v4_sym_B_top50\ckpts\final_pi_B_specialist_4v4_sym_B_top50.zip'
if (Test-Path $final) {
  Write-Host "ALREADY BUILT: $final -- not training again."
  exit 0
}
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$B = 'artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip'
$sha = '021342c84bbe3aa37d70ed609d898bba3ea85affb53852324f92922c72f70e12'
$spec = 'artifacts/strategic_demand/sppo/SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json'
$common = @(
  'experiments/train_specialist_scale.py','--team-size','4','--policy','B',
  '--device','cuda','--entity-repair-enabled','--entity-hidden-dim','32',
  '--role-conditioning-enabled','--role-hold-ticks','8','--role-fixed-for-episode','--role-k-defend','2',
  '--split-attack-defend-enabled',
  '--split-attack-defend-frozen-ckpt',$B,'--split-attack-defend-frozen-ckpt-sha256',$sha,
  '--load-path',$B,
  '--defend-teacher-lambda','0.1','--defend-teacher-lambda-end','0.0',
  '--defend-teacher-decay-start-step','50000','--defend-teacher-decay-end-step','150000',
  '--defend-teacher-cadence','4','--run-label-suffix','_sym_B_top50',
  '--symmetric-role-spec',$spec
)

if ($args -contains '-SkipSmoke') {
  Write-Host 'Skipping smoke (-SkipSmoke)'
} else {
  Write-Host '=== 5k smoke ==='
  $smokeArgs = $common + @('--seed','99901004','--smoke','--total-timesteps','5000')
  $s = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList $smokeArgs -NoNewWindow -Wait -PassThru `
       -RedirectStandardOutput "$ld\smoke_4v4_B.log" -RedirectStandardError "$ld\smoke_4v4_B.log.err"
  if ($s.ExitCode -ne 0) { Write-Host "STOP: smoke failed exit $($s.ExitCode)"; exit $s.ExitCode }
  Write-Host 'smoke PASS'
}

Write-Host '=== 200k full train seed 26800001 ==='
$trainArgs = $common + @(
  '--seed','26800001',
  '--experiment-id','SYMMETRIC_ROLE_TOP50_4V4_B_DEFENDER_TRAINING',
  '--total-timesteps','200000'
)
$p = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList $trainArgs -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput "$ld\train_4v4_B.log" -RedirectStandardError "$ld\train_4v4_B.log.err"
Write-Host ("launched pid={0}; watch: Get-Content {1}\train_4v4_B.log.err -Tail 5 -Wait" -f $p.Id, $ld)
