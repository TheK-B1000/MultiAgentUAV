# 4v4 symmetric-role top-50: after pi_DB finishes, dry-run then run the symmetric evaluation.
# Detached and unattended. Progress: sym_4v4_chain.log next to this file.
# Amendment_1: 200 fresh crossover episodes (all four cells); no A-cell reuse.
# HARD STOP: defender-only chain is exploratory ablation under DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.
$ErrorActionPreference = 'Continue'
$allowAblation = $args -contains '-AllowExploratoryAblation'
$filtered = @($args | Where-Object { $_ -ne '-AllowExploratoryAblation' })
if (-not $allowAblation) {
  Write-Host 'REFUSING: run_4v4_after_training.ps1 is defender-only TOP50 eval chain.'
  Write-Host 'Main candidate is DUAL_BRANCH_ROLE_COMPOSITE_V1_SPEC.json'
  Write-Host 'Re-run with -AllowExploratoryAblation only for intentional ablation.'
  exit 2
}
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$ld = 'artifacts\strategic_demand\sppo\symmetric_role_top50'
$log = "$ld\sym_4v4_chain.log"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }

$trainPid = [int]$filtered[0]
Log "waiting for 4v4 pi_DB training pid=$trainPid"
while (Get-Process -Id $trainPid -EA SilentlyContinue) { Start-Sleep -Seconds 120 }

$ck = 'artifacts/scale_4v4_specialists/pi_B_specialist_4v4_sym_B_top50/ckpts/final_pi_B_specialist_4v4_sym_B_top50.zip'
if (-not (Test-Path $ck)) { Log "STOP: training ended without $ck"; exit 1 }
Log "pi_DB final checkpoint sha256=$((Get-FileHash $ck).Hash.ToLower())"

$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$A = 'artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip'
$B = 'artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip'
$DA = 'artifacts/scale_4v4_specialists/pi_A_specialist_4v4_a4_split_attack_defend_v1/ckpts/final_pi_A_specialist_4v4_a4_split_attack_defend_v1.zip'
$spec = 'artifacts/strategic_demand/sppo/SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json'
$seeds = 'artifacts/strategic_demand/sppo/symmetric_role_top50/4v4_ours_top50_seed_ids.json'
$a = "experiments/eval_specialist_crossover_scaled.py --team-size 4 --spec $spec --post-hoc-ablation-spec $spec " +
     "--seed-list $seeds --registry-experiment-id DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1_EVAL " +
     "--label TOP50_4V4_SYMMETRIC_OURS --device cuda --pi-a-path $DA --pi-b-path $ck --role-fixed-for-episode " +
     "--role-k-defend 2 --frozen-attack-path $A --frozen-attack-path-sha256 94dde69d091a79344db3390d5464dbb4bcf51677a175df93969ab252b2e0f478 " +
     "--frozen-attack-path-b $B --frozen-attack-path-b-sha256 021342c84bbe3aa37d70ed609d898bba3ea85affb53852324f92922c72f70e12"
Log "eval command: $a"

$d = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList "$a --dry-run" -NoNewWindow -Wait -PassThru `
     -RedirectStandardOutput "$ld\sym_4v4_dryrun.log" -RedirectStandardError "$ld\sym_4v4_dryrun.log.err"
if ($d.ExitCode -ne 0) { Log "STOP: symmetric 4v4 dry-run failed (exit $($d.ExitCode)); see sym_4v4_dryrun.log*"; exit 1 }
Log "dry-run PASS"

$p = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList "$a --resume" -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput "$ld\sym_4v4_eval.log" -RedirectStandardError "$ld\sym_4v4_eval.log.err"
Log "symmetric 4v4 eval launched pid=$($p.Id)"
while (Get-Process -Id $p.Id -EA SilentlyContinue) { Start-Sleep -Seconds 120 }
$r = 'artifacts\strategic_demand\sppo\TOP50_4V4_SYMMETRIC_OURS_SPECIALIST_CROSSOVER_EVAL_RESULT.json'
if (Test-Path $r) { Log "SEALED $r"; "DONE" | Set-Content "$ld\SYM_4V4_DONE.txt" } else { Log "STOP: eval exited without a result; see sym_4v4_eval.log.err" }
