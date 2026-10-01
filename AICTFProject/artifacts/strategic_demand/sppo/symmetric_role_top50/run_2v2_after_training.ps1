# 2v2 symmetric-role top-50: after pi_DB finishes, dry-run then run the symmetric evaluation.
# Detached and unattended (no tool timeout). Progress: sym_2v2_chain.log next to this file.
$ErrorActionPreference = 'Continue'
$root = 'K:\MultiAgentUAV\AICTFProject'
Set-Location $root
$ld = 'artifacts\strategic_demand\sppo\symmetric_role_top50'
$log = "$ld\sym_2v2_chain.log"
function Log($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }

$trainPid = [int]$args[0]
Log "waiting for 2v2 pi_DB training pid=$trainPid"
while (Get-Process -Id $trainPid -EA SilentlyContinue) { Start-Sleep -Seconds 120 }

$ck = 'artifacts/scale_2v2_specialists/pi_B_specialist_2v2_sym_B_top50/ckpts/final_pi_B_specialist_2v2_sym_B_top50.zip'
if (-not (Test-Path $ck)) { Log "STOP: training ended without $ck"; exit 1 }
Log "pi_DB final checkpoint sha256=$((Get-FileHash $ck).Hash.ToLower())"

$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$A = 'artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip'
$B = 'artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip'
$DA = 'artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_split_defend_k1/ckpts/final_pi_A_specialist_2v2_std_split_defend_k1.zip'
$spec = 'artifacts/strategic_demand/sppo/SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json'
$seeds = 'artifacts/strategic_demand/sppo/symmetric_role_top50/2v2_ours_top50_seed_ids.json'
$a = "experiments/eval_specialist_crossover_scaled.py --team-size 2 --spec $spec --post-hoc-ablation-spec $spec " +
     "--seed-list $seeds --registry-experiment-id STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER " +
     "--label TOP50_2V2_SYMMETRIC_OURS --device cuda --pi-a-path $DA --pi-b-path $ck --role-fixed-for-episode " +
     "--role-k-defend 1 --frozen-attack-path $A --frozen-attack-path-sha256 858805dde3588a686868ae15cfcc4b4c2d63f30d23222dfe6ae421ab7a563b7f " +
     "--frozen-attack-path-b $B --frozen-attack-path-b-sha256 9ee024ad6356ea79ee765c7d55bba184a510a73fe8a4738a1f7203630636fd0d"
Log "eval command: $a"

$d = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList "$a --dry-run" -NoNewWindow -Wait -PassThru `
     -RedirectStandardOutput "$ld\sym_2v2_dryrun.log" -RedirectStandardError "$ld\sym_2v2_dryrun.log.err"
if ($d.ExitCode -ne 0) { Log "STOP: symmetric 2v2 dry-run failed (exit $($d.ExitCode)); see sym_2v2_dryrun.log*"; exit 1 }
Log "dry-run PASS"

$p = Start-Process -FilePath .venv\Scripts\python.exe -ArgumentList "$a --resume" -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput "$ld\sym_2v2_eval.log" -RedirectStandardError "$ld\sym_2v2_eval.log.err"
Log "symmetric 2v2 eval launched pid=$($p.Id)"
while (Get-Process -Id $p.Id -EA SilentlyContinue) { Start-Sleep -Seconds 120 }
$r = 'artifacts\strategic_demand\sppo\TOP50_2V2_SYMMETRIC_OURS_SPECIALIST_CROSSOVER_EVAL_RESULT.json'
if (Test-Path $r) { Log "SEALED $r"; "DONE" | Set-Content "$ld\SYM_2V2_DONE.txt" } else { Log "STOP: eval exited without a result; see sym_2v2_eval.log.err" }
