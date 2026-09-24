# SCHOOL_PC_6V6_LOCKED_PIPELINE — unattended grind. No exploratory branches.
# Usage (from AICTFProject):
#   powershell -ExecutionPolicy Bypass -File experiments/run_school_pc_6v6_locked_pipeline.ps1
# Optional:
#   -ParallelRepair   run A and B entity-repair at once (needs 2 GPUs; set CUDA_VISIBLE_DEVICES per job)
#   -SkipRepair       start at split (repair finals already present)
#   -SkipSplit        start at exploratory eval (split final already present)
#   -DryRun           print steps only

param(
  [switch]$ParallelRepair,
  [switch]$SkipRepair,
  [switch]$SkipSplit,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "experiments\train_specialist_scale.py"))) {
  $Root = (Get-Location).Path
}
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { throw "missing venv python: $Py" }
$LogDir = Join-Path $Root "artifacts\strategic_demand\sppo"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# Fail-closed preflight before any long grind.
Write-Host "=== preflight foundation ==="
if (-not $DryRun) {
  & $Py experiments/run_school_pc_6v6_preflight.py --stage foundation
  if ($LASTEXITCODE -ne 0) { throw "foundation preflight FAIL — refusing school-PC grind" }
} else {
  Write-Host "(DryRun) would run: experiments/run_school_pc_6v6_preflight.py --stage foundation"
}

$ABase = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip"
$BBase = "artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip"
$ARepair = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip"
$BRepair = "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip"
$PiD = "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/ckpts/final_pi_A_specialist_6v6_split_defend_k1_v1.zip"
$Spec = "artifacts/strategic_demand/sppo/SCHOOL_PC_6V6_LOCKED_PIPELINE.json"

function Write-Seal($Name, $Path) {
  if (-not (Test-Path $Path)) { throw "missing checkpoint: $Path" }
  $h = (Get-FileHash $Path -Algorithm SHA256).Hash.ToLower()
  $obj = [ordered]@{
    utc = (Get-Date).ToUniversalTime().ToString("o")
    path = $Path
    sha256 = $h
    bytes = (Get-Item $Path).Length
  }
  $out = Join-Path $LogDir $Name
  ($obj | ConvertTo-Json) | Set-Content $out -Encoding utf8
  Write-Host "SEAL $Name sha256=$h"
  return $h
}

function Invoke-Logged($Tag, $ArgList) {
  $log = Join-Path $LogDir "school_pc_6v6_$Tag.log"
  $err = Join-Path $LogDir "school_pc_6v6_$Tag.err"
  Write-Host "=== $Tag ==="
  Write-Host ($ArgList -join " ")
  if ($DryRun) { return }
  $p = Start-Process -FilePath $Py -ArgumentList $ArgList -WorkingDirectory $Root `
    -RedirectStandardOutput $log -RedirectStandardError $err -Wait -PassThru -NoNewWindow
  if ($p.ExitCode -ne 0) {
    Get-Content $err -Tail 40
    throw "$Tag failed exit=$($p.ExitCode)"
  }
}

if (-not $SkipRepair) {
  $argsA = @(
    "experiments/train_specialist_scale.py",
    "--team-size","6","--policy","A","--seed","22500001","--device","cuda",
    "--total-timesteps","1000000",
    "--entity-repair-enabled","--entity-hidden-dim","32",
    "--load-path",$ABase,
    "--run-label-suffix","_c2_entity_repair"
  )
  $argsB = @(
    "experiments/train_specialist_scale.py",
    "--team-size","6","--policy","B","--seed","22500002","--device","cuda",
    "--total-timesteps","1000000",
    "--entity-repair-enabled","--entity-hidden-dim","32",
    "--load-path",$BBase,
    "--run-label-suffix","_c2_entity_repair"
  )

  if ($ParallelRepair) {
    Write-Host "ParallelRepair: launching A and B (requires two GPUs / enough VRAM)."
    if ($DryRun) {
      Write-Host ($argsA -join " ")
      Write-Host ($argsB -join " ")
    } else {
      $pA = Start-Process -FilePath $Py -ArgumentList $argsA -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $LogDir "school_pc_6v6_repair_A.log") `
        -RedirectStandardError (Join-Path $LogDir "school_pc_6v6_repair_A.err") -PassThru -NoNewWindow
      $pB = Start-Process -FilePath $Py -ArgumentList $argsB -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $LogDir "school_pc_6v6_repair_B.log") `
        -RedirectStandardError (Join-Path $LogDir "school_pc_6v6_repair_B.err") -PassThru -NoNewWindow
      Wait-Process -Id $pA.Id, $pB.Id
      if ($pA.ExitCode -ne 0 -or $pB.ExitCode -ne 0) {
        throw "parallel repair failed A=$($pA.ExitCode) B=$($pB.ExitCode)"
      }
    }
  } else {
    Invoke-Logged "repair_A" $argsA
    $null = Write-Seal "SCHOOL_PC_6V6_PI_A_REPAIR_SEAL.json" $ARepair
    Invoke-Logged "repair_B" $argsB
  }
  $null = Write-Seal "SCHOOL_PC_6V6_PI_A_REPAIR_SEAL.json" $ARepair
  $null = Write-Seal "SCHOOL_PC_6V6_PI_B_REPAIR_SEAL.json" $BRepair
} else {
  $null = Write-Seal "SCHOOL_PC_6V6_PI_A_REPAIR_SEAL.json" $ARepair
  $null = Write-Seal "SCHOOL_PC_6V6_PI_B_REPAIR_SEAL.json" $BRepair
}

$ASha = (Get-FileHash $ARepair -Algorithm SHA256).Hash.ToLower()

# Fail-closed split preflight after repair seals, before 200k.
Write-Host "=== preflight split ==="
if (-not $DryRun) {
  & $Py experiments/run_school_pc_6v6_preflight.py --stage split
  if ($LASTEXITCODE -ne 0) { throw "split preflight FAIL — refusing 200k grind" }
} else {
  Write-Host "(DryRun) would run: experiments/run_school_pc_6v6_preflight.py --stage split"
}

if (-not $SkipSplit) {
  $argsSplit = @(
    "experiments/train_specialist_scale.py",
    "--team-size","6","--policy","A","--seed","22600001","--device","cuda",
    "--total-timesteps","200000",
    "--entity-repair-enabled","--entity-hidden-dim","32",
    "--role-conditioning-enabled","--role-fixed-for-episode","--role-k-defend","1",
    "--split-attack-defend-enabled",
    "--split-attack-defend-frozen-ckpt",$ARepair,
    "--split-attack-defend-frozen-ckpt-sha256",$ASha,
    "--load-path",$ARepair,
    "--defend-teacher-lambda","0.1",
    "--defend-teacher-lambda-end","0.0",
    "--defend-teacher-decay-start-step","50000",
    "--defend-teacher-decay-end-step","150000",
    "--defend-teacher-cadence","4",
    "--run-label-suffix","_split_defend_k1_v1"
  )
  Invoke-Logged "split_k1" $argsSplit
  $null = Write-Seal "SCHOOL_PC_6V6_SPLIT_K1_SEAL.json" $PiD
} else {
  $null = Write-Seal "SCHOOL_PC_6V6_SPLIT_K1_SEAL.json" $PiD
}

$argsEval = @(
  "experiments/eval_specialist_crossover_scaled.py",
  "--team-size","6",
  "--spec",$Spec,
  "--pi-a-path",$PiD,
  "--pi-b-path",$BRepair,
  "--frozen-attack-path",$ARepair,
  "--frozen-attack-path-sha256",$ASha,
  "--role-fixed-for-episode","--role-k-defend","1",
  "--seed-base","22700001","--n-seeds","64",
  "--label","EXPLORATORY_6V6_SPLIT_K1",
  "--device","cuda"
)
Invoke-Logged "crossover_exploratory" $argsEval

$ExplResult = Join-Path $Root "artifacts\strategic_demand\EXPLORATORY_6V6_SPLIT_K1_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
if (-not $DryRun) {
  if (-not (Test-Path $ExplResult)) {
    # eval may write under artifacts/strategic_demand/sppo depending on script; search
    $found = Get-ChildItem (Join-Path $Root "artifacts\strategic_demand") -Recurse -Filter "*EXPLORATORY_6V6_SPLIT_K1*RESULT*.json" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) { $ExplResult = $found.FullName }
  }
  Write-Host "Exploratory result path: $ExplResult"
  Write-Host "If PASS (Delta_A>0, Delta_B>0, both LCB95>0), run confirmatory with seed-base 22800001 n=128 label CONFIRMATORY_6V6_SPLIT_K1."
  Write-Host "Do NOT auto-spend confirmatory without reading the exploratory seal."
}

Write-Host "SCHOOL_PC_6V6_LOCKED_PIPELINE finished through exploratory eval."
