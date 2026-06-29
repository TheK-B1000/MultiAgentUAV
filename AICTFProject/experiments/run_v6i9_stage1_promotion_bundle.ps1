<#
.SYNOPSIS
  V6I9 Stage 1 promotion bundle: competence gate, map-awareness eval, optional Stage 2 launch.

.DESCRIPTION
  1. Run experiments/gate_v6i9_map_aware.py (Gates 1-4, --allow-saturated-wr).
  2. Run experiments/eval_v6i9_map_awareness.py (V6I8 vs V6I9 promotion eval).
  3. If promotion verdict starts with "READY FOR STAGE B", launch Stage 2 training.

  -SkipStage2: run steps 1-2 only (eval bundle, no training).
  -ForceStage2: launch Stage 2 even if promotion verdict is not READY (manual override).

.EXAMPLE
  Full pipeline including Stage 2 when promotion passes:
  .\experiments\run_v6i9_stage1_promotion_bundle.ps1

.EXAMPLE
  Eval only (no Stage 2 training):
  .\experiments\run_v6i9_stage1_promotion_bundle.ps1 -SkipStage2

.EXAMPLE
  Eval plus manual Stage 2 launch decision later:
  .\experiments\run_v6i9_stage1_promotion_bundle.ps1 -SkipStage2
  # then run the printed train_ppo command yourself

.EXAMPLE
  Dry run:
  .\experiments\run_v6i9_stage1_promotion_bundle.ps1 -DryRun
#>
param(
    [string]$Baseline = "checkpoints/2v2/ckpt_v6i8-adapter-balanced-hardpool-refactor-r1-seed1_2v2_750000.zip",
    [string]$Candidate = "checkpoints/2v2/ckpt_v6i9-mapaware-generalist-hardpool-refactor-r1-seed1_2v2_1000000.zip",
    [string[]]$Maps = @("map_a_open", "map_b_split_lane"),
    [string[]]$Opponents = @("OP8", "OP9", "OP10"),
    [int]$GateEpisodes = 5,
    [int]$PromoEpisodes = 20,
    [int]$PromoSeedStart = 7000,
    [string]$PromoOutDir = "artifacts/v6i9_map_awareness_refactor_r1",
    [string]$Device = "cuda",
    [string]$Stage2Preset = "v6i9_mapaware_repertoire_hardpool",
    [string]$Stage2RunTag = "v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1",
    [int]$Stage2Seed = 1,
    [int]$Stage2Agents = 2,
    [long]$Stage2AdditionalSteps = 500000,
    [switch]$SkipStage2,
    [switch]$ForceStage2,
    [switch]$AllowSaturatedPool,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $projectRoot $Path))
}

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$Command,
        [string]$LogPath = ""
    )
    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host $Label
    Write-Host ("=" * 72)
    if ($DryRun) {
        Write-Host ("DRY RUN: " + ($Command -join " "))
        return 0
    }
    if ($LogPath) {
        & $Command[0] $Command[1..($Command.Length - 1)] 2>&1 | Tee-Object -FilePath $LogPath
    } else {
        & $Command[0] $Command[1..($Command.Length - 1)]
    }
    $exitCode = 0
    if ($null -ne $LASTEXITCODE) {
        $exitCode = [int]$LASTEXITCODE
    }
    return $exitCode
}

function Test-GateVerdictPass {
    param([string]$LogPath)
    if (-not (Test-Path $LogPath)) {
        return $false
    }
    $text = Get-Content $LogPath -Raw
    return ($text -match "VERDICT: PASS")
}

function Test-Stage2ReadyVerdict {
    param([string]$Verdict)
    if ([string]::IsNullOrWhiteSpace($Verdict)) { return $false }
    return $Verdict.StartsWith("READY FOR STAGE B")
}

$baselinePath = Resolve-ProjectPath $Baseline
$candidatePath = Resolve-ProjectPath $Candidate
$promoOutPath = Resolve-ProjectPath $PromoOutDir

foreach ($path in @($baselinePath, $candidatePath)) {
    if (-not (Test-Path $path)) {
        throw "Missing checkpoint: $path"
    }
}

$bundleLogDir = Resolve-ProjectPath ("artifacts/v6i9_promotion_bundle_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $bundleLogDir | Out-Null
    New-Item -ItemType Directory -Force -Path $promoOutPath | Out-Null
}

Write-Host "V6I9 Stage 1 promotion bundle"
Write-Host ("  Project root : " + $projectRoot)
Write-Host ("  Baseline     : " + $baselinePath)
Write-Host ("  Candidate    : " + $candidatePath)
Write-Host ("  Promo output : " + $promoOutPath)
if (-not $DryRun) {
    Write-Host ("  Bundle log   : " + $bundleLogDir)
}

# Step 1: competence gate
$gateLogPath = Join-Path $bundleLogDir "gate_output.txt"
$gateCmd = @(
    "uv", "run", "python", "experiments/gate_v6i9_map_aware.py",
    "--checkpoint", $candidatePath,
    "--device", $Device,
    "--episodes", "$GateEpisodes",
    "--allow-saturated-wr"
)
$gateExit = Invoke-Step -Label "STEP 1/3 - V6I9 competence gate" -Command $gateCmd -LogPath $gateLogPath
if (-not $DryRun) {
    $gatePassed = (Test-GateVerdictPass $gateLogPath) -or ($gateExit -eq 0)
    if (-not $gatePassed) {
        Write-Host ""
        Write-Host "Gate check failed. Fix gates or inspect logs before promotion eval."
        Write-Host ("Gate log: " + $gateLogPath)
        exit $(if ($gateExit -ne 0) { $gateExit } else { 1 })
    }
}

# Step 2: promotion eval
$promoCmd = @(
    "uv", "run", "python", "experiments/eval_v6i9_map_awareness.py",
    "--baseline", $baselinePath,
    "--candidate", $candidatePath,
    "--maps"
) + $Maps + @(
    "--opponents"
) + $Opponents + @(
    "--episodes", "$PromoEpisodes",
    "--seed-start", "$PromoSeedStart",
    "--device", $Device,
    "--output-dir", $promoOutPath
)
if ($AllowSaturatedPool) {
    $promoCmd += "--allow-saturated-pool"
}
$promoExit = Invoke-Step -Label "STEP 2/3 - V6I9 map-awareness promotion eval" -Command $promoCmd
if ($promoExit -ne 0 -and -not $DryRun -and -not $ForceStage2) {
    Write-Host ""
    Write-Host "Promotion eval reported NOT READY. Review: $promoOutPath\summary.json"
    Write-Host "Override with -ForceStage2 to launch Stage 2 anyway."
    exit $promoExit
}

$summaryPath = Join-Path $promoOutPath "summary.json"
$verdict = $null
if ($DryRun) {
    $verdict = "(dry-run)"
} elseif (Test-Path $summaryPath) {
    $verdict = (Get-Content $summaryPath -Raw | ConvertFrom-Json).verdict
    Copy-Item -Force $summaryPath (Join-Path $bundleLogDir "promotion_summary.json")
} else {
    throw "Promotion eval finished but summary.json was not written: $summaryPath"
}

Write-Host ""
Write-Host ("Promotion verdict: " + $verdict)

if ($SkipStage2) {
    Write-Host "Skipping Stage 2 training (-SkipStage2)."
    if (Test-Stage2ReadyVerdict $verdict) {
        Write-Host ""
        Write-Host "Manual Stage 2 launch command:"
        Write-Host ("  uv run python rl/train_ppo.py --preset {0} --load {1} --load-weights-only --additional-steps {2} --seed {3} --agents {4} --device {5} --run-tag {6} --fresh-metrics-csv" -f $Stage2Preset, $candidatePath, $Stage2AdditionalSteps, $Stage2Seed, $Stage2Agents, $Device, $Stage2RunTag)
    }
    exit 0
}

$readyForStage2 = ($ForceStage2 -or (Test-Stage2ReadyVerdict $verdict))
if (-not $readyForStage2) {
    Write-Host ""
    Write-Host "Stage 2 NOT launched - promotion verdict is not READY FOR STAGE B."
    Write-Host "Review: $summaryPath"
    Write-Host "Override with -ForceStage2 if you accept the risk."
    exit 2
}

# Step 3: Stage 2 repertoire training
$trainCmd = @(
    "uv", "run", "python", "rl/train_ppo.py",
    "--preset", $Stage2Preset,
    "--load", $candidatePath,
    "--load-weights-only",
    "--additional-steps", "$Stage2AdditionalSteps",
    "--seed", "$Stage2Seed",
    "--agents", "$Stage2Agents",
    "--device", $Device,
    "--run-tag", $Stage2RunTag,
    "--fresh-metrics-csv"
)
$trainExit = Invoke-Step -Label "STEP 3/3 - V6I9 Stage 2 repertoire training" -Command $trainCmd
exit $trainExit
