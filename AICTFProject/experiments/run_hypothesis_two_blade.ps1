<#
.SYNOPSIS
  Default "real hypothesis" training pair: opponent-randomized flat vs latent Option B + λp + stronger strategy coef.

.DESCRIPTION
  Fixed OP3 is a sanity courtyard; this script runs the minimum contrast where latent strategy can earn its keep
  under opponent diversity (training pool OP1-OP3; OP4 reserved for eval). Uses train_ppo hypothesis_* presets (see rl/train_ppo.py).

.EXAMPLE
  .\experiments\run_hypothesis_two_blade.ps1 -Seed 42 -Timesteps 1000000

.EXAMPLE
  Dry-run commands only:
  .\experiments\run_hypothesis_two_blade.ps1 -DryRun -Seed 42
#>
param(
    [string]$Seed = "42",
    [string]$Agents = "2",
    [string]$Timesteps = "1000000",
    [string]$TrainEpisodesLogEvery = "1000",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$expRoot = Join-Path $projectRoot ("experiments\hypothesis_runs\" + $timestamp)
$ckptRoot = Join-Path $expRoot "checkpoints"
New-Item -ItemType Directory -Force -Path $ckptRoot | Out-Null
$runDir = Join-Path $ckptRoot ($Agents + "v" + $Agents)
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Write-Host ("Hypothesis experiment root: " + $expRoot)

$runs = @(
    @{ Key = "hypothesis_flat_opprand"; Preset = "hypothesis_flat_opprand" },
    @{ Key = "hypothesis_latent_opprand_optionb_lamp_coef05"; Preset = "hypothesis_latent_opprand_optionb_lamp_coef05" }
)

foreach ($r in $runs) {
    $runTag = ("research_" + $r.Key + "_seed" + $Seed + "_" + $Agents + "v" + $Agents)
    $trainCmd = @(
        $python, "rl/train_ppo.py",
        "--preset", $r.Preset,
        "--seed", $Seed,
        "--agents", $Agents,
        "--total-steps", $Timesteps,
        "--episode-log-every", $TrainEpisodesLogEvery,
        "--checkpoint-dir", $runDir,
        "--run-tag", $runTag
    )
    # Blade 2 (Option B + mid-episode z refresh): per-step E3 CSV for E2 switch analysis; flat blade skips latent-only flag.
    if ($r.Key -like "hypothesis_latent_*") {
        $trainCmd += "--e3-step-telemetry"
    }
    Write-Host ""
    Write-Host ("[TRAIN] " + $r.Key + " -> " + $runTag)
    if ($DryRun) {
        Write-Host ("DRY RUN: " + ($trainCmd -join " "))
    } else {
        & $trainCmd[0] $trainCmd[1..($trainCmd.Length - 1)]
    }
}

Write-Host ""
Write-Host "Next: eval with experiments\run_week_eval_bundle.ps1 on each final_*.zip (train+eval maps, OP1-OP4)."
Write-Host ("Checkpoints: " + $runDir)
