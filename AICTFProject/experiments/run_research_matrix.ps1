param(
    [string[]]$Seeds = @("42", "43", "44", "45", "46"),
    [string]$Agents = "2",
    [string]$Timesteps = "1000000",
    [string]$TrainEpisodesLogEvery = "1000",
    [string]$EvalEpisodes = "200",
    [string]$Device = "cuda",
    [string[]]$Opponents = @("OP3", "OP4"),
    [string[]]$MapSets = @("train", "eval"),
    [switch]$SkipTrain,
    [switch]$SkipEval,
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
$expRoot = Join-Path $projectRoot ("experiments\research_runs\" + $timestamp)
$ckptRoot = Join-Path $expRoot "checkpoints"
$evalRoot = Join-Path $expRoot "eval_csv"

New-Item -ItemType Directory -Force -Path $expRoot | Out-Null
New-Item -ItemType Directory -Force -Path $ckptRoot | Out-Null
New-Item -ItemType Directory -Force -Path $evalRoot | Out-Null

Write-Host ("Experiment root: " + $expRoot)
Write-Host ("Using python: " + $python)

# Sanity-check matrix: fixed scripted OP3 only (learning/stability courtyard — not the main hypothesis test).
# Real-hypothesis runs use experiments/run_hypothesis_two_blade.ps1 (opponent-randomized training).
# Option B heavy preset remains manual until eval evidence warrants it.
$baselines = @(
    @{
        Key = "sanity_latent_optiona_op3"
        TrainArgs = @("--preset", "plan_option_a")
    },
    @{
        Key = "sanity_flat_op3"
        TrainArgs = @("--preset", "plan_option_a", "--no-latent-strategy")
    },
    @{
        Key = "sanity_curriculum_jacob"
        TrainArgs = @("--preset", "plan_option_a", "--mode", "CURRICULUM", "--no-latent-strategy")
    },
    @{
        Key = "sanity_fixed_latent_z0_op3"
        TrainArgs = @("--preset", "plan_option_a", "--fixed-latent-strategy", "--fixed-latent-id", "0")
    }
)

$runManifest = @()

foreach ($seed in $Seeds) {
    foreach ($b in $baselines) {
        $runTag = ("research_" + $b.Key + "_seed" + $seed + "_" + $Agents + "v" + $Agents)
        $runDir = Join-Path $ckptRoot ($Agents + "v" + $Agents)
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $finalCkpt = Join-Path $runDir ("final_" + $runTag + ".zip")

        $trainCmd = @(
            $python, "rl/train_ppo.py",
            "--seed", $seed,
            "--agents", $Agents,
            "--total-steps", $Timesteps,
            "--episode-log-every", $TrainEpisodesLogEvery,
            "--checkpoint-dir", $runDir,
            "--run-tag", $runTag
        ) + $b.TrainArgs

        $runManifest += [pscustomobject]@{
            seed = $seed
            baseline = $b.Key
            run_tag = $runTag
            checkpoint = $finalCkpt
        }

        if (-not $SkipTrain) {
            Write-Host ""
            Write-Host ("[TRAIN] " + $b.Key + " seed=" + $seed + " run_tag=" + $runTag)
            if ($DryRun) {
                Write-Host ("DRY RUN: " + ($trainCmd -join " "))
            } else {
                & $trainCmd[0] $trainCmd[1..($trainCmd.Length - 1)]
            }
        }
    }
}

$manifestPath = Join-Path $expRoot "run_manifest.csv"
$runManifest | Export-Csv -NoTypeInformation -Path $manifestPath
Write-Host ("Manifest: " + $manifestPath)

if (-not $SkipEval) {
    $opponentArgs = @()
    foreach ($o in $Opponents) {
        $opponentArgs += $o
    }
    $mapSetArgs = @()
    foreach ($m in $MapSets) {
        $mapSetArgs += $m
    }

    foreach ($row in $runManifest) {
        Write-Host ""
        Write-Host ("[EVAL] " + $row.baseline + " seed=" + $row.seed)
        if (-not (Test-Path $row.checkpoint)) {
            Write-Warning ("Checkpoint missing, skipping eval: " + $row.checkpoint)
            continue
        }

        $evalCmd = @(
            $python, "plot/eval_checkpoint.py",
            "--checkpoint", $row.checkpoint,
            "--label", $row.run_tag,
            "--agents", $Agents,
            "--episodes", $EvalEpisodes,
            "--device", $Device,
            "--out-dir", $evalRoot,
            "--opponents"
        ) + $opponentArgs + @("--map-sets") + $mapSetArgs

        if ($DryRun) {
            Write-Host ("DRY RUN: " + ($evalCmd -join " "))
        } else {
            & $evalCmd[0] $evalCmd[1..($evalCmd.Length - 1)]
        }
    }
}

Write-Host ""
Write-Host "Completed."
Write-Host ("Checkpoints: " + $ckptRoot)
Write-Host ("Eval CSVs:   " + $evalRoot)
