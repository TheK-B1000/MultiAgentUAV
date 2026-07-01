<#
.SYNOPSIS
  Free eval bundle: E4 (train vs eval maps) + cross-opponent zero-shot probes.

.DESCRIPTION
  Runs plot/eval_checkpoint.py for each checkpoint with --map-sets train eval and a wide opponent set.
  Use this week before committing to Option B training (~4h). Compare train→eval WR gaps and cross-OPP WR.

.EXAMPLE
  .\experiments\run_week_eval_bundle.ps1 `
    -Checkpoints @(
      "experiments\research_runs\20260508_081842\checkpoints\2v2\final_research_latent_a1_seed42_2v2.zip",
      "experiments\research_runs\20260508_081842\checkpoints\2v2\final_research_flat_ppo_seed42_2v2.zip"
    )

.EXAMPLE
  After curriculum finishes, append its final_*.zip path to -Checkpoints and re-run.
#>
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Checkpoints,
    [string]$Agents = "2",
    [string]$Episodes = "200",
    [string]$Device = "cuda",
    [string[]]$Opponents = @("OP3", "OP4", "OP5_RUSHER"),
    [string[]]$MapSets = @("train", "eval"),
    [string]$OutDir = "",
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

if (-not $OutDir) {
    $OutDir = Join-Path $projectRoot ("csv\week_eval_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
$OutDir = [System.IO.Path]::GetFullPath($OutDir)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host ("Output CSV dir: " + $OutDir)
Write-Host ("Opponents: " + ($Opponents -join ", "))
Write-Host ("Map sets: " + ($MapSets -join ", "))

$opponentArgs = @()
foreach ($o in $Opponents) {
    $opponentArgs += $o
}
$mapSetArgs = @()
foreach ($m in $MapSets) {
    $mapSetArgs += $m
}

foreach ($raw in $Checkpoints) {
    $ckpt = $raw.Trim()
    if (-not [System.IO.Path]::IsPathRooted($ckpt)) {
        $ckpt = Join-Path $projectRoot $ckpt
    }
    $ckpt = [System.IO.Path]::GetFullPath($ckpt)
    if (-not $ckpt.EndsWith(".zip")) {
        $ckpt = $ckpt + ".zip"
    }
    if (-not (Test-Path $ckpt)) {
        Write-Warning ("Skipping missing checkpoint: " + $ckpt)
        continue
    }
    $label = [System.IO.Path]::GetFileNameWithoutExtension($ckpt)

    $evalCmd = @(
        $python, "plot/eval_checkpoint.py",
        "--checkpoint", $ckpt,
        "--label", $label,
        "--agents", $Agents,
        "--episodes", $Episodes,
        "--device", $Device,
        "--out-dir", $OutDir,
        "--opponents"
    ) + $opponentArgs + @("--map-sets") + $mapSetArgs

    Write-Host ""
    Write-Host ("[EVAL] " + $label)
    if ($DryRun) {
        Write-Host ("DRY RUN: " + ($evalCmd -join " "))
    } else {
        & $evalCmd[0] $evalCmd[1..($evalCmd.Length - 1)]
    }
}

Write-Host ""
Write-Host ("Done. Aggregate summary (if written): " + (Join-Path $OutDir "eval_*_aggregate.csv"))
