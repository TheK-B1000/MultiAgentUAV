# Full-day deployment communication ablation suite (eval only, no retraining).
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File K:\MultiAgentUAV\AICTFProject\scripts\run_deploy_comm_ablations.ps1
#
# Or from AICTFProject:
#   .\scripts\run_deploy_comm_ablations.ps1

$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Latent = Join-Path $Root "checkpoints\4v4\final_latent_ep_hardpool_4v4_seed1_4v4.zip"
$Flat = Join-Path $Root "checkpoints\4v4\final_flat_hardpool_4v4_seed1_4v4.zip"

if (-not (Test-Path $Latent)) {
    Write-Error "Missing latent checkpoint: $Latent"
    exit 1
}

$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("deploy_comm_ablations_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

function Invoke-EvalRun {
    param(
        [string]$Checkpoint,
        [string]$Label,
        [string[]]$ExtraArgs
    )
    Write-Log "======== START $Label ========"
    $args = @(
        "plot/eval_checkpoint.py",
        "--checkpoint", $Checkpoint,
        "--opponents", "OP3", "OP5_RUSHER", "OP6", "OP7", "OP4",
        "--map-sets", "train", "eval",
        "--episodes", "500",
        "--label", $Label
    ) + $ExtraArgs
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & python @args 2>&1 | Tee-Object -FilePath $LogFile -Append
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -ne 0) {
        Write-Log "FAILED $Label (exit $code) after $($sw.Elapsed)"
    } else {
        Write-Log "OK $Label in $($sw.Elapsed)"
    }
    Write-Log ""
    return $code
}

Write-Log "Deploy comm ablation suite | cwd=$Root"
Write-Log "Log file: $LogFile"
Write-Log ""

# --- Optional baselines (comment out if you already have these CSVs) ---
if (Test-Path $Flat) {
    [void](Invoke-EvalRun -Checkpoint $Flat -Label "flat_comm_off" -ExtraArgs @())
} else {
    Write-Log "SKIP flat_comm_off (no checkpoint: $Flat)"
}

[void](Invoke-EvalRun -Checkpoint $Latent -Label "latent_comm_on" -ExtraArgs @())

# --- Core deploy ablations (latent checkpoint) ---
[void](Invoke-EvalRun -Checkpoint $Latent -Label "latent_deploy_stoch_z" -ExtraArgs @("--stochastic"))
[void](Invoke-EvalRun -Checkpoint $Latent -Label "latent_deploy_fix_z0" -ExtraArgs @("--fixed-latent-id", "0"))
[void](Invoke-EvalRun -Checkpoint $Latent -Label "latent_deploy_fix_z1" -ExtraArgs @("--fixed-latent-id", "1"))
[void](Invoke-EvalRun -Checkpoint $Latent -Label "latent_deploy_fix_z2" -ExtraArgs @("--fixed-latent-id", "2"))
[void](Invoke-EvalRun -Checkpoint $Latent -Label "latent_deploy_fix_z3" -ExtraArgs @("--fixed-latent-id", "3"))
[void](Invoke-EvalRun -Checkpoint $Latent -Label "latent_deploy_z20" -ExtraArgs @("--latent-resample-every", "20"))

Write-Log "======== ALL RUNS FINISHED ========"
Write-Log "CSVs: $Root\csv\eval_*"
Write-Log "Compare aggregate files: csv\eval_*_4v4_aggregate.csv"
