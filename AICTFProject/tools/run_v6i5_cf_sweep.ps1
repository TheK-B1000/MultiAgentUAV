param(
    [string]$Python = "C:\Users\K-B\AppData\Local\Programs\Python\Python312\python.exe",
    [string]$ProjectRoot = "K:\MultiAgentUAV\AICTFProject",
    [string]$CheckpointDir = "checkpoints\4v4_diag",
    [int]$TotalSteps = 150000,
    [int]$NSteps = 256
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Set-Location $ProjectRoot

$runs = @(
    @{ Multiplier = "2x"; Coef = "2.0" },
    @{ Multiplier = "4x"; Coef = "4.0" },
    @{ Multiplier = "8x"; Coef = "8.0" }
)

foreach ($run in $runs) {
    $tag = "v6i5_cf_sweep_$($run.Multiplier)_150k"
    $metrics = Join-Path $CheckpointDir "$($tag)_4v4_metrics.csv"
    if (Test-Path $metrics) {
        Write-Host "SKIP $tag metrics already exists: $metrics"
        continue
    }

    Write-Host "START $tag latent_cf_coef_max=$($run.Coef)"
    & $Python -u rl\train_ppo.py `
        --preset v6i5 `
        --agents 4 `
        --total-steps $TotalSteps `
        --n-steps $NSteps `
        --run-tag $tag `
        --checkpoint-dir $CheckpointDir `
        --fresh-metrics-csv `
        --no-progress-bar `
        --periodic-checkpoint-steps 50000 `
        --latent-cf-coef-max $run.Coef
    if ($LASTEXITCODE -ne 0) {
        throw "$tag failed with exit code $LASTEXITCODE"
    }
    Write-Host "DONE $tag"
}
