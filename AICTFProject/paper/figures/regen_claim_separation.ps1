# Regenerate Claim A / Claim B separation figures + regenerable PNG previews.
# Run from AICTFProject/:
#   powershell -NoProfile -File paper/figures/regen_claim_separation.ps1
#
# Prefer Python 3.12 with matplotlib if `python`/`uv` lacks it.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..\..

$py = "C:\Users\K-B\AppData\Local\Programs\Python\Python312\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$scripts = @(
  "paper/figures/build_claim_a_2v2.py",
  "paper/figures/build_claim_b_spp_ablation.py",
  "paper/figures/build_sharing_ladder_2v2.py",
  "paper/figures/build_2v2_measurement_hierarchy.py",
  "paper/figures/build_specialist_crossover_2v2.py",
  "paper/figures/build_2v2_baseline_sharing_performance.py",
  "paper/figures/build_strategy_existence_2v2.py"
)

foreach ($s in $scripts) {
  if (-not (Test-Path $s)) {
    Write-Warning "skip missing: $s"
    continue
  }
  Write-Host "==> $s"
  & $py $s
  if ($LASTEXITCODE -ne 0) { throw "failed: $s" }
}

Write-Host "OK: claim separation figures regenerated under paper/generated/"
