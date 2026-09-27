# Detached: standardized 2v2 Separated crossover, exploratory n=64
# (STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json). Runs the spec's LAUNCH.command verbatim
# after refusing unless the spec is tracked, has no staged/unstaged changes, and is FROZEN.
# -SpecRel / -Stem reuse it for the confirmatory pass (-Stem std_crossover_confirmatory).
# SAFE BY DEFAULT: without -Launch it runs the spec checks and the evaluator's own --dry-run only.
param([switch]$Launch, [string]$SpecRel = "artifacts/strategic_demand/sppo/STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json",
      [string]$Stem = "std_crossover_exploratory")
$ErrorActionPreference = "Continue"
$Root = "K:\MultiAgentUAV\AICTFProject"
$Dir = Join-Path $Root "artifacts\scale_2v2_specialists"
$Watch = Join-Path $Dir "$($Stem)_watch.log"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONPATH = $Root
Set-Location $Root
function Log([string]$m) { Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $m" }
Log "$Stem start (commit $(git -C K:\MultiAgentUAV log -1 --format=%h))"
git -C $Root ls-files --error-unmatch -- $SpecRel *> $null
if ($LASTEXITCODE -ne 0) { Log "STOP: spec not tracked"; exit 3 }
git -C $Root diff --quiet -- $SpecRel; if ($LASTEXITCODE -ne 0) { Log "STOP: spec has unstaged changes"; exit 3 }
git -C $Root diff --cached --quiet -- $SpecRel; if ($LASTEXITCODE -ne 0) { Log "STOP: spec has staged changes"; exit 3 }
$specObj = Get-Content -Raw -LiteralPath (Join-Path $Root $SpecRel) | ConvertFrom-Json
if (-not ([string]$specObj.status).StartsWith("FROZEN")) { Log "STOP: spec status $($specObj.status)"; exit 3 }
$parts = ([string]$specObj.LAUNCH.command).Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
$exe = Join-Path $Root $parts[0]
$args_ = $parts[1..($parts.Length - 1)]
if (-not $Launch) {
  Log "CHECK-ONLY: spec committed+clean+FROZEN; running evaluator --dry-run"
  & $exe @args_ --dry-run *> $null
  $rc = $LASTEXITCODE
  Log "CHECK-ONLY dry-run exited $rc (pass -Launch to spend seeds)"
  exit $rc
}
Log "run: $($specObj.LAUNCH.command)"
& $exe @args_ 1> (Join-Path $Dir "$($Stem).log") 2> (Join-Path $Dir "$($Stem).log.err")
$rc = $LASTEXITCODE
Log "$Stem exited $rc"
exit $rc
