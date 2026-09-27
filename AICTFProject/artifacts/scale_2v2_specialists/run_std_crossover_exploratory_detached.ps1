# Detached: standardized 2v2 Separated crossover, exploratory n=64
# (STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json). Runs the spec's LAUNCH.command verbatim
# after refusing unless the spec is tracked, has no staged/unstaged changes, and is FROZEN.
$ErrorActionPreference = "Continue"
$Root = "K:\MultiAgentUAV\AICTFProject"
$SpecRel = "artifacts/strategic_demand/sppo/STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json"
$Dir = Join-Path $Root "artifacts\scale_2v2_specialists"
$Watch = Join-Path $Dir "std_crossover_exploratory_watch.log"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONPATH = $Root
Set-Location $Root
function Log([string]$m) { Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $m" }
Log "crossover exploratory start (commit $(git -C K:\MultiAgentUAV log -1 --format=%h))"
git -C $Root ls-files --error-unmatch -- $SpecRel *> $null
if ($LASTEXITCODE -ne 0) { Log "STOP: spec not tracked"; exit 3 }
git -C $Root diff --quiet -- $SpecRel; if ($LASTEXITCODE -ne 0) { Log "STOP: spec has unstaged changes"; exit 3 }
git -C $Root diff --cached --quiet -- $SpecRel; if ($LASTEXITCODE -ne 0) { Log "STOP: spec has staged changes"; exit 3 }
$specObj = Get-Content -Raw -LiteralPath (Join-Path $Root $SpecRel) | ConvertFrom-Json
if (-not ([string]$specObj.status).StartsWith("FROZEN")) { Log "STOP: spec status $($specObj.status)"; exit 3 }
$parts = ([string]$specObj.LAUNCH.command).Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
$exe = Join-Path $Root $parts[0]
$args_ = $parts[1..($parts.Length - 1)]
Log "run: $($specObj.LAUNCH.command)"
& $exe @args_ 1> (Join-Path $Dir "std_crossover_exploratory.log") 2> (Join-Path $Dir "std_crossover_exploratory.log.err")
$rc = $LASTEXITCODE
Log "crossover exploratory exited $rc"
exit $rc
