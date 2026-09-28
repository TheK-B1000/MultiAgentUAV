# Detached: suite distillation dataset collection for one team size (SUITE_DISTILLATION_<N>V<N>_SPEC.json).
# SAFE BY DEFAULT: without -Launch it only checks that the spec is tracked, clean (no staged/unstaged changes)
# and FROZEN, and that the Rule-9 collection blocks are registered and RESERVED. With -Launch it runs the spec's
# LAUNCH.command verbatim plus --resume (fingerprint-exact shard reuse after an interrupted run), then requires
# the FROZEN_DATASET manifest to exist.
# -DatasetTag names a new collection beside an older frozen one (e.g. V2 for the corrected 4v4); its spec
# LAUNCH.command must carry the same --dataset-tag.
param([switch]$Launch, [Parameter(Mandatory = $true)][ValidateSet(2, 4, 6)][int]$TeamSize, [string]$DatasetTag = "")
$ErrorActionPreference = "Continue"
$Root = "K:\MultiAgentUAV\AICTFProject"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$SD = Join-Path $Root "artifacts\strategic_demand\sppo"
$T = if ($DatasetTag) { "_$($DatasetTag.ToUpper())" } else { "" }
$SpecRel = "artifacts/strategic_demand/sppo/SUITE_DISTILLATION_$($TeamSize)V$($TeamSize)$($T)_SPEC.json"
$Manifest = Join-Path $SD "SUITE_DISTILLATION_$($TeamSize)V$($TeamSize)$($T)_DATASET.json"
$Stem = "suite_distillation_$($TeamSize)v$($TeamSize)$($T.ToLower())_collect"
$Watch = Join-Path $SD "$($Stem)_watch.log"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONPATH = $Root
Set-Location $Root
function Log([string]$m) { Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $m" }
$mode = if ($Launch) { "LAUNCH" } else { "CHECK-ONLY" }
Log "$Stem $mode start (commit $(git -C K:\MultiAgentUAV log -1 --format=%h))"
git -C $Root ls-files --error-unmatch -- $SpecRel *> $null
if ($LASTEXITCODE -ne 0) { Log "STOP: spec not tracked"; exit 3 }
git -C $Root diff --quiet -- $SpecRel; if ($LASTEXITCODE -ne 0) { Log "STOP: spec has unstaged changes"; exit 3 }
git -C $Root diff --cached --quiet -- $SpecRel; if ($LASTEXITCODE -ne 0) { Log "STOP: spec has staged changes"; exit 3 }
$specObj = Get-Content -Raw -LiteralPath (Join-Path $Root $SpecRel) | ConvertFrom-Json
if (-not ([string]$specObj.status).StartsWith("FROZEN")) { Log "STOP: spec status $($specObj.status)"; exit 3 }
$chk = & $Py -c "import json,sys; from experiments.collect_suite_distillation_states import check_collection_seeds as c; print(c(json.load(open(sys.argv[1],encoding='utf-8'))))" (Join-Path $Root $SpecRel) 2>&1
$rc = $LASTEXITCODE
Log "  rule9: $chk"
if ($rc -ne 0) { Log "STOP: Rule-9 collection blocks refused"; exit 3 }
if (-not $Launch) { Log "CHECK-ONLY PASS (pass -Launch to collect)"; exit 0 }
$parts = ([string]$specObj.LAUNCH.command).Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
$exe = Join-Path $Root $parts[0]
$args_ = $parts[1..($parts.Length - 1)]
Log "run: $($specObj.LAUNCH.command) --resume"
& $exe @args_ --resume 1>> (Join-Path $SD "$($Stem).log") 2>> (Join-Path $SD "$($Stem).log.err")
$rc = $LASTEXITCODE
Log "$Stem exited $rc"
if (-not (Test-Path -LiteralPath $Manifest)) { Log "STOP: no dataset manifest written"; exit 4 }
$st = (Get-Content -Raw -LiteralPath $Manifest | ConvertFrom-Json).status
Log "dataset manifest status $st"
if ($st -ne "FROZEN_DATASET") { exit 4 }
Log "$Stem DONE"
exit 0
