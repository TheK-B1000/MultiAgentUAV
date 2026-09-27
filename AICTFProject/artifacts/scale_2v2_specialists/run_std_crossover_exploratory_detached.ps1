# Detached: standardized 2v2 Separated crossover, exploratory n=64
# (STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json). Runs the spec's LAUNCH.command verbatim
# after refusing unless the spec is tracked, has no staged/unstaged changes, and is FROZEN.
# -SpecRel / -Stem reuse it for the confirmatory pass (-Stem std_crossover_confirmatory).
# SAFE BY DEFAULT: without -Launch it runs the spec checks and the evaluator's own --dry-run only.
param([switch]$Launch, [string]$SpecRel = "artifacts/strategic_demand/sppo/STANDARDIZED_2V2_SEPARATED_CROSSOVER_EXPLORATORY_SPEC.json",
      [string]$Stem = "std_crossover_exploratory",
      # several LAUNCH keys run in order (e.g. a paired diagnostic); the evaluator exits 1 when its
      # own gate fails, so success is judged by a SEALED/AUDIT_FAILED result record per label.
      [string[]]$CommandKeys = @("command"))
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
# powershell -File passes an array argument as ONE string ("a,b"); split it explicitly.
$CommandKeys = @($CommandKeys | ForEach-Object { $_ -split "," } | Where-Object { $_ })
foreach ($key in $CommandKeys) {
  $cmd = [string]$specObj.LAUNCH.$key
  if (-not $cmd) { Log "STOP: spec has no LAUNCH.$key"; exit 3 }
  $parts = $cmd.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
  $exe = Join-Path $Root $parts[0]
  $args_ = $parts[1..($parts.Length - 1)]
  if (-not $Launch) {
    & $exe @args_ --dry-run *> $null
    $rc = $LASTEXITCODE
    Log "CHECK-ONLY $key dry-run exited $rc"
    if ($rc -ne 0) { exit $rc }
    continue
  }
  $label = $parts[[array]::IndexOf($parts, "--label") + 1]
  $sfx = if ($CommandKeys.Count -gt 1) { "_$key" } else { "" }
  Log "run $key : $cmd"
  & $exe @args_ 1> (Join-Path $Dir "$($Stem)$sfx.log") 2> (Join-Path $Dir "$($Stem)$sfx.log.err")
  $rc = $LASTEXITCODE
  Log "$Stem $key exited $rc"
  $res = Join-Path $Root "artifacts\strategic_demand\sppo\$($label)_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
  if (-not (Test-Path -LiteralPath $res)) { Log "STOP: no result record for $label after $key"; exit 4 }
  $st = (Get-Content -Raw -LiteralPath $res | ConvertFrom-Json).status
  if ($st -ne "SEALED" -and $st -ne "AUDIT_FAILED") { Log "STOP: $label record status $st"; exit 4 }
  Log "$label record status $st"
}
if (-not $Launch) { Log "CHECK-ONLY PASS (pass -Launch to spend seeds)"; exit 0 }
Log "$Stem DONE"
exit 0
