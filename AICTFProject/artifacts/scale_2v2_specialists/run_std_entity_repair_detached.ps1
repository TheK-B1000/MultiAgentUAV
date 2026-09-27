# Detached: standardized 2v2 entity repair, pi_A then pi_B (STANDARDIZED_2V2_ENTITY_REPAIR_SPEC.json).
#
# SAFE BY DEFAULT: without -Launch this only runs the preflight and exits; nothing is created and
# no Python training process starts. Training needs the explicit -Launch switch, so an unbound or
# mistyped switch can never start a run (an unsafe-by-default guard test spent seed 23200001).
#
# Every launch decision is made by experiments/frozen_launch_guard.py (one tested implementation:
# spec tracked + no staged/unstaged changes + FROZEN, LAUNCH seed/experiment-id agree with SEEDS,
# registry block RESERVED, seed never trained, warm-start path + sha256 match the pins).
# Preflight checks BOTH policies before anything starts; each policy is re-checked immediately
# before its own launch. The spec's LAUNCH command is run verbatim.
# -Policies limits the run to a subset (e.g. B alone after A finished); default is A then B.
# NOTE: PowerShell variable names are case-insensitive -- never reuse $Spec for the parsed JSON.
param([switch]$Launch, [string[]]$Policies = @("A", "B"))
$ErrorActionPreference = "Continue"
$Root = "K:\MultiAgentUAV\AICTFProject"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Guard = Join-Path $Root "experiments\frozen_launch_guard.py"
$SpecRel = "artifacts/strategic_demand/sppo/STANDARDIZED_2V2_ENTITY_REPAIR_SPEC.json"
$Spec = Join-Path $Root $SpecRel
$Dir = Join-Path $Root "artifacts\scale_2v2_specialists"
$Watch = Join-Path $Dir "std_entity_repair_watch.log"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONPATH = $Root
Set-Location $Root
function Log([string]$m) { Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $m" }
function Guard([string]$pol) {
  $out = & $Py $Guard --spec $Spec --policy $pol --repo-root $Root 2>&1
  $rc = $LASTEXITCODE
  foreach ($line in $out) { Log "  guard pi_$pol $line" }
  return $rc
}

$mode = if ($Launch) { "LAUNCH" } else { "CHECK-ONLY" }
$mode = "$mode policies=$($Policies -join ',')"
Log "entity repair $mode start (commit $(git -C K:\MultiAgentUAV log -1 --format=%h))"
foreach ($pol in $Policies) {
  if ((Guard $pol) -ne 0) { Log "STOP ($mode preflight): guard refused pi_$pol; nothing started"; exit 3 }
}
if (-not $Launch) { Log "CHECK-ONLY PASS: both policies authorized; no training started (pass -Launch to train)"; exit 0 }

foreach ($pol in $Policies) {
  if ((Guard $pol) -ne 0) { Log "STOP: guard refused pi_$pol immediately before launch"; exit 3 }
  $specObj = Get-Content -Raw -LiteralPath $Spec | ConvertFrom-Json
  $parts = ([string]$specObj.LAUNCH.$pol).Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
  $exe = Join-Path $Root $parts[0]
  $args_ = $parts[1..($parts.Length - 1)]
  Log "train pi_$pol : $($specObj.LAUNCH.$pol)"
  & $exe @args_ 1> (Join-Path $Dir "std_entity_repair_$pol.log") 2> (Join-Path $Dir "std_entity_repair_$pol.log.err")
  $rc = $LASTEXITCODE
  Log "pi_$pol exited $rc"
  if ($rc -ne 0) { Log "STOP: pi_$pol failed; later policies not started"; exit $rc }
}
Log "entity repair DONE"
exit 0
