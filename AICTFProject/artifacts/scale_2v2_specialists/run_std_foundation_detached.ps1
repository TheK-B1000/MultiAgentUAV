# Detached: standardized 2v2 foundation, pi_A then pi_B (STANDARDIZED_2V2_FOUNDATION_SPEC.json).
$ErrorActionPreference = "Continue"
$Root = "K:\MultiAgentUAV\AICTFProject"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Tr = Join-Path $Root "experiments\train_specialist_scale.py"
$Dir = Join-Path $Root "artifacts\scale_2v2_specialists"
$Watch = Join-Path $Dir "std_foundation_watch.log"
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONPATH = $Root
Set-Location $Root
function Log([string]$m) { Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $m" }
Log "foundation start (commit $(git -C K:\MultiAgentUAV log -1 --format=%h))"
foreach ($job in @(@("A", "23100001"), @("B", "23100002"))) {
  $pol = $job[0]; $seed = $job[1]
  Log "train pi_$pol seed=$seed"
  & $Py -u $Tr --team-size 2 --policy $pol --seed $seed --device cuda --total-timesteps 1000000 --run-label-suffix _std 1> (Join-Path $Dir "std_foundation_$pol.log") 2> (Join-Path $Dir "std_foundation_$pol.log.err")
  $rc = $LASTEXITCODE
  Log "pi_$pol exited $rc"
  if ($rc -ne 0) { Log "STOP: pi_$pol failed; pi_B not started"; exit $rc }
}
Log "foundation DONE"
exit 0
