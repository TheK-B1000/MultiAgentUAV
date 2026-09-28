# Detached: exploratory crossover for Share-Backbone then Share-Macro at 4v4.
$ErrorActionPreference = "Continue"
$Root = "k:\MultiAgentUAV\AICTFProject"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Eval = Join-Path $Root "experiments\eval_suite_sharing_crossover_4v4.py"
$Sp = Join-Path $Root "artifacts\strategic_demand\sppo"
$Watch = Join-Path $Sp "suite_sharing\4v4\depth_crossover_watch.log"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = $Root
Set-Location $Root

function Log([string]$msg) {
  Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $msg"
}

function Run-Eval([string]$arm) {
  $dir = Join-Path $Sp "suite_sharing\4v4\$arm"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $out = Join-Path $dir "crossover_eval.log"
  $err = Join-Path $dir "crossover_eval.log.err"
  Remove-Item -LiteralPath $out,$err -Force -ErrorAction SilentlyContinue
  Log "dry-run $arm"
  & $Py -u $Eval --arm $arm --dry-run --device cuda 1> $out 2> $err
  if ($LASTEXITCODE -ne 0) { Log "dry-run FAIL $arm exit=$LASTEXITCODE"; exit $LASTEXITCODE }
  Log "eval $arm"
  & $Py -u $Eval --arm $arm --device cuda 1> $out 2> $err
  if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 1) {
    # exit 1 = gate FAIL is still a completed sealed/flagged write in some paths;
    # integrity FLAG returns 0. Nonzero other = crash.
    Log "eval ended $arm exit=$LASTEXITCODE (check RESULT/FLAG)"
  } else {
    Log "eval finished $arm exit=$LASTEXITCODE"
  }
}

Log "depth crossover start"
Run-Eval "share_backbone"
Run-Eval "share_macro"
Log "depth crossover DONE"
exit 0
