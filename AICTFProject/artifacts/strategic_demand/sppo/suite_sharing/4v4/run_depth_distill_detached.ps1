# Detached: distill Share-Backbone then Share-Macro at 4v4, then pin SPEC shas.
$ErrorActionPreference = "Continue"
$Root = "k:\MultiAgentUAV\AICTFProject"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Distill = Join-Path $Root "experiments\run_suite_sharing_distillation.py"
$Sp = Join-Path $Root "artifacts\strategic_demand\sppo"
$Watch = Join-Path $Sp "suite_sharing\4v4\depth_extension_watch.log"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONPATH = $Root
Set-Location $Root

function Log([string]$msg) {
  Add-Content -LiteralPath $Watch -Value "$(Get-Date -Format o) $msg"
}

function Run-Arm([string]$arm) {
  $tag = $arm
  $dir = Join-Path $Sp "suite_sharing\4v4\$tag"
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  $out = Join-Path $dir "distill.log"
  $err = Join-Path $dir "distill.log.err"
  Remove-Item -LiteralPath $out,$err -Force -ErrorAction SilentlyContinue
  Log "preflight $arm"
  & $Py -u $Distill --arm $arm --team-size 4 --preflight --device cuda *>> $out
  if ($LASTEXITCODE -ne 0) { Log "preflight FAIL $arm exit=$LASTEXITCODE"; exit $LASTEXITCODE }
  Log "distill $arm"
  & $Py -u $Distill --arm $arm --team-size 4 --device cuda *>> $out
  if ($LASTEXITCODE -ne 0) { Log "distill FAIL $arm exit=$LASTEXITCODE"; exit $LASTEXITCODE }
  $frozen = Join-Path $dir "STUDENT_FROZEN.json"
  if (-not (Test-Path -LiteralPath $frozen)) { Log "missing frozen $arm"; exit 2 }
  Log "frozen $arm ok"
}

Log "depth extension start"
Run-Arm "share_backbone"
Run-Arm "share_macro"

# Pin shas into crossover SPEC via Python (preserve JSON structure)
$pinPy = @'
import json
from pathlib import Path
sp = Path(r"k:\MultiAgentUAV\AICTFProject\artifacts\strategic_demand\sppo")
spec_path = sp / "SUITE_SHARING_4V4_CROSSOVER_EVAL_SPEC.json"
spec = json.loads(spec_path.read_text(encoding="utf-8"))
for arm in ("share_backbone", "share_macro"):
    frozen = json.loads((sp / f"suite_sharing/4v4/{arm}/STUDENT_FROZEN.json").read_text(encoding="utf-8"))
    spec["ARMS"][arm]["sha256"] = frozen["sha256"]
    print(f"pinned {arm} {frozen['sha256'][:12]}...")
spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
print("SPEC sha pins written")
'@
$pinFile = Join-Path $Sp "suite_sharing\4v4\_pin_depth_shas.py"
Set-Content -LiteralPath $pinFile -Value $pinPy -Encoding utf8
& $Py -u $pinFile
if ($LASTEXITCODE -ne 0) { Log "pin FAIL exit=$LASTEXITCODE"; exit $LASTEXITCODE }
Log "SPEC sha pins written"
Log "depth distill DONE; next crossover evals"
exit 0
