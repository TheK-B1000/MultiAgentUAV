$ErrorActionPreference = "Stop"
Set-Location "K:\MultiAgentUAV\AICTFProject"
$py = ".\.venv\Scripts\python.exe"
& $py experiments/eval_suite_sharing_crossover_4v4.py --arm fully_shared --device cuda *>&1 |
  Tee-Object -FilePath "artifacts/strategic_demand/sppo/suite_sharing/4v4/fully_shared_z/crossover_eval.log"
if ($LASTEXITCODE -notin 0,1) { exit $LASTEXITCODE }
& $py experiments/eval_suite_sharing_crossover_4v4.py --arm share_encoder --device cuda *>&1 |
  Tee-Object -FilePath "artifacts/strategic_demand/sppo/suite_sharing/4v4/share_encoder/crossover_eval.log"
exit $LASTEXITCODE
