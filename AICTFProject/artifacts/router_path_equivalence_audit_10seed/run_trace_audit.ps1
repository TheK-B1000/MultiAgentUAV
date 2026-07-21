$env:PYTHONUNBUFFERED='1'
Set-Location -LiteralPath 'K:\MultiAgentUAV\AICTFProject'
& 'C:\Users\K-B\AppData\Local\Programs\Python\Python312\python.exe' experiments\eval_v6i9_router_diagnostic_ablation.py --episodes 10 --base-seed 9100 --device cuda --trace-audit --out-dir artifacts\router_path_equivalence_audit_10seed
exit $LASTEXITCODE
