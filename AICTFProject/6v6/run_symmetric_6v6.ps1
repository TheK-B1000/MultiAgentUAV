# Start the 6v6 symmetric-role diagnostic detached (safe to close this window or log off the console).
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File <repo>\AICTFProject\6v6\run_symmetric_6v6.ps1
$proj = Split-Path -Parent $PSScriptRoot
Set-Location $proj
$env:FOR_DISABLE_CONSOLE_CTRL_HANDLER = '1'
$py = Join-Path $proj '.venv\Scripts\python.exe'

& $py '6v6\run_symmetric_6v6.py' --check
if ($LASTEXITCODE -ne 0) { Write-Host "`nNot started: fix the problems above (usually: git pull)."; exit 1 }

$p = Start-Process -FilePath $py -ArgumentList '6v6\run_symmetric_6v6.py' -WorkingDirectory $proj -WindowStyle Hidden -PassThru `
     -RedirectStandardOutput '6v6\symmetric_6v6.stdout' -RedirectStandardError '6v6\symmetric_6v6.stderr'
Write-Host "Started (pid $($p.Id)). Progress:"
Write-Host "  Get-Content $proj\6v6\symmetric_6v6.log -Wait -Tail 20"
Write-Host "When it says DONE, send: $proj\6v6\symmetric_results.zip"
Write-Host "If the PC restarts, run this same command again -- it continues where it stopped."
