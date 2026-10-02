# Pack the professor-facing 6v6 package into one zip.
# From AICTFProject:
#   powershell -ExecutionPolicy Bypass -File experiments/pack_6v6_handoff.ps1
#
# Preference order:
#   1) 6v6/FOR_PROFESSOR/          (current dual-branch + Stage-4 package)
#   2) 6v6/dual_branch_6v6_bundle/ (alias of the same)
#   3) entire 6v6/ folder          (fallback; includes older split-k1 handoff bits)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "6v6\README.md"))) {
  $Root = (Get-Location).Path
}
$Hand = Join-Path $Root "6v6"
if (-not (Test-Path $Hand)) { throw "missing $Hand — run the pipeline first" }

$src = $null
foreach ($cand in @("FOR_PROFESSOR", "dual_branch_6v6_bundle")) {
  $p = Join-Path $Hand $cand
  if (Test-Path $p) { $src = $p; break }
}
if (-not $src) {
  Write-Host "WARNING: FOR_PROFESSOR not built yet; packing entire 6v6/ (may include old split-k1 files)."
  $src = $Hand
}

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$zip = Join-Path $Hand "6v6_handoff_$stamp.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

Add-Type -AssemblyName System.IO.Compression.FileSystem
$tmp = Join-Path $env:TEMP "6v6_handoff_pack_$stamp"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $tmp -Recurse -Force
Get-ChildItem $tmp -Filter "6v6_handoff_*.zip" -ErrorAction SilentlyContinue | Remove-Item -Force
Get-ChildItem $tmp -Filter "dual_branch_6v6_results.zip" -ErrorAction SilentlyContinue | Remove-Item -Force
[System.IO.Compression.ZipFile]::CreateFromDirectory(
  $tmp,
  $zip,
  [System.IO.Compression.CompressionLevel]::Optimal,
  $false
)
Remove-Item $tmp -Recurse -Force

Write-Host "SOURCE $src"
Write-Host "WROTE  $zip"
Write-Host "Size:" ((Get-Item $zip).Length / 1MB).ToString("0.0") "MB"
Write-Host "Also present when pipeline finishes: $(Join-Path $Hand 'dual_branch_6v6_results.zip')"
