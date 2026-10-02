# Optional. Preferred workflow: zip 6v6/FOR_PROFESSOR in File Explorer.
# This script only exists if someone prefers a CLI zip of the same folder.
#
# From AICTFProject:
#   powershell -ExecutionPolicy Bypass -File experiments/pack_6v6_handoff.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "6v6\START_HERE.txt"))) {
  $Root = (Get-Location).Path
}
$Hand = Join-Path $Root "6v6"
$src = Join-Path $Hand "FOR_PROFESSOR"
if (-not (Test-Path $src)) {
  throw "missing $src — wait until the dual-branch pipeline finishes (look for READY_TO_ZIP.txt)."
}
if (-not (Test-Path (Join-Path $src "READY_TO_ZIP.txt"))) {
  Write-Host "WARNING: READY_TO_ZIP.txt missing; folder may be incomplete."
}

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$zip = Join-Path $Hand "FOR_PROFESSOR_$stamp.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
  $src,
  $zip,
  [System.IO.Compression.CompressionLevel]::Optimal,
  $false
)

Write-Host "SOURCE $src"
Write-Host "WROTE  $zip"
Write-Host "Size:" ((Get-Item $zip).Length / 1MB).ToString("0.0") "MB"
Write-Host "Preferred: right-click FOR_PROFESSOR in File Explorer -> Compress to ZIP file"
