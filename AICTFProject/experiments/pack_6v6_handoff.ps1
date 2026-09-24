# Pack AICTFProject/6v6 into a single zip for email / Drive / USB.
# From AICTFProject:
#   powershell -ExecutionPolicy Bypass -File experiments/pack_6v6_handoff.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "6v6\README.md"))) {
  $Root = (Get-Location).Path
}
$Hand = Join-Path $Root "6v6"
if (-not (Test-Path $Hand)) { throw "missing $Hand — run the pipeline first (or create folder)" }

$stamp = Get-Date -Format "yyyyMMdd_HHmm"
$zip = Join-Path $Hand "6v6_handoff_$stamp.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }

# Prefer .NET zip so long paths / large model zips are fine on Windows.
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
  $Hand,
  $zip,
  [System.IO.Compression.CompressionLevel]::Optimal,
  $false
)

# Re-open and drop any nested handoff zips from inside the archive? Simpler: exclude *.zip when packing.
# Recreate excluding other handoff zips:
Remove-Item $zip -Force
$tmp = Join-Path $env:TEMP "6v6_handoff_pack_$stamp"
if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null
Copy-Item -Path (Join-Path $Hand "*") -Destination $tmp -Recurse -Force
Get-ChildItem $tmp -Filter "6v6_handoff_*.zip" -ErrorAction SilentlyContinue | Remove-Item -Force
[System.IO.Compression.ZipFile]::CreateFromDirectory(
  $tmp,
  $zip,
  [System.IO.Compression.CompressionLevel]::Optimal,
  $false
)
Remove-Item $tmp -Recurse -Force

Write-Host "WROTE $zip"
Write-Host "Send that file. Size:" ((Get-Item $zip).Length / 1MB).ToString("0.0") "MB"
