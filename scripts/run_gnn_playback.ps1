$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$ExportScript = Join-Path $RootDir "scripts\export_gnn_playback_data.py"
$PlaybackSource = Join-Path $RootDir "outputs\roi\gnn_training_history.npz"
$AppDir = Join-Path $RootDir "apps\gnn-playback"
$NodeModules = Join-Path $AppDir "node_modules"

if (-not (Test-Path $VenvPython)) {
  throw "Expected virtualenv python at $VenvPython. Run .\setup.ps1 first."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
  throw "npm was not found on PATH. Install Node.js, then re-run this script."
}

if (-not (Test-Path $PlaybackSource)) {
  throw "Missing $PlaybackSource. Run .\.venv\Scripts\python.exe scripts\run_pipeline.py first to generate GNN training history."
}

Set-Location $RootDir
& $VenvPython $ExportScript
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

Set-Location $AppDir
if (-not (Test-Path $NodeModules)) {
  Write-Host "Installing React app dependencies..."
  & npm install
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

& npm run dev
exit $LASTEXITCODE
