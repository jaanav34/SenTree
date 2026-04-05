$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$AppPath = Join-Path $RootDir "src\dashboard\app.py"

if (-not (Test-Path $VenvPython)) {
  throw "Expected virtualenv python at $VenvPython. Run .\setup.ps1 first."
}

Set-Location $RootDir
& $VenvPython -m streamlit run $AppPath
exit $LASTEXITCODE
