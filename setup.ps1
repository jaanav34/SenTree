$ErrorActionPreference = "Stop"

Write-Host "=== SenTree Setup (Windows PowerShell) ==="

function Resolve-Python {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) { return @("py", "-3") }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) { return @("python") }

  throw "Python not found. Install Python 3.x (and ensure `python` or `py` is on PATH)."
}

$pythonCmd = Resolve-Python

if (-not (Test-Path ".venv")) {
  Write-Host "Creating virtual environment at .venv ..."
  & $pythonCmd[0] @($pythonCmd[1..($pythonCmd.Length-1)]) -m venv .venv
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Virtual environment Python not found at $venvPython"
}

Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing requirements.txt (venv-only)..."
& $venvPython -m pip install --require-virtualenv -r requirements.txt

# PyG CPU extras install (best-effort)
Write-Host "Installing PyG CPU extras (best-effort)..."
try {
  & $venvPython -m pip install --require-virtualenv torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-2.1.0+cpu.html" | Out-Host
} catch {
  Write-Warning "PyG extras install failed - GCNConv may still work without them"
}

Write-Host "Creating output directories..."
New-Item -ItemType Directory -Force -Path "data/raw","data/processed","outputs/videos","outputs/roi","outputs/embeddings" | Out-Null

Write-Host ""
Write-Host "=== Setup Complete ==="
Write-Host "Next steps:"
Write-Host "  1. .\.venv\Scripts\python.exe data\generate_synthetic.py"
Write-Host "  2. `$env:GOOGLE_API_KEY = 'your-key'"
Write-Host "  3. .\.venv\Scripts\python.exe scripts\run_pipeline.py"
Write-Host "  4. .\.venv\Scripts\python.exe scripts\index_videos.py"
Write-Host "  5. .\.venv\Scripts\python.exe -m streamlit run src\dashboard\app.py"

