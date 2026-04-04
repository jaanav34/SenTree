$ErrorActionPreference = "Stop"

Write-Host "=== SenTree Setup (Windows PowerShell) ==="

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
  )
  & $Exe @Args
  if ($LASTEXITCODE -ne 0) {
    throw "Command failed ($LASTEXITCODE): $Exe $($Args -join ' ')"
  }
}

function Resolve-Python {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    foreach ($ver in @("3.12", "3.13", "3.11", "3.10", "3")) {
      $flag = "-$ver"
      & py $flag -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
      if ($LASTEXITCODE -eq 0) {
        return @("py", $flag)
      }
    }
  }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) { return @("python") }

  throw "Python not found. Install Python 3.x (and ensure `python` or `py` is on PATH)."
}

$pythonCmd = Resolve-Python
$pyExe = $pythonCmd[0]
$pyArgs = @()
if ($pythonCmd.Length -gt 1) { $pyArgs = @($pythonCmd[1]) }

$versionStr = (& $pyExe @pyArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($versionStr -match '^(\d+)\.(\d+)$') {
  $maj = [int]$Matches[1]
  $min = [int]$Matches[2]
  if (-not $env:SENTREE_SKIP_PY_VERSION_CHECK -and $maj -eq 3 -and $min -ge 14) {
    throw "Python $versionStr is too new for reliable wheels (numpy/torch often missing). Install Python 3.12 or 3.13 and re-run setup."
  }
}

if (-not (Test-Path ".venv")) {
  Write-Host "Creating virtual environment at .venv ..."
  Invoke-Checked $pyExe @pyArgs "-m" "venv" ".venv"
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Virtual environment Python not found at $venvPython"
}

function Ensure-VenvPip {
  & $venvPython -m pip --version 2>$null
  if ($LASTEXITCODE -eq 0) { return }

  Write-Host "Bootstrapping pip inside .venv..."
  New-Item -ItemType Directory -Force -Path ".tmp" | Out-Null

  $oldTemp = $env:TEMP
  $oldTmp = $env:TMP
  $tmpDir = (Resolve-Path ".tmp").Path
  $env:TEMP = $tmpDir
  $env:TMP = $tmpDir

  & $venvPython -m ensurepip --upgrade --default-pip 2>$null
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "ensurepip failed; extracting bundled pip wheel as a fallback..."
    Invoke-Checked $venvPython "-c" "import ensurepip, pathlib, zipfile, ensurepip._bundled; import os; site=r'$(Resolve-Path '.\\.venv\\Lib\\site-packages')'; p=next(pathlib.Path(ensurepip._bundled.__path__[0]).glob('pip-*.whl')); zipfile.ZipFile(p).extractall(site); print('pip extracted from', p)"
  }

  $env:TEMP = $oldTemp
  $env:TMP = $oldTmp

  & $venvPython -m pip --version 2>$null
  if ($LASTEXITCODE -ne 0) {
    throw "pip could not be installed into .venv (even after fallback)."
  }
}

Ensure-VenvPip

Write-Host "Upgrading pip..."
Invoke-Checked $venvPython "-m" "pip" "install" "--upgrade" "pip"

Write-Host "Installing requirements.txt (venv-only)..."
Invoke-Checked $venvPython "-m" "pip" "install" "--require-virtualenv" "-r" "requirements.txt"

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
