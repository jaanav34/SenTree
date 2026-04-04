$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue) {
  $global:PSNativeCommandUseErrorActionPreference = $false
}

Write-Host "=== SenTree Setup (Windows PowerShell) ==="

function Invoke-Proc {
  param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
  )
  function Quote-Arg([string]$arg) {
    if ($null -eq $arg -or $arg.Length -eq 0) { return '""' }
    if ($arg -notmatch '[\s"]') { return $arg }

    $sb = New-Object System.Text.StringBuilder
    [void]$sb.Append('"')
    $backslashes = 0
    foreach ($ch in $arg.ToCharArray()) {
      if ($ch -eq '\') {
        $backslashes++
        continue
      }
      if ($ch -eq '"') {
        [void]$sb.Append(('\' * ($backslashes * 2 + 1)))
        [void]$sb.Append('"')
        $backslashes = 0
        continue
      }
      if ($backslashes -gt 0) {
        [void]$sb.Append(('\' * $backslashes))
        $backslashes = 0
      }
      [void]$sb.Append($ch)
    }
    if ($backslashes -gt 0) {
      [void]$sb.Append(('\' * ($backslashes * 2)))
    }
    [void]$sb.Append('"')
    return $sb.ToString()
  }

  $argLine = ($Args | ForEach-Object { Quote-Arg $_ }) -join ' '
  $p = Start-Process -FilePath $Exe -ArgumentList $argLine -NoNewWindow -Wait -PassThru
  return $p.ExitCode
}

function Invoke-Checked {
  param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Args
  )
  $code = Invoke-Proc $Exe @Args
  if ($code -ne 0) {
    throw "Command failed ($code): $Exe $($Args -join ' ')"
  }
}

function Resolve-Python {
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    foreach ($ver in @("3.12", "3.13", "3.11", "3.10", "3")) {
      $flag = "-$ver"
      $LASTEXITCODE = 1
      try {
        & py $flag -c "pass" 2>$null
      } catch { }
      if ($LASTEXITCODE -eq 0) {
        return @("py", $flag)
      }
    }
  }

  $python = Get-Command python -ErrorAction SilentlyContinue
  if ($python) { return @("python") }

  throw "Python not found. Install Python 3.x (and ensure `python` or `py` is on PATH)."
}

if ($env:SENTREE_PYTHON) {
  $pyExe = $env:SENTREE_PYTHON
  $pyArgs = @()
} else {
  $pythonCmd = @(Resolve-Python)
  $pyExe = $pythonCmd[0]
  $pyArgs = @()
  if ($pythonCmd.Length -gt 1) { $pyArgs = @($pythonCmd[1]) }
}

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
  New-Item -ItemType Directory -Force -Path ".tmp" | Out-Null
  $oldTemp = $env:TEMP
  $oldTmp = $env:TMP
  $tmpDir = (Resolve-Path ".tmp").Path
  $env:TEMP = $tmpDir
  $env:TMP = $tmpDir

  # `venv` may fail when trying to run ensurepip in locked-down temp dirs; we bootstrap pip ourselves below.
  Invoke-Checked $pyExe @pyArgs "-m" "venv" "--without-pip" ".venv"

  $env:TEMP = $oldTemp
  $env:TMP = $oldTmp
}

$venvPython = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  throw "Virtual environment Python not found at $venvPython"
}

function Ensure-VenvPip {
  $code = Invoke-Proc $venvPython "-c" "import pip; print(pip.__version__)"
  if ($code -eq 0) { return }

  Write-Host "Bootstrapping pip inside .venv..."
  New-Item -ItemType Directory -Force -Path ".tmp" | Out-Null

  $oldTemp = $env:TEMP
  $oldTmp = $env:TMP
  $tmpDir = (Resolve-Path ".tmp").Path
  $env:TEMP = $tmpDir
  $env:TMP = $tmpDir

  # Avoid `ensurepip` because it relies on creating a temp directory which is often blocked by endpoint protection.
  $site = (Resolve-Path ".\\.venv\\Lib\\site-packages").Path
  Invoke-Checked $venvPython "-c" "import ensurepip, pathlib, zipfile, ensurepip._bundled; site=r'$site'; p=next(pathlib.Path(ensurepip._bundled.__path__[0]).glob('pip-*.whl')); zipfile.ZipFile(p).extractall(site); print('pip extracted from', p)"

  $env:TEMP = $oldTemp
  $env:TMP = $oldTmp

  $code = Invoke-Proc $venvPython "-c" "import pip; print(pip.__version__)"
  if ($code -ne 0) {
    throw "pip could not be installed into .venv (even after fallback)."
  }
}

Ensure-VenvPip

Write-Host "Upgrading pip..."
Invoke-Checked $venvPython "-m" "pip" "install" "--upgrade" "pip"

Write-Host "Installing requirements.txt (venv-only)..."
Invoke-Checked $venvPython "-m" "pip" "install" "--require-virtualenv" "-r" "requirements.txt"

# System dependency: ffmpeg (required to render MP4s)
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue) -and -not $env:SENTREE_FFMPEG_PATH -and -not $env:SENTREE_FFMPEG) {
  Write-Warning "ffmpeg not found on PATH. MP4 rendering will fail until you install it. Try: winget install Gyan.FFmpeg"
}

# PyG CPU extras install (best-effort)
Write-Host "Installing PyG CPU extras (best-effort)..."
try {
  $pygUrl = (& $venvPython -c "import torch; v=torch.__version__.split('+')[0]; cuda=getattr(torch.version,'cuda',None); tag='cu' + cuda.replace('.','') if cuda else 'cpu'; print('https://data.pyg.org/whl/torch-' + v + '+' + tag + '.html')").Trim()
  & $venvPython -m pip install --require-virtualenv --only-binary=:all: torch-scatter torch-sparse -f $pygUrl | Out-Host
} catch {
  Write-Warning "PyG extras install failed - GCNConv may still work without them"
}

Write-Host "Creating output directories..."
New-Item -ItemType Directory -Force -Path "data/raw","data/processed","outputs/videos","outputs/roi","outputs/embeddings" | Out-Null

Write-Host ""
Write-Host "=== Setup Complete ==="
Write-Host "Next steps:"
Write-Host "  1. .\.venv\Scripts\python.exe data\generate_synthetic.py"
Write-Host "  2. Set your Gemini key (either):"
Write-Host "       - Add `GOOGLE_API_KEY=...` to .env, or"
Write-Host "       - `$env:GOOGLE_API_KEY = 'your-key'"
Write-Host "  3. .\.venv\Scripts\python.exe scripts\run_pipeline.py"
Write-Host "  4. .\.venv\Scripts\python.exe scripts\index_videos.py"
Write-Host "  5. .\.venv\Scripts\python.exe -m streamlit run src\dashboard\app.py"
