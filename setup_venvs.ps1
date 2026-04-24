<#
.SYNOPSIS
    Creates a base Python 3.11 venv with shared packages, then 4 child venvs
    that inherit from it via a .pth symlink to the base site-packages.

.DESCRIPTION
    Run this script from: d:\STUDY 2\test\
    Re-run safely — it skips venv creation if .venv\ already exists (use -Force to recreate).

    HOW IT WORKS:
    1. Base venv is created normally with Python 3.11
    2. Heavy shared packages (torch CUDA, numpy, etc.) installed in base
    3. Child venvs are plain venvs (no --system-site-packages)
    4. A .pth file is placed in each child's site-packages to add
       the base venv's site-packages to the module search path
    5. Each child installs ONLY its own lightweight, unique deps
    6. pip constraints prevent children from re-installing shared packages

.PARAMETER Force
    If set, deletes and recreates all venvs from scratch.

.EXAMPLE
    .\setup_venvs.ps1
    .\setup_venvs.ps1 -Force
#>

param(
    [switch]$Force
)

$ErrorActionPreference = "Continue"
$ROOT = $PSScriptRoot  # d:\STUDY 2\test

# ── Paths ────────────────────────────────────────────────────────────────────
$BASE_VENV       = Join-Path $ROOT ".venv"
$BASE_SITE_PKGS  = Join-Path $BASE_VENV "Lib\site-packages"

$CHILD_VENVS = @(
    @{ Name = "app";              Path = Join-Path $ROOT "app\.venv";                               Reqs = Join-Path $ROOT "app\requirements.txt" },
    @{ Name = "cohere-asr";       Path = Join-Path $ROOT "transcribe\cohere-asr\.venv";             Reqs = Join-Path $ROOT "transcribe\cohere-asr\requirements.txt" },
    @{ Name = "qwen-asr";         Path = Join-Path $ROOT "transcribe\qwen-asr\.venv";               Reqs = Join-Path $ROOT "transcribe\qwen-asr\requirements.txt" },
    @{ Name = "whisperx";   Path = Join-Path $ROOT "transcribe\whisperx\.venv";     Reqs = Join-Path $ROOT "transcribe\whisperx\requirements.txt" }
)

$PY311 = "py"  # Windows py launcher

# Shared packages that should NEVER be installed in child venvs
# (children inherit them from base via .pth)
$SHARED_PACKAGES = @(
    "torch", "torchvision", "torchaudio",
    "numpy", "fastapi", "uvicorn", "requests",
    "python-dotenv", "librosa", "soundfile",
    "pydantic", "pydantic-core"
)

# ── Helper ───────────────────────────────────────────────────────────────────
function Write-Step($msg) { Write-Host "`n$('='*60)`n  $msg`n$('='*60)" -ForegroundColor Cyan }

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — BASE VENV
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "STEP 1: Base venv  ->  $BASE_VENV"

if ($Force -and (Test-Path $BASE_VENV)) {
    Write-Host "  [Force] Removing existing base venv..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $BASE_VENV
}

if (-not (Test-Path $BASE_VENV)) {
    Write-Host "  Creating base venv (Python 3.11)..."
    & $PY311 -3.11 -m venv $BASE_VENV
} else {
    Write-Host "  Base venv already exists -- skipping creation." -ForegroundColor Green
}

# Install shared packages in base
$basePip = Join-Path $BASE_VENV "Scripts\pip.exe"
$baseReqs = Join-Path $ROOT "requirements.txt"
Write-Host "  Installing shared packages in base venv..."
& $basePip install -r $baseReqs --quiet
Write-Host "  Base venv ready!" -ForegroundColor Green

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CREATE CONSTRAINT FILE
# ══════════════════════════════════════════════════════════════════════════════
# This prevents children from pulling in their own copy of shared packages.
# pip's --constraint flag pins packages to already-installed versions.
Write-Step "STEP 2: Creating constraints file"

$constraintsFile = Join-Path $ROOT "constraints.txt"
$basePython = Join-Path $BASE_VENV "Scripts\python.exe"

# Generate constraints from what's currently installed in base
$installed = & $basePip freeze 2>$null
$constraintLines = @()
foreach ($line in $installed) {
    # Only constrain shared packages
    foreach ($pkg in $SHARED_PACKAGES) {
        $pkgNorm = $pkg.ToLower().Replace("-", "_").Replace(".", "_")
        $lineNorm = $line.Split("==")[0].ToLower().Replace("-", "_").Replace(".", "_")
        if ($lineNorm -eq $pkgNorm) {
            $constraintLines += $line
        }
    }
}
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[IO.File]::WriteAllLines($constraintsFile, $constraintLines, $utf8NoBom)
Write-Host "  Constraints file: $constraintsFile"
Write-Host "  Constrained packages: $($constraintLines.Count)"

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — CHILD VENVS
# ══════════════════════════════════════════════════════════════════════════════
foreach ($child in $CHILD_VENVS) {
    Write-Step "STEP 3: Child venv [$($child.Name)]  ->  $($child.Path)"

    if ($Force -and (Test-Path $child.Path)) {
        Write-Host "  [Force] Removing existing child venv..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $child.Path
    }

    if (-not (Test-Path $child.Path)) {
        Write-Host "  Creating child venv..."
        & $PY311 -3.11 -m venv $child.Path
    } else {
        Write-Host "  Child venv already exists -- skipping creation." -ForegroundColor Green
    }

    # ── Drop .pth file to link child -> base site-packages ──
    $childSitePkgs = Join-Path $child.Path "Lib\site-packages"
    $pthFile = Join-Path $childSitePkgs "_base_venv.pth"
    # Write .pth WITHOUT BOM — Python's site.py silently ignores BOM-prefixed .pth files
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($pthFile, ($BASE_SITE_PKGS + "`n"), $utf8NoBom)
    Write-Host "  Linked to base via: $pthFile"

    # ── Install child-specific deps (with constraints to skip shared pkgs) ──
    $childPip = Join-Path $child.Path "Scripts\pip.exe"

    if (Test-Path $child.Reqs) {
        Write-Host "  Installing module-specific packages..."
        & $childPip install -r $child.Reqs --quiet
    }

    Write-Host "  [$($child.Name)] ready!" -ForegroundColor Green
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CLEAN UP: Remove any duplicate shared packages from children
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "STEP 4: Cleaning duplicate shared packages from children"

foreach ($child in $CHILD_VENVS) {
    $childPip = Join-Path $child.Path "Scripts\pip.exe"
    $childFrozen = & $childPip freeze --local 2>$null  # --local = only this venv

    foreach ($line in $childFrozen) {
        foreach ($pkg in $SHARED_PACKAGES) {
            $pkgNorm = $pkg.ToLower().Replace("-", "_").Replace(".", "_")
            $lineNorm = $line.Split("==")[0].ToLower().Replace("-", "_").Replace(".", "_")
            if ($lineNorm -eq $pkgNorm) {
                Write-Host "  [$($child.Name)] Removing duplicate: $($line.Split('==')[0])" -ForegroundColor Yellow
                & $childPip uninstall $line.Split("==")[0] -y --quiet 2>$null
            }
        }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════
Write-Step "STEP 5: Verification"

Write-Host "`n  Checking torch (from base) in each child venv:" -ForegroundColor Cyan
foreach ($child in $CHILD_VENVS) {
    $childPython = Join-Path $child.Path "Scripts\python.exe"
    $torchCheck = & $childPython -c "import torch; print('torch', torch.__version__, 'CUDA:', torch.cuda.is_available())" 2>&1
    $color = if ($torchCheck -match "cu128") { "Green" } else { "Red" }
    Write-Host "    [$($child.Name)] $torchCheck" -ForegroundColor $color
}

Write-Host "`n  Checking module-specific packages:" -ForegroundColor Cyan

$specifics = @(
    @{ Name = "app";              Cmd = "import demucs; print('demucs OK')" },
    @{ Name = "cohere-asr";       Cmd = "import transformers; print('transformers', transformers.__version__)" },
    @{ Name = "qwen-asr";         Cmd = "import qwen_asr; print('qwen_asr OK')" },
    @{ Name = "whisperx";   Cmd = "import whisperx; print('whisperx OK')" },
)

foreach ($spec in $specifics) {
    $venvPath = ($CHILD_VENVS | Where-Object { $_.Name -eq $spec.Name }).Path
    $childPython = Join-Path $venvPath "Scripts\python.exe"
    $result = & $childPython -c $spec.Cmd 2>&1
    Write-Host "    [$($spec.Name)] $result"
}

Write-Step "ALL DONE!"
Write-Host ""
Write-Host "  Base venv:    $BASE_VENV"
Write-Host "  Child venvs:  app, cohere-asr, qwen-asr, whisperx"
Write-Host ""
Write-Host "  To activate a child venv manually:"
Write-Host "    .\app\.venv\Scripts\Activate.ps1"
Write-Host "    .\transcribe\cohere-asr\.venv\Scripts\Activate.ps1"
Write-Host "    .\transcribe\qwen-asr\.venv\Scripts\Activate.ps1"
Write-Host "    .\transcribe\whisperx\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  Or use VS Code:  Ctrl+Shift+P -> Tasks: Run Task -> Start All Services"
