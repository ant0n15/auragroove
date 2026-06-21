@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ==================================================
echo   Auragroove  -  Windows Installer
echo ==================================================
echo.
echo This builds the Python environment and downloads the
echo model weights for the bundled ACE-Step engine.
echo.
echo Requirements:
echo   - NVIDIA GPU + recent driver (CUDA 12.8 runtime ships with PyTorch)
echo   - Internet: ~4-5 GB (PyTorch) + ~12 GB (models) on first install
echo   - ~25 GB free disk
echo.

REM ---- locate uv (bundled first, then PATH, then install it) ----
set "UV="
if exist "%~dp0tools\uv.exe" set "UV=%~dp0tools\uv.exe"
if not defined UV ( where uv >nul 2>nul && set "UV=uv" )
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
if not defined UV (
    echo Installing uv package manager...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)
if not defined UV (
    echo.
    echo ERROR: could not find or install uv.
    echo Install it from https://docs.astral.sh/uv/ then re-run this.
    pause & exit /b 1
)
echo Using uv: !UV!
echo.

REM ---- build the venv from the exact lockfile ----
echo Building environment in acestep_engine\.venv ...
echo (downloads Python 3.12 + PyTorch/CUDA the first time - please wait)
echo.
"!UV!" sync --project "%~dp0acestep_engine" --no-dev
if errorlevel 1 (
    echo.
    echo ******** INSTALL FAILED ********
    echo Check the messages above ^(usually network or disk space^).
    pause & exit /b 1
)

REM ---- download model weights if missing ----
echo.
echo Checking / downloading model weights (first time pulls ~12 GB)...
"%~dp0acestep_engine\.venv\Scripts\python.exe" "%~dp0fetch_models.py"
if errorlevel 1 (
    echo WARNING: some model downloads failed - check your connection and re-run install.bat.
)

REM ---- quick sanity check ----
echo.
echo Verifying...
"%~dp0acestep_engine\.venv\Scripts\python.exe" -c "import torch, gradio, acestep; print('  torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
if errorlevel 1 (
    echo WARNING: verification import failed - the UI may still work, check above.
)

echo.
echo ==================================================
echo   Done!  Double-click  run_auragroove.bat  to start.
echo ==================================================
pause
