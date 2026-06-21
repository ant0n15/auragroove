@echo off
cd /d "%~dp0"
REM Auragroove launcher. Self-contained: uses the bundled acestep_engine venv. No Pinokio.

if not exist "%~dp0acestep_engine\.venv\Scripts\python.exe" (
    echo Environment not found - running first-time install...
    echo.
    call "%~dp0install.bat"
    if errorlevel 1 exit /b 1
)

"%~dp0acestep_engine\.venv\Scripts\python.exe" auragroove.py
pause
