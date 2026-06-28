@echo off
cd /d "%~dp0"
setlocal
REM Fine-tune ACE-Step on your own tracks (LoRA). Uses the bundled engine venv.

set "PY=%~dp0acestep_engine\.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo Environment not found - running first-time install...
    echo.
    call "%~dp0install.bat"
    if errorlevel 1 exit /b 1
)

REM ---- Step 1: scaffold dataset.json on first run, then let the user edit it ----
if not exist "%~dp0finetune\dataset.json" (
    echo.
    echo No dataset.json yet - scanning finetune\dataset for audio...
    "%PY%" "%~dp0finetune\prepare_dataset.py" --audio-dir "%~dp0finetune\dataset" --trigger agphonk
    if errorlevel 1 exit /b 1
    echo.
    echo ================================================================
    echo  Created finetune\dataset.json
    echo  Put your tracks in finetune\dataset\  (mp3/wav/flac).
    echo  For vocals: add a same-name .txt next to each track with its
    echo  lyrics, and optionally a .caption.txt with a style description.
    echo  Edit finetune\dataset.json if you like, then run this again.
    echo ================================================================
    pause
    exit /b 0
)

REM ---- Step 2: preprocess + train the LoRA ----
echo.
echo Training LoRA from finetune\dataset.json ...
"%PY%" "%~dp0finetune\run_finetune.py" --dataset-json "%~dp0finetune\dataset.json" --name myphonk --variant base --rank 16 --epochs 30
echo.
pause
