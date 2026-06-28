@echo off
setlocal
cd /d "%~dp0"
REM Fine-tune ACE-Step on your own tracks (LoRA). Uses the bundled engine venv.

set "PY=%~dp0acestep_engine\.venv\Scripts\python.exe"
if not exist "%PY%" goto :noenv

if not exist "%~dp0finetune\dataset.json" goto :scaffold
goto :train

:scaffold
echo.
echo No dataset.json yet - scanning finetune\dataset for audio...
"%PY%" "%~dp0finetune\prepare_dataset.py" --audio-dir "%~dp0finetune\dataset" --trigger agphonk
if errorlevel 1 goto :fail
echo.
echo ================================================================
echo  Created finetune\dataset.json
echo  Put your tracks in finetune\dataset\  (mp3/wav/flac).
echo  For vocals: add a same-name .txt next to each track with its
echo  lyrics, and optionally a .caption.txt with a style description.
echo  Edit finetune\dataset.json if you like, then run this again.
echo ================================================================
goto :end

:train
echo.
echo Training LoRA (turbo, rank 24) from finetune\dataset.json ...
"%PY%" "%~dp0finetune\run_finetune.py" --dataset-json "%~dp0finetune\dataset.json" --name myphonk_turbo --variant turbo --tensors "%~dp0finetune\cache\tensors_turbo" --rank 24 --epochs 30 --max-duration 30
if errorlevel 1 goto :fail
echo.
echo ================================================================
echo  Done!  Restart run_auragroove.bat, then open the LoRA panel and
echo  set  LoRA = myphonk_turbo  (DiT Model = stock acestep-v15-turbo).
echo  Put your trigger word (agphonk) in the caption.
echo ================================================================
goto :end

:noenv
echo Environment not found - running first-time install...
echo.
call "%~dp0install.bat"
goto :end

:fail
echo.
echo Something failed - see the messages above.

:end
echo.
pause
