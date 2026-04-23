@echo off
REM GTNM Model Training Script for Windows
REM =======================================
REM Optimized for RTX 3090 (Ampere SM_8.6) + TensorFlow 1.x

echo.
echo ========================================
echo   GTNM Method Name Recommendation
echo   Training Script (RTX 3090)
echo ========================================
echo.

REM Record overall start time
set "START_TIME=%time%"

REM Activate conda environment
echo [INFO] Activating conda environment: GTNM...
call conda activate GTNM
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate conda environment 'GTNM'
    echo [INFO] Please ensure conda is installed and GTNM environment exists
    pause
    exit /b 1
)
echo [SUCCESS] Conda environment activated: GTNM
echo.

cd /d "%~dp0"

echo [INFO] Starting training...
echo [INFO] Working directory: %cd%
echo [INFO] Data path: ./predata/
echo [INFO] Model output: ./saved/
echo [INFO] Batch size: 64
echo [INFO] Total epochs: 50 (default, can be changed with --num_epochs)
echo.

python model/train.py --gpu 0 --pro True --batch_size 64

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo   Training Completed Successfully!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo   Training Failed or Interrupted!
    echo ========================================
    echo.
)

echo [INFO] Started at: %START_TIME%
echo [INFO] Finished at: %time%
echo.
pause
