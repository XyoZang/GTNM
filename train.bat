@echo off
REM GTNM Model Training Script for Windows
REM =======================================

echo.
echo ========================================
echo   GTNM Method Name Recommendation
echo   Training Script
echo ========================================
echo.

REM Activate conda environment (uncomment if needed)
REM call conda activate GTNM

cd /d "%~dp0model"

echo [INFO] Starting training...
echo [INFO] Data path: ./predata/
echo [INFO] Model output: ./saved/
echo.

python train.py --gpu 0 --pro True

echo.
echo [DONE] Training completed!
pause
