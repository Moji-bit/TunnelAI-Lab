@echo off
setlocal

REM Go to project root (folder of this .bat)
cd /d %~dp0

REM --- Find conda and activate ---
call conda --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Conda not found. Open "Anaconda Prompt" and run this file from there.
  pause
  exit /b 1
)

REM --- Create env if missing ---
call conda env list | findstr /i "tunnelai" >nul
if errorlevel 1 (
  echo [INFO] Creating conda env "tunnelai" from environment.yml ...
  call conda env create -f environment.yml
)

REM --- Activate env ---
call conda activate tunnelai

REM --- Run app ---
streamlit run ui/dashboard.py

endlocal
pause