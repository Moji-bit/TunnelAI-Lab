@echo off
setlocal

REM Usage: scripts\verify_clean_machine.bat
REM Windows equivalent of verify_clean_machine.sh

cd /d %~dp0\..

where conda >nul 2>&1
if errorlevel 1 (
  echo [ERROR] conda not found on PATH.
  exit /b 1
)

set ENV_NAME=tunnelai

echo [1/5] Removing old env if exists: %ENV_NAME%
call conda env remove -n %ENV_NAME% -y >nul 2>&1

echo [2/5] Creating env from environment.lock.yml
call conda env create -f environment.lock.yml
if errorlevel 1 exit /b 1

echo [3/5] Running import sanity checks
call conda run -n %ENV_NAME% python -c "import numpy,pandas,yaml,streamlit; print('imports_ok')"
if errorlevel 1 exit /b 1

echo [4/5] Running matrix planner
call conda run -n %ENV_NAME% python scripts/run_experiment_matrix.py --mode plan
if errorlevel 1 exit /b 1

echo [5/5] Running recorder smoke test
call conda run -n %ENV_NAME% python streaming/run_record.py --scenario scenarios/stau_case_00.json --out data/raw/_clean_check.csv --max-seconds 2
if errorlevel 1 exit /b 1

echo [OK] Clean-machine verification passed.
endlocal
