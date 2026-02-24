@echo off
setlocal
cd /d %~dp0\..
python scripts\run_reports_after_batch.py
endlocal
