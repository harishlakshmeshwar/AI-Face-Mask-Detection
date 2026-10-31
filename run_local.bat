@echo off
REM Start the AI Face Mask Detection server on Windows.
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set PY=.venv\Scripts\python.exe
) else (
  set PY=python
)

echo Using %PY%
"%PY%" app.py
pause
