@echo off
setlocal
set "SHENLUN_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%SHENLUN_PYTHON%" (
    python -m venv "%~dp0.venv" || exit /b 1
)
"%SHENLUN_PYTHON%" -m pip install -r "%~dp0requirements.txt"
