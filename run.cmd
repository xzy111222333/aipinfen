@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "SHENLUN_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%SHENLUN_PYTHON%" (
    echo Run setup.cmd first.
    exit /b 1
)
"%SHENLUN_PYTHON%" "%~dp0shenlun.py" %*
