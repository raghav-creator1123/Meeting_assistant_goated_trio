@echo off
:: Meeting Assistant launcher (Windows)
:: Usage: run_app.bat [--share] [--port 8080] [--api]
::
:: Uses .venv\Scripts\python.exe if a virtual environment exists, otherwise "python".
:: Set PYTHON yourself to use a specific interpreter, e.g.
::   set PYTHON=C:\Python311\python.exe

cd /d "%~dp0"
if not defined PYTHON (
    if exist ".venv\Scripts\python.exe" (
        set PYTHON=.venv\Scripts\python.exe
    ) else (
        set PYTHON=python
    )
)
echo [Launcher] Using %PYTHON%
"%PYTHON%" app.py %*
