@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
    echo Snachala zapustite:  powershell -ExecutionPolicy Bypass -File windows\build.ps1
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py
