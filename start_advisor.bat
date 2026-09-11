@echo off
rem Starts the live advisor with no console window and opens the panel.
cd /d %~dp0
start "" ".venv\Scripts\pythonw.exe" -m uvicorn app.main:app
timeout /t 3 >nul
start http://127.0.0.1:8000/live
