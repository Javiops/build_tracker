@echo off
rem In-game build advisor overlay (League must be Borderless or Windowed).
cd /d %~dp0
start "" .venv\Scripts\pythonw.exe -m app.overlay
