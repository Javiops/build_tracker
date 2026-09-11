@echo off
rem Stops the background advisor server started by start_advisor.bat.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" | Where-Object { $_.CommandLine -match 'uvicorn' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
echo Advisor stopped.
pause
