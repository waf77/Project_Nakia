@echo off
title Nakia Session Manager (Minimized)

:: Auto-minimize this launcher window immediately so it stays out of your way
if not "%1"=="min" start /min cmd /c "%~f0" min & exit /b

:: Ensure Nakia's shared folder exists
if not exist "D:\Nakia Stuff" mkdir "D:\Nakia Stuff"

:: Give Ollama a 24K context window so all 8 MCP servers fit comfortably alongside chat history
set OLLAMA_CONTEXT_LENGTH=24576

:: 1. Start Ollama if it isn't running already
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I /N "ollama.exe">NUL
if "%ERRORLEVEL%"=="1" (
    start "Ollama Server" /min ollama serve
    timeout /t 3 /nobreak >nul
)

:: 2. Start ZONOS3-Mini TTS API minimized
cd /d "C:\Users\wayne\AppData\Local\Programs\zonos3"
if exist ".venv\Scripts\python.exe" (
    start "ZONOS3-Mini TTS" /min cmd /c ".\.venv\Scripts\python.exe zonos_api.py"
) else (
    start "ZONOS3-Mini TTS" /min cmd /c "python zonos_api.py"
)

:: Give ZONOS3-Mini 5 seconds to load onto your RTX 5060
timeout /t 5 /nobreak >nul

:: 3. Start Open-LLM-VTuber backend minimized
cd /d "C:\Users\wayne\AppData\Local\Programs\open-llm-vtuber\Open-LLM-VTuber"
start "Open-LLM-VTuber Server" /min cmd /c ".\.venv\Scripts\python.exe run_server.py"

:: Wait until Open-LLM-VTuber is actively listening on port 12393 (up to 60 seconds for first-time tool downloads)
powershell -NoProfile -Command "for ($i=0; $i -lt 60; $i++) { if ((Test-NetConnection -ComputerName 127.0.0.1 -Port 12393 -InformationLevel Quiet -WarningAction SilentlyContinue)) { break }; Start-Sleep -Seconds 1 }"

:: 4. Launch Nakia Desktop App and WAIT until you exit it
start /wait "" "C:\Users\wayne\AppData\Local\Programs\open-llm-vtuber\open-llm-vtuber-electron.exe"

:: ===================================================
:: CLEANUP: Runs automatically as soon as you exit Nakia
:: ===================================================
taskkill /FI "WINDOWTITLE eq ZONOS3-Mini TTS*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Open-LLM-VTuber Server*" /T /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Ollama Server*" /T /F >nul 2>&1
taskkill /IM ollama.exe /F >nul 2>&1

exit
