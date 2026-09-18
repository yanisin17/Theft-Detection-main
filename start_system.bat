@echo off
title Theft Guard AI Launcher
color 0A

set "PROJECT_DIR=%~dp0"
set "DASHBOARD_DIR=%PROJECT_DIR%dashboard"

echo ==================================================
echo   THEFT GUARD AI - Theft Detection System
echo   Starting up... Please wait.
echo ==================================================
echo.

:: 1. Check and install dashboard dependencies if needed
echo [1/3] Checking Frontend Dependencies...
if exist "%DASHBOARD_DIR%\node_modules" (
    echo   node_modules already installed.
) else (
    echo   Installing npm packages... Please wait.
    cd /d "%DASHBOARD_DIR%"
    call npm install
    cd /d "%PROJECT_DIR%"
)

:: 2. Start Backend (in new window, using test02 virtualenv)
echo [2/3] Starting Backend Service (Python/FastAPI, venv: .venv)...
start "Theft Guard Backend" cmd /k "cd /d "%PROJECT_DIR%" && "%PROJECT_DIR%.venv\Scripts\python.exe" backend.py"

:: Wait for backend to initialize (YOLO + MediaPipe + XGBoost loading takes longer)
timeout /t 12 /nobreak >nul

:: 3. Start Frontend (in new window)
echo [3/3] Starting Dashboard (Next.js)...
start "Theft Guard Dashboard" cmd /k "cd /d "%DASHBOARD_DIR%" && npm run dev"

:: Wait for frontend compilation
timeout /t 5 /nobreak >nul

:: 4. Open Browser
echo. Opening Browser (http://localhost:3000)...
start http://localhost:3000

echo.
echo ==================================================
echo   SYSTEM ACTIVE!
echo   Backend:  http://localhost:8000/docs
echo   Dashboard: http://localhost:3000
echo.
echo   To shutdown, close the opened command windows.
echo ==================================================
pause
