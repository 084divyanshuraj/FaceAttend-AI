@echo off
title FaceAttend AI - Smart Attendance System
color 0B
echo ============================================================
echo   FaceAttend AI - AI-Powered Attendance Management System
echo ============================================================
echo.

cd /d "%~dp0"

:: 1. Verify Python Installation
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10+ and add it to your system PATH.
    pause
    exit /b 1
)

:: 2. Ensure Pretrained Models Exist
echo [*] Checking pretrained ONNX models (YuNet and SFace)...
python scripts\download_models.py
if errorlevel 1 (
    echo [ERROR] Failed to verify models.
    pause
    exit /b 1
)

:: 3. Launch Default Browser after 2 seconds
start "" powershell -Command "Start-Sleep -Seconds 2; Start-Process 'http://localhost:5000'"

:: 4. Start Flask Application Server
echo.
echo [*] Starting FaceAttend AI server on http://localhost:5000 ...
echo [*] Press CTRL+C in this window to stop the server.
echo ============================================================
echo.

python app.py

pause

