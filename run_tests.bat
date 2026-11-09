@echo off
title FaceAttend AI - Master Test Suite Runner
color 0A
echo ============================================================
echo   FaceAttend AI - Automated Test Suite Runner
echo ============================================================
echo.

cd /d "%~dp0"
python tests\run_all_tests.py

echo.
pause
