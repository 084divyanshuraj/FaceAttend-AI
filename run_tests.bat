@echo off
title FaceAttend AI - Master Test Suite Runner
color 0A
echo ============================================================
echo   FaceAttend AI - Automated Test Suite Runner
echo ============================================================
echo.

cd /d "%~dp0"

python -c "
import subprocess, sys

test_scripts = [
    'tests/test_core_cv.py',
    'tests/test_database.py',
    'tests/test_registration.py',
    'tests/test_attendance_engine.py',
    'tests/test_api_endpoints.py',
    'tests/test_views.py',
    'tests/test_audit_and_reports.py'
]

print('=' * 65)
print('Running all 7 verification test suites...')
print('=' * 65)

all_passed = True
for script in test_scripts:
    res = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if res.returncode == 0:
        print(f' [PASS] {script}')
    else:
        print(f' [FAIL] {script}')
        print(res.stderr)
        print(res.stdout)
        all_passed = False

print('=' * 65)
if all_passed:
    print(' ALL 7 TEST SUITES PASSED WITH 100% SUCCESS!')
else:
    print(' SOME TESTS FAILED. Inspect output above.')
print('=' * 65)
sys.exit(0 if all_passed else 1)
"

echo.
pause

