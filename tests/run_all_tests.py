import subprocess
import sys

def main():
    test_scripts = [
        'tests/test_core_cv.py',
        'tests/test_database.py',
        'tests/test_registration.py',
        'tests/test_attendance_engine.py',
        'tests/test_api_endpoints.py',
        'tests/test_views.py',
        'tests/test_audit_and_reports.py',
        'tests/test_custom_model_integration.py'
    ]

    print("=" * 65)
    print(" Running all 8 verification test suites...")
    print("=" * 65)

    all_passed = True
    for script in test_scripts:
        res = subprocess.run([sys.executable, script], capture_output=True, text=True)
        if res.returncode == 0:
            print(f" [PASS] {script}")
        else:
            print(f" [FAIL] {script}")
            if res.stderr:
                print(res.stderr)
            if res.stdout:
                print(res.stdout)
            all_passed = False

    print("=" * 65)
    if all_passed:
        print(" ALL 8 TEST SUITES PASSED WITH 100% SUCCESS!")
    else:
        print(" SOME TESTS FAILED. Inspect output above.")
    print("=" * 65)
    sys.exit(0 if all_passed else 1)

if __name__ == '__main__':
    main()

