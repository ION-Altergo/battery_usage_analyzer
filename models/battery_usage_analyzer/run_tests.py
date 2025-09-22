"""
Simple test runner for the Battery Usage Analyzer model.
"""

import sys
import os

# Add current directory to path to handle imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_battery_usage_analyzer import run_all_tests

if __name__ == "__main__":
    print("Running Battery Usage Analyzer tests...")
    passed, failed = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if failed == 0 else 1)
