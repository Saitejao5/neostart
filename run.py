#!/usr/bin/env python3
"""
run.py — Safe startup script with diagnostics.

This script validates configuration before starting Streamlit.
Usage: python run.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Check diagnostics first
print("\n🔍 Running diagnostics...\n")
result = subprocess.run([sys.executable, "diagnose.py"], cwd=Path(__file__).parent)

if result.returncode != 0:
    print("\n❌ Diagnostics failed. Fix the issues above before running Streamlit.")
    sys.exit(1)

# Start Streamlit
print("\n🚀 Starting ScholarBot...\n")
subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--logger.level=debug"
])
