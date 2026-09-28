"""
Pytest configuration for FootballIQ.

Pytest auto-discovers this file and runs it before collecting tests.
Its job here is to add the project root to sys.path, so test files
can import project packages (cv, backend, analytics, ai) the same
way scripts/process_video.py already does manually.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))