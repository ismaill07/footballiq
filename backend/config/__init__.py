"""
Central configuration loader for FootballIQ.

Reads environment variables from a .env file so secrets and
machine-specific settings never get hard-coded into source files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
