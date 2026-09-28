"""
Central configuration loader for FootballIQ.

Reads environment variables from a .env file so secrets and
machine-specific settings never get hard-coded into source files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# --- Phase 1: video processing ---

# Keep 1 out of every N frames when extracting from a video.
# Higher = faster processing, fewer frames to work with downstream.
VIDEO_SAMPLE_RATE: int = int(os.getenv("VIDEO_SAMPLE_RATE", "5"))

# Resize extracted frames to this width (pixels), preserving aspect
# ratio. Set to 0 in .env to disable resizing.
_raw_resize_width = int(os.getenv("FRAME_RESIZE_WIDTH", "640"))
FRAME_RESIZE_WIDTH: int | None = _raw_resize_width if _raw_resize_width > 0 else None

RAW_VIDEO_DIR: str = os.getenv("RAW_VIDEO_DIR", "data/raw")
PROCESSED_FRAMES_DIR: str = os.getenv("PROCESSED_FRAMES_DIR", "data/processed")
