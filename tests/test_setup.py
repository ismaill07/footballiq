"""
Phase 0 smoke test.

This doesn't test any FootballIQ logic yet - it only proves that
the environment (Python + OpenCV) is correctly installed and
importable, which is the actual Phase 0 deliverable.
"""

import cv2


def test_opencv_importable():
    assert cv2.__version__ is not None


def test_opencv_version_format():
    # Sanity check that the version string looks like "x.y.z"
    parts = cv2.__version__.split(".")
    assert len(parts) >= 2
