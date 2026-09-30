"""
Phase 4 tests for jersey color extraction.

These use synthetic solid-color frames rather than real photos - the
extraction logic (cropping, grass masking, color averaging) is fully
testable in isolation without needing actual football footage.
"""

import numpy as np

from cv.team_classification.color_extractor import extract_torso_crop, dominant_jersey_color


def test_extract_torso_crop_degenerate_bbox_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert extract_torso_crop(frame, [50, 50, 50, 50]) is None  # zero-area bbox


def test_extract_torso_crop_returns_expected_band():
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    crop = extract_torso_crop(frame, [0, 0, 60, 100])
    assert crop is not None
    # Torso band is 20%-55% of height = rows 20 to 55 = 35 rows.
    assert crop.shape[0] == 35
    assert crop.shape[1] == 60


def test_dominant_jersey_color_returns_none_for_pure_grass():
    # Pure grass-green (BGR), filling the entire torso crop.
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    frame[:, :] = (40, 150, 40)
    color = dominant_jersey_color(frame, [0, 0, 60, 100])
    assert color is None


def test_dominant_jersey_color_detects_solid_red_jersey():
    # Solid strong red (BGR), no grass present at all.
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    frame[:, :] = (0, 0, 200)
    color = dominant_jersey_color(frame, [0, 0, 60, 100])
    assert color is not None
    b, g, r = color
    assert r > 150
    assert g < 50
    assert b < 50
