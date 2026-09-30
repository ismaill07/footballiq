"""
Jersey color extraction for FootballIQ team classification.

Extracts a representative jersey color for a detected person's
bounding box, with two adjustments to reduce noise:

  1. Crop to the torso region only (roughly 20%-55% down the box,
     shoulders to waist) - avoids the head and legs/socks/shoes,
     which vary a lot (black shorts, colored socks) and don't
     reliably indicate team.
  2. Mask out pitch-green pixels from that crop - grass visible
     around a player's edges would otherwise skew the average color,
     especially for smaller/farther-away detections.

This is deliberately a simple, explainable heuristic (mean color of
a masked crop), not a learned embedding - matching the project's
"simplest reliable approach first" principle (ARCHITECTURE.md
section 8). Visual embeddings or a trained classifier are documented
future upgrades, not built here.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# HSV range for typical pitch grass green. Lighting varies between
# clips, so this is a reasonably wide band, not a precise match -
# it will occasionally over- or under-mask on unusual footage.
GRASS_HSV_LOWER = np.array([35, 40, 40])
GRASS_HSV_UPPER = np.array([85, 255, 255])


def extract_torso_crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """
    Crop the torso region of a person's bounding box: full box width,
    vertically the 20%-55% band from the top of the box.

    Returns None if the bbox is degenerate (zero or negative area
    after clamping to the frame) rather than raising - callers should
    treat None as "no color available for this detection."
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None

    height = y2 - y1
    torso_top = y1 + int(height * 0.20)
    torso_bottom = y1 + int(height * 0.55)

    if torso_bottom <= torso_top:
        return None

    return frame[torso_top:torso_bottom, x1:x2]


def dominant_jersey_color(frame: np.ndarray, bbox) -> Optional[Tuple[int, int, int]]:
    """
    Return the dominant BGR color of a player's jersey, or None if no
    reliable color could be extracted (degenerate crop, or the crop
    is almost entirely grass after masking - e.g. a very small or
    poorly-cropped detection).
    """
    crop = extract_torso_crop(frame, bbox)
    if crop is None or crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass_mask = cv2.inRange(hsv, GRASS_HSV_LOWER, GRASS_HSV_UPPER)
    non_grass_mask = cv2.bitwise_not(grass_mask)

    non_grass_pixels = crop[non_grass_mask > 0]

    # If almost everything got masked out as grass, this crop probably
    # doesn't contain a clear jersey view - report failure honestly
    # rather than returning a near-meaningless average.
    min_pixels = max(10, crop.shape[0] * crop.shape[1] * 0.1)
    if len(non_grass_pixels) < min_pixels:
        return None

    mean_color = non_grass_pixels.mean(axis=0)
    return tuple(int(c) for c in mean_color)  # BGR
