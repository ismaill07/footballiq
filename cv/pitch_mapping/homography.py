"""
Pitch mapping for FootballIQ: converts image-plane pixel coordinates
into an approximate top-down ("bird's eye") pitch-plane coordinate
system, using a homography computed from 4 manually-selected point
correspondences.

IMPORTANT, stated up front rather than discovered later: output
coordinates are in a NORMALIZED 0-1 range describing relative position
within the user-selected quadrilateral, NOT real-world meters.
Converting to true meters requires knowing the actual physical
dimensions of the pitch area the 4 points bound, which is not
known/confirmed for arbitrary footage. If real pitch dimensions
become known later, UNIT_SQUARE below can be changed from a unit
square to actual meter coordinates with no other code changes -
everything downstream (position mapping, future heatmaps/zones)
keeps working the same way.

See ARCHITECTURE.md section 9: "These values should be clearly
described as estimates when exact calibration is unavailable."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

# Destination is a unit square: (0,0) top-left, (1,0) top-right,
# (1,1) bottom-right, (0,1) bottom-left of the mapped region.
# Order MUST match the order points are given in PitchCalibration.
UNIT_SQUARE = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    dtype=np.float32,
)


@dataclass
class PitchCalibration:
    """
    4 image-plane points, in order: top-left, top-right, bottom-right,
    bottom-left of a REAL rectangle on the pitch (e.g. penalty box
    corners), as seen in the camera's perspective. Order matters - it
    must match UNIT_SQUARE's order above.
    """

    image_points: List[Tuple[float, float]]

    def __post_init__(self):
        if len(self.image_points) != 4:
            raise ValueError(
                f"PitchCalibration needs exactly 4 points, got {len(self.image_points)}"
            )


class PitchMapper:
    """
    Computes and applies a homography from image pixel coordinates to
    normalized (0-1) pitch-plane coordinates, based on a 4-point
    calibration.
    """

    def __init__(self, calibration: PitchCalibration):
        self.calibration = calibration
        src = np.array(calibration.image_points, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(src, UNIT_SQUARE)

        if self.homography_matrix is None:
            raise ValueError(
                "Could not compute homography from the given points - they "
                "may be collinear or otherwise degenerate. Pick 4 points "
                "that form a real, non-degenerate quadrilateral."
            )

    def image_to_pitch(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Transform one image-plane (x, y) pixel coordinate to normalized pitch-plane (x, y)."""
        src = np.array([[point]], dtype=np.float32)  # shape (1, 1, 2)
        dst = cv2.perspectiveTransform(src, self.homography_matrix)
        x, y = dst[0][0]
        return float(x), float(y)

    def is_within_bounds(self, pitch_point: Tuple[float, float], margin: float = 0.05) -> bool:
        """
        Sanity check: is this normalized point roughly within the
        mapped region (allowing a small margin, since players can
        legitimately stand just outside the 4 calibration points)?
        """
        x, y = pitch_point
        return (-margin <= x <= 1 + margin) and (-margin <= y <= 1 + margin)


def ground_contact_point(bbox) -> Tuple[float, float]:
    """
    Approximate where a player's feet touch the ground: the
    bottom-center of their bounding box. This is a standard
    approximation in sports computer vision - the box top is roughly
    the head, the box bottom edge is closest to actual ground contact.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)
