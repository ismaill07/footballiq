"""
Phase 5 tests for pitch mapping homography.

Pure math - fully testable with synthetic point coordinates, no
images or GUI needed.
"""

import pytest

from cv.pitch_mapping.homography import PitchCalibration, PitchMapper, ground_contact_point


def test_calibration_requires_exactly_four_points():
    with pytest.raises(ValueError):
        PitchCalibration(image_points=[(0, 0), (100, 0), (100, 100)])


def test_mapper_transforms_known_square_correctly():
    # A simple axis-aligned square: TL, TR, BR, BL.
    calibration = PitchCalibration(image_points=[(0, 0), (200, 0), (200, 200), (0, 200)])
    mapper = PitchMapper(calibration)

    center_x, center_y = mapper.image_to_pitch((100, 100))
    assert center_x == pytest.approx(0.5, abs=0.01)
    assert center_y == pytest.approx(0.5, abs=0.01)

    top_left_x, top_left_y = mapper.image_to_pitch((0, 0))
    assert top_left_x == pytest.approx(0.0, abs=0.01)
    assert top_left_y == pytest.approx(0.0, abs=0.01)

    bottom_right_x, bottom_right_y = mapper.image_to_pitch((200, 200))
    assert bottom_right_x == pytest.approx(1.0, abs=0.01)
    assert bottom_right_y == pytest.approx(1.0, abs=0.01)


def test_mapper_handles_perspective_distortion():
    # A trapezoid simulating a camera-angle perspective: the "far" edge
    # (top) is narrower than the "near" edge (bottom) - typical of a
    # real pitch rectangle viewed at an angle, not straight overhead.
    calibration = PitchCalibration(image_points=[(50, 0), (150, 0), (200, 100), (0, 100)])
    mapper = PitchMapper(calibration)

    # The trapezoid's own geometric center should still map close to (0.5, 0.5).
    center_x, center_y = mapper.image_to_pitch((100, 50))
    assert 0.3 < center_x < 0.7
    assert 0.3 < center_y < 0.7


def test_is_within_bounds():
    calibration = PitchCalibration(image_points=[(0, 0), (200, 0), (200, 200), (0, 200)])
    mapper = PitchMapper(calibration)

    assert mapper.is_within_bounds((0.5, 0.5)) is True
    assert mapper.is_within_bounds((2.0, 2.0)) is False
    assert mapper.is_within_bounds((-0.02, 0.5)) is True  # within default margin


def test_degenerate_points_raise_clear_error():
    # 4 collinear points - cannot form a valid quadrilateral/homography.
    calibration = PitchCalibration(image_points=[(0, 0), (50, 0), (100, 0), (150, 0)])
    with pytest.raises(ValueError):
        PitchMapper(calibration)


def test_ground_contact_point_is_bottom_center():
    bbox = [10, 20, 50, 100]  # x1, y1, x2, y2
    x, y = ground_contact_point(bbox)
    assert x == 30.0  # horizontal center
    assert y == 100.0  # bottom edge
