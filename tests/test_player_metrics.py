"""
Phase 6 tests for player analytics. Pure math + synthetic frame data,
no model or GUI dependency.
"""

import math

import pytest

from analytics.player_metrics import build_trajectories, compute_player_metrics


def test_build_trajectories_groups_by_track_id():
    frames_data = [
        {"frame": 0, "objects": [
            {"track_id": 1, "class": "person", "team": "Team A", "bbox": [0, 0, 10, 10]},
            {"track_id": 2, "class": "person", "team": "Team B", "bbox": [50, 0, 60, 10]},
            {"track_id": None, "class": "ball", "team": None, "bbox": [30, 5, 35, 10]},
        ]},
        {"frame": 5, "objects": [
            {"track_id": 1, "class": "person", "team": "Team A", "bbox": [5, 0, 15, 10]},
        ]},
    ]
    trajectories = build_trajectories(frames_data)

    assert set(trajectories.keys()) == {1, 2}
    assert len(trajectories[1].samples) == 2
    assert len(trajectories[2].samples) == 1


def test_build_trajectories_skips_detections_without_track_id():
    frames_data = [
        {"frame": 0, "objects": [
            {"track_id": None, "class": "person", "team": "Team A", "bbox": [0, 0, 10, 10]},
        ]},
    ]
    trajectories = build_trajectories(frames_data)
    assert trajectories == {}


def test_compute_player_metrics_known_distance():
    # Ground point moves from (0,0) to (6,8) - a 6-8-10 right triangle,
    # so the straight-line distance is exactly 10.
    frames_data = [
        {"frame": 0, "objects": [{"track_id": 1, "class": "person", "team": "Team A", "bbox": [0, 0, 0, 0]}]},
        {"frame": 5, "objects": [{"track_id": 1, "class": "person", "team": "Team A", "bbox": [6, 8, 6, 8]}]},
        # bbox [x1,y1,x2,y2]=[6,8,6,8] -> ground_contact_point = ((6+6)/2, 8) = (6, 8)
    ]
    trajectories = build_trajectories(frames_data)
    metrics = compute_player_metrics(trajectories[1])

    assert metrics.frames_tracked == 2
    assert metrics.total_pixel_distance == pytest.approx(10.0, abs=0.01)  # distance (0,0)->(6,8)


def test_compute_player_metrics_single_sample_has_zero_distance():
    frames_data = [
        {"frame": 0, "objects": [{"track_id": 1, "class": "person", "team": "Team A", "bbox": [0, 0, 10, 10]}]},
    ]
    trajectories = build_trajectories(frames_data)
    metrics = compute_player_metrics(trajectories[1])

    assert metrics.frames_tracked == 1
    assert metrics.total_pixel_distance == 0.0
    assert metrics.avg_pixels_per_sample_step == 0.0


def test_compute_player_metrics_raises_on_empty_trajectory():
    from analytics.player_metrics import PlayerTrajectory
    empty = PlayerTrajectory(track_id=99, team="Team A")
    with pytest.raises(ValueError):
        compute_player_metrics(empty)
