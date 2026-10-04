"""
Phase 6 tests for team analytics. Pure math, no model or GUI dependency.
"""

import pytest

from analytics.team_metrics import compute_team_frame_metrics, summarize_team_metrics, TeamFrameMetrics


def test_returns_none_for_fewer_than_two_players():
    assert compute_team_frame_metrics(0, "Team A", []) is None
    assert compute_team_frame_metrics(0, "Team A", [(10, 10)]) is None


def test_known_width_and_compactness():
    # Two players at (0, 0) and (10, 0): width = 10, both players are
    # 5 pixels from the centroid (5, 0), so compactness = 5.
    metrics = compute_team_frame_metrics(0, "Team A", [(0, 0), (10, 0)])

    assert metrics is not None
    assert metrics.player_count == 2
    assert metrics.avg_position == (5.0, 0.0)
    assert metrics.width_px == pytest.approx(10.0)
    assert metrics.compactness_px == pytest.approx(5.0)


def test_summarize_empty_list_reports_no_data():
    summary = summarize_team_metrics([])
    assert summary["frames_with_data"] == 0
    assert summary["avg_width_px"] is None
    assert summary["avg_compactness_px"] is None


def test_summarize_averages_across_frames():
    m1 = TeamFrameMetrics(frame_index=0, team="Team A", player_count=2,
                           avg_position=(0, 0), width_px=10.0, compactness_px=5.0)
    m2 = TeamFrameMetrics(frame_index=5, team="Team A", player_count=2,
                           avg_position=(0, 0), width_px=20.0, compactness_px=7.0)

    summary = summarize_team_metrics([m1, m2])

    assert summary["frames_with_data"] == 2
    assert summary["avg_width_px"] == pytest.approx(15.0)
    assert summary["avg_compactness_px"] == pytest.approx(6.0)
