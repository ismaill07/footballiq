"""
Phase 3 tests for player/ball tracking.

Tracking wraps a real pretrained model + tracker, so tests are split
the same way as Phase 2's: pure logic tests (data structures) that
always run, and a real end-to-end smoke test that loads the model
and confirms tracking runs without error across a short SEQUENCE of
frames using one shared tracker instance (required for track_id
persistence). It does not assert specific track IDs, since synthetic
frames have no real football content for the model to recognize.
"""

import numpy as np
import pytest

from cv.tracking.player_tracker import TrackedObject, FrameTracks


def test_frame_tracks_holds_multiple_tracks():
    tracks = [
        TrackedObject(track_id=1, label="person", confidence=0.9, bbox=[0, 0, 10, 10]),
        TrackedObject(track_id=2, label="ball", confidence=0.6, bbox=[20, 20, 25, 25]),
    ]
    frame_tracks = FrameTracks(frame_index=3, tracks=tracks)

    assert frame_tracks.frame_index == 3
    assert len(frame_tracks.tracks) == 2
    assert frame_tracks.tracks[0].track_id == 1


def test_frame_tracks_defaults_to_empty_list():
    frame_tracks = FrameTracks(frame_index=0)
    assert frame_tracks.tracks == []


def test_tracker_runs_across_a_sequence_without_error():
    """
    Real end-to-end smoke test: loads the actual model + ByteTrack and
    runs it across several synthetic frames IN ORDER, reusing ONE
    tracker instance. Confirms the wiring and persist=True behavior
    work without erroring - does not assert specific IDs, since
    random frames have no real content to track.
    """
    pytest.importorskip("ultralytics")
    from cv.tracking.player_tracker import PlayerTracker

    tracker = PlayerTracker(confidence_threshold=0.3)

    for i in range(3):
        frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
        result = tracker.track(frame, frame_index=i)

        assert result.frame_index == i
        assert isinstance(result.tracks, list)
        for t in result.tracks:
            assert t.label in ("person", "ball")
            assert 0.0 <= t.confidence <= 1.0
            assert len(t.bbox) == 4
