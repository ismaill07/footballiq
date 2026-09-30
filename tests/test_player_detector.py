"""
Phase 2 tests for player/ball detection.

The detector wraps a real pretrained model, so these tests are split
into two kinds:
  - Pure logic tests (drawing, data structures) that need no model
    and always run.
  - A real inference smoke test that loads yolo26n.pt and confirms
    the wiring works end-to-end. This downloads ~5MB on first run
    and is skipped automatically if ultralytics isn't installed.
"""

import numpy as np
import pytest

from cv.detection.player_detector import Detection, FrameDetections


def test_frame_detections_holds_multiple_detections():
    detections = [
        Detection(label="person", confidence=0.91, bbox=[10, 20, 50, 100]),
        Detection(label="ball", confidence=0.55, bbox=[200, 150, 215, 165]),
    ]
    frame_detections = FrameDetections(frame_index=42, detections=detections)

    assert frame_detections.frame_index == 42
    assert len(frame_detections.detections) == 2
    assert frame_detections.detections[0].label == "person"
    assert frame_detections.detections[1].label == "ball"


def test_frame_detections_defaults_to_empty_list():
    frame_detections = FrameDetections(frame_index=0)
    assert frame_detections.detections == []


def test_detector_runs_inference_without_error():
    """
    Real end-to-end smoke test: loads the actual model and runs it on
    a synthetic frame. Doesn't assert specific detections (a random
    image has no real football content) - just confirms the model
    loads, runs, and returns results in the expected shape.
    """
    ultralytics = pytest.importorskip("ultralytics")
    from cv.detection.player_detector import PlayerDetector

    detector = PlayerDetector(confidence_threshold=0.3)
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)

    result = detector.detect(frame, frame_index=7)

    assert result.frame_index == 7
    assert isinstance(result.detections, list)
    # Every detection, if any, must be a properly-shaped Detection.
    for det in result.detections:
        assert det.label in ("person", "ball")
        assert 0.0 <= det.confidence <= 1.0
        assert len(det.bbox) == 4
