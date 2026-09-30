"""
Player and ball detection for FootballIQ, using a pretrained YOLO model.

This wraps Ultralytics' YOLO detector behind a small, typed interface
so the rest of the pipeline (and later, the tracker in Phase 3) does
not depend on Ultralytics' API directly - if we ever swap detectors,
only this file changes. Output follows the schema documented in
ARCHITECTURE.md section 6.

IMPORTANT LIMITATION: COCO's pretrained "person" class covers players,
referees, and goalkeepers alike - the model cannot yet tell them apart.
That distinction is Phase 4's job (team/role classification), not this
module's. Detections are labelled "person" and "ball" here, deliberately
NOT "player", to avoid implying a distinction the model doesn't make.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ultralytics import YOLO

# COCO class IDs relevant to football.
COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32

CLASS_ID_TO_LABEL = {
    COCO_PERSON_CLASS_ID: "person",
    COCO_SPORTS_BALL_CLASS_ID: "ball",
}


@dataclass
class Detection:
    """A single detected object in one frame."""

    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates


@dataclass
class FrameDetections:
    """All detections found in one frame."""

    frame_index: int
    detections: List[Detection] = field(default_factory=list)


class PlayerDetector:
    """
    Runs a pretrained YOLO model and returns only football-relevant
    detections (person, sports ball), discarding every other COCO class.
    """

    def __init__(self, model_name: str = "yolo26n.pt", confidence_threshold: float = 0.3):
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold

    def detect(self, frame, frame_index: int = 0) -> FrameDetections:
        """Run detection on a single frame (a numpy array, e.g. from OpenCV)."""
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            classes=list(CLASS_ID_TO_LABEL.keys()),
            verbose=False,
        )

        detections: List[Detection] = []
        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = CLASS_ID_TO_LABEL.get(class_id, "unknown")
            confidence = float(box.conf[0])
            bbox = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(Detection(label=label, confidence=confidence, bbox=bbox))

        return FrameDetections(frame_index=frame_index, detections=detections)
