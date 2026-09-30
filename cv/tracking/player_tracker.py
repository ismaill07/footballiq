"""
Multi-object tracking for FootballIQ, built on Ultralytics' built-in
ByteTrack/BoT-SORT integration (bundled with the `ultralytics` package
already installed in Phase 2 - no new detector, just a new mode).

Tracking assigns a persistent track_id to each detected person/ball so
we can follow the SAME object across frames, instead of detecting it
fresh (with no memory) each time. Output follows the schema documented
in ARCHITECTURE.md section 7.

IMPORTANT LIMITATIONS, stated up front rather than discovered later:

1. We track on SAMPLED frames (e.g. every 5th frame from Phase 1), not
   every raw frame. Larger position jumps between tracked frames make
   ID switches more likely than tracking every frame would. This is a
   documented trade-off (speed vs. tracking stability), not a bug.

2. Detections are still labelled "person"/"ball", not "player" - the
   underlying detector still cannot distinguish players from referees
   or goalkeepers. Tracking gives us continuity of identity, not role.

3. track_id values are only meaningful within ONE PlayerTracker
   instance's lifetime, processing frames of ONE video IN ORDER.
   Creating a new instance, or skipping/reordering frames, resets or
   corrupts tracking state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ultralytics import YOLO

COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32

CLASS_ID_TO_LABEL = {
    COCO_PERSON_CLASS_ID: "person",
    COCO_SPORTS_BALL_CLASS_ID: "ball",
}


@dataclass
class TrackedObject:
    """A single tracked object in one frame."""

    track_id: Optional[int]
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates


@dataclass
class FrameTracks:
    """All tracked objects found in one frame."""

    frame_index: int
    tracks: List[TrackedObject] = field(default_factory=list)


class PlayerTracker:
    """
    Runs Ultralytics' built-in tracker across a SEQUENCE of frames,
    maintaining track_id continuity between calls via persist=True.

    Create ONE instance per video and reuse it, calling .track() once
    per frame IN ORDER. A fresh instance resets all tracking state.
    """

    def __init__(
        self,
        model_name: str = "yolo26n.pt",
        tracker: str = "bytetrack.yaml",
        confidence_threshold: float = 0.3,
    ):
        self.model = YOLO(model_name)
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold

    def track(self, frame, frame_index: int = 0) -> FrameTracks:
        """Run tracking on a single frame. Call in frame order, reusing this instance."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.confidence_threshold,
            classes=list(CLASS_ID_TO_LABEL.keys()),
            verbose=False,
        )

        tracks: List[TrackedObject] = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return FrameTracks(frame_index=frame_index, tracks=tracks)

        for box in result.boxes:
            class_id = int(box.cls[0])
            label = CLASS_ID_TO_LABEL.get(class_id, "unknown")
            confidence = float(box.conf[0])
            bbox = [float(v) for v in box.xyxy[0].tolist()]
            track_id = int(box.id[0]) if box.id is not None else None

            tracks.append(
                TrackedObject(track_id=track_id, label=label, confidence=confidence, bbox=bbox)
            )

        return FrameTracks(frame_index=frame_index, tracks=tracks)
