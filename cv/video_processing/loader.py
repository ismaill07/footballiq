"""
Video loading and metadata utilities for FootballIQ.

Wraps OpenCV's VideoCapture with clear error handling and a simple,
typed interface for the rest of the pipeline to build on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2


@dataclass
class VideoMetadata:
    """Metadata describing a loaded video file."""

    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


class VideoLoader:
    """
    Opens a video file and exposes its metadata and frames.

    Usage:
        with VideoLoader("match.mp4") as loader:
            print(loader.metadata)
            for index, frame in loader.frames():
                ...
    """

    def __init__(self, video_path: str):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self._capture = cv2.VideoCapture(video_path)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"OpenCV could not open video file: {video_path}. "
                "The file may be corrupted or use an unsupported codec."
            )

        self.metadata = VideoMetadata(
            path=video_path,
            fps=self._capture.get(cv2.CAP_PROP_FPS),
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def frames(self):
        """Yield (frame_index, frame) tuples for every frame in the video."""
        index = 0
        while True:
            success, frame = self._capture.read()
            if not success:
                break
            yield index, frame
            index += 1

    def release(self):
        self._capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
