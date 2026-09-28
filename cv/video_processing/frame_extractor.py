"""
Frame sampling, resizing, and saving for FootballIQ's video pipeline.

Frame sampling exists because processing every frame of a match video
is usually unnecessary and expensive: consecutive frames barely differ
at typical broadcast frame rates, so we only keep every Nth frame
(configurable via VIDEO_SAMPLE_RATE in backend/config).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

import cv2

from cv.video_processing.loader import VideoLoader, VideoMetadata


@dataclass
class ExtractionResult:
    """Summary of a single frame-extraction run."""

    total_frames_in_video: int
    frames_sampled: int
    frames_saved: int
    processing_seconds: float
    output_dir: Optional[str]

    @property
    def processing_fps(self) -> float:
        if self.processing_seconds <= 0:
            return 0.0
        return self.frames_sampled / self.processing_seconds


class FrameExtractor:
    """
    Samples and optionally resizes frames from a video.

    sample_rate=5 keeps 1 out of every 5 frames (frame indices
    0, 5, 10, 15, ...). sample_rate=1 keeps every frame.
    """

    def __init__(self, sample_rate: int = 1, resize_width: Optional[int] = None):
        if sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")
        self.sample_rate = sample_rate
        self.resize_width = resize_width

    def _resize_if_needed(self, frame):
        if self.resize_width is None:
            return frame
        height, width = frame.shape[:2]
        if width == self.resize_width:
            return frame
        scale = self.resize_width / width
        new_height = int(height * scale)
        return cv2.resize(frame, (self.resize_width, new_height))

    def extract(self, video_path: str, output_dir: Optional[str] = None) -> ExtractionResult:
        """
        Run extraction over a video file.

        If output_dir is given, sampled (and resized) frames are saved
        there as JPEGs. If output_dir is None, frames are processed
        but not written to disk (useful for benchmarking throughput
        without disk I/O skewing the measurement).
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        start_time = time.perf_counter()
        frames_sampled = 0
        frames_saved = 0
        metadata: Optional[VideoMetadata] = None

        with VideoLoader(video_path) as loader:
            metadata = loader.metadata

            for index, frame in loader.frames():
                if index % self.sample_rate != 0:
                    continue

                frames_sampled += 1
                frame = self._resize_if_needed(frame)

                if output_dir:
                    filename = os.path.join(output_dir, f"frame_{index:06d}.jpg")
                    cv2.imwrite(filename, frame)
                    frames_saved += 1

        elapsed = time.perf_counter() - start_time

        return ExtractionResult(
            total_frames_in_video=metadata.frame_count,
            frames_sampled=frames_sampled,
            frames_saved=frames_saved,
            processing_seconds=elapsed,
            output_dir=output_dir,
        )
