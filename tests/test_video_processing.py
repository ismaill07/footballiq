"""
Phase 1 tests for the video processing pipeline.

These generate a small synthetic video at runtime rather than
depending on the user's downloaded match_sample.mp4, since that file
is git-ignored (large raw data shouldn't live in version control)
and therefore won't exist in every environment, e.g. CI.
"""

import os

import cv2
import numpy as np
import pytest

from cv.video_processing.loader import VideoLoader
from cv.video_processing.frame_extractor import FrameExtractor


@pytest.fixture
def synthetic_video(tmp_path):
    """Create a tiny synthetic video: 20 frames, 10 fps, 100x80 pixels."""
    video_path = str(tmp_path / "synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, 10, (100, 80))

    for i in range(20):
        frame = np.full((80, 100, 3), fill_value=i * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    return video_path


def test_loader_reads_correct_metadata(synthetic_video):
    with VideoLoader(synthetic_video) as loader:
        assert loader.metadata.frame_count == 20
        assert loader.metadata.width == 100
        assert loader.metadata.height == 80
        assert loader.metadata.fps == pytest.approx(10, rel=0.1)


def test_loader_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoLoader("does_not_exist.mp4")


def test_extractor_sample_rate(synthetic_video):
    extractor = FrameExtractor(sample_rate=5)
    result = extractor.extract(synthetic_video, output_dir=None)
    # 20 frames, keeping every 5th (indices 0, 5, 10, 15) = 4 frames
    assert result.frames_sampled == 4
    assert result.frames_saved == 0  # no output_dir given, so nothing written


def test_extractor_saves_and_resizes(tmp_path, synthetic_video):
    output_dir = str(tmp_path / "out")
    extractor = FrameExtractor(sample_rate=1, resize_width=50)
    result = extractor.extract(synthetic_video, output_dir=output_dir)

    assert result.frames_saved == 20
    saved_files = os.listdir(output_dir)
    assert len(saved_files) == 20

    sample_frame = cv2.imread(os.path.join(output_dir, saved_files[0]))
    assert sample_frame.shape[1] == 50  # resized to target width
