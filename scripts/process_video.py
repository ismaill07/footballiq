"""
Phase 1 CLI: run the video processing pipeline on a single match video.

Usage (from the project root, with venv activated):
    python scripts/process_video.py
    python scripts/process_video.py --video data/raw/match_sample.mp4
    python scripts/process_video.py --video data/raw/match_sample.mp4 --sample-rate 3
"""

import argparse
import os
import sys

# Allow running this script directly without installing the project as a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import (
    VIDEO_SAMPLE_RATE,
    FRAME_RESIZE_WIDTH,
    RAW_VIDEO_DIR,
    PROCESSED_FRAMES_DIR,
)
from cv.video_processing.loader import VideoLoader
from cv.video_processing.frame_extractor import FrameExtractor


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 1 video pipeline")
    parser.add_argument(
        "--video",
        default=os.path.join(RAW_VIDEO_DIR, "match_sample.mp4"),
        help="Path to the input video file",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=VIDEO_SAMPLE_RATE,
        help="Keep 1 out of every N frames",
    )
    args = parser.parse_args()

    print(f"Loading video: {args.video}")
    try:
        with VideoLoader(args.video) as loader:
            meta = loader.metadata
            print(f"  FPS:        {meta.fps:.2f}")
            print(f"  Frames:     {meta.frame_count}")
            print(f"  Resolution: {meta.width}x{meta.height}")
            print(f"  Duration:   {meta.duration_seconds:.2f}s")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join(PROCESSED_FRAMES_DIR, video_name)

    print(
        f"\nExtracting frames (sample_rate={args.sample_rate}, "
        f"resize_width={FRAME_RESIZE_WIDTH})..."
    )
    extractor = FrameExtractor(sample_rate=args.sample_rate, resize_width=FRAME_RESIZE_WIDTH)
    result = extractor.extract(args.video, output_dir=output_dir)

    print("\nDone.")
    print(f"  Total frames in video: {result.total_frames_in_video}")
    print(f"  Frames sampled:        {result.frames_sampled}")
    print(f"  Frames saved:          {result.frames_saved}")
    print(f"  Processing time:       {result.processing_seconds:.2f}s")
    print(f"  Processing throughput: {result.processing_fps:.2f} frames/sec (measured)")
    print(f"  Output directory:      {output_dir}")


if __name__ == "__main__":
    main()
