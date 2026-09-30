"""
Phase 3 CLI: run multi-object tracking on frames already extracted by
Phase 1's process_video.py. Saves annotated images (boxes + track IDs,
each ID drawn in a consistent color) and a tracks.json file matching
the schema in ARCHITECTURE.md section 7.

Usage (from the project root, with venv activated):
    python scripts/track_players.py --frames-dir data/processed/match_sample2
    python scripts/track_players.py --frames-dir data/processed/match_sample2 --tracker botsort.yaml
"""

import argparse
import colorsys
import glob
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from cv.tracking.player_tracker import PlayerTracker, FrameTracks


def color_for_id(track_id):
    """Deterministic, visually distinct BGR color per track ID (for OpenCV)."""
    if track_id is None:
        return (128, 128, 128)
    hue = (track_id * 0.15) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_tracks(frame, frame_tracks: FrameTracks):
    for t in frame_tracks.tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox]
        color = color_for_id(t.track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        id_text = f"ID {t.track_id}" if t.track_id is not None else "ID ?"
        label_text = f"{id_text} {t.label} {t.confidence:.2f}"
        cv2.putText(
            frame, label_text, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 3 tracking")
    parser.add_argument("--frames-dir", default="data/processed/match_sample2")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument(
        "--tracker", default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
    )
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg")))
    if not frame_paths:
        print(f"ERROR: no frames found in {args.frames_dir}.")
        print("Run scripts/process_video.py first to generate frames.")
        sys.exit(1)

    print(f"Found {len(frame_paths)} frames in {args.frames_dir}")
    print(f"Loading model: {args.model} with tracker: {args.tracker}")
    tracker = PlayerTracker(
        model_name=args.model, tracker=args.tracker, confidence_threshold=args.confidence
    )

    annotated_dir = args.frames_dir.rstrip("/\\") + "_tracked"
    os.makedirs(annotated_dir, exist_ok=True)

    all_results = []
    id_frame_counts = defaultdict(int)
    total_tracks = 0
    start_time = time.perf_counter()

    # IMPORTANT: frames must be processed in order, with the SAME tracker
    # instance, for track_id continuity to mean anything.
    for path in frame_paths:
        frame = cv2.imread(path)
        frame_index = int(os.path.splitext(os.path.basename(path))[0].split("_")[1])

        result = tracker.track(frame, frame_index=frame_index)
        total_tracks += len(result.tracks)

        for t in result.tracks:
            if t.track_id is not None:
                id_frame_counts[t.track_id] += 1

        annotated = draw_tracks(frame.copy(), result)
        out_path = os.path.join(annotated_dir, os.path.basename(path))
        cv2.imwrite(out_path, annotated)

        all_results.append({
            "frame": result.frame_index,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class": t.label,
                    "confidence": round(t.confidence, 3),
                    "bbox": t.bbox,
                }
                for t in result.tracks
            ],
        })

    elapsed = time.perf_counter() - start_time
    fps = len(frame_paths) / elapsed if elapsed > 0 else 0.0

    json_path = os.path.join(annotated_dir, "tracks.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    unique_ids = len(id_frame_counts)
    avg_track_length = (sum(id_frame_counts.values()) / unique_ids) if unique_ids > 0 else 0.0
    single_frame_ids = sum(1 for count in id_frame_counts.values() if count == 1)

    print("\nDone.")
    print(f"  Frames processed:          {len(frame_paths)}")
    print(f"  Total tracked detections:  {total_tracks}")
    print(f"  Unique track IDs assigned: {unique_ids}")
    print(f"  Avg frames per track ID:   {avg_track_length:.2f}")
    print(f"  Track IDs seen only once:  {single_frame_ids} (possible fragmentation, descriptive only)")
    print(f"  Processing time:           {elapsed:.2f}s")
    print(f"  Throughput:                {fps:.2f} frames/sec (measured)")
    print(f"  Annotated frames:          {annotated_dir}")
    print(f"  Tracks JSON:               {json_path}")
    print("\nNote: IDF1/MOTA/ID-switch metrics require ground-truth annotations")
    print("      and are NOT computed here - the counts above are descriptive only,")
    print("      not a formal tracking-quality measurement.")


if __name__ == "__main__":
    main()
