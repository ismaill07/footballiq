"""
Phase 2 CLI: run player/ball detection on frames already extracted by
Phase 1's process_video.py. Saves annotated images (boxes drawn) and a
detections.json file matching the schema in ARCHITECTURE.md section 6.

Usage (from the project root, with venv activated):
    python scripts/detect_players.py
    python scripts/detect_players.py --frames-dir data/processed/match_sample
    python scripts/detect_players.py --confidence 0.4
"""

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from cv.detection.player_detector import PlayerDetector, FrameDetections


def draw_detections(frame, frame_detections: FrameDetections):
    """Draw bounding boxes + labels on a copy of the frame."""
    for det in frame_detections.detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        # Green for people, orange for the ball - just for visual clarity.
        color = (0, 255, 0) if det.label == "person" else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            frame, label_text, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 2 detection")
    parser.add_argument(
        "--frames-dir",
        default="data/processed/match_sample",
        help="Directory of frames produced by scripts/process_video.py",
    )
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--model", default="yolo26n.pt")
    args = parser.parse_args()

    frame_paths = sorted(glob.glob(os.path.join(args.frames_dir, "frame_*.jpg")))
    if not frame_paths:
        print(f"ERROR: no frames found in {args.frames_dir}.")
        print("Run scripts/process_video.py first to generate frames.")
        sys.exit(1)

    print(f"Found {len(frame_paths)} frames in {args.frames_dir}")
    print(f"Loading model: {args.model} (first run downloads weights, may take a minute)")
    detector = PlayerDetector(model_name=args.model, confidence_threshold=args.confidence)

    annotated_dir = args.frames_dir.rstrip("/\\") + "_detections"
    os.makedirs(annotated_dir, exist_ok=True)

    all_results = []
    total_detections = 0
    start_time = time.perf_counter()

    for path in frame_paths:
        frame = cv2.imread(path)
        frame_index = int(os.path.splitext(os.path.basename(path))[0].split("_")[1])

        result = detector.detect(frame, frame_index=frame_index)
        total_detections += len(result.detections)

        annotated = draw_detections(frame.copy(), result)
        out_path = os.path.join(annotated_dir, os.path.basename(path))
        cv2.imwrite(out_path, annotated)

        all_results.append({
            "frame": result.frame_index,
            "detections": [
                {"class": d.label, "confidence": round(d.confidence, 3), "bbox": d.bbox}
                for d in result.detections
            ],
        })

    elapsed = time.perf_counter() - start_time
    fps = len(frame_paths) / elapsed if elapsed > 0 else 0.0

    json_path = os.path.join(annotated_dir, "detections.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDone.")
    print(f"  Frames processed:      {len(frame_paths)}")
    print(f"  Total detections:      {total_detections}")
    print(f"  Avg detections/frame:  {total_detections / len(frame_paths):.2f}")
    print(f"  Processing time:       {elapsed:.2f}s")
    print(f"  Inference throughput:  {fps:.2f} frames/sec (measured)")
    print(f"  Annotated frames:      {annotated_dir}")
    print(f"  Detections JSON:       {json_path}")


if __name__ == "__main__":
    main()
