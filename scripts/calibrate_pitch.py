"""
Phase 5 CLI, step 1 of 2: interactively click 4 points on a frame to
calibrate pitch mapping.

IMPORTANT: this opens a real GUI window and needs mouse input - it
only works run directly on your machine, not through any remote/
automated environment.

Click 4 points, IN THIS ORDER, on a REAL rectangle visible on the
pitch (e.g. the 4 corners of a penalty box, or wherever two sideline/
boundary lines cross to form a rectangle you can identify):
  1. top-left
  2. top-right
  3. bottom-right
  4. bottom-left

The 4 points do NOT need to look like a rectangle on your screen
(camera perspective will distort it) - they need to BE a real
rectangle on the actual pitch. Order matters: going around the
rectangle consistently (clockwise from top-left) is what lets the
math correct for perspective.

Controls:
  - Left-click to place a point (up to 4)
  - 'r' to reset and start over
  - 'q' to quit without saving

Usage:
    python scripts/calibrate_pitch.py --frame data/processed/match_sample2/frame_000010.jpg
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

clicked_points = []


def on_mouse_click(event, x, y, flags, param):
    frame = param
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        print(f"Point {len(clicked_points)}: ({x}, {y})")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            frame, str(len(clicked_points)), (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
        )


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 5 pitch calibration")
    parser.add_argument("--frame", required=True, help="Path to a representative frame image")
    parser.add_argument(
        "--output", default="data/calibration/pitch_calibration.json",
        help="Where to save the calibration JSON",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.frame):
        print(f"ERROR: frame not found: {args.frame}")
        sys.exit(1)

    original = cv2.imread(args.frame)
    if original is None:
        print(f"ERROR: could not read image: {args.frame}")
        sys.exit(1)

    frame = original.copy()

    window_name = "Click 4 points: TL, TR, BR, BL (r=reset, q=quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click, param=frame)

    print("Click 4 points, in order: top-left, top-right, bottom-right, bottom-left")
    print("of a REAL rectangle on the pitch (e.g. penalty box corners).")
    print("Press 'r' to reset and start over. Press 'q' to quit without saving.")

    while True:
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            clicked_points.clear()
            frame = original.copy()
            cv2.setMouseCallback(window_name, on_mouse_click, param=frame)
            print("Reset. Click again.")
        elif key == ord('q'):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            sys.exit(0)
        elif len(clicked_points) == 4:
            cv2.imshow(window_name, frame)
            cv2.waitKey(500)  # brief pause so you can see the 4th point before the window closes
            break

    cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    calibration_data = {
        "source_frame": args.frame,
        "image_points": clicked_points,
        "note": (
            "4 points: top-left, top-right, bottom-right, bottom-left of a "
            "real rectangle on the pitch. Mapped to a normalized 0-1 unit "
            "square, NOT real-world meters (physical pitch dimensions for "
            "this footage are not known/confirmed)."
        ),
    }
    with open(args.output, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\nSaved calibration to {args.output}")
    print(f"Points: {clicked_points}")


if __name__ == "__main__":
    main()
