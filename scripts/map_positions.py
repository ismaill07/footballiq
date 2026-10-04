"""
Phase 5 CLI, step 2 of 2: apply a saved pitch calibration to convert
tracked player positions from image pixel coordinates into normalized
top-down pitch-plane coordinates. Also draws simple top-down
visualizations for a quick visual sanity check.

Prerequisites: run calibrate_pitch.py first (produces the calibration
JSON), and classify_teams.py before that (produces teams.json).

Usage (from the project root, with venv activated):
    python scripts/map_positions.py --frames-dir data/processed/match_sample2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from cv.pitch_mapping.homography import PitchMapper, PitchCalibration, ground_contact_point

TEAM_COLORS_BGR = {
    "Team A": (0, 0, 255),
    "Team B": (255, 100, 0),
    "unclassified": (128, 128, 128),
}
CANVAS_WIDTH, CANVAS_HEIGHT = 600, 400  # pixels, purely for visualization


def draw_topdown_canvas(positions, output_path):
    """positions: list of (pitch_x, pitch_y, team_label) in normalized 0-1 coordinates."""
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), (40, 120, 40), dtype=np.uint8)
    cv2.rectangle(canvas, (10, 10), (CANVAS_WIDTH - 10, CANVAS_HEIGHT - 10), (255, 255, 255), 2)

    for pitch_x, pitch_y, team_label in positions:
        px = int(pitch_x * CANVAS_WIDTH)
        py = int(pitch_y * CANVAS_HEIGHT)
        color = TEAM_COLORS_BGR.get(team_label, (200, 200, 200))
        cv2.circle(canvas, (px, py), 6, color, -1)

    cv2.imwrite(output_path, canvas)


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 5 position mapping")
    parser.add_argument("--frames-dir", default="data/processed/match_sample2")
    parser.add_argument("--calibration", default="data/calibration/pitch_calibration.json")
    args = parser.parse_args()

    if not os.path.isfile(args.calibration):
        print(f"ERROR: calibration file not found: {args.calibration}")
        print("Run scripts/calibrate_pitch.py first.")
        sys.exit(1)

    with open(args.calibration) as f:
        calib_data = json.load(f)

    calibration = PitchCalibration(image_points=[tuple(p) for p in calib_data["image_points"]])
    mapper = PitchMapper(calibration)

    teams_dir = args.frames_dir.rstrip("/\\") + "_teams"
    teams_json_path = os.path.join(teams_dir, "teams.json")

    if not os.path.isfile(teams_json_path):
        print(f"ERROR: {teams_json_path} not found.")
        print("Run scripts/classify_teams.py first.")
        sys.exit(1)

    with open(teams_json_path) as f:
        frames_data = json.load(f)

    output_dir = args.frames_dir.rstrip("/\\") + "_pitch"
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    out_of_bounds_count = 0
    total_mapped = 0
    all_positions_for_viz = []

    for frame_entry in frames_data:
        frame_index = frame_entry["frame"]
        frame_positions = []

        for obj in frame_entry["objects"]:
            if obj["class"] != "person":
                continue  # Phase 5 maps player positions; ball mapping is a future extension

            ground_point = ground_contact_point(obj["bbox"])
            pitch_x, pitch_y = mapper.image_to_pitch(ground_point)
            in_bounds = mapper.is_within_bounds((pitch_x, pitch_y))

            total_mapped += 1
            if not in_bounds:
                out_of_bounds_count += 1

            frame_positions.append({
                "track_id": obj.get("track_id"),
                "team": obj.get("team"),
                "pitch_x": round(pitch_x, 4),
                "pitch_y": round(pitch_y, 4),
                "within_calibrated_bounds": in_bounds,
            })
            all_positions_for_viz.append((pitch_x, pitch_y, obj.get("team")))

        all_results.append({"frame": frame_index, "positions": frame_positions})

        frame_viz_path = os.path.join(output_dir, f"topdown_frame_{frame_index:06d}.jpg")
        draw_topdown_canvas(
            [(p["pitch_x"], p["pitch_y"], p["team"]) for p in frame_positions],
            frame_viz_path,
        )

    # A combined view across the whole clip - a crude but genuinely useful
    # sanity check: do positions cluster in a sensible region, or are they
    # scattered nonsensically (a sign the calibration points were wrong)?
    combined_viz_path = os.path.join(output_dir, "topdown_all_frames.jpg")
    draw_topdown_canvas(all_positions_for_viz, combined_viz_path)

    json_path = os.path.join(output_dir, "pitch_positions.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDone.")
    print(f"  Positions mapped:        {total_mapped}")
    print(f"  Outside calibrated area: {out_of_bounds_count} (players near/beyond the 4 calibration points)")
    print(f"  Per-frame top-down images and combined view: {output_dir}")
    print(f"  Positions JSON:          {json_path}")
    print("\nNote: pitch_x / pitch_y are NORMALIZED (0-1) coordinates within the")
    print("      calibrated quadrilateral, NOT real-world meters. Ground position")
    print("      is approximated as the bottom-center of each bounding box.")


if __name__ == "__main__":
    main()
