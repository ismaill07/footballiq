"""
Phase 4 CLI: classify tracked players into Team A / Team B / unclassified
using K-Means clustering on jersey colors extracted from each detection's
bounding box. Reads tracks.json produced by Phase 3's track_players.py.

Usage (from the project root, with venv activated):
    python scripts/classify_teams.py --frames-dir data/processed/match_sample2

Prerequisite: run track_players.py on the same --frames-dir first, since
this script reads its tracks.json output (looks for
<frames-dir>_tracked/tracks.json automatically).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from cv.team_classification.color_extractor import dominant_jersey_color
from cv.team_classification.team_classifier import TeamClassifier, UNCLASSIFIED_LABEL

TEAM_COLORS_BGR = {
    "Team A": (0, 0, 255),
    "Team B": (255, 100, 0),
    UNCLASSIFIED_LABEL: (128, 128, 128),
}
BALL_COLOR_BGR = (0, 165, 255)


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 4 team classification")
    parser.add_argument(
        "--frames-dir", default="data/processed/match_sample2",
        help="Directory of RAW frames from process_video.py (not the _tracked one)",
    )
    parser.add_argument("--outlier-threshold", type=float, default=60.0)
    args = parser.parse_args()

    tracks_dir = args.frames_dir.rstrip("/\\") + "_tracked"
    tracks_json_path = os.path.join(tracks_dir, "tracks.json")

    if not os.path.isfile(tracks_json_path):
        print(f"ERROR: {tracks_json_path} not found.")
        print("Run scripts/track_players.py on this frames-dir first.")
        sys.exit(1)

    with open(tracks_json_path) as f:
        frames_data = json.load(f)

    print(f"Loaded {len(frames_data)} frames of tracking data from {tracks_json_path}")

    # Pass 1: extract jersey colors for every "person" detection across all frames.
    color_samples = []
    per_detection_color = {}

    for frame_entry in frames_data:
        frame_index = frame_entry["frame"]
        frame_path = os.path.join(args.frames_dir, f"frame_{frame_index:06d}.jpg")
        if not os.path.isfile(frame_path):
            continue
        frame = cv2.imread(frame_path)

        for i, t in enumerate(frame_entry["tracks"]):
            if t["class"] != "person":
                continue
            color = dominant_jersey_color(frame, t["bbox"])
            per_detection_color[(frame_index, i)] = color
            if color is not None:
                color_samples.append(color)

    print(f"Extracted {len(color_samples)} valid jersey color samples")

    if len(color_samples) < 2:
        print("ERROR: not enough valid jersey color samples to fit team clusters.")
        print("This can happen with very few/small player detections, or if most")
        print("crops were mostly grass after masking.")
        sys.exit(1)

    classifier = TeamClassifier(outlier_distance_threshold=args.outlier_threshold)
    classifier.fit(color_samples)

    # Pass 2: assign teams and draw annotated output.
    output_dir = args.frames_dir.rstrip("/\\") + "_teams"
    os.makedirs(output_dir, exist_ok=True)

    team_counts = {"Team A": 0, "Team B": 0, UNCLASSIFIED_LABEL: 0}
    all_results = []

    for frame_entry in frames_data:
        frame_index = frame_entry["frame"]
        frame_path = os.path.join(args.frames_dir, f"frame_{frame_index:06d}.jpg")
        if not os.path.isfile(frame_path):
            continue
        frame = cv2.imread(frame_path)
        annotated = frame.copy()

        frame_result = {"frame": frame_index, "objects": []}

        for i, t in enumerate(frame_entry["tracks"]):
            bbox = t["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            if t["class"] == "ball":
                color = BALL_COLOR_BGR
                team_label = None
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated, "ball", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )
            else:
                jersey_color = per_detection_color.get((frame_index, i))
                if jersey_color is None:
                    team_label = UNCLASSIFIED_LABEL
                else:
                    assignment = classifier.predict(jersey_color)
                    team_label = assignment.label
                team_counts[team_label] = team_counts.get(team_label, 0) + 1

                color = TEAM_COLORS_BGR[team_label]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                id_text = f"ID {t.get('track_id')}" if t.get("track_id") is not None else "ID ?"
                cv2.putText(
                    annotated, f"{id_text} {team_label}", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                )

            frame_result["objects"].append({
                "track_id": t.get("track_id"),
                "class": t["class"],
                "team": team_label,
                "bbox": bbox,
            })

        out_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(out_path, annotated)
        all_results.append(frame_result)

    json_path = os.path.join(output_dir, "teams.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nDone.")
    print(f"  Team A assignments:  {team_counts['Team A']}")
    print(f"  Team B assignments:  {team_counts['Team B']}")
    print(f"  Unclassified:        {team_counts[UNCLASSIFIED_LABEL]}")
    print(f"  Annotated frames:    {output_dir}")
    print(f"  Teams JSON:          {json_path}")
    print("\nNote: 'Team A' / 'Team B' are arbitrary cluster labels, not tied to")
    print("      real team names or specific jersey colors. 'unclassified' likely")
    print("      includes referees, a 3rd-kit goalkeeper, or noisy crops - this is")
    print("      a documented limitation, not resolved in this phase.")


if __name__ == "__main__":
    main()
