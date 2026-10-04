"""
Phase 6 CLI: compute deterministic player and team analytics from
Phase 4's teams.json (tracked, team-classified detections).

IMPORTANT: all spatial values here are PIXEL-space measurements, NOT
real-world meters. See LIMITATIONS.md - pitch calibration for this
project's current clip is documented as unreliable, so this script
deliberately does not convert to real-world units rather than present
fabricated precision.

Usage (from the project root, with venv activated):
    python scripts/compute_analytics.py --frames-dir data/processed/match_sample2
"""

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.player_metrics import build_trajectories, compute_player_metrics
from analytics.team_metrics import compute_team_frame_metrics, summarize_team_metrics
from cv.pitch_mapping.homography import ground_contact_point


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 6 football analytics")
    parser.add_argument("--frames-dir", default="data/processed/match_sample2")
    args = parser.parse_args()

    teams_dir = args.frames_dir.rstrip("/\\") + "_teams"
    teams_json_path = os.path.join(teams_dir, "teams.json")

    if not os.path.isfile(teams_json_path):
        print(f"ERROR: {teams_json_path} not found.")
        print("Run scripts/classify_teams.py first.")
        sys.exit(1)

    with open(teams_json_path) as f:
        frames_data = json.load(f)

    # --- Player metrics ---
    trajectories = build_trajectories(frames_data)
    player_metrics = [compute_player_metrics(t) for t in trajectories.values()]
    player_metrics.sort(key=lambda m: m.track_id)

    # --- Team metrics ---
    # Pre-populate every team label actually seen in the data, even if it
    # never qualifies for a frame metric (fewer than 2 players at once) -
    # this guarantees every real team gets an explicit "no data" summary
    # instead of silently vanishing, which would be ambiguous (missing
    # because it doesn't exist, or missing because it lacks data?).
    team_frame_metrics = defaultdict(list)
    all_team_labels = {
        obj.get("team")
        for frame_entry in frames_data
        for obj in frame_entry["objects"]
        if obj["class"] == "person" and obj.get("team") is not None
    }
    for team in all_team_labels:
        team_frame_metrics[team]  # touch the key so it exists even if empty

    for frame_entry in frames_data:
        frame_index = frame_entry["frame"]
        positions_by_team = defaultdict(list)

        for obj in frame_entry["objects"]:
            if obj["class"] != "person":
                continue
            team = obj.get("team")
            if team is None:
                continue
            x, y = ground_contact_point(obj["bbox"])
            positions_by_team[team].append((x, y))

        for team, positions in positions_by_team.items():
            metrics = compute_team_frame_metrics(frame_index, team, positions)
            if metrics is not None:
                team_frame_metrics[team].append(metrics)

    team_summaries = {
        team: summarize_team_metrics(metrics_list)
        for team, metrics_list in team_frame_metrics.items()
    }

    output_dir = args.frames_dir.rstrip("/\\") + "_analytics"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "units_note": (
            "All spatial values are PIXEL-space measurements, NOT real-world "
            "meters - see LIMITATIONS.md. Pitch calibration for this clip is "
            "documented as unreliable."
        ),
        "player_metrics": [
            {
                "track_id": m.track_id,
                "team": m.team,
                "frames_tracked": m.frames_tracked,
                "total_pixel_distance": m.total_pixel_distance,
                "avg_position_px": m.avg_position,
                "avg_pixels_per_sample_step": m.avg_pixels_per_sample_step,
            }
            for m in player_metrics
        ],
        "team_metrics": team_summaries,
    }

    json_path = os.path.join(output_dir, "analytics.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nDone.")
    print(f"  Players with trajectories: {len(player_metrics)}")
    for m in player_metrics:
        print(
            f"    track_id={m.track_id:<3} team={m.team:<12} frames={m.frames_tracked:<3} "
            f"total_px_dist={m.total_pixel_distance:<8} avg_pace_px/step={m.avg_pixels_per_sample_step}"
        )

    print("\n  Team summaries:")
    for team, summary in team_summaries.items():
        print(f"    {team}: {summary}")

    print(f"\n  Analytics JSON: {json_path}")
    print("\nNote: all distances are in PIXEL units within the original frame,")
    print("      NOT real-world meters - see LIMITATIONS.md for why.")


if __name__ == "__main__":
    main()
