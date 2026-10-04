"""
Player-level football analytics for FootballIQ.

All distance/pace values here are computed in PIXEL space from
tracked bounding-box ground-contact points, NOT real-world meters.
See LIMITATIONS.md: pitch calibration for this project's current
clip is documented as unreliable, so this module deliberately does
NOT convert these to real-world units rather than present fabricated
precision.

These are honest, measured, deterministic calculations - matching
ARCHITECTURE.md section 11: "The analytics engine should NOT depend
on the LLM. It should use deterministic calculations wherever
possible."

IMPORTANT caveat on trajectory.team: a player's team is recorded from
the FIRST frame we see their track_id in - but Phase 4 recomputes
team color independently every frame, not tied to track identity. If
a player's color classification flipped between frames (unlikely but
possible, especially near the Team A / Team B cluster boundary), this
trajectory's reported team may not match every individual frame. This
is a known approximation, not silently hidden.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cv.pitch_mapping.homography import ground_contact_point


@dataclass
class PositionSample:
    frame_index: int
    x: float
    y: float


@dataclass
class PlayerTrajectory:
    track_id: int
    team: Optional[str]
    samples: List[PositionSample] = field(default_factory=list)

    def add_sample(self, frame_index: int, x: float, y: float):
        self.samples.append(PositionSample(frame_index=frame_index, x=x, y=y))


@dataclass
class PlayerMetrics:
    track_id: int
    team: Optional[str]
    frames_tracked: int
    total_pixel_distance: float
    avg_position: Tuple[float, float]
    avg_pixels_per_sample_step: float  # a "pace" proxy, NOT a real speed


def build_trajectories(frames_data: List[dict]) -> Dict[int, PlayerTrajectory]:
    """
    Build a PlayerTrajectory per track_id from a list of per-frame
    detection dicts (as produced by classify_teams.py's teams.json),
    using each detection's bounding-box ground-contact point.

    Detections with no track_id (None) are skipped - a trajectory
    requires a stable identity to be meaningful.
    """
    trajectories: Dict[int, PlayerTrajectory] = {}

    for frame_entry in frames_data:
        frame_index = frame_entry["frame"]
        for obj in frame_entry["objects"]:
            if obj["class"] != "person":
                continue
            track_id = obj.get("track_id")
            if track_id is None:
                continue

            if track_id not in trajectories:
                trajectories[track_id] = PlayerTrajectory(track_id=track_id, team=obj.get("team"))

            x, y = ground_contact_point(obj["bbox"])
            trajectories[track_id].add_sample(frame_index, x, y)

    return trajectories


def compute_player_metrics(trajectory: PlayerTrajectory) -> PlayerMetrics:
    """
    Compute distance/position metrics for one player's trajectory.
    A trajectory must have at least 1 sample; distance is 0 for a
    single-sample trajectory (nothing to measure movement between).
    """
    if not trajectory.samples:
        raise ValueError(f"Trajectory for track_id={trajectory.track_id} has no samples")

    total_distance = 0.0
    for prev, curr in zip(trajectory.samples, trajectory.samples[1:]):
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        total_distance += math.hypot(dx, dy)

    avg_x = sum(s.x for s in trajectory.samples) / len(trajectory.samples)
    avg_y = sum(s.y for s in trajectory.samples) / len(trajectory.samples)

    steps = len(trajectory.samples) - 1
    avg_pace = (total_distance / steps) if steps > 0 else 0.0

    return PlayerMetrics(
        track_id=trajectory.track_id,
        team=trajectory.team,
        frames_tracked=len(trajectory.samples),
        total_pixel_distance=round(total_distance, 2),
        avg_position=(round(avg_x, 2), round(avg_y, 2)),
        avg_pixels_per_sample_step=round(avg_pace, 2),
    )
