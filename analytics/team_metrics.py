"""
Team-level football analytics for FootballIQ.

Like player_metrics.py, all spatial values are PIXEL-space, not
real-world meters - see LIMITATIONS.md. "Width" and "compactness"
here are simple, documented proxies computed directly from pixel
positions, not calibrated pitch measurements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class TeamFrameMetrics:
    frame_index: int
    team: str
    player_count: int
    avg_position: Tuple[float, float]
    width_px: float  # spread along the image x-axis - a proxy, not calibrated pitch width
    compactness_px: float  # average distance from each player to the team's centroid


def compute_team_frame_metrics(
    frame_index: int, team: str, positions: List[Tuple[float, float]]
) -> Optional[TeamFrameMetrics]:
    """
    Compute one team's shape metrics for a single frame, given that
    team's player ground positions in that frame. Returns None if
    fewer than 2 players are present - width/compactness aren't
    meaningful for 0 or 1 players, so this is reported as "no data"
    for that frame rather than a misleading 0.
    """
    if len(positions) < 2:
        return None

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    avg_x = sum(xs) / len(xs)
    avg_y = sum(ys) / len(ys)

    width_px = max(xs) - min(xs)

    compactness_px = sum(math.hypot(x - avg_x, y - avg_y) for x, y in positions) / len(positions)

    return TeamFrameMetrics(
        frame_index=frame_index,
        team=team,
        player_count=len(positions),
        avg_position=(round(avg_x, 2), round(avg_y, 2)),
        width_px=round(width_px, 2),
        compactness_px=round(compactness_px, 2),
    )


def summarize_team_metrics(per_frame_metrics: List[TeamFrameMetrics]) -> dict:
    """
    Average a list of per-frame TeamFrameMetrics (all for the same
    team) into a single clip-level summary. Returns a "no data"
    summary if the list is empty (that team never had 2+ players
    detected together in any frame) rather than raising - a
    genuinely possible, honestly-reportable outcome.
    """
    if not per_frame_metrics:
        return {
            "frames_with_data": 0,
            "avg_width_px": None,
            "avg_compactness_px": None,
        }

    avg_width = sum(m.width_px for m in per_frame_metrics) / len(per_frame_metrics)
    avg_compactness = sum(m.compactness_px for m in per_frame_metrics) / len(per_frame_metrics)

    return {
        "frames_with_data": len(per_frame_metrics),
        "avg_width_px": round(avg_width, 2),
        "avg_compactness_px": round(avg_compactness, 2),
    }
