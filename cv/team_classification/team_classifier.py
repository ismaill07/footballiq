"""
Team assignment via K-Means clustering on jersey colors.

Deliberately the simplest reliable approach, per the project's
"simplest reliable approach should be implemented first" principle
(PROJECT_SPEC.md section 6): cluster extracted jersey colors into 2
groups (Team A / Team B), and flag colors sitting far from BOTH
cluster centers as "unclassified" - a cheap, explainable stand-in for
referees/goalkeepers wearing a visually distinct kit.

IMPORTANT: this is NOT a trained referee detector. It is a distance
threshold heuristic. With only one or two referee/goalkeeper examples
in a clip, they cannot form their own reliable cluster, so this is
the documented, honest limitation of this phase - not silently
patched over. See PROJECT_SPEC.md section 6 for the acknowledged gap.

"Team A" / "Team B" are arbitrary cluster labels assigned by K-Means'
internal ordering - they carry no meaning about which real-world team
is which, and are not guaranteed to stay consistent (e.g. cluster 0
is not always the same side) across separate runs or separate clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans

UNCLASSIFIED_LABEL = "unclassified"


@dataclass
class TeamAssignment:
    label: str  # "Team A", "Team B", or "unclassified"
    distance_to_nearest_center: float


class TeamClassifier:
    """
    Fit on a batch of jersey colors (BGR tuples) collected across a
    clip, then assign each color to Team A, Team B, or "unclassified"
    based on distance to the two learned cluster centers.
    """

    def __init__(self, outlier_distance_threshold: float = 60.0):
        self.outlier_distance_threshold = outlier_distance_threshold
        self.kmeans: Optional[KMeans] = None

    def fit(self, colors: List[Tuple[int, int, int]]) -> "TeamClassifier":
        if len(colors) < 2:
            raise ValueError("Need at least 2 jersey color samples to fit team clusters")

        X = np.array(colors, dtype=np.float64)
        self.kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        self.kmeans.fit(X)
        return self

    def predict(self, color: Tuple[int, int, int]) -> TeamAssignment:
        if self.kmeans is None:
            raise RuntimeError("TeamClassifier.fit() must be called before predict()")

        x = np.array([color], dtype=np.float64)
        cluster_id = int(self.kmeans.predict(x)[0])
        center = self.kmeans.cluster_centers_[cluster_id]
        distance = float(np.linalg.norm(x[0] - center))

        if distance > self.outlier_distance_threshold:
            return TeamAssignment(label=UNCLASSIFIED_LABEL, distance_to_nearest_center=distance)

        label = "Team A" if cluster_id == 0 else "Team B"
        return TeamAssignment(label=label, distance_to_nearest_center=distance)
