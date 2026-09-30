"""
Phase 4 tests for K-Means-based team classification.
"""

import pytest

from cv.team_classification.team_classifier import TeamClassifier, UNCLASSIFIED_LABEL


def test_fit_requires_at_least_two_samples():
    classifier = TeamClassifier()
    with pytest.raises(ValueError):
        classifier.fit([(255, 0, 0)])


def test_two_well_separated_colors_get_different_labels():
    # Simulate two teams: reds and blues, tightly clustered around each color.
    red_samples = [(0, 0, 200 + i) for i in range(5)]
    blue_samples = [(200 + i, 0, 0) for i in range(5)]

    classifier = TeamClassifier(outlier_distance_threshold=100.0)
    classifier.fit(red_samples + blue_samples)

    red_assignment = classifier.predict((0, 0, 200))
    blue_assignment = classifier.predict((200, 0, 0))

    assert red_assignment.label in ("Team A", "Team B")
    assert blue_assignment.label in ("Team A", "Team B")
    assert red_assignment.label != blue_assignment.label


def test_outlier_far_from_both_clusters_is_unclassified():
    red_samples = [(0, 0, 200 + i) for i in range(5)]
    blue_samples = [(200 + i, 0, 0) for i in range(5)]

    classifier = TeamClassifier(outlier_distance_threshold=30.0)
    classifier.fit(red_samples + blue_samples)

    # Bright yellow-ish - far from both the red and blue clusters.
    outlier_assignment = classifier.predict((0, 255, 255))
    assert outlier_assignment.label == UNCLASSIFIED_LABEL


def test_predict_before_fit_raises():
    classifier = TeamClassifier()
    with pytest.raises(RuntimeError):
        classifier.predict((0, 0, 0))
