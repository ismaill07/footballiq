"""
Phase 9 tests for the experimental groundedness heuristic.
Pure string/regex logic - no API call, no network needed.
"""

import json

from ai.validation.grounding_check import check_groundedness, extract_numbers


def test_extract_numbers_finds_integers_and_decimals():
    numbers = extract_numbers("Team A covered 476.9 pixels across 10 frames, team B had 14 ids.")
    assert "476.9" in numbers
    assert "10" in numbers
    assert "14" in numbers


def test_all_numbers_found_when_grounded():
    evidence = '{"total_pixel_distance": 476.9, "frames_tracked": 10}'
    result = check_groundedness("The player covered 476.9 pixels over 10 frames.", evidence)
    assert result.unverified_count == 0
    assert "476.9" in result.numbers_found_in_evidence


def test_invented_number_is_flagged():
    evidence = '{"total_pixel_distance": 476.9, "frames_tracked": 10}'
    result = check_groundedness("The player covered 9999 pixels, an incredible distance.", evidence)
    assert "9999" in result.numbers_not_found_in_evidence
    assert result.unverified_count == 1


def test_no_numbers_in_answer_is_fine():
    result = check_groundedness("Insufficient data to determine this.", '{"total_pixel_distance": 476.9}')
    assert result.numbers_in_answer == []
    assert result.unverified_count == 0


def test_thousands_separator_is_not_split():
    # "1,706.04" must be read as one number, not "1" and "706.04".
    result = check_groundedness("Team B covered 1,706.04 pixels.", '{"a": 5}')
    assert result.numbers_in_answer == ["1706.04"]


def test_list_numbering_is_ignored():
    answer = "UNCERTAINTY:\n1. First point\n2. Second point\n3) Third point"
    result = check_groundedness(answer, '{"a": 5}')
    assert result.numbers_in_answer == []


def test_decimal_at_line_start_is_not_mistaken_for_list_marker():
    result = check_groundedness("476.9 pixels in total", '{"d": 476.9}')
    assert result.numbers_in_answer == ["476.9"]
    assert result.unverified_count == 0


def test_numbers_compared_by_value_not_text():
    result = check_groundedness("The distance was 80 pixels.", '{"d": 80.0}')
    assert result.unverified_count == 0


def test_regression_real_gemini_answer_only_flags_the_two_derived_sums():
    """
    Regression test built from the first REAL answer the analyst gave.
    Both team totals in it were verified by hand to be correct sums of
    per-track values. The check should flag exactly those two derived
    totals (it cannot verify sums) and nothing else.
    """
    team_b = {1: 476.9, 6: 461.88, 14: 63.6, 16: 77.45, 19: 329.36, 23: 199.47, 29: 97.38}
    team_a = {2: 56.91, 3: 63.53, 15: 128.47, 25: 0.0, 31: 97.02}
    frames = {1: 10, 2: 6, 3: 7, 6: 9, 14: 3, 15: 5, 16: 5, 19: 11, 23: 8, 25: 1, 29: 6, 31: 4}
    evidence = {"player_metrics": [
        {"track_id": tid, "team": team, "frames_tracked": frames[tid], "total_pixel_distance": dist}
        for team, tracks in (("Team B", team_b), ("Team A", team_a))
        for tid, dist in tracks.items()
    ]}

    answer = (
        "Based on the aggregate `total_pixel_distance` of all tracked players:\n\n"
        "*   **FACT:** Team B covered a total of **1,706.04 pixels**.\n"
        "*   **FACT:** Team A covered a total of **345.93 pixels**.\n\n"
        "**UNCERTAINTY:**\n"
        "1. The evidence contains more player track segments for Team B (7) than Team A (5).\n"
        "2. Because track IDs can be fragmented, these totals are not a complete record.\n"
        "3. All measurements are in pixel units.\n"
    )

    result = check_groundedness(answer, json.dumps(evidence))

    assert result.numbers_not_found_in_evidence == ["1706.04", "345.93"]
    assert "7" in result.numbers_found_in_evidence
    assert "5" in result.numbers_found_in_evidence
