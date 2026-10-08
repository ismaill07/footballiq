"""
Lightweight, EXPERIMENTAL groundedness check for FootballIQ's AI
Analyst responses (Phase 9).

This is NOT a rigorous hallucination detector. It's a simple
heuristic: extract numeric values mentioned in the LLM's answer and
check whether each one appears somewhere in the evidence JSON.
Numbers that don't appear are flagged as "unverified".

KNOWN WEAKNESSES (found by testing against a real answer):
  * A number the model DERIVED correctly from the evidence - a sum,
    an average, a difference - is flagged, because it never appears
    literally in the evidence. In the first real test, both flagged
    team totals were correct sums, checked by hand. So a flag means
    "a human should check this", NOT "this is invented".
  * The reverse is also possible: an invented number that happens to
    equal some unrelated evidence value would pass. This check cannot
    tell whether a number is used in the RIGHT context.

Parsing fixes made after that first real test:
  * Thousands separators ("1,706.04") are normalised before reading
    the answer, so they aren't split into "1" and "706.04".
  * List numbering at the start of a line ("1. ", "2) ") is ignored,
    so it isn't mistaken for a statistic.
  * Numbers are compared by value, so "80" matches "80.0".

Documented as experimental per the project's AI Engineering Rules:
"report limitations rather than fabricate."
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

# "1. ", "2) ", "- 3. " at the very start of a line = list numbering.
LIST_MARKER_PATTERN = re.compile(r"^[ \t]*(?:[-*][ \t]*)?\d+[.)][ \t]+", re.MULTILINE)

# A comma between digits followed by exactly 3 digits = thousands separator.
THOUSANDS_SEPARATOR_PATTERN = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


@dataclass
class GroundednessCheck:
    numbers_in_answer: List[str]
    numbers_found_in_evidence: List[str]
    numbers_not_found_in_evidence: List[str]

    @property
    def unverified_count(self) -> int:
        return len(self.numbers_not_found_in_evidence)


def normalize_answer_text(text: str) -> str:
    """Remove list numbering and thousands separators from the ANSWER text."""
    text = LIST_MARKER_PATTERN.sub("", text)
    text = THOUSANDS_SEPARATOR_PATTERN.sub("", text)
    return text


def extract_numbers(text: str) -> List[str]:
    return NUMBER_PATTERN.findall(text)


def check_groundedness(answer_text: str, evidence_json_text: str) -> GroundednessCheck:
    """
    EXPERIMENTAL heuristic. See the module docstring for known
    weaknesses before treating any result here as a hallucination verdict.

    Normalisation is applied to the answer only - the evidence JSON is
    machine-generated and read as-is.
    """
    answer_numbers = extract_numbers(normalize_answer_text(answer_text))
    evidence_values = {round(float(n), 6) for n in extract_numbers(evidence_json_text)}

    found = [n for n in answer_numbers if round(float(n), 6) in evidence_values]
    not_found = [n for n in answer_numbers if round(float(n), 6) not in evidence_values]

    return GroundednessCheck(
        numbers_in_answer=answer_numbers,
        numbers_found_in_evidence=found,
        numbers_not_found_in_evidence=not_found,
    )
