"""
Phase 9 CLI: ask the AI Football Analyst a question, grounded in
Phase 6's analytics.json.

Requires GEMINI_API_KEY to be set in your .env file (create a key in
Google AI Studio if you don't have one).

Usage (from the project root, with venv activated):
    python scripts/ask_analyst.py --question "Which team covered more ground?"
    python scripts/ask_analyst.py --frames-dir data/processed/match_sample2 --question "..."
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import GEMINI_API_KEY, LLM_MODEL
from ai.llm.analyst import FootballAnalyst
from ai.validation.grounding_check import check_groundedness


def main():
    parser = argparse.ArgumentParser(description="FootballIQ Phase 9 AI Football Analyst")
    parser.add_argument("--frames-dir", default="data/processed/match_sample2")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model", default=LLM_MODEL)
    args = parser.parse_args()

    analytics_dir = args.frames_dir.rstrip("/\\") + "_analytics"
    analytics_json_path = os.path.join(analytics_dir, "analytics.json")

    if not os.path.isfile(analytics_json_path):
        print(f"ERROR: {analytics_json_path} not found.")
        print("Run scripts/compute_analytics.py first.")
        sys.exit(1)

    with open(analytics_json_path) as f:
        evidence = json.load(f)

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set.")
        print("Add it to your .env file: GEMINI_API_KEY=your-key-here")
        sys.exit(1)

    analyst = FootballAnalyst(model=args.model, api_key=GEMINI_API_KEY)

    print(f"Question: {args.question}\n")
    print("Thinking...\n")

    try:
        response = analyst.ask(args.question, evidence)
    except Exception as e:
        print(f"ERROR calling the Gemini API: {e}")
        print("Check your API key is valid, and that the model name is current")
        print("(model names change - see https://ai.google.dev/gemini-api/docs/models).")
        sys.exit(1)

    print("=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(response.answer)
    print("=" * 60)

    evidence_json_text = json.dumps(evidence)
    check = check_groundedness(response.answer, evidence_json_text)

    print(f"\n[Experimental groundedness check]")
    print(f"  Numbers mentioned in answer: {len(check.numbers_in_answer)}")
    print(f"  Found in evidence:           {len(check.numbers_found_in_evidence)}")
    print(f"  NOT found in evidence:       {check.unverified_count}", end="")
    if check.unverified_count > 0:
        print(f" -> {check.numbers_not_found_in_evidence}")
        print("  (These MAY be invented, or may be legitimate derived values")
        print("   like sums/averages/rounding - review manually. This check")
        print("   is a heuristic, not a definitive verdict.)")
    else:
        print()


if __name__ == "__main__":
    main()
