"""
AI Football Analyst for FootballIQ (Phase 9).

The LLM here is strictly downstream of the deterministic analytics
engine (ARCHITECTURE.md section 13): it receives the EXACT structured
JSON evidence already computed in Phase 6 (analytics.json) and is
instructed to interpret it, never invent new statistics or events.

Grounding rules enforced via the system prompt (PROJECT_SPEC.md
section 10 / AI Engineering Rules):
  1. Never invent statistics or events.
  2. Never claim a player performed an action unless supported by
     the provided evidence.
  3. Clearly distinguish measured FACT from INTERPRETATION from
     UNCERTAINTY.
  4. Say "Insufficient data to determine this." when the evidence
     doesn't support a confident answer.
  5. All spatial/distance values are PIXEL-space estimates, not real
     meters (per LIMITATIONS.md) - the LLM must not imply otherwise.
  6. Track IDs may be fragmented (Phase 6's documented finding) - the
     LLM should be cautious about per-player claims as a result.

PROVIDER NOTE: this module currently uses Google's Gemini API via the
`google-genai` SDK. The prompt (SYSTEM_PROMPT) and the evidence
formatting (build_user_message) are provider-independent and kept
separate from the network call on purpose - swapping providers later
means changing only FootballAnalyst.__init__ and .ask(), nothing else.

The prompt-building logic is separate from the network call so it can
be unit-tested without an API key or network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from google import genai
from google.genai import types

SYSTEM_PROMPT = """You are the AI Football Analyst for FootballIQ.

You will be given structured JSON evidence computed by a deterministic
computer-vision and analytics pipeline - NOT raw video. You did not
watch the match. You only know what is in the evidence provided.

STRICT RULES, follow all of them:
1. Never invent a statistic, event, or player action that is not
   directly present in the evidence JSON.
2. If the evidence does not contain what's needed to answer a
   question, say exactly: "Insufficient data to determine this."
   Do not guess or approximate to sound more helpful.
3. Clearly label your claims as FACT (a number taken directly from
   the evidence), INTERPRETATION (your reasoning about what the facts
   might mean), or UNCERTAINTY (where the evidence is incomplete,
   noisy, or contradictory).
4. All spatial values (positions, distances, width, compactness) are
   in PIXEL units within the original video frame, NOT real-world
   meters. Never state or imply a real-world distance.
5. Track IDs may be fragmented (the same real player split across
   multiple track_id values due to tracking limitations) - if asked
   about "a player," prefer answering at the team level unless a
   specific track_id is given, and mention this limitation if it's
   relevant to the question.
6. Keep answers concise and grounded. Do not pad with generic
   football commentary not tied to the evidence.
7. If you answer "Insufficient data to determine this.", briefly name
   which data is missing, then stop. Do not offer speculative
   explanations for the question asked. You may add directly relevant
   measured facts, labelled FACT, but never draw tactical conclusions
   (for example about defensive structure or intent) from pixel-space
   metrics alone.
8. The number of track IDs is NOT the number of players. Track IDs are
   fragmented, so a team can have more track IDs than real players.
   Never infer how many players were on the pitch, or how complete the
   tracking was, from track counts.
9. avg_pixels_per_sample_step is average movement per sampled frame,
   not a speed. Call it "average movement per sample step" and never
   describe it as speed or peak speed.
"""


@dataclass
class AnalystResponse:
    answer: str
    model: str


def build_user_message(question: str, evidence: dict) -> str:
    """
    Build the user-turn message sent to the model: the evidence JSON
    followed by the question. Pure string-building, no network call -
    kept separate so it's independently testable.
    """
    evidence_json = json.dumps(evidence, indent=2)
    return f"""EVIDENCE (structured JSON from the analytics pipeline):
{evidence_json}

QUESTION:
{question}"""


class FootballAnalyst:
    """Sends a user's question plus structured evidence to Gemini, under strict grounding rules."""

    def __init__(self, model: str = "gemini-3-flash-preview", api_key: Optional[str] = None):
        self.model = model
        # If api_key is None, the SDK falls back to the GEMINI_API_KEY /
        # GOOGLE_API_KEY environment variable on its own.
        self.client = genai.Client(api_key=api_key)

    def ask(self, question: str, evidence: dict) -> AnalystResponse:
        user_message = build_user_message(question, evidence)

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_message,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )

        # response.text can be None/empty if the reply was blocked or
        # contained no text parts - report that honestly instead of
        # returning a blank answer that looks like a real one.
        answer_text = response.text
        if not answer_text:
            answer_text = (
                "[No text returned by the model. The response may have been "
                "blocked or empty - try rephrasing the question.]"
            )

        return AnalystResponse(answer=answer_text, model=self.model)
