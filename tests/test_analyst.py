"""
Phase 9 tests for the AI analyst.

Two kinds of tests:
  1. build_user_message - pure string formatting, always runnable.
  2. FootballAnalyst.ask - tested with a FAKE client swapped in, so we
     verify OUR code (what it sends, how it handles the reply) without
     any network call or API key.

What is NOT tested here: the real network call to Google's API and
the actual quality/groundedness of real model answers. Those need a
real GEMINI_API_KEY and must be checked by hand - see ask_analyst.py.
"""

from ai.llm.analyst import FootballAnalyst, SYSTEM_PROMPT, build_user_message


# ---------- build_user_message ----------

def test_user_message_includes_question():
    message = build_user_message("Which team covered more ground?", {"player_metrics": []})
    assert "Which team covered more ground?" in message


def test_user_message_includes_evidence_as_json():
    message = build_user_message("test question", {"total_pixel_distance": 476.9, "track_id": 1})
    assert "476.9" in message
    assert "track_id" in message


def test_user_message_structure_is_stable():
    message = build_user_message("q", {"a": 1})
    assert "EVIDENCE" in message
    assert "QUESTION" in message
    assert message.index("EVIDENCE") < message.index("QUESTION")


# ---------- ask() with a fake client ----------

class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, reply_text):
        self.reply_text = reply_text
        self.last_call = None

    def generate_content(self, model, contents, config):
        self.last_call = {"model": model, "contents": contents, "config": config}
        return _FakeResponse(self.reply_text)


class _FakeClient:
    def __init__(self, reply_text):
        self.models = _FakeModels(reply_text)


def _analyst_with_fake_reply(reply_text):
    analyst = FootballAnalyst(model="test-model", api_key="fake-key")
    analyst.client = _FakeClient(reply_text)
    return analyst


def test_ask_sends_system_prompt_evidence_and_question():
    analyst = _analyst_with_fake_reply("Team A covered 80.0 pixels.")
    evidence = {"total_pixel_distance": 80.0}

    analyst.ask("Who moved most?", evidence)
    call = analyst.client.models.last_call

    assert call["model"] == "test-model"
    assert "Who moved most?" in call["contents"]
    assert "80.0" in call["contents"]
    assert call["config"].system_instruction == SYSTEM_PROMPT


def test_ask_returns_model_text():
    analyst = _analyst_with_fake_reply("Insufficient data to determine this.")
    response = analyst.ask("What was possession?", {"a": 1})

    assert response.answer == "Insufficient data to determine this."
    assert response.model == "test-model"


def test_ask_handles_empty_response_honestly():
    # A blocked/empty reply must NOT come back as a blank answer that
    # looks real - it should say that nothing was returned.
    analyst = _analyst_with_fake_reply(None)
    response = analyst.ask("anything", {"a": 1})

    assert "No text returned" in response.answer


def test_system_prompt_contains_key_grounding_rules():
    assert "Insufficient data to determine this." in SYSTEM_PROMPT
    assert "PIXEL" in SYSTEM_PROMPT
    assert "Never invent" in SYSTEM_PROMPT


def _flat_prompt():
    # Collapse line breaks so phrase checks don't depend on wrapping.
    return " ".join(SYSTEM_PROMPT.split())


def test_system_prompt_stops_speculation_after_insufficient_data():
    prompt = _flat_prompt()
    assert "Do not offer speculative explanations" in prompt
    assert "never draw tactical conclusions" in prompt


def test_system_prompt_forbids_inferring_players_from_track_counts():
    prompt = _flat_prompt()
    assert "number of track IDs is NOT the number of players" in prompt


def test_system_prompt_forbids_calling_pace_a_speed():
    prompt = _flat_prompt()
    assert "not a speed" in prompt
