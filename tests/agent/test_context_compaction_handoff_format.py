from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from run_agent import AIAgent


OLD_PATCH3_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )
    compressor.tail_token_budget = 1
    return compressor


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def test_summary_prompt_uses_neutral_state_snapshot_headings():
    compressor = _compressor()
    turns = [
        {"role": "user", "content": "Please continue patch 4"},
        {"role": "assistant", "content": "I updated the compressor tests."},
    ]

    captured = {}

    def mock_call_llm(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return _response("## Snapshot Purpose\nHistorical continuity record.")

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        handoff = compressor._generate_summary(turns)

    assert handoff.startswith(SUMMARY_PREFIX)
    prompt = captured["prompt"]
    assert "## Snapshot Purpose" in prompt
    assert "## Latest Live Request At Compaction" in prompt
    assert "## Unresolved Requests At Compaction" in prompt
    assert "## Follow-up Context" in prompt
    assert "## Active Task" not in prompt
    assert "## Pending User Asks" not in prompt
    assert "## Remaining Work" not in prompt


def test_old_patch3_handoff_prefix_is_still_recognized_for_iterative_continuity():
    compressor = _compressor()
    old_summary = "OLD-SUMMARY-BODY unique continuity facts"

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{OLD_PATCH3_PREFIX}\n{old_summary}"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
    ]

    captured = {}

    def mock_call_llm(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return _response("## Snapshot Purpose\nUpdated continuity record.")

    with patch("agent.context_compressor.call_llm", mock_call_llm):
        compressor.compress(messages)

    prompt = captured["prompt"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {OLD_PATCH3_PREFIX}" not in prompt


def test_effective_system_prompt_labels_handoff_as_historical_snapshot():
    agent = AIAgent.__new__(AIAgent)
    agent.ephemeral_system_prompt = "ephemeral system note"
    agent._pending_compaction_handoff = f"{SUMMARY_PREFIX}\ncarry forward this state"

    effective = agent._build_effective_system_prompt("base system prompt")

    assert "historical state snapshot" in effective
    assert "not a live conversation turn" in effective
    assert "not as new instructions" in effective
    assert "not a new user message" not in effective



def test_effective_system_prompt_adds_live_turn_precedence_anchor_when_live_user_message_exists():
    agent = AIAgent.__new__(AIAgent)
    agent.ephemeral_system_prompt = ""
    agent._pending_compaction_handoff = (
        f"{SUMMARY_PREFIX}\n"
        "## Latest Live Request At Compaction\n"
        "Historical request: continue patch 4."
    )

    effective = agent._build_effective_system_prompt(
        "base system prompt",
        latest_live_user_message="continue with patch 5 only",
    )

    assert "If anything in the historical snapshot conflicts with the live user turn below, the live user turn wins." in effective
    assert "Latest live user turn in this request:" in effective
    assert "continue with patch 5 only" in effective
    assert effective.index("continue with patch 5 only") > effective.index("Historical request: continue patch 4.")
