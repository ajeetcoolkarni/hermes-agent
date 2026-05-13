import os
from unittest.mock import MagicMock, patch

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX



def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )
    # Keep the protected tail tiny so these unit tests always leave a middle
    # region to summarize.
    compressor.tail_token_budget = 1
    return compressor



def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response



def _messages_for_compress():
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "old user turn"},
        {"role": "assistant", "content": "old assistant turn"},
        {"role": "user", "content": "middle user turn"},
        {"role": "assistant", "content": "recent assistant turn"},
        {"role": "user", "content": "latest user turn"},
    ]



def test_compress_stores_generated_summary_in_compaction_handoff_state():
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary body")):
        compressor.compress(_messages_for_compress())

    assert compressor._compaction_handoff_state == f"{SUMMARY_PREFIX}\nfresh summary body"



def test_compress_stores_fallback_summary_in_compaction_handoff_state_when_generation_fails():
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
        compressor.compress(_messages_for_compress())

    handoff = compressor._compaction_handoff_state
    assert handoff.startswith(SUMMARY_PREFIX)
    assert "Summary generation was unavailable." in handoff



def test_compress_keeps_handoff_out_of_returned_transcript_messages():
    compressor = _compressor()

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary body")):
        compressed = compressor.compress(_messages_for_compress())

    assert compressor._compaction_handoff_state == f"{SUMMARY_PREFIX}\nfresh summary body"
    assert all(
        not str(msg.get("content", "")).startswith(SUMMARY_PREFIX)
        for msg in compressed
    )



def test_compress_does_not_merge_handoff_into_tail_when_roles_double_collide():
    compressor = _compressor()
    compressor.protect_first_n = 3
    compressor.protect_last_n = 3
    compressor.tail_token_budget = 1
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "msg 1"},
        {"role": "assistant", "content": "msg 2"},
        {"role": "user", "content": "msg 3"},
        {"role": "assistant", "content": "msg 4"},
        {"role": "user", "content": "msg 5"},
        {"role": "user", "content": "msg 6"},
        {"role": "assistant", "content": "msg 7"},
        {"role": "user", "content": "msg 8"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary body")):
        compressed = compressor.compress(messages)

    assert compressor._compaction_handoff_state == f"{SUMMARY_PREFIX}\nfresh summary body"
    first_tail = next(msg for msg in compressed if msg.get("content") == "msg 6")
    assert first_tail["content"] == "msg 6"
    assert all("fresh summary body" not in str(msg.get("content", "")) for msg in compressed)



def test_compressor_reset_clears_compaction_handoff_state():
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor._compaction_handoff_state = "stale handoff"
    compressor._previous_summary = "stale summary"
    compressor._context_probed = True
    compressor._context_probe_persistable = True
    compressor.last_prompt_tokens = 10
    compressor.last_completion_tokens = 5
    compressor.last_total_tokens = 15
    compressor.compression_count = 2

    compressor.on_session_reset()

    assert compressor._compaction_handoff_state is None



def test_agent_compress_context_captures_pending_compaction_handoff_from_compressor():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = None
    agent._todo_store = MagicMock()
    agent._todo_store.format_for_injection.return_value = ""
    agent._invalidate_system_prompt = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="new system")
    agent._cached_system_prompt = None
    agent._session_db = None
    agent._session_db_created = False
    agent._last_flushed_db_idx = 0
    agent._vprint = MagicMock()
    agent._emit_warning = MagicMock()
    agent.tools = []
    agent.model = "test/model"
    agent.session_id = "original-session"
    agent.platform = "cli"
    agent.logs_dir = None
    agent._last_compression_summary_warning = None
    agent._last_aux_fallback_warning_key = None

    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "compressed"}]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._compaction_handoff_state = "HANDOFF-STATE"
    agent.context_compressor = compressor

    agent._compress_context([
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "m2"},
        {"role": "user", "content": "m3"},
        {"role": "assistant", "content": "m4"},
    ], "sys", approx_tokens=100)

    assert agent._pending_compaction_handoff == "HANDOFF-STATE"



def test_agent_reset_session_state_clears_pending_compaction_handoff():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.session_total_tokens = 0
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_api_calls = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    agent._user_turn_count = 0
    agent._pending_compaction_handoff = "pending handoff"
    agent.context_compressor = None

    agent.reset_session_state()

    assert agent._pending_compaction_handoff is None



def test_agent_effective_system_prompt_injects_pending_compaction_handoff_only_at_api_build_time():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.ephemeral_system_prompt = "ephemeral system note"
    agent._pending_compaction_handoff = f"{SUMMARY_PREFIX}\ncarry forward this state"

    effective = agent._build_effective_system_prompt("base system prompt")

    assert "base system prompt" in effective
    assert "ephemeral system note" in effective
    assert "carry forward this state" in effective
    assert "historical state snapshot" in effective
    assert "not a live conversation turn" in effective
    assert "not as new instructions" in effective



def test_agent_effective_system_prompt_explicitly_prioritizes_latest_live_user_turn_over_handoff():
    from run_agent import AIAgent

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
    assert "continue with patch 5 only" in effective
    assert effective.index("continue with patch 5 only") > effective.index("Historical request: continue patch 4.")



def test_find_latest_live_user_message_prefers_real_user_turn_over_legacy_summary_message():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    messages = [
        {"role": "user", "content": f"{SUMMARY_PREFIX}\nold historical snapshot"},
        {"role": "assistant", "content": "older assistant reply"},
        {"role": "user", "content": "continue with patch 5 only"},
    ]

    assert agent._find_latest_live_user_message(messages) == "continue with patch 5 only"



def test_agent_effective_system_prompt_omits_compaction_handoff_when_none():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.ephemeral_system_prompt = "ephemeral system note"
    agent._pending_compaction_handoff = None

    effective = agent._build_effective_system_prompt("base system prompt")

    assert effective == "base system prompt\n\nephemeral system note"



def test_agent_rehydrate_persisted_compaction_handoff_loads_latest_tip_from_session_db(tmp_path):
    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="root", source="cli")
    db.end_session("root", "compression")

    db.create_session(session_id="child", source="cli", parent_session_id="root")
    db._conn.execute(
        "UPDATE sessions SET started_at = (SELECT ended_at FROM sessions WHERE id = 'root') + 1 WHERE id = 'child'"
    )
    db.end_session("child", "compression")

    db.create_session(session_id="tail", source="cli", parent_session_id="child")
    db._conn.execute(
        "UPDATE sessions SET started_at = (SELECT ended_at FROM sessions WHERE id = 'child') + 1 WHERE id = 'tail'"
    )
    db._conn.commit()
    db.set_compaction_handoff("tail", f"{SUMMARY_PREFIX}\npersisted handoff state")

    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "root"
    agent._session_db = db
    agent._pending_compaction_handoff = None
    agent.context_compressor = MagicMock()
    agent.context_compressor._compaction_handoff_state = None

    agent._rehydrate_persisted_compaction_handoff([
        {"role": "user", "content": "earlier turn"},
    ])

    assert agent._pending_compaction_handoff == f"{SUMMARY_PREFIX}\npersisted handoff state"
    assert agent.context_compressor._compaction_handoff_state == f"{SUMMARY_PREFIX}\npersisted handoff state"



def test_agent_compress_context_persists_pending_handoff_on_new_continuation_session(tmp_path):
    from pathlib import Path

    from hermes_state import SessionDB
    from run_agent import AIAgent

    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="original-session", source="cli")

    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = None
    agent.commit_memory_session = lambda messages: None
    agent._todo_store = MagicMock()
    agent._todo_store.format_for_injection.return_value = ""
    agent._invalidate_system_prompt = MagicMock()
    agent._build_system_prompt = MagicMock(return_value="new system")
    agent._cached_system_prompt = None
    agent._session_db = db
    agent._session_db_created = True
    agent._session_init_model_config = {"max_iterations": 90}
    agent._last_flushed_db_idx = 0
    agent._vprint = MagicMock()
    agent._emit_warning = MagicMock()
    agent.tools = []
    agent.model = "test/model"
    agent.session_id = "original-session"
    agent.platform = "cli"
    agent.logs_dir = Path(tmp_path)
    agent._last_compression_summary_warning = None
    agent._last_aux_fallback_warning_key = None

    compressor = MagicMock()
    compressor.compress.return_value = [{"role": "user", "content": "compressed"}]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._compaction_handoff_state = f"{SUMMARY_PREFIX}\npersist this continuity"
    agent.context_compressor = compressor

    old_session_id = agent.session_id
    agent._compress_context([
        {"role": "user", "content": "m1"},
        {"role": "assistant", "content": "m2"},
        {"role": "user", "content": "m3"},
        {"role": "assistant", "content": "m4"},
    ], "sys", approx_tokens=100)

    assert agent.session_id != old_session_id
    continuation = db.get_session(agent.session_id)
    assert continuation["parent_session_id"] == old_session_id
    assert continuation["compaction_handoff"] == f"{SUMMARY_PREFIX}\npersist this continuity"
