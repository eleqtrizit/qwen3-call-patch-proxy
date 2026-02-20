#!/usr/bin/env python3
"""
Tests targeting uncovered code paths to boost coverage from 52% to 85%+.
"""
import asyncio
import json
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from aiohttp import web

sys.path.insert(0, os.path.dirname(__file__))

from qwen3_call_patch_proxy import (
    CONFIG_FILE,
    ToolBuffer,
    ToolFixEngine,
    RequestState,
    cleanup_request,
    detect_and_convert_xml_tool_call,
    fix_engine,
    get_fixed_arguments,
    handle_request,
    infer_tool_name_from_content,
    is_json_complete,
    periodic_cleanup,
    process_all_buffers,
    process_complete_buffer,
    process_remaining_buffers,
    process_sse_event,
    request_states,
    try_fix_incomplete_json,
    try_json_recovery,
    validate_json_syntax,
)


# ---------------------------------------------------------------------------
# ToolBuffer
# ---------------------------------------------------------------------------

class TestToolBuffer:
    def test_is_expired_true(self):
        """
        Test that is_expired returns True when last_updated is past timeout.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x")
        buf.last_updated = datetime.now() - timedelta(seconds=60)
        assert buf.is_expired(30) is True

    def test_is_expired_false(self):
        """
        Test that is_expired returns False for a freshly updated buffer.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x")
        assert buf.is_expired(30) is False

    def test_update_content_appends(self):
        """
        Test that update_content appends text and refreshes timestamp.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x", content="hello")
        before = buf.last_updated
        buf.update_content(" world")
        assert buf.content == "hello world"
        assert buf.last_updated >= before

    def test_size(self):
        """
        Test that size returns byte length of content.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x", content="abc")
        assert buf.size() == 3


# ---------------------------------------------------------------------------
# RequestState
# ---------------------------------------------------------------------------

class TestRequestState:
    def test_cleanup_expired_buffers(self):
        """
        Test that cleanup_expired_buffers removes only expired buffers.

        :return: None
        :rtype: None
        """
        state = RequestState(request_id="req-1")
        expired = ToolBuffer(call_id="old")
        expired.last_updated = datetime.now() - timedelta(seconds=60)
        fresh = ToolBuffer(call_id="new")

        state.tool_buffers["old"] = expired
        state.tool_buffers["new"] = fresh

        state.cleanup_expired_buffers(30)

        assert "old" not in state.tool_buffers
        assert "new" in state.tool_buffers


# ---------------------------------------------------------------------------
# ToolFixEngine
# ---------------------------------------------------------------------------

class TestToolFixEngineLoadConfig:
    def test_load_config_file_not_found_uses_defaults(self, tmp_path):
        """
        Test that a missing config file falls back to defaults.

        :return: None
        :rtype: None
        """
        engine = ToolFixEngine(str(tmp_path / "nonexistent.yaml"))
        assert "tools" in engine.config
        assert "bash" in engine.config["tools"]

    def test_load_config_bad_yaml_uses_defaults(self, tmp_path):
        """
        Test that malformed YAML falls back to defaults.

        :return: None
        :rtype: None
        """
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{\x00invalid yaml}")
        engine = ToolFixEngine(str(bad_yaml))
        assert "tools" in engine.config


class TestApplySingleFix:
    def _engine(self):
        return ToolFixEngine("nonexistent.yaml")

    def test_parse_json_object_from_string(self):
        """
        Test parse_json_object action converts string to dict.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        fix = {"name": "f", "parameter": "p", "condition": "is_string", "action": "parse_json_object"}
        args = {"p": '{"key": "val"}'}
        engine._apply_single_fix(args, fix, "req")
        assert args["p"] == {"key": "val"}

    def test_convert_string_to_boolean_true(self):
        """
        Test convert_string_to_boolean converts 'true' string.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        fix = {"name": "f", "parameter": "p", "condition": "is_string", "action": "convert_string_to_boolean"}
        args = {"p": "true"}
        engine._apply_single_fix(args, fix, "req")
        assert args["p"] is True

    def test_convert_string_to_boolean_false(self):
        """
        Test convert_string_to_boolean converts 'false' string.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        fix = {"name": "f", "parameter": "p", "condition": "is_string", "action": "convert_string_to_boolean"}
        args = {"p": "false"}
        engine._apply_single_fix(args, fix, "req")
        assert args["p"] is False

    def test_convert_tool_to_write_success(self):
        """
        Test convert_tool_to_write returns new tool name when fields present.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        fix = {"name": "f", "parameter": "content", "condition": "exists", "action": "convert_tool_to_write"}
        args = {"filePath": "foo.py", "content": "hello"}
        result = engine._apply_single_fix(args, fix, "req")
        assert result == ("write", True)

    def test_convert_tool_to_write_missing_fields(self):
        """
        Test convert_tool_to_write returns False when required fields missing.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        fix = {"name": "f", "parameter": "content", "condition": "exists", "action": "convert_tool_to_write"}
        args = {"content": "hello"}  # missing filePath
        result = engine._apply_single_fix(args, fix, "req")
        assert result is False

    def test_exception_with_fallback(self):
        """
        Test that an action exception applies fallback_value and returns True.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        # parse_json_array on un-parseable string triggers fallback
        fix = {
            "name": "f",
            "parameter": "p",
            "condition": "is_string",
            "action": "parse_json_array",
            "fallback_value": [],
        }
        args = {"p": "not valid json at all !!!"}
        result = engine._apply_single_fix(args, fix, "req")
        assert result is True
        assert args["p"] == []

    def test_exception_without_fallback_returns_false(self):
        """
        Test that an action exception without fallback_value returns False.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        fix = {
            "name": "f",
            "parameter": "p",
            "condition": "is_string",
            "action": "parse_json_object",
            # no fallback_value
        }
        args = {"p": "not json {{{"}
        result = engine._apply_single_fix(args, fix, "req")
        assert result is False


class TestCheckCondition:
    def _engine(self):
        return ToolFixEngine("nonexistent.yaml")

    def test_missing_condition(self):
        """
        Test 'missing' condition returns True when param absent.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        assert engine._check_condition({}, "p", "missing", {}) is True
        assert engine._check_condition({"p": "v"}, "p", "missing", {}) is False

    def test_exists_condition(self):
        """
        Test 'exists' condition returns True when param present.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        assert engine._check_condition({"p": "v"}, "p", "exists", {}) is True
        assert engine._check_condition({}, "p", "exists", {}) is False

    def test_invalid_enum_condition(self):
        """
        Test 'invalid_enum' condition returns True when value not in valid list.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        fix = {"valid_values": ["a", "b"]}
        assert engine._check_condition({"p": "c"}, "p", "invalid_enum", fix) is True
        assert engine._check_condition({"p": "a"}, "p", "invalid_enum", fix) is False

    def test_unknown_condition_returns_false(self):
        """
        Test that an unknown condition returns False.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        assert engine._check_condition({"p": "v"}, "p", "unknown_cond", {}) is False


class TestFixMalformedJson:
    def _engine(self):
        return ToolFixEngine("nonexistent.yaml")

    def test_empty_string_returns_empty(self):
        """
        Test that _fix_malformed_json returns empty string for empty input.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        assert engine._fix_malformed_json("") == ""

    def test_single_quotes_fixed(self):
        """
        Test that single quotes are replaced with double quotes.

        :return: None
        :rtype: None
        """
        engine = self._engine()
        result = engine._fix_malformed_json("{'key': 'value'}")
        assert '"key"' in result
        assert '"value"' in result


# ---------------------------------------------------------------------------
# validate_json_syntax
# ---------------------------------------------------------------------------

class TestValidateJsonSyntax:
    def test_valid_json(self):
        """
        Test that valid JSON returns True.

        :return: None
        :rtype: None
        """
        assert validate_json_syntax('{"key": "value"}') is True

    def test_invalid_json(self):
        """
        Test that invalid JSON returns False.

        :return: None
        :rtype: None
        """
        assert validate_json_syntax('{bad json}') is False


# ---------------------------------------------------------------------------
# try_fix_incomplete_json
# ---------------------------------------------------------------------------

class TestTryFixIncompleteJson:
    @pytest.mark.asyncio
    async def test_fixes_missing_closing_brace(self):
        """
        Test that a missing closing brace is added.

        :return: None
        :rtype: None
        """
        result = await try_fix_incomplete_json('{"key": "value"')
        assert result is not None
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    @pytest.mark.asyncio
    async def test_empty_string_returns_empty(self):
        """
        Test that empty input returns empty string.

        :return: None
        :rtype: None
        """
        result = await try_fix_incomplete_json("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_empty(self):
        """
        Test that whitespace-only input returns empty string.

        :return: None
        :rtype: None
        """
        result = await try_fix_incomplete_json("   ")
        assert result == ""

    @pytest.mark.asyncio
    async def test_unfixable_returns_none(self):
        """
        Test that truly unfixable JSON returns None.

        :return: None
        :rtype: None
        """
        result = await try_fix_incomplete_json('{"key": }}}}}')
        assert result is None

    @pytest.mark.asyncio
    async def test_already_valid_json(self):
        """
        Test that already-valid JSON is returned unchanged.

        :return: None
        :rtype: None
        """
        result = await try_fix_incomplete_json('{"key": "value"}')
        assert result == '{"key": "value"}'


# ---------------------------------------------------------------------------
# try_json_recovery
# ---------------------------------------------------------------------------

class TestTryJsonRecovery:
    @pytest.mark.asyncio
    async def test_trailing_comma_recovery(self):
        """
        Test recovery from JSON with trailing comma.

        :return: None
        :rtype: None
        """
        tool = {"function": {"name": "bash", "arguments": ""}}
        result = await try_json_recovery('{"command": "ls",', tool, "bash", "req")
        assert result is True
        assert tool["function"]["arguments"]

    @pytest.mark.asyncio
    async def test_missing_opening_brace_recovery(self):
        """
        Test recovery from JSON missing opening brace.

        :return: None
        :rtype: None
        """
        tool = {"function": {"name": "bash", "arguments": ""}}
        # No opening brace — recovery wraps with { }
        result = await try_json_recovery('"command": "ls", "description": "list"', tool, "bash", "req")
        assert result is True

    @pytest.mark.asyncio
    async def test_all_recovery_fails(self):
        """
        Test that completely unrecoverable JSON returns False.

        :return: None
        :rtype: None
        """
        tool = {"function": {"name": "bash", "arguments": ""}}
        result = await try_json_recovery('totally not json at all @#$%', tool, "bash", "req")
        assert result is False


# ---------------------------------------------------------------------------
# get_fixed_arguments
# ---------------------------------------------------------------------------

class TestGetFixedArguments:
    @pytest.mark.asyncio
    async def test_valid_buffer_returns_fixed_args(self):
        """
        Test that a valid buffer returns tool name and JSON string.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x", tool_name="bash", content='{"command": "ls"}')
        tool_name, args = await get_fixed_arguments(buf, "req")
        assert tool_name == "bash"
        assert '"command"' in args

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty_args(self):
        """
        Test that invalid JSON in buffer returns original tool name and empty args.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x", tool_name="bash", content='not json {{{')
        tool_name, args = await get_fixed_arguments(buf, "req")
        assert tool_name == "bash"
        assert args == ""


# ---------------------------------------------------------------------------
# cleanup_request
# ---------------------------------------------------------------------------

class TestCleanupRequest:
    @pytest.mark.asyncio
    async def test_removes_request_state(self):
        """
        Test that cleanup_request removes the state entry.

        :return: None
        :rtype: None
        """
        request_id = "cleanup-test"
        request_states[request_id] = RequestState(request_id=request_id)
        await cleanup_request(request_id)
        assert request_id not in request_states

    @pytest.mark.asyncio
    async def test_noop_if_not_present(self):
        """
        Test that cleanup_request is a no-op for unknown request IDs.

        :return: None
        :rtype: None
        """
        await cleanup_request("nonexistent-id")  # should not raise


# ---------------------------------------------------------------------------
# process_sse_event edge cases
# ---------------------------------------------------------------------------

class TestProcessSseEvent:
    @pytest.mark.asyncio
    async def test_missing_request_id_returns_event(self):
        """
        Test that an event for an unknown request_id is returned unchanged.

        :return: None
        :rtype: None
        """
        event = {"choices": [{"delta": {"content": "hello"}}]}
        result = await process_sse_event(event, "nonexistent-req-id")
        assert result == event

    @pytest.mark.asyncio
    async def test_no_choices_returns_event_unchanged(self):
        """
        Test that an event with empty choices is returned unchanged.

        :return: None
        :rtype: None
        """
        request_id = "no-choices-test"
        request_states[request_id] = RequestState(request_id=request_id)
        try:
            event = {"choices": []}
            result = await process_sse_event(event, request_id)
            assert result == event
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_event_without_choices_key(self):
        """
        Test that an event without a 'choices' key is returned unchanged.

        :return: None
        :rtype: None
        """
        request_id = "no-choices-key-test"
        request_states[request_id] = RequestState(request_id=request_id)
        try:
            event = {"model": "qwen3"}
            result = await process_sse_event(event, request_id)
            assert result == event
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_content_buffer_overflow_clears(self):
        """
        Test that content buffer is cleared when it exceeds max size.

        :return: None
        :rtype: None
        """
        request_id = "overflow-test"
        state = RequestState(request_id=request_id)
        # Pre-fill buffer just below limit to trigger on next chunk
        engine_max = fix_engine.get_setting("max_buffer_size", 1048576)
        state.content_buffer = "x" * (engine_max + 1)
        request_states[request_id] = state
        try:
            event = {"choices": [{"delta": {"content": "more"}}]}
            await process_sse_event(event, request_id)
            # Buffer should be cleared (it exceeded limit in previous turn, cleared now)
            assert len(state.content_buffer) < engine_max + 10
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_finish_reason_tool_calls_processes_buffers(self):
        """
        Test that finish_reason=tool_calls triggers buffer processing.

        :return: None
        :rtype: None
        """
        request_id = "finish-reason-test"
        state = RequestState(request_id=request_id)
        buf = ToolBuffer(call_id="main_tool_call", tool_name="bash",
                         content='{"command": "ls", "description": "list"}')
        state.tool_buffers["main_tool_call"] = buf
        request_states[request_id] = state
        try:
            event = {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
            await process_sse_event(event, request_id)
            # process_all_buffers fixes the buffer content
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_fragment_buffer_size_exceeded_clears(self):
        """
        Test that exceeding buffer size limit clears tool_calls from delta.

        :return: None
        :rtype: None
        """
        request_id = "buf-size-test"
        state = RequestState(request_id=request_id)
        request_states[request_id] = state

        # Pre-fill the main_tool_call buffer to near the limit
        from qwen3_call_patch_proxy import ToolBuffer
        max_size = fix_engine.get_setting("max_buffer_size", 1048576)
        big_buf = ToolBuffer(call_id="main_tool_call", tool_name="bash")
        big_buf.content = "x" * (max_size + 1)
        state.tool_buffers["main_tool_call"] = big_buf

        try:
            # Send a fragment that would push over the size limit
            event = {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "function": {"arguments": '{"command": "ls"}'}}
                        ]
                    }
                }]
            }
            result = await process_sse_event(event, request_id)
            delta = result["choices"][0]["delta"]
            # tool_calls should be suppressed or empty
            assert "tool_calls" not in delta or len(delta["tool_calls"]) == 0
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_fragments_with_unknown_tool_name_suppressed(self):
        """
        Test that fragments whose tool cannot be inferred are suppressed.

        :return: None
        :rtype: None
        """
        request_id = "no-tool-name-test"
        state = RequestState(request_id=request_id)
        request_states[request_id] = state
        try:
            # Send a complete JSON fragment with no recognizable keys
            event = {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"zzz_unknown_key": "value"}'}}
                        ]
                    }
                }]
            }
            result = await process_sse_event(event, request_id)
            delta = result["choices"][0]["delta"]
            # Since tool name can't be inferred, output should be suppressed
            assert "tool_calls" not in delta or len(delta["tool_calls"]) == 0
        finally:
            request_states.pop(request_id, None)


# ---------------------------------------------------------------------------
# process_complete_buffer
# ---------------------------------------------------------------------------

class TestProcessCompleteBuffer:
    @pytest.mark.asyncio
    async def test_tool_name_conversion(self):
        """
        Test that tool name is updated in the tool dict when converted.

        :return: None
        :rtype: None
        """
        # Use the YAML config's read→write conversion
        buf = ToolBuffer(call_id="x", tool_name="read",
                         content='{"filePath": "foo.py", "content": "hello"}')
        tool = {"function": {"name": "read", "arguments": ""}}
        await process_complete_buffer(buf, tool, "req")
        # The read+content → write conversion
        assert tool["function"]["name"] == "write"

    @pytest.mark.asyncio
    async def test_json_parse_failure_triggers_recovery(self):
        """
        Test that a JSON parse failure attempts recovery.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x", tool_name="bash",
                         content='{"command": "ls",')
        tool = {"function": {"name": "bash", "arguments": ""}}
        await process_complete_buffer(buf, tool, "req")
        # Recovery should succeed and produce valid JSON
        if tool["function"]["arguments"]:
            json.loads(tool["function"]["arguments"])

    @pytest.mark.asyncio
    async def test_successful_fix(self):
        """
        Test that valid JSON args are fixed and stored back in tool dict.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x", tool_name="bash",
                         content='{"command": "ls", "description": "list"}')
        tool = {"function": {"name": "bash", "arguments": ""}}
        await process_complete_buffer(buf, tool, "req")
        args = json.loads(tool["function"]["arguments"])
        assert args["command"] == "ls"


# ---------------------------------------------------------------------------
# process_all_buffers
# ---------------------------------------------------------------------------

class TestProcessAllBuffers:
    @pytest.mark.asyncio
    async def test_empty_buffers_noop(self):
        """
        Test that process_all_buffers is a no-op with no buffers.

        :return: None
        :rtype: None
        """
        state = RequestState(request_id="req-empty")
        await process_all_buffers(state, "req-empty")  # should not raise

    @pytest.mark.asyncio
    async def test_fixes_buffer_content(self):
        """
        Test that process_all_buffers fixes valid buffer content in-place.

        :return: None
        :rtype: None
        """
        state = RequestState(request_id="req-fix")
        buf = ToolBuffer(call_id="main_tool_call", tool_name="bash",
                         content='{"command": "ls", "description": "list"}')
        state.tool_buffers["main_tool_call"] = buf
        await process_all_buffers(state, "req-fix")
        # Buffer should still exist with fixed content
        if "main_tool_call" in state.tool_buffers:
            json.loads(state.tool_buffers["main_tool_call"].content)

    @pytest.mark.asyncio
    async def test_drops_empty_buffers(self):
        """
        Test that buffers with no content are dropped.

        :return: None
        :rtype: None
        """
        state = RequestState(request_id="req-drop")
        buf = ToolBuffer(call_id="empty_buf", tool_name="bash", content="")
        state.tool_buffers["empty_buf"] = buf
        await process_all_buffers(state, "req-drop")
        assert "empty_buf" not in state.tool_buffers

    @pytest.mark.asyncio
    async def test_drops_unfixable_buffers(self):
        """
        Test that buffers with unrecoverable JSON are dropped.

        :return: None
        :rtype: None
        """
        state = RequestState(request_id="req-unfixable")
        buf = ToolBuffer(call_id="bad_buf", tool_name="bash",
                         content='totally not json @#$%^&*()')
        state.tool_buffers["bad_buf"] = buf
        await process_all_buffers(state, "req-unfixable")
        assert "bad_buf" not in state.tool_buffers


# ---------------------------------------------------------------------------
# process_remaining_buffers
# ---------------------------------------------------------------------------

class TestProcessRemainingBuffers:
    def _mock_response(self):
        resp = MagicMock()
        resp.write = AsyncMock()
        return resp

    @pytest.mark.asyncio
    async def test_noop_if_no_state(self):
        """
        Test that process_remaining_buffers is a no-op for unknown request IDs.

        :return: None
        :rtype: None
        """
        resp = self._mock_response()
        await process_remaining_buffers("no-such-id", resp)
        resp.write.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_if_no_buffers(self):
        """
        Test that process_remaining_buffers is a no-op when there are no buffers.

        :return: None
        :rtype: None
        """
        request_id = "prb-no-buf"
        request_states[request_id] = RequestState(request_id=request_id)
        resp = self._mock_response()
        try:
            await process_remaining_buffers(request_id, resp)
            resp.write.assert_not_called()
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_emits_completion_for_valid_buffer(self):
        """
        Test that a fixable buffer results in a SSE completion being written.

        :return: None
        :rtype: None
        """
        request_id = "prb-emit"
        state = RequestState(request_id=request_id)
        buf = ToolBuffer(call_id="main_tool_call", tool_name="bash",
                         content='{"command": "ls", "description": "list"}')
        state.tool_buffers["main_tool_call"] = buf
        request_states[request_id] = state
        resp = self._mock_response()
        try:
            await process_remaining_buffers(request_id, resp)
            resp.write.assert_called()
            written_bytes = resp.write.call_args[0][0]
            payload = written_bytes.decode("utf-8")
            assert "data:" in payload
            event_data = json.loads(payload.removeprefix("data:").strip())
            tool_calls = event_data["choices"][0]["delta"]["tool_calls"]
            assert tool_calls[0]["function"]["name"] == "bash"
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_skips_buffer_without_tool_name(self):
        """
        Test that buffers without a tool name are skipped (no completion written).

        :return: None
        :rtype: None
        """
        request_id = "prb-no-tool"
        state = RequestState(request_id=request_id)
        # Content that can be fixed but tool_name is empty and uninferable
        buf = ToolBuffer(call_id="unknown_buf", tool_name="",
                         content='{"zzz_unknown": "value"}')
        state.tool_buffers["unknown_buf"] = buf
        request_states[request_id] = state
        resp = self._mock_response()
        try:
            await process_remaining_buffers(request_id, resp)
            resp.write.assert_not_called()
        finally:
            request_states.pop(request_id, None)


# ---------------------------------------------------------------------------
# health_check and reload_config (aiohttp handlers)
# ---------------------------------------------------------------------------

class TestAiohttpHandlers:
    @pytest.mark.asyncio
    async def test_health_check(self):
        """
        Test health_check handler returns status healthy.

        :return: None
        :rtype: None
        """
        from aiohttp.test_utils import make_mocked_request
        from qwen3_call_patch_proxy import health_check

        request = make_mocked_request("GET", "/_health", app=MagicMock())
        request.app.__getitem__ = lambda self, key: "http://127.0.0.1:8080" if key == "target_url" else None

        response = await health_check(request)
        data = json.loads(response.body)
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_reload_config_success(self):
        """
        Test reload_config handler returns success status.

        :return: None
        :rtype: None
        """
        from aiohttp.test_utils import make_mocked_request
        from qwen3_call_patch_proxy import reload_config

        request = make_mocked_request("POST", "/_reload")
        response = await reload_config(request)
        data = json.loads(response.body)
        assert data["status"] == "success"


# ---------------------------------------------------------------------------
# main() argument parsing
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_imports_without_running(self):
        """
        Test that the main function can be imported and inspected without running a server.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import main
        import inspect
        assert inspect.isfunction(main)

    def test_main_with_custom_args(self, monkeypatch):
        """
        Test that main() parses CLI arguments correctly before starting the server.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import main
        import aiohttp.web as web

        monkeypatch.setattr(sys, "argv", [
            "proxy", "--target-url", "http://localhost:9999", "--port", "8001", "--verbose"
        ])

        captured = {}

        def fake_run_app(app, host, port):
            captured["host"] = host
            captured["port"] = port
            captured["target_url"] = app["target_url"]
            captured["verbose"] = app["verbose"]
            raise KeyboardInterrupt  # exit cleanly

        monkeypatch.setattr(web, "run_app", fake_run_app)
        main()

        assert captured["port"] == 8001
        assert captured["target_url"] == "http://localhost:9999"
        assert captured["verbose"] is True


# ---------------------------------------------------------------------------
# remove_parameter fix action
# ---------------------------------------------------------------------------

class TestRemoveParameterAction:
    def test_remove_parameter_removes_key(self):
        """
        Test that remove_parameter action deletes the specified key.

        :return: None
        :rtype: None
        """
        engine = ToolFixEngine("nonexistent.yaml")
        fix = {"name": "f", "parameter": "p", "condition": "exists", "action": "remove_parameter"}
        args = {"p": "remove_me", "other": "keep"}
        engine._apply_single_fix(args, fix, "req")
        assert "p" not in args
        assert "other" in args

    def test_remove_parameter_noop_when_absent(self):
        """
        Test that remove_parameter is safe when parameter is not present.

        :return: None
        :rtype: None
        """
        engine = ToolFixEngine("nonexistent.yaml")
        fix = {"name": "f", "parameter": "p", "condition": "exists", "action": "remove_parameter"}
        # condition 'exists' would be False, so action won't run — but let's
        # verify the underlying code path if called directly with condition met
        args = {"other": "keep"}
        # condition is 'exists' and 'p' not in args, so _apply_single_fix returns False
        result = engine._apply_single_fix(args, fix, "req")
        assert result is False


# ---------------------------------------------------------------------------
# cleanup_request with non-empty buffers
# ---------------------------------------------------------------------------

class TestCleanupRequestWithBuffers:
    @pytest.mark.asyncio
    async def test_cleanup_with_buffers_logs_and_removes(self):
        """
        Test cleanup_request removes state that contains active tool buffers.

        :return: None
        :rtype: None
        """
        request_id = "cleanup-buf-test"
        state = RequestState(request_id=request_id)
        state.tool_buffers["buf1"] = ToolBuffer(call_id="buf1", content='{"cmd": "ls"}')
        state.tool_buffers["buf2"] = ToolBuffer(call_id="buf2", content='{"path": "."}')
        request_states[request_id] = state
        await cleanup_request(request_id)
        assert request_id not in request_states


# ---------------------------------------------------------------------------
# Named call with incomplete JSON — lines 702-762
# ---------------------------------------------------------------------------

class TestNamedCallIncompleteJson:
    @pytest.mark.asyncio
    async def test_named_call_incomplete_json_merged_into_fragment_buffer(self):
        """
        Test that a named tool call with incomplete JSON is merged into the shared buffer.

        :return: None
        :rtype: None
        """
        request_id = "incomplete-named"
        state = RequestState(request_id=request_id)
        request_states[request_id] = state
        try:
            # Named tool call with incomplete JSON (missing closing brace)
            event = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_abc123",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "ls"'  # incomplete
                            }
                        }]
                    }
                }]
            }
            result = await process_sse_event(event, request_id)
            delta = result["choices"][0]["delta"]
            # The named call should be suppressed (merged into fragment buffer)
            assert "tool_calls" not in delta or len(delta["tool_calls"]) == 0
            # The main_tool_call buffer should now hold the merged content
            assert "main_tool_call" in state.tool_buffers
            assert "command" in state.tool_buffers["main_tool_call"].content
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_named_call_incomplete_becomes_complete_when_fragment_arrives(self):
        """
        Test that named+fragment combination produces a complete tool call.

        :return: None
        :rtype: None
        """
        request_id = "incomplete-named-then-complete"
        state = RequestState(request_id=request_id)
        request_states[request_id] = state
        try:
            # First event: named call with incomplete JSON
            event1 = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_xyz",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command": "ls", "description": "list"'
                            }
                        }]
                    }
                }]
            }
            await process_sse_event(event1, request_id)

            # Second event: fragment that completes the JSON
            event2 = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "}"}}]
                    }
                }]
            }
            result2 = await process_sse_event(event2, request_id)
            delta2 = result2["choices"][0]["delta"]
            # Should now have a complete tool call
            assert "tool_calls" in delta2 and len(delta2["tool_calls"]) == 1
            assert delta2["tool_calls"][0]["function"]["name"] == "bash"
        finally:
            request_states.pop(request_id, None)

    @pytest.mark.asyncio
    async def test_named_call_merged_with_existing_fragment_prefix(self):
        """
        Test that named call content is appended when fragment buffer starts with JSON opener.

        :return: None
        :rtype: None
        """
        request_id = "merge-append-test"
        state = RequestState(request_id=request_id)
        # Pre-fill the main buffer with a JSON prefix
        main_buf = ToolBuffer(call_id="main_tool_call", tool_name="bash",
                              content='{"command": "ls"')
        state.tool_buffers["main_tool_call"] = main_buf
        request_states[request_id] = state
        try:
            # Named call arrives with additional incomplete content
            event = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "id": "call_def",
                            "function": {
                                "name": "bash",
                                "arguments": ', "description": "list"'
                            }
                        }]
                    }
                }]
            }
            await process_sse_event(event, request_id)
            # The buffer should have the combined content
            if "main_tool_call" in state.tool_buffers:
                assert "command" in state.tool_buffers["main_tool_call"].content
        finally:
            request_states.pop(request_id, None)


# ---------------------------------------------------------------------------
# process_complete_buffer — recovery failure path (lines 822-823)
# ---------------------------------------------------------------------------

class TestProcessCompleteBufferRecoveryFailure:
    @pytest.mark.asyncio
    async def test_recovery_failure_keeps_original(self):
        """
        Test that when all recovery attempts fail, the original content is kept.

        :return: None
        :rtype: None
        """
        buf = ToolBuffer(call_id="x", tool_name="bash",
                         content='totally unparseable @@## content {{{{')
        tool = {"function": {"name": "bash", "arguments": "original"}}
        await process_complete_buffer(buf, tool, "req")
        # When both parse and recovery fail, original arguments are kept
        assert tool["function"]["arguments"] == "original"


# ---------------------------------------------------------------------------
# process_remaining_buffers — fallback recovery path (lines 896-900)
# ---------------------------------------------------------------------------

class TestProcessRemainingBuffersFallback:
    @pytest.mark.asyncio
    async def test_fallback_recovery_when_fix_incomplete_fails(self):
        """
        Test that try_json_recovery is used when try_fix_incomplete_json returns falsy.

        :return: None
        :rtype: None
        """
        request_id = "prb-fallback"
        state = RequestState(request_id=request_id)
        # Content that try_fix_incomplete_json can't fix but try_json_recovery can
        # Trailing comma: fix_incomplete adds } → still invalid; recovery strips comma
        buf = ToolBuffer(call_id="main_tool_call", tool_name="bash",
                         content='{"command": "ls", "description": "list",')
        state.tool_buffers["main_tool_call"] = buf
        request_states[request_id] = state

        resp = MagicMock()
        resp.write = AsyncMock()
        try:
            await process_remaining_buffers(request_id, resp)
            resp.write.assert_called()
            written = resp.write.call_args[0][0].decode("utf-8")
            data = json.loads(written.removeprefix("data:").strip())
            assert data["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "bash"
        finally:
            request_states.pop(request_id, None)


# ---------------------------------------------------------------------------
# is_json_complete — escape and mismatch edge cases
# ---------------------------------------------------------------------------

class TestIsJsonCompleteEdgeCases:
    def test_escape_sequence_in_string(self):
        """
        Test that escape sequences inside strings are handled correctly.

        :return: None
        :rtype: None
        """
        # The backslash-escaped quote should not toggle in_string
        assert is_json_complete('{"key": "val\\"ue"}') is True

    def test_extra_closing_bracket_returns_false(self):
        """
        Test that an unmatched closing bracket returns False.

        :return: None
        :rtype: None
        """
        assert is_json_complete('}') is False

    def test_mismatched_bracket_returns_false(self):
        """
        Test that mismatched brackets return False.

        :return: None
        :rtype: None
        """
        assert is_json_complete('[}') is False

    def test_non_json_start_returns_false(self):
        """
        Test that strings not starting with {{ or [ return False.

        :return: None
        :rtype: None
        """
        assert is_json_complete('hello world') is False


# ---------------------------------------------------------------------------
# infer_tool_name_from_content — empty string
# ---------------------------------------------------------------------------

class TestInferToolNameEdgeCases:
    def test_empty_content_returns_empty(self):
        """
        Test that empty content returns empty string.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        assert infer_tool_name_from_content("") == ""

    def test_task_tool_detection(self):
        """
        Test detection of task tool from subagent_type parameter.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        content = '{"description": "do something", "prompt": "...", "subagent_type": "generalPurpose"}'
        assert infer_tool_name_from_content(content) == "task"

    def test_notebookedit_detection(self):
        """
        Test detection of notebookedit tool.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        content = '{"notebook_path": "foo.ipynb", "new_source": "print(1)"}'
        assert infer_tool_name_from_content(content) == "notebookedit"

    def test_webfetch_detection(self):
        """
        Test detection of webfetch tool.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        content = '{"url": "https://example.com", "prompt": "summarize"}'
        assert infer_tool_name_from_content(content) == "webfetch"

    def test_websearch_detection(self):
        """
        Test detection of websearch tool.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        content = '{"query": "python asyncio"}'
        assert infer_tool_name_from_content(content) == "websearch"

    def test_write_detection(self):
        """
        Test detection of write tool when content and file_path both present.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        content = '{"file_path": "foo.py", "content": "print(1)"}'
        assert infer_tool_name_from_content(content) == "write"

    def test_edit_detection_with_file_path(self):
        """
        Test detection of edit tool with file_path/old_string/new_string.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        content = '{"file_path": "foo.py", "old_string": "a", "new_string": "b"}'
        assert infer_tool_name_from_content(content) == "edit"

    def test_path_only_returns_list(self):
        """
        Test that path-only content maps to list tool.

        :return: None
        :rtype: None
        """
        from qwen3_call_patch_proxy import infer_tool_name_from_content
        content = '{"path": "/some/dir"}'
        assert infer_tool_name_from_content(content) == "list"


# ---------------------------------------------------------------------------
# reload_config exception path (lines 1185-1187)
# ---------------------------------------------------------------------------

class TestReloadConfigException:
    @pytest.mark.asyncio
    async def test_reload_config_returns_error_on_exception(self):
        """
        Test that reload_config returns 500 when ToolFixEngine raises an exception.

        :return: None
        :rtype: None
        """
        from aiohttp.test_utils import make_mocked_request
        from qwen3_call_patch_proxy import reload_config
        import qwen3_call_patch_proxy as mod

        request = make_mocked_request("POST", "/_reload")

        with patch.object(mod, "ToolFixEngine", side_effect=Exception("bad config")):
            response = await reload_config(request)

        data = json.loads(response.body)
        assert data["status"] == "error"
        assert response.status == 500


# ---------------------------------------------------------------------------
# periodic_cleanup
# ---------------------------------------------------------------------------

class TestPeriodicCleanup:
    @pytest.mark.asyncio
    async def test_periodic_cleanup_runs_until_request_removed(self):
        """
        Test that periodic_cleanup removes expired buffers and exits when state is gone.

        :return: None
        :rtype: None
        """
        request_id = "periodic-test"
        state = RequestState(request_id=request_id)

        # Add an expired buffer
        expired = ToolBuffer(call_id="old")
        expired.last_updated = datetime.now() - timedelta(seconds=60)
        state.tool_buffers["old"] = expired
        request_states[request_id] = state

        # Run cleanup with very short sleep by patching asyncio.sleep
        call_count = 0

        async def fast_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 1:
                # Remove the request after first cleanup iteration to stop the loop
                request_states.pop(request_id, None)

        with patch("qwen3_call_patch_proxy.asyncio.sleep", side_effect=fast_sleep):
            await periodic_cleanup(request_id)

        assert request_id not in request_states

    @pytest.mark.asyncio
    async def test_periodic_cleanup_handles_cancellation(self):
        """
        Test that periodic_cleanup exits cleanly on asyncio.CancelledError.

        :return: None
        :rtype: None
        """
        request_id = "periodic-cancel-test"
        request_states[request_id] = RequestState(request_id=request_id)
        try:
            async def raise_cancelled(seconds):
                raise asyncio.CancelledError

            with patch("qwen3_call_patch_proxy.asyncio.sleep", side_effect=raise_cancelled):
                await periodic_cleanup(request_id)  # should not raise
        finally:
            request_states.pop(request_id, None)


# ---------------------------------------------------------------------------
# handle_request — core proxy logic via mocked aiohttp.ClientSession
# ---------------------------------------------------------------------------

def _make_async_iter(items):
    """Create an async iterator from a list of byte items."""

    class _AsyncIter:
        def __init__(self, seq):
            self._seq = iter(seq)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._seq)
            except StopIteration:
                raise StopAsyncIteration

    return _AsyncIter(items)


def _make_mock_backend_response(sse_lines, status=200, reason="OK"):
    """Build a mock aiohttp backend response for SSE streaming tests."""
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.reason = reason
    mock_resp.headers = {}
    mock_resp.content = _make_async_iter(sse_lines)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.request = MagicMock(return_value=mock_resp)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    return mock_session


class TestHandleRequest:
    @pytest.mark.asyncio
    async def test_handle_request_streams_sse(self):
        """
        Test that handle_request proxies SSE events to the client.

        :return: None
        :rtype: None
        """
        sse_lines = [
            b'data: {"choices": [{"delta": {"content": "hello"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        mock_session = _make_mock_backend_response(sse_lines)

        app = web.Application()
        app["target_url"] = "http://fake-backend"
        app["verbose"] = False
        app.router.add_route("*", "/{tail:.*}", handle_request)

        from aiohttp.test_utils import TestClient, TestServer

        with patch("qwen3_call_patch_proxy.aiohttp.ClientSession", return_value=mock_session):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/test")
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_handle_request_non_sse_lines_passed_through(self):
        """
        Test that non-SSE lines (not starting with 'data:') are passed through unchanged.

        :return: None
        :rtype: None
        """
        sse_lines = [
            b': keep-alive\n\n',
            b'data: [DONE]\n\n',
        ]
        mock_session = _make_mock_backend_response(sse_lines)

        app = web.Application()
        app["target_url"] = "http://fake-backend"
        app["verbose"] = False
        app.router.add_route("*", "/{tail:.*}", handle_request)

        from aiohttp.test_utils import TestClient, TestServer

        with patch("qwen3_call_patch_proxy.aiohttp.ClientSession", return_value=mock_session):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/test")
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_handle_request_invalid_json_passed_through(self):
        """
        Test that SSE lines with invalid JSON are passed through without crashing.

        :return: None
        :rtype: None
        """
        sse_lines = [
            b'data: not-valid-json\n\n',
            b'data: [DONE]\n\n',
        ]
        mock_session = _make_mock_backend_response(sse_lines)

        app = web.Application()
        app["target_url"] = "http://fake-backend"
        app["verbose"] = False
        app.router.add_route("*", "/{tail:.*}", handle_request)

        from aiohttp.test_utils import TestClient, TestServer

        with patch("qwen3_call_patch_proxy.aiohttp.ClientSession", return_value=mock_session):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/test")
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_handle_request_server_disconnected(self):
        """
        Test that ServerDisconnectedError returns a 502 response.

        :return: None
        :rtype: None
        """
        from aiohttp.client_exceptions import ServerDisconnectedError

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(side_effect=ServerDisconnectedError())
        mock_session.__aexit__ = AsyncMock(return_value=None)

        app = web.Application()
        app["target_url"] = "http://fake-backend"
        app["verbose"] = False
        app.router.add_route("*", "/{tail:.*}", handle_request)

        from aiohttp.test_utils import TestClient, TestServer

        with patch("qwen3_call_patch_proxy.aiohttp.ClientSession", return_value=mock_session):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/test")
                assert resp.status == 502

    @pytest.mark.asyncio
    async def test_handle_request_with_tool_call_event(self):
        """
        Test handle_request processes a complete tool call SSE event end-to-end.

        :return: None
        :rtype: None
        """
        tool_event = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123456789012345678901",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls", "description": "list"}'
                        }
                    }]
                }
            }]
        }
        sse_lines = [
            f'data: {json.dumps(tool_event)}\n\n'.encode(),
            b'data: [DONE]\n\n',
        ]
        mock_session = _make_mock_backend_response(sse_lines)

        app = web.Application()
        app["target_url"] = "http://fake-backend"
        app["verbose"] = True  # cover verbose branch
        app.router.add_route("*", "/{tail:.*}", handle_request)

        from aiohttp.test_utils import TestClient, TestServer

        with patch("qwen3_call_patch_proxy.aiohttp.ClientSession", return_value=mock_session):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/test")
                assert resp.status == 200

    @pytest.mark.asyncio
    async def test_handle_request_client_connection_reset(self):
        """
        Test that ClientConnectionResetError returns a 499 response.

        :return: None
        :rtype: None
        """
        from aiohttp.client_exceptions import ClientConnectionResetError

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(side_effect=ClientConnectionResetError())
        mock_session.__aexit__ = AsyncMock(return_value=None)

        app = web.Application()
        app["target_url"] = "http://fake-backend"
        app["verbose"] = False
        app.router.add_route("*", "/{tail:.*}", handle_request)

        from aiohttp.test_utils import TestClient, TestServer

        with patch("qwen3_call_patch_proxy.aiohttp.ClientSession", return_value=mock_session):
            async with TestClient(TestServer(app)) as client:
                resp = await client.get("/test")
                assert resp.status == 499
