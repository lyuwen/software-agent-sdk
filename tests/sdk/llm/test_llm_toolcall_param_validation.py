"""Tests for the static tool-call parameter validation guardrail in LLM.

The guardrail is gated behind the ``OH_VALIDATE_TOOLCALL_PARAMS`` env var. When
enabled, a response whose ``file_editor``/``terminal`` tool call carries invalid
parameters raises ``LLMNoResponseError`` inside the retry boundary, prompting a
re-issue of the inference request (same behavior as the malformed-pattern
check).
"""

from unittest.mock import patch

import pytest
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message as LiteLLMMessage,
    ModelResponse,
    Usage,
)
from pydantic import SecretStr

from openhands.sdk.llm import LLM, LLMResponse, Message, TextContent
from openhands.sdk.llm.exceptions import LLMNoResponseError


def _tool_call_response(
    name: str, arguments: str, response_id: str = "tc-1"
) -> ModelResponse:
    return ModelResponse(
        id=response_id,
        choices=[
            Choices(
                finish_reason="tool_calls",
                index=0,
                message=LiteLLMMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(name=name, arguments=arguments),
                        )
                    ],
                ),
            )
        ],
        created=1,
        model="gpt-4o",
        object="chat.completion",
        system_fingerprint="t",
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


# A str_replace with new_str == old_str is statically invalid.
_INVALID_ARGS = (
    '{"command": "str_replace", "path": "/a/b.py", '
    '"old_str": "x = 1", "new_str": "x = 1"}'
)
_VALID_ARGS = (
    '{"command": "str_replace", "path": "/a/b.py", '
    '"old_str": "x = 1", "new_str": "x = 2"}'
)


@pytest.fixture
def base_llm() -> LLM:
    return LLM(
        usage_id="test-llm",
        model="gpt-4o",
        api_key=SecretStr("test_key"),
        num_retries=2,
        retry_min_wait=1,
        retry_max_wait=2,
        temperature=0.0,
    )


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_invalid_params_retries_then_succeeds(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("str_replace_editor", _INVALID_ARGS, "bad-1"),
        _tool_call_response("str_replace_editor", _VALID_ARGS, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 2  # initial invalid + 1 retry


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_invalid_params_exhausts_retries(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("str_replace_editor", _INVALID_ARGS, "bad-1"),
        _tool_call_response("str_replace_editor", _INVALID_ARGS, "bad-2"),
    ]

    with pytest.raises(LLMNoResponseError):
        base_llm.completion(
            messages=[Message(role="user", content=[TextContent(text="hi")])]
        )

    assert mock_completion.call_count == base_llm.num_retries


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_disabled_by_default_lets_invalid_params_through(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    # Env var unset -> guardrail is a no-op, invalid params are NOT retried.
    monkeypatch.delenv("OH_VALIDATE_TOOLCALL_PARAMS", raising=False)
    mock_completion.side_effect = [
        _tool_call_response("str_replace_editor", _INVALID_ARGS, "bad-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 1  # no retry


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_valid_params_pass_when_enabled(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("str_replace_editor", _VALID_ARGS, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 1


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_unknown_tool_is_skipped(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    # A tool with no dedicated checker and arbitrary (but well-formed JSON) args
    # must not be flagged. `browser` has no checker in the dispatch table.
    mock_completion.side_effect = [
        _tool_call_response("browser", '{"anything": "goes"}', "b-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 1


# ---------------------------------------------------------------------------
# Generic JSON gate: applies to *every* tool call, including tools without a
# dedicated schema checker. Guards against the provider failure mode where raw
# backslash escapes (e.g. ``\Box``) make ``function.arguments`` un-parseable.
# ---------------------------------------------------------------------------

# Genuinely invalid JSON: ``\B`` is not a legal JSON escape sequence. This is
# the shape that produced ``Invalid \escape`` at agent-side json.loads time.
_INVALID_ESCAPE_ARGS = r'{"query": "see \Box here"}'


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_unknown_tool_with_bad_json_is_retried(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    # Even for a tool with no dedicated checker (`browser`), un-parseable
    # arguments must be caught by the generic gate and retried.
    mock_completion.side_effect = [
        _tool_call_response("browser", _INVALID_ESCAPE_ARGS, "bad-1"),
        _tool_call_response("browser", '{"query": "done"}', "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 2  # invalid JSON + 1 retry


# ---------------------------------------------------------------------------
# task_tracker schema checker.
# ---------------------------------------------------------------------------

# The real failing payload: a raw ``\boxed`` / ``\Box`` inside a notes field
# makes the whole arguments string invalid JSON.
_TASK_TRACKER_BAD_ESCAPE = (
    r'{"command": "plan", "task_list": '
    r'[{"title": "Find \boxed and \Box", "status": "todo", "notes": "x"}]}'
)
_TASK_TRACKER_VALID = (
    '{"command": "plan", "task_list": '
    '[{"title": "explore", "status": "in_progress", "notes": "look around"}]}'
)
# `plan` with no task_list is schema-invalid (task_list is required for plan).
_TASK_TRACKER_PLAN_NO_LIST = '{"command": "plan"}'


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_task_tracker_bad_escape_is_retried(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("task_tracker", _TASK_TRACKER_BAD_ESCAPE, "bad-1"),
        _tool_call_response("task_tracker", _TASK_TRACKER_VALID, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 2


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_task_tracker_plan_without_list_is_retried(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("task_tracker", _TASK_TRACKER_PLAN_NO_LIST, "bad-1"),
        _tool_call_response("task_tracker", _TASK_TRACKER_VALID, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 2


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_task_tracker_valid_plan_passes(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("task_tracker", _TASK_TRACKER_VALID, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 1


# ---------------------------------------------------------------------------
# Builtin finish / think schema checkers.
# ---------------------------------------------------------------------------

# The real failing payload: `finish` called with only the tolerated `summary`
# metadata key and no required `message` field.
_FINISH_MISSING_MESSAGE = (
    '{"summary": "All issues addressed: 19 new tests pass"}'
)
_FINISH_VALID = '{"message": "Done: added methods, fixed behavior, tests pass."}'
_THINK_MISSING_THOUGHT = '{"summary": "reasoning about the fix"}'
_THINK_VALID = '{"thought": "I should check the failing tests first."}'


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_finish_missing_message_is_retried(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("finish", _FINISH_MISSING_MESSAGE, "bad-1"),
        _tool_call_response("finish", _FINISH_VALID, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 2


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_finish_valid_passes(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("finish", _FINISH_VALID, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 1


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_think_missing_thought_is_retried(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("think", _THINK_MISSING_THOUGHT, "bad-1"),
        _tool_call_response("think", _THINK_VALID, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 2


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_think_valid_passes(
    mock_completion, base_llm: LLM, monkeypatch
) -> None:
    monkeypatch.setenv("OH_VALIDATE_TOOLCALL_PARAMS", "1")
    mock_completion.side_effect = [
        _tool_call_response("think", _THINK_VALID, "good-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 1

