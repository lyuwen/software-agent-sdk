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
    # A non-editor/terminal tool with arbitrary args must not be flagged.
    mock_completion.side_effect = [
        _tool_call_response("finish", '{"message": "done"}', "fin-1"),
    ]

    resp = base_llm.completion(
        messages=[Message(role="user", content=[TextContent(text="hi")])]
    )

    assert isinstance(resp, LLMResponse)
    assert mock_completion.call_count == 1
