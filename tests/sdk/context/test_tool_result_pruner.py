"""Tests for deterministic tool-result pruning.

Covers the configuration contract, the content transform, the message-level
application, and the agent-level hook in ``prepare_llm_messages``.
"""

import pytest

from openhands.sdk.agent.utils import prepare_llm_messages
from openhands.sdk.context.tool_result_pruner import (
    DEFAULT_HEAD_CHARS,
    DEFAULT_TAIL_CHARS,
    DEFAULT_THRESHOLD_CHARS,
    ENV_ENABLED,
    ENV_HEAD_CHARS,
    ENV_TAIL_CHARS,
    ENV_THRESHOLD_CHARS,
    PRUNE_MARKER,
    ToolResultPruneConfig,
    code_point_length,
    config_from_env,
    is_enabled,
    maybe_prune_tool_result_messages,
    measure_content,
    prune_content,
    prune_tool_result_messages,
    resolve_config,
)
from openhands.sdk.event.llm_convertible import (
    ActionEvent,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.llm import (
    ImageContent,
    Message,
    MessageToolCall,
    TextContent,
)
from openhands.sdk.mcp.definition import MCPToolAction, MCPToolObservation


SMALL = ToolResultPruneConfig(threshold_chars=50, head_chars=4, tail_chars=3)
MARKER_LEN = code_point_length(PRUNE_MARKER)


@pytest.fixture(autouse=True)
def _clear_pruner_env(monkeypatch: pytest.MonkeyPatch):
    """Keep the ambient environment from enabling the pruner in tests."""
    for name in (ENV_ENABLED, ENV_THRESHOLD_CHARS, ENV_HEAD_CHARS, ENV_TAIL_CHARS):
        monkeypatch.delenv(name, raising=False)


def _image() -> ImageContent:
    return ImageContent(image_urls=["data:image/png;base64,AAAA"])


def _tool_message(text: str) -> Message:
    return Message(
        role="tool",
        content=[TextContent(text=text)],
        name="bash",
        tool_call_id="call_1",
    )


def _message_event(content: str) -> MessageEvent:
    return MessageEvent(
        llm_message=Message(role="user", content=[TextContent(text=content)]),
        source="user",
    )


def _action_event(tool_call_id: str = "call_1") -> ActionEvent:
    return ActionEvent(
        thought=[TextContent(text="Test thought")],
        action=MCPToolAction(data={}),
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        tool_call=MessageToolCall(
            id=tool_call_id, name="test_tool", arguments="{}", origin="completion"
        ),
        llm_response_id="resp_1",
        source="agent",
    )


def _observation_event(
    tool_call_id: str = "call_1", text: str = "ok"
) -> ObservationEvent:
    return ObservationEvent(
        observation=MCPToolObservation.from_text(text=text, tool_name="test_tool"),
        tool_name="test_tool",
        tool_call_id=tool_call_id,
        action_id="action_event_id",
        source="environment",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_resolve_config_defaults():
    resolved = resolve_config()
    assert resolved == ToolResultPruneConfig(
        threshold_chars=DEFAULT_THRESHOLD_CHARS,
        head_chars=DEFAULT_HEAD_CHARS,
        tail_chars=DEFAULT_TAIL_CHARS,
    )
    assert resolved == resolve_config(ToolResultPruneConfig())


@pytest.mark.parametrize(
    "config, message",
    [
        (ToolResultPruneConfig(threshold_chars=0), "threshold_chars"),
        (ToolResultPruneConfig(head_chars=-1), "head_chars"),
        (ToolResultPruneConfig(tail_chars=-1), "tail_chars"),
        (
            ToolResultPruneConfig(threshold_chars=50, head_chars=20, tail_chars=20),
            "must be at most threshold_chars",
        ),
        (
            ToolResultPruneConfig(threshold_chars=1.5),  # type: ignore[arg-type]
            "threshold_chars",
        ),
    ],
)
def test_resolve_config_rejects_invalid_budgets(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_config(config)


def test_code_point_length_counts_code_points():
    assert code_point_length("a😀b") == 3
    assert code_point_length(PRUNE_MARKER) == MARKER_LEN


# ---------------------------------------------------------------------------
# Content transform
# ---------------------------------------------------------------------------


def test_measure_content_counts_only_text_blocks():
    content = [TextContent(text="a😀b"), _image()]
    assert measure_content(content) == 3


def test_prune_content_skips_content_within_threshold():
    content = [TextContent(text="a😀b"), _image()]
    assert prune_content(content, SMALL) is None


def test_prune_content_keeps_head_and_tail_without_splitting_code_points():
    content: list[TextContent | ImageContent] = [TextContent(text="😀" * 60)]
    result = prune_content(content, SMALL)

    assert result is not None
    assert result == [
        TextContent(text=f"{'😀' * 4}{PRUNE_MARKER}{'😀' * 3}"),
    ]
    assert measure_content(result) <= SMALL.threshold_chars
    assert isinstance(result[0], TextContent)
    assert "\ufffd" not in result[0].text


def test_prune_content_preserves_non_text_blocks_in_order():
    first_image, second_image = _image(), _image()
    content: list[TextContent | ImageContent] = [
        TextContent(text="A" * 40),
        first_image,
        TextContent(text="B" * 30),
        second_image,
        TextContent(text="C" * 30),
    ]

    result = prune_content(content, SMALL)

    # The middle text block collapses entirely; rich blocks keep their order.
    assert result is not None
    assert result == [
        TextContent(text=f"AAAA{PRUNE_MARKER}"),
        first_image,
        second_image,
        TextContent(text="CCC"),
    ]
    assert measure_content(result) <= SMALL.threshold_chars


def test_prune_content_supports_zero_head_and_tail():
    config = ToolResultPruneConfig(
        threshold_chars=MARKER_LEN, head_chars=0, tail_chars=0
    )
    result = prune_content([TextContent(text="x" * 100)], config)

    assert result is not None
    assert result == [TextContent(text=PRUNE_MARKER)]
    assert measure_content(result) == config.threshold_chars


def test_prune_content_is_idempotent():
    content = [TextContent(text="x" * 100)]
    once = prune_content(content, SMALL)

    assert once is not None
    assert prune_content(once, SMALL) is None


# ---------------------------------------------------------------------------
# Message transform
# ---------------------------------------------------------------------------


def test_prune_tool_result_messages_touches_only_over_budget_tool_messages():
    user = Message(role="user", content=[TextContent(text="hi")])
    short_tool = _tool_message("short")
    long_tool = _tool_message("x" * 100)
    assistant = Message(role="assistant", content=[TextContent(text="thinking")])

    result = prune_tool_result_messages([user, short_tool, long_tool, assistant], SMALL)

    assert result[0] is user
    assert result[1] is short_tool
    assert result[2] is not long_tool
    assert result[3] is assistant
    assert measure_content(result[2].content) <= SMALL.threshold_chars
    # The original message is not mutated.
    assert long_tool.content == [TextContent(text="x" * 100)]


def test_maybe_prune_is_noop_when_disabled():
    messages = [_tool_message("x" * 100)]
    assert is_enabled() is False
    assert maybe_prune_tool_result_messages(messages) is messages


def test_maybe_prune_uses_env_budgets(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_ENABLED, "true")
    monkeypatch.setenv(ENV_THRESHOLD_CHARS, str(MARKER_LEN + 4))
    monkeypatch.setenv(ENV_HEAD_CHARS, "2")
    monkeypatch.setenv(ENV_TAIL_CHARS, "2")

    assert is_enabled() is True
    assert config_from_env() == ToolResultPruneConfig(
        threshold_chars=MARKER_LEN + 4, head_chars=2, tail_chars=2
    )

    result = maybe_prune_tool_result_messages([_tool_message("x" * 100)])
    assert result[0].content == [TextContent(text=f"xx{PRUNE_MARKER}xx")]


def test_invalid_env_integer_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_THRESHOLD_CHARS, "not-a-number")

    with pytest.raises(ValueError, match=ENV_THRESHOLD_CHARS):
        maybe_prune_tool_result_messages([_tool_message("x" * 100)])


def test_invalid_env_budget_combination_fails_loudly(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_THRESHOLD_CHARS, "10")
    monkeypatch.setenv(ENV_HEAD_CHARS, "20")
    monkeypatch.setenv(ENV_TAIL_CHARS, "20")

    with pytest.raises(ValueError, match="head_chars \\+ marker \\+ tail_chars"):
        maybe_prune_tool_result_messages([_tool_message("x" * 100)])


def test_invalid_env_is_ignored_while_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_THRESHOLD_CHARS, "not-a-number")
    messages = [_tool_message("x" * 100)]
    assert maybe_prune_tool_result_messages(messages) is messages


# ---------------------------------------------------------------------------
# Agent hook
# ---------------------------------------------------------------------------


def test_prepare_llm_messages_prunes_tool_results_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_ENABLED, "1")
    oversized = "x" * (DEFAULT_THRESHOLD_CHARS + 100)
    observation = _observation_event(text=oversized)
    events = [
        _message_event("run the tool"),
        _action_event(),
        observation,
    ]

    messages = prepare_llm_messages(events, condenser=None)

    tool_messages = [message for message in messages if message.role == "tool"]
    assert len(tool_messages) == 1
    assert measure_content(tool_messages[0].content) <= DEFAULT_THRESHOLD_CHARS
    assert tool_messages[0].content == [
        TextContent(
            text=f"{'x' * DEFAULT_HEAD_CHARS}{PRUNE_MARKER}{'x' * DEFAULT_TAIL_CHARS}"
        )
    ]
    # The event history keeps the full-fidelity observation.
    assert measure_content(observation.observation.to_llm_content) == len(oversized)


def test_prepare_llm_messages_leaves_tool_results_alone_when_disabled():
    oversized = "x" * (DEFAULT_THRESHOLD_CHARS + 100)
    observation = _observation_event(text=oversized)
    events = [_message_event("run the tool"), _action_event(), observation]

    messages = prepare_llm_messages(events, condenser=None)

    tool_messages = [message for message in messages if message.role == "tool"]
    assert tool_messages[0].content == observation.observation.to_llm_content
