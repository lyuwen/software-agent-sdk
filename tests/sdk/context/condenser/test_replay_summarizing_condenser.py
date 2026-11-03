from typing import Literal, cast
from unittest.mock import MagicMock

import pytest
from litellm.types.utils import ModelResponse

from openhands.sdk import ReplaySummarizingCondenser as ExportedCondenser
from openhands.sdk.context.condenser.base import (
    CondensationRequirement,
    NoCondensationAvailableException,
)
from openhands.sdk.context.condenser.llm_summarizing_condenser import Reason
from openhands.sdk.context.condenser.replay_summarizing_condenser import (
    CHECKPOINT_PREAMBLE,
    SUMMARY_CLOSE_TAG,
    SUMMARY_OPEN_TAG,
    ReplaySummarizingCondenser,
)
from openhands.sdk.context.view import View
from openhands.sdk.event.base import Event
from openhands.sdk.event.condenser import Condensation, CondensationRequest
from openhands.sdk.event.llm_convertible import MessageEvent, SystemPromptEvent
from openhands.sdk.llm import LLM, LLMResponse, Message, MetricsSnapshot, TextContent


CHARS_PER_TOKEN = 4
CHARS_PER_EVENT = 400
AGENT_WINDOW = 10_000


def message_event(
    content: str, source: Literal["user", "agent"] = "user"
) -> MessageEvent:
    role: Literal["user", "assistant"] = "user" if source == "user" else "assistant"
    return MessageEvent(
        llm_message=Message(role=role, content=[TextContent(text=content)]),
        source=source,
    )


def char_token_count(messages: list[Message]) -> int:
    """Deterministic token counter: four text characters per token."""
    total_chars = 0
    for message in messages:
        for block in message.content:
            if isinstance(block, TextContent):
                total_chars += len(block.text)
    return total_chars // CHARS_PER_TOKEN


def make_agent_llm(window: int | None = AGENT_WINDOW) -> LLM:
    agent_llm = MagicMock(spec=LLM)
    agent_llm.max_input_tokens = window
    agent_llm.get_token_count.side_effect = char_token_count
    return agent_llm


def make_response(content: str) -> LLMResponse:
    message = Message(role="assistant", content=[TextContent(text=content)])
    metrics = MetricsSnapshot(
        model_name="test-model",
        accumulated_cost=0.0,
        max_budget_per_task=None,
        accumulated_token_usage=None,
    )
    raw_response = MagicMock(spec=ModelResponse)
    raw_response.id = "mock-response-id"
    return LLMResponse(message=message, metrics=metrics, raw_response=raw_response)


def make_condenser_llm(*responses: str) -> LLM:
    """Mock LLM carrying the attributes LLM's own validation requires."""
    condenser_llm = MagicMock(spec=LLM)
    cast(MagicMock, condenser_llm.completion).side_effect = [
        make_response(r) for r in responses
    ]
    condenser_llm.openrouter_site_url = "https://docs.all-hands.dev/"
    condenser_llm.openrouter_app_name = "OpenHands"
    condenser_llm.aws_access_key_id = None
    condenser_llm.aws_secret_access_key = None
    condenser_llm.aws_region_name = None
    condenser_llm.metrics = None
    condenser_llm.model = "test-model"
    condenser_llm.log_completions = False
    condenser_llm.log_completions_folder = None
    condenser_llm.custom_tokenizer = None
    condenser_llm.base_url = None
    condenser_llm.reasoning_effort = None
    condenser_llm.litellm_extra_body = {}
    condenser_llm.temperature = 0.0
    condenser_llm.input_cost_per_token = None
    condenser_llm.output_cost_per_token = None
    condenser_llm._metrics = None
    return condenser_llm


def completion_mock(condenser_llm: LLM) -> MagicMock:
    return cast(MagicMock, condenser_llm.completion)


def replayed_messages(condenser_llm: LLM, call: int = -1) -> list[Message]:
    messages: list[Message] = (
        completion_mock(condenser_llm).call_args_list[call].kwargs["messages"]
    )
    return messages


def message_text(message: Message) -> str:
    assert len(message.content) == 1
    block = message.content[0]
    assert isinstance(block, TextContent)
    return block.text


def joined_text(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextContent):
                parts.append(block.text)
    return "\n".join(parts)


def big_view(event_count: int = 100) -> View:
    return View.from_events(
        [message_event("A" * CHARS_PER_EVENT) for _ in range(event_count)]
    )


def test_defaults() -> None:
    condenser = ReplaySummarizingCondenser(llm=make_condenser_llm("summary"))

    assert condenser.threshold_ratio == 0.8
    assert condenser.max_tokens is None
    assert condenser.retain_ratio == 0.16
    assert condenser.retain_tokens is None
    assert condenser.keep_first == 2
    assert condenser.max_event_chars is None
    assert condenser.compaction_retries == 1
    assert condenser.checkpoint_framing is True
    assert condenser.handles_condensation_requests() is True


def test_retention_config_is_validated() -> None:
    condenser_llm = make_condenser_llm("summary")

    with pytest.raises(ValueError):
        ReplaySummarizingCondenser(
            llm=condenser_llm, retain_ratio=0.2, retain_tokens=10
        )
    with pytest.raises(ValueError):
        ReplaySummarizingCondenser(llm=condenser_llm, threshold_ratio=0.1)
    with pytest.raises(ValueError):
        ReplaySummarizingCondenser(llm=condenser_llm, keep_first=0)


def test_token_pressure_drives_soft_requirement() -> None:
    condenser = ReplaySummarizingCondenser(llm=make_condenser_llm("summary"))
    agent_llm = make_agent_llm()

    small_view = View.from_events(
        [message_event("A" * CHARS_PER_EVENT) for _ in range(10)]
    )
    assert condenser.condensation_requirement(small_view, agent_llm) is None
    assert condenser.get_condensation_reasons(small_view, agent_llm) == set()

    view = big_view()
    assert Reason.TOKENS in condenser.get_condensation_reasons(view, agent_llm)
    assert (
        condenser.condensation_requirement(view, agent_llm)
        == CondensationRequirement.SOFT
    )


def test_absolute_token_cap_works_without_model_capacity() -> None:
    condenser = ReplaySummarizingCondenser(
        llm=make_condenser_llm("summary"), max_tokens=1_000
    )
    agent_llm = make_agent_llm(window=None)

    # No context window is known, so only the absolute cap can trigger pressure.
    assert condenser.condensation_requirement(View.from_events([]), agent_llm) is None
    view = big_view(event_count=20)  # ~2,000 tokens
    assert (
        condenser.condensation_requirement(view, agent_llm)
        == CondensationRequirement.SOFT
    )


def test_condensation_request_is_hard_requirement() -> None:
    condenser = ReplaySummarizingCondenser(llm=make_condenser_llm("summary"))
    agent_llm = make_agent_llm()
    view = View.from_events(
        [message_event("A" * CHARS_PER_EVENT) for _ in range(10)]
        + [CondensationRequest()]
    )

    assert view.unhandled_condensation_request is True
    assert (
        condenser.condensation_requirement(view, agent_llm)
        == CondensationRequirement.HARD
    )


def test_request_forces_condensation_below_pressure() -> None:
    condenser_llm = make_condenser_llm("summary")
    condenser = ReplaySummarizingCondenser(llm=condenser_llm)
    agent_llm = make_agent_llm()

    # Only ~1,000 tokens, far below the 8,000-token pressure threshold.
    events = [message_event("A" * CHARS_PER_EVENT) for _ in range(10)]
    events_with_request: list[Event] = [*events, CondensationRequest()]
    result = condenser.condense(
        View.from_events(events_with_request), agent_llm=agent_llm
    )

    assert isinstance(result, Condensation)
    assert result.summary_offset == condenser.keep_first
    # An explicit request frees at least half of the summarizable history.
    assert len(result.forgotten_event_ids) >= len(events[condenser.keep_first :]) // 2


def test_condensation_pins_prefix_and_keeps_recent_tail() -> None:
    condenser_llm = make_condenser_llm("Task is X")
    condenser = ReplaySummarizingCondenser(llm=condenser_llm)
    agent_llm = make_agent_llm()

    events = [message_event("A" * CHARS_PER_EVENT) for _ in range(100)]
    result = condenser.condense(View.from_events(events), agent_llm=agent_llm)

    assert isinstance(result, Condensation)
    assert result.summary_offset == condenser.keep_first
    forgotten = set(result.forgotten_event_ids)
    assert events[0].id not in forgotten
    assert events[1].id not in forgotten
    assert events[-1].id not in forgotten
    # The forgotten span is the head of the summarizable history, not a middle gap.
    assert {event.id for event in events[2:50]} <= forgotten
    assert result.llm_response_id == "mock-response-id"
    assert result.summary is not None
    assert result.summary.startswith(CHECKPOINT_PREAMBLE)
    assert SUMMARY_OPEN_TAG in result.summary
    assert SUMMARY_CLOSE_TAG in result.summary
    assert "Task is X" in result.summary


def test_summarization_replays_real_messages_and_pinned_prefix() -> None:
    condenser_llm = make_condenser_llm("summary")
    condenser = ReplaySummarizingCondenser(llm=condenser_llm)
    agent_llm = make_agent_llm()

    long_tool_output = "TOOL OUTPUT: " + "x" * 800
    events: list[MessageEvent | SystemPromptEvent] = [
        SystemPromptEvent(
            source="agent", system_prompt=TextContent(text="SYSTEM"), tools=[]
        ),
        message_event("TASK: fix the bug"),
    ]
    events += [message_event("A" * CHARS_PER_EVENT) for _ in range(100)]
    events[5] = message_event(long_tool_output, source="agent")

    result = condenser.condense(View.from_events(events), agent_llm=agent_llm)

    assert isinstance(result, Condensation)
    messages = replayed_messages(condenser_llm)
    assert messages[0].role == "system"
    assert message_text(messages[1]) == "TASK: fix the bug"
    assert messages[-1].role == "user"
    assert "compaction engine" in message_text(messages[-1])
    # Real message content is replayed, not str(event) previews truncated at 500.
    replayed_text = joined_text(messages)
    assert "SYSTEM" in replayed_text
    assert long_tool_output in replayed_text


def test_condensation_records_summarizer_input_and_output() -> None:
    condenser_llm = make_condenser_llm("summary text")
    condenser = ReplaySummarizingCondenser(llm=condenser_llm)
    agent_llm = make_agent_llm()

    events = [message_event("A" * CHARS_PER_EVENT) for _ in range(100)]
    result = condenser.condense(View.from_events(events), agent_llm=agent_llm)

    assert isinstance(result, Condensation)
    # The exact summarizer prompt and response are recoverable from the event.
    assert result.request_messages == replayed_messages(condenser_llm)
    assert result.response_message is not None
    assert result.response_message.role == "assistant"
    assert message_text(result.response_message) == "summary text"


def test_previous_checkpoint_is_merged_into_next_summary() -> None:
    condenser_llm = make_condenser_llm("FIRST", "SECOND")
    condenser = ReplaySummarizingCondenser(llm=condenser_llm)
    agent_llm = make_agent_llm()

    events = [message_event("A" * CHARS_PER_EVENT) for _ in range(100)]
    first = condenser.condense(View.from_events(events), agent_llm=agent_llm)
    assert isinstance(first, Condensation)

    later_events: list[Event] = [*events, first]
    later_events += [message_event("B" * CHARS_PER_EVENT) for _ in range(100)]
    second = condenser.condense(View.from_events(later_events), agent_llm=agent_llm)

    assert isinstance(second, Condensation)
    assert second.summary_offset == condenser.keep_first
    replayed_text = joined_text(replayed_messages(condenser_llm))
    # The previous checkpoint is replayed verbatim, so the summarizer can merge it
    # forward instead of carrying a separate previous-summary section.
    assert "FIRST" in replayed_text


def test_non_shrinking_summary_is_rejected() -> None:
    condenser = ReplaySummarizingCondenser(
        llm=make_condenser_llm("S" * 40_000), compaction_retries=0
    )

    with pytest.raises(NoCondensationAvailableException):
        condenser.get_condensation(big_view(), agent_llm=make_agent_llm())


def test_soft_failure_returns_uncondensed_view() -> None:
    condenser = ReplaySummarizingCondenser(
        llm=make_condenser_llm("S" * 40_000), compaction_retries=0
    )
    view = big_view()

    result = condenser.condense(view, agent_llm=make_agent_llm())

    assert result is view


def test_empty_summary_is_rejected() -> None:
    condenser = ReplaySummarizingCondenser(llm=make_condenser_llm("   "))

    with pytest.raises(NoCondensationAvailableException):
        condenser.get_condensation(big_view(), agent_llm=make_agent_llm())


def test_retry_forgets_a_larger_span_when_summary_does_not_shrink() -> None:
    condenser_llm = make_condenser_llm("S" * 40_000, "concise summary")
    condenser = ReplaySummarizingCondenser(llm=condenser_llm, compaction_retries=1)

    result = condenser.condense(big_view(), agent_llm=make_agent_llm())

    assert isinstance(result, Condensation)
    assert completion_mock(condenser_llm).call_count == 2
    first_messages = replayed_messages(condenser_llm, call=0)
    second_messages = replayed_messages(condenser_llm, call=1)
    assert len(second_messages) > len(first_messages)
    assert result.summary is not None
    assert "concise summary" in result.summary


def test_max_event_chars_only_limits_the_summarizer_input() -> None:
    condenser_llm = make_condenser_llm("summary")
    condenser = ReplaySummarizingCondenser(llm=condenser_llm, max_event_chars=50)

    result = condenser.condense(big_view(), agent_llm=make_agent_llm())

    assert isinstance(result, Condensation)
    limit = 50 + len("\n...[truncated for summarization]")
    for message in replayed_messages(condenser_llm)[:-1]:
        for block in message.content:
            assert isinstance(block, TextContent)
            assert len(block.text) <= limit
    # Only the summarizer input is truncated; the committed condensation still
    # forgets whole original events.
    assert result.forgotten_event_ids


def test_checkpoint_framing_can_be_disabled() -> None:
    condenser = ReplaySummarizingCondenser(
        llm=make_condenser_llm("raw summary"), checkpoint_framing=False
    )

    result = condenser.condense(big_view(), agent_llm=make_agent_llm())

    assert isinstance(result, Condensation)
    assert result.summary == "raw summary"


def test_condenser_is_exported_from_sdk_root() -> None:
    assert ExportedCondenser is ReplaySummarizingCondenser
