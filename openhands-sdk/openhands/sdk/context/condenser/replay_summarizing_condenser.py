"""Token-budgeted condenser that summarizes by replaying the real conversation.

`LLMSummarizingCondenser` renders every forgotten event with `str(event)` and pastes
the previews into a single templated prompt. That is cheap, but lossy: tool-call
arguments never reach the summarizer and every event is cut at the display preview
length.

This backend instead rebuilds the actual `Message` objects for the forgotten span,
prefixed by the events that stay pinned at the front of the conversation (system
prompt, and whatever the caller pins with `keep_first`), and appends one
summarization directive as the final user message. The resulting request is a
genuine prefix of the conversation's own request, so providers that cache prefixes
only have to process the directive and the summary output.

Pressure is measured in tokens against the agent LLM's context window rather than in
event count, the retained tail is priced in tokens, and a summary that would not
shrink the span it replaces is rejected instead of committed.
"""

import logging
import os

from pydantic import Field, model_validator

from openhands.sdk.context.condenser.base import (
    CondensationRequirement,
    NoCondensationAvailableException,
    RollingCondenser,
)
from openhands.sdk.context.condenser.llm_summarizing_condenser import Reason
from openhands.sdk.context.condenser.utils import (
    get_suffix_length_for_token_reduction,
    get_total_token_count,
)
from openhands.sdk.context.prompts import render_template
from openhands.sdk.context.view import View
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.event.condenser import Condensation, CondensationSummaryEvent
from openhands.sdk.llm import LLM, ImageContent, Message, TextContent
from openhands.sdk.observability.laminar import observe


logger = logging.getLogger(__name__)

CHECKPOINT_PREAMBLE = (
    "This is an automatically generated checkpoint condensing an earlier span of the "
    "conversation to free up context. Treat the captured context as established "
    "background and build on it without restating it. Continue the task directly from "
    "the messages that follow, without acknowledging this checkpoint."
)

SUMMARY_OPEN_TAG = "<compacted-summary>"
SUMMARY_CLOSE_TAG = "</compacted-summary>"

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_FILE = "replay_summarizing_prompt.j2"


class ReplaySummarizingCondenser(RollingCondenser):
    """Condenser that summarizes a token-budgeted span of the real messages.

    The condenser measures pressure from the agent LLM's context window
    (`threshold_ratio`), retains the newest `retain_ratio`/`retain_tokens` of the
    conversation verbatim, and replaces everything between the pinned front events
    and the retained tail with one LLM-written summary.

    The summarization call replays the conversation's own messages for that span, so
    tool calls, tool results and long user instructions reach the summarizer intact
    instead of as truncated display strings.
    """

    llm: LLM
    """LLM used to write the summary. It may be a different route than the agent's."""

    threshold_ratio: float = Field(default=0.8, gt=0.0, le=1.0)
    """Fragment of the agent LLM's input-token budget that triggers condensation."""

    max_tokens: int | None = Field(default=None, gt=0)
    """Absolute pressure cap, used alone or as a stricter bound on the ratio."""

    retain_ratio: float | None = Field(default=0.16, gt=0.0, lt=1.0)
    """Fragment of the context window kept verbatim in the recent tail."""

    retain_tokens: int | None = Field(default=None, gt=0)
    """Absolute recent-tail budget. Mutually exclusive with `retain_ratio`."""

    keep_first: int = Field(default=2, ge=1)
    """Number of leading events (system prompt first) that are never condensed."""

    max_event_chars: int | None = Field(default=None, gt=0)
    """Optional per-message character cap applied only to the summarization call."""

    compaction_retries: int = Field(default=1, ge=0)
    """Extra summarization attempts with a larger forgotten span when a summary
    would not shrink its source."""

    checkpoint_framing: bool = True
    """Whether to wrap the summary in the checkpoint preamble and tags."""

    @model_validator(mode="after")
    def _validate_config(self) -> "ReplaySummarizingCondenser":
        if self.retain_ratio is not None and self.retain_tokens is not None:
            raise ValueError("retain_ratio and retain_tokens are mutually exclusive")
        if self.retain_ratio is not None and self.retain_ratio >= self.threshold_ratio:
            raise ValueError("retain_ratio must be less than threshold_ratio")
        return self

    def handles_condensation_requests(self) -> bool:
        return True

    def _threshold_tokens(self, agent_llm: LLM) -> int | None:
        """Return the token count that triggers pressure condensation, if known."""
        limits: list[int] = []
        if agent_llm.max_input_tokens:
            limits.append(int(agent_llm.max_input_tokens * self.threshold_ratio))
        if self.max_tokens is not None:
            limits.append(self.max_tokens)
        return min(limits) if limits else None

    def _retain_budget(self, agent_llm: LLM) -> int:
        """Return how many tokens of recent history must stay verbatim."""
        if self.retain_tokens is not None:
            return self.retain_tokens
        ratio = self.retain_ratio or 0.0
        if agent_llm.max_input_tokens:
            return int(agent_llm.max_input_tokens * ratio)
        if self.max_tokens is not None:
            return int(self.max_tokens * ratio)
        return 0

    def get_condensation_reasons(
        self, view: View, agent_llm: LLM | None = None
    ) -> set[Reason]:
        """Return why, if at all, this view should be condensed."""
        reasons: set[Reason] = set()

        if view.unhandled_condensation_request:
            reasons.add(Reason.REQUEST)

        if agent_llm is not None:
            threshold = self._threshold_tokens(agent_llm)
            if threshold is not None:
                total_tokens = get_total_token_count(view.events, agent_llm)
                if total_tokens > threshold:
                    reasons.add(Reason.TOKENS)

        return reasons

    def condensation_requirement(
        self, view: View, agent_llm: LLM | None = None
    ) -> CondensationRequirement | None:
        reasons = self.get_condensation_reasons(view, agent_llm)
        if not reasons:
            return None
        # An explicit request means the conversation cannot continue as-is.
        if Reason.REQUEST in reasons:
            return CondensationRequirement.HARD
        # Token pressure is a soft bound: skip when no balanced span is available
        # and try again on a later step.
        return CondensationRequirement.SOFT

    def _select_region(
        self,
        view: View,
        agent_llm: LLM,
        retain_scale: float = 1.0,
        for_request: bool = False,
    ) -> tuple[int, int]:
        """Select the head-anchored span to forget, respecting retained tail budget.

        Returns the slice `[start, end)` of `view.events` that will be replaced by
        the summary. `start` is a manipulation-index boundary at or after
        `keep_first`; `end` is the first boundary at or after the point where the
        retained tail begins. An explicit condensation request frees at least half
        of the summarizable history even when token pressure is low, because the
        caller needs room for its next request right away.
        """
        total_tokens = get_total_token_count(view.events, agent_llm)
        retained_budget = int(self._retain_budget(agent_llm) * retain_scale)
        token_reduction = max(total_tokens - retained_budget, 0)
        if for_request:
            pinned_tokens = get_total_token_count(
                view.events[: self.keep_first], agent_llm
            )
            token_reduction = max(token_reduction, (total_tokens - pinned_tokens) // 2)
        suffix_length = get_suffix_length_for_token_reduction(
            events=view.events[self.keep_first :],
            llm=agent_llm,
            token_reduction=token_reduction,
        )

        start = view.find_next_manipulation_index(self.keep_first, strict=False)
        end = view.find_next_manipulation_index(len(view) - suffix_length, strict=False)
        if end <= start:
            raise NoCondensationAvailableException(
                "Cannot condense 0 events: no balanced range fits the retained budget."
            )
        return start, end

    def _build_summarization_messages(
        self, view: View, start: int, end: int
    ) -> list[Message]:
        """Replay the pinned prefix plus the forgotten span, then the directive."""
        prefix = list(view.events[:start])
        forgotten = list(view.events[start:end])
        messages = LLMConvertibleEvent.events_to_messages(prefix + forgotten)
        if self.max_event_chars is not None:
            messages = [self._truncate_message(message) for message in messages]
        prompt = render_template(PROMPT_DIR, PROMPT_FILE)
        messages.append(Message(role="user", content=[TextContent(text=prompt)]))
        return messages

    def _truncate_message(self, message: Message) -> Message:
        """Cap text content of one replayed message at `max_event_chars`."""
        assert self.max_event_chars is not None
        content: list[TextContent | ImageContent] = []
        truncated = False
        for block in message.content:
            if (
                isinstance(block, TextContent)
                and len(block.text) > self.max_event_chars
            ):
                content.append(
                    TextContent(
                        text=block.text[: self.max_event_chars]
                        + "\n...[truncated for summarization]"
                    )
                )
                truncated = True
            else:
                content.append(block)
        if not truncated:
            return message
        return message.model_copy(update={"content": content})

    def _summarize(
        self, view: View, start: int, end: int
    ) -> tuple[str, str, list[Message], Message]:
        """Ask the summarizer LLM for a checkpoint.

        Returns:
            A tuple of the summary text, the response id, the exact messages sent
            to the summarizer, and the summarizer's response message.
        """
        messages = self._build_summarization_messages(view, start, end)
        response = self.llm.completion(messages=messages)
        for block in response.message.content:
            if isinstance(block, TextContent) and block.text.strip():
                return block.text, response.id, messages, response.message
        raise NoCondensationAvailableException(
            "Summarization LLM returned no text content."
        )

    def _frame_summary(self, summary: str) -> str:
        if not self.checkpoint_framing:
            return summary
        return (
            f"{CHECKPOINT_PREAMBLE}\n\n"
            f"{SUMMARY_OPEN_TAG}\n{summary}\n{SUMMARY_CLOSE_TAG}"
        )

    @observe(ignore_inputs=["view", "agent_llm"])
    def get_condensation(
        self, view: View, agent_llm: LLM | None = None
    ) -> Condensation:
        if agent_llm is None:
            raise NoCondensationAvailableException(
                "ReplaySummarizingCondenser needs the agent LLM to measure pressure."
            )

        for_request = view.unhandled_condensation_request
        for attempt in range(self.compaction_retries + 1):
            start, end = self._select_region(
                view,
                agent_llm,
                retain_scale=0.5**attempt,
                for_request=for_request,
            )
            forgotten_events = list(view.events[start:end])
            summary, response_id, request_messages, response_message = self._summarize(
                view, start, end
            )
            framed_summary = self._frame_summary(summary)

            summary_tokens = get_total_token_count(
                [CondensationSummaryEvent(summary=framed_summary)], agent_llm
            )
            forgotten_tokens = get_total_token_count(forgotten_events, agent_llm)
            if summary_tokens < forgotten_tokens:
                return Condensation(
                    forgotten_event_ids=[event.id for event in forgotten_events],
                    summary=framed_summary,
                    summary_offset=start,
                    llm_response_id=response_id,
                    request_messages=request_messages,
                    response_message=response_message,
                )
            logger.warning(
                "Replay summary did not shrink its source (%d >= %d tokens); "
                "retrying with a larger forgotten span",
                summary_tokens,
                forgotten_tokens,
            )

        raise NoCondensationAvailableException(
            "Condensation rejected: summaries did not shrink the forgotten events."
        )
