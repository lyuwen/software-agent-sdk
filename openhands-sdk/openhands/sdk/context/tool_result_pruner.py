"""Deterministic tool-result pruning for agent LLM requests.

Port of the DeepSeek Harness ``compaction-tool-result-pruner`` behavior to the
SDK's agent loop. An oversized tool result is replaced, for the LLM request
only, by a bounded head, a fixed middle-elision marker, and a bounded tail:

    <head chars>\\n\\n[... tool result middle pruned ...]\\n\\n<tail chars>

The transform is deterministic and idempotent: every pruned result has at most
``threshold_chars`` text code points, and a pruned result is never pruned again.
Only text blocks are measured and sliced; rich blocks (images) keep their
relative order. Slicing is by Unicode code point, so it never splits a
surrogate pair.

The conversation history itself is never modified: events keep their
full-fidelity output, so history dumps, replay, and evaluation artifacts remain
exact while the model sees the smaller request. The pruner is a pure message
transform invoked by :func:`openhands.sdk.agent.utils.prepare_llm_messages`.

Configuration comes from environment variables so a deployment can opt in
without changing agent construction. The variables are read on each request and
an invalid value fails loudly rather than silently falling back to defaults:

- ``OH_TOOL_RESULT_PRUNER``: enable the pruner (``1``, ``true``, ``yes``, ``on``).
- ``OH_TOOL_RESULT_PRUNER_THRESHOLD_CHARS``: prune text above this many
  Unicode code points (default ``8192``).
- ``OH_TOOL_RESULT_PRUNER_HEAD_CHARS``: leading code points retained (default
  ``4096``).
- ``OH_TOOL_RESULT_PRUNER_TAIL_CHARS``: trailing code points retained (default
  ``1024``).

Unlike DeepSeek Harness, where the pruner runs only once a compaction trigger
qualifies, this implementation prunes on every agent LLM request while enabled.
The transform is cheap and convergent, so the difference only affects when an
oversized result first reaches the model; gate the call from a condenser if
compaction-pressure semantics are required.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from openhands.sdk.llm import ImageContent, Message, TextContent
from openhands.sdk.logger import get_logger


logger = get_logger(__name__)

ENV_ENABLED = "OH_TOOL_RESULT_PRUNER"
ENV_THRESHOLD_CHARS = "OH_TOOL_RESULT_PRUNER_THRESHOLD_CHARS"
ENV_HEAD_CHARS = "OH_TOOL_RESULT_PRUNER_HEAD_CHARS"
ENV_TAIL_CHARS = "OH_TOOL_RESULT_PRUNER_TAIL_CHARS"

PRUNE_MARKER = "\n\n[... tool result middle pruned ...]\n\n"
"""Fixed marker substituted for every removed middle span."""

DEFAULT_THRESHOLD_CHARS = 8192
DEFAULT_HEAD_CHARS = 4096
DEFAULT_TAIL_CHARS = 1024

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

type ContentBlock = TextContent | ImageContent


@dataclass(frozen=True)
class ToolResultPruneConfig:
    """Character budgets for deterministic tool-result pruning.

    Attributes:
        threshold_chars: Prune a tool result whose combined text exceeds this
            many Unicode code points.
        head_chars: Maximum leading code points retained.
        tail_chars: Maximum trailing code points retained.
    """

    threshold_chars: int = DEFAULT_THRESHOLD_CHARS
    head_chars: int = DEFAULT_HEAD_CHARS
    tail_chars: int = DEFAULT_TAIL_CHARS


def code_point_length(text: str) -> int:
    """Return the number of Unicode code points in ``text``.

    Python strings are sequences of code points, so this is ``len``; it exists
    to document the unit used by the character budgets and to mirror the
    upstream JavaScript implementation.
    """
    return len(text)


def resolve_config(
    config: ToolResultPruneConfig | None = None,
) -> ToolResultPruneConfig:
    """Validate a pruning configuration and return it unchanged.

    Args:
        config: Configuration to validate. Defaults to the built-in budgets.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If a budget is not an integer, is out of range, or the
            emitted head + marker + tail exceeds ``threshold_chars``.
    """
    resolved = config if config is not None else ToolResultPruneConfig()
    _assert_integer("threshold_chars", resolved.threshold_chars)
    _assert_integer("head_chars", resolved.head_chars)
    _assert_integer("tail_chars", resolved.tail_chars)
    if resolved.threshold_chars <= 0:
        raise ValueError(
            "tool-result prune: threshold_chars "
            f"({resolved.threshold_chars}) must be a positive integer"
        )
    if resolved.head_chars < 0:
        raise ValueError(
            "tool-result prune: head_chars "
            f"({resolved.head_chars}) must be a non-negative integer"
        )
    if resolved.tail_chars < 0:
        raise ValueError(
            "tool-result prune: tail_chars "
            f"({resolved.tail_chars}) must be a non-negative integer"
        )
    emitted = (
        resolved.head_chars + code_point_length(PRUNE_MARKER) + resolved.tail_chars
    )
    if emitted > resolved.threshold_chars:
        raise ValueError(
            "tool-result prune: head_chars + marker + tail_chars "
            f"({emitted}) must be at most threshold_chars "
            f"({resolved.threshold_chars})"
        )
    return resolved


def is_enabled() -> bool:
    """Return whether the pruner is enabled through the environment."""
    return os.environ.get(ENV_ENABLED, "").strip().lower() in _TRUE_VALUES


def config_from_env() -> ToolResultPruneConfig:
    """Resolve and validate the pruning configuration from the environment.

    Returns:
        The configuration described by the ``OH_TOOL_RESULT_PRUNER_*``
        variables, using the default budgets for unset variables.

    Raises:
        ValueError: If a variable is set to a non-integer value or the resolved
            budgets are invalid.
    """
    return resolve_config(
        ToolResultPruneConfig(
            threshold_chars=_env_int(ENV_THRESHOLD_CHARS, DEFAULT_THRESHOLD_CHARS),
            head_chars=_env_int(ENV_HEAD_CHARS, DEFAULT_HEAD_CHARS),
            tail_chars=_env_int(ENV_TAIL_CHARS, DEFAULT_TAIL_CHARS),
        )
    )


def measure_content(content: Sequence[ContentBlock]) -> int:
    """Measure the text content of a tool result in Unicode code points.

    Non-text blocks cost zero.

    Args:
        content: Tool-result content blocks to measure.

    Returns:
        Total code points across all text blocks.
    """
    return sum(
        code_point_length(block.text)
        for block in content
        if isinstance(block, TextContent)
    )


def prune_content(
    content: Sequence[ContentBlock],
    config: ToolResultPruneConfig | None = None,
) -> list[ContentBlock] | None:
    """Replace an over-budget text middle while retaining block order.

    Args:
        content: Original tool-result content blocks.
        config: Character budgets to apply. Defaults to the built-in budgets.

    Returns:
        Pruned content blocks, or ``None`` when the text is within budget.

    Raises:
        ValueError: If ``config`` is invalid.
        RuntimeError: If the configured budgets fail to locate a removable
            text span, or if the replacement is not smaller and within budget.
            This can only mean a bug in this module.
    """
    resolved = resolve_config(config)
    total_chars = measure_content(content)
    if total_chars <= resolved.threshold_chars:
        return None

    removed_start = resolved.head_chars
    removed_end = total_chars - resolved.tail_chars
    pruned: list[ContentBlock] = []
    consumed = 0
    marker_inserted = False

    for block in content:
        if not isinstance(block, TextContent):
            pruned.append(block)
            continue

        text = block.text
        block_start = consumed
        block_end = block_start + code_point_length(text)
        head_end = min(len(text), max(0, removed_start - block_start))
        tail_start = min(len(text), max(0, removed_end - block_start))
        intersects_removed = block_start < removed_end and block_end > removed_start
        marker = PRUNE_MARKER if intersects_removed and not marker_inserted else ""
        if marker:
            marker_inserted = True
        pruned_text = text[:head_end] + marker + text[tail_start:]
        if pruned_text:
            pruned.append(block.model_copy(update={"text": pruned_text}))
        consumed = block_end

    if not marker_inserted:
        raise RuntimeError("tool-result prune: failed to locate the removed text span")
    chars_after = measure_content(pruned)
    if chars_after > resolved.threshold_chars or chars_after >= total_chars:
        raise RuntimeError(
            "tool-result prune: replacement must be smaller and within threshold"
        )
    return pruned


def prune_tool_result_messages(
    messages: Sequence[Message],
    config: ToolResultPruneConfig | None = None,
) -> list[Message]:
    """Prune the content of every over-budget tool-role message.

    Non-tool messages, and tool messages within budget, are returned unchanged.

    Args:
        messages: LLM messages for one agent request.
        config: Character budgets to apply. Defaults to the built-in budgets.

    Returns:
        New message list with pruned copies of over-budget tool messages.
    """
    resolved = resolve_config(config)
    pruned_messages: list[Message] = []
    for message in messages:
        if message.role != "tool":
            pruned_messages.append(message)
            continue
        pruned_content = prune_content(message.content, resolved)
        if pruned_content is None:
            pruned_messages.append(message)
        else:
            pruned_messages.append(
                message.model_copy(update={"content": pruned_content})
            )
    return pruned_messages


def maybe_prune_tool_result_messages(messages: list[Message]) -> list[Message]:
    """Prune tool results when the pruner is enabled through the environment.

    This is the agent-loop entry point. It is a no-op unless
    ``OH_TOOL_RESULT_PRUNER`` is truthy, so default SDK behavior is unchanged.

    Args:
        messages: LLM messages for one agent request.

    Returns:
        The original list when disabled, otherwise the pruned message list.

    Raises:
        ValueError: If the pruner is enabled but its budgets are invalid.
    """
    if not is_enabled():
        return messages
    config = config_from_env()
    pruned = prune_tool_result_messages(messages, config)
    rewritten = [
        (original, replacement)
        for original, replacement in zip(messages, pruned, strict=True)
        if original is not replacement
    ]
    if rewritten:
        chars_removed = sum(
            measure_content(original.content) - measure_content(replacement.content)
            for original, replacement in rewritten
        )
        logger.debug(
            "tool-result pruner: trimmed %d characters from %d tool result(s) "
            "for this request (threshold=%d, head=%d, tail=%d)",
            chars_removed,
            len(rewritten),
            config.threshold_chars,
            config.head_chars,
            config.tail_chars,
        )
    return pruned


def _assert_integer(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"tool-result prune: {name} ({value!r}) must be an integer")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"tool-result prune: {name} must be an integer, got {raw!r}"
        ) from None
