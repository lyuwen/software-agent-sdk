"""DeepSeek-harness-compatible persistent bash tool.

Model-facing name: ``bash``
Schema: command (required), timeoutMs (optional), reset (optional)

Wraps the existing OpenHands TerminalExecutor so the underlying PTY session and
all of its persistence semantics are reused unchanged.  Only the model-facing
schema and the output format differ from the native ``terminal`` tool.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from pydantic import Field

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

# DSH default: 16 000 characters before clipping.
_MAX_OUTPUT_CHARS = 16_000
_TRUNCATED_MESSAGE = (
    "<response clipped>"
    "<NOTE>To save on context only part of this file has been shown to you. "
    "You should retry this tool after you have searched inside the file with "
    "`grep -n` in order to find the line numbers of what you are looking for.</NOTE>"
)
_SHELL_RESET_MESSAGE = (
    "The persistent bash shell was reset; "
    "the next bash call starts from the workspace with a fresh current directory "
    "and environment."
)


def _render_output(text: str, exit_code: int | None) -> str:
    """Format command output to match DSH's renderCaptured convention.

    Appends ``[exit code: N]`` only when the exit code is non-zero (or unknown
    and the command appeared to fail).  Truncates at _MAX_OUTPUT_CHARS.
    """
    if len(text) > _MAX_OUTPUT_CHARS:
        text = text[:_MAX_OUTPUT_CHARS] + _TRUNCATED_MESSAGE

    if exit_code is not None and exit_code != 0:
        marker = f"[exit code: {exit_code}]"
        return f"{text}\n{marker}" if text else marker
    return text


class DeepSeekBashAction(Action):
    """Input schema for the DeepSeek-compatible ``bash`` tool."""

    command: str = Field(
        description="The bash command to run. Relative path is preferred in the command.",
    )
    timeoutMs: float | None = Field(  # noqa: N815  — matches DSH argument name exactly
        default=None,
        gt=0,
        description=(
            "Timeout in milliseconds for this command, overriding the deployment "
            "default.  The shell resets after a timeout."
        ),
    )
    reset: bool = Field(
        default=False,
        description=(
            "Reset the persistent shell before running: closes the current session "
            "and starts a fresh one from the workspace, losing all environment and "
            "working-directory state."
        ),
    )


class DeepSeekBashObservation(Observation):
    """Output schema for the DeepSeek-compatible ``bash`` tool."""

    pass


class DeepSeekBashExecutor(ToolExecutor):
    """Adapter that drives TerminalExecutor with DeepSeek-harness calling conventions."""

    def __init__(self, terminal_executor) -> None:  # type: ignore[annotation]
        self._terminal = terminal_executor

    def __call__(
        self,
        action: DeepSeekBashAction,
        conversation=None,
    ) -> DeepSeekBashObservation:
        from openhands.tools.terminal.definition import TerminalAction

        if not action.command.strip():
            raise ValueError("command must be a non-empty string")

        # Convert timeoutMs → timeout in seconds (TerminalAction uses seconds).
        timeout_seconds: float | None = None
        if action.timeoutMs is not None:
            timeout_seconds = action.timeoutMs / 1000.0

        terminal_action = TerminalAction(
            command=action.command,
            is_input=False,
            timeout=timeout_seconds,
            reset=action.reset,
        )

        obs = self._terminal(terminal_action, conversation)

        # Build the model-facing text from the raw output text and exit code.
        # We deliberately skip the OpenHands-specific metadata lines
        # ([Current working directory: …], [Python interpreter: …], etc.)
        # because DSH's bash tool does not emit them.
        raw_text = obs.text or ""
        exit_code = None
        if hasattr(obs, "metadata") and obs.metadata is not None:
            ec = obs.metadata.exit_code
            # -1 means "still running" in OpenHands; treat as None here.
            if ec is not None and ec != -1:
                exit_code = ec

        result_text = _render_output(raw_text, exit_code)

        # Append the reset notice when the terminal was just reset with no command.
        if action.reset and not action.command.strip():
            result_text = "\n".join(
                part for part in [result_text, _SHELL_RESET_MESSAGE] if part
            )

        return DeepSeekBashObservation.from_text(text=result_text)

    def close(self) -> None:
        self._terminal.close()


class DeepSeekBashTool(ToolDefinition[DeepSeekBashAction, DeepSeekBashObservation]):
    """DeepSeek-harness-compatible ``bash`` tool backed by OpenHands TerminalExecutor."""

    # Override automatic name derivation so the model always sees "bash",
    # never "deep_seek_bash".
    name: ClassVar[str] = "bash"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        executor: ToolExecutor | None = None,
    ) -> Sequence["DeepSeekBashTool"]:
        from openhands.tools.terminal.impl import TerminalExecutor

        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        if executor is None:
            terminal_executor = TerminalExecutor(
                working_dir=working_dir,
                full_output_save_dir=conv_state.env_observation_persistence_dir,
            )
        else:
            terminal_executor = executor

        deepseek_executor = DeepSeekBashExecutor(terminal_executor)

        return [
            cls(
                action_type=DeepSeekBashAction,
                observation_type=DeepSeekBashObservation,
                description=(
                    "Run commands in a bash shell\n"
                    "* When invoking this tool, the contents of the \"command\" parameter does NOT need to be XML-escaped.\n"
                    "* You don't have access to the internet via this tool.\n"
                    "* You do have access to a mirror of common linux and python packages via apt and pip.\n"
                    "* State is persistent across command calls and discussions with the user.\n"
                    "* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.\n"
                    "* Please avoid commands that may produce a very large amount of output.\n"
                    "* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background."
                ),
                annotations=ToolAnnotations(
                    title="bash",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
                executor=deepseek_executor,
            )
        ]


register_tool(DeepSeekBashTool.name, DeepSeekBashTool)
