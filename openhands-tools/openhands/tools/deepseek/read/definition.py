"""DeepSeek-harness-compatible read tool.

Model-facing name: ``read``
Forks view from FileEditorExecutor but outputs the DSH envelope format:
  <path>...</path>
  <type>file</type>
  <content>
  line-numbered content
  </content>
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

_READ_LIMIT = 2000


class DeepSeekReadAction(Action):
    """Input schema for the DeepSeek-compatible ``read`` tool."""

    file_path: str = Field(
        description="Path to read, resolved by the filesystem backend.",
    )
    offset: int | None = Field(
        default=None,
        ge=1,
        description="1-based first line to return. Defaults to 1.",
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        description=f"Maximum number of lines to return. Defaults to {_READ_LIMIT}.",
    )


class DeepSeekReadObservation(Observation):
    """Output schema for the DeepSeek-compatible ``read`` tool."""
    pass


class DeepSeekReadExecutor(ToolExecutor):
    """Executor for the DeepSeek-compatible read tool."""

    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = workspace_root

    def __call__(
        self,
        action: DeepSeekReadAction,
        conversation=None,  # noqa: ARG002
    ) -> DeepSeekReadObservation:
        from openhands.tools.file_editor.definition import FileEditorAction

        if not action.file_path.strip():
            return DeepSeekReadObservation.from_text(
                text="file_path must be a non-empty string", is_error=True
            )

        offset = action.offset if action.offset is not None else 1
        limit = action.limit if action.limit is not None else _READ_LIMIT

        if limit > _READ_LIMIT:
            return DeepSeekReadObservation.from_text(
                text=f"limit must be less than or equal to {_READ_LIMIT}", is_error=True
            )

        # Use view with view_range to get line-numbered content.
        view_range = None if (offset == 1 and limit == _READ_LIMIT) else [offset, offset + limit - 1]

        editor_action = FileEditorAction(
            command="view",
            path=action.file_path,
            view_range=view_range,
        )

        from openhands.tools.file_editor.impl import FileEditorExecutor
        executor = FileEditorExecutor(workspace_root=self.workspace_root)
        obs = executor(editor_action)

        if obs.is_error:
            return DeepSeekReadObservation.from_text(text=obs.text or "", is_error=True)

        # Reformat the cat -n style output into the DSH envelope.
        raw_text = obs.text or ""
        formatted = self._format_as_dsh_envelope(action.file_path, raw_text)
        return DeepSeekReadObservation.from_text(text=formatted)

    def _format_as_dsh_envelope(self, path: str, cat_n_output: str) -> str:
        """Wrap line-numbered content in the DSH read envelope."""
        return f"<path>{path}</path>\n<type>file</type>\n<content>\n{cat_n_output}\n</content>"


class DeepSeekReadTool(ToolDefinition[DeepSeekReadAction, DeepSeekReadObservation]):
    """DeepSeek-harness-compatible ``read`` tool."""

    name: ClassVar[str] = "read"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        executor: ToolExecutor | None = None,
    ) -> Sequence["DeepSeekReadTool"]:
        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        return [
            cls(
                action_type=DeepSeekReadAction,
                observation_type=DeepSeekReadObservation,
                description="Read a UTF-8 text file and return line-numbered content.",
                annotations=ToolAnnotations(
                    title="read",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor or DeepSeekReadExecutor(working_dir),
            )
        ]


register_tool(DeepSeekReadTool.name, DeepSeekReadTool)
