"""DeepSeek-harness-compatible write tool.

Model-facing name: ``write``
Forks create from FileEditorExecutor, but allows overwriting existing files
(DSH write is create-or-replace, not create-only).
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


class DeepSeekWriteAction(Action):
    """Input schema for the DeepSeek-compatible ``write`` tool."""

    file_path: str = Field(
        description="Path to write, resolved by the filesystem backend.",
    )
    content: str = Field(
        description="Full UTF-8 text content to write.",
    )


class DeepSeekWriteObservation(Observation):
    """Output schema for the DeepSeek-compatible ``write`` tool."""
    pass


def _format_write_output(path: str, operation: str) -> str:
    """DSH write envelope: Created/Updated file."""
    verb = "Created" if operation == "create" else "Updated"
    return f"<path>{path}</path>\n<type>file</type>\n<content>\n{verb} file\n</content>"


class DeepSeekWriteExecutor(ToolExecutor):
    """Executor for the DeepSeek-compatible write tool."""

    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = workspace_root

    def __call__(
        self,
        action: DeepSeekWriteAction,
        conversation=None,  # noqa: ARG002
    ) -> DeepSeekWriteObservation:
        from openhands.tools.file_editor.definition import FileEditorAction
        from openhands.tools.file_editor.impl import FileEditorExecutor

        if not action.file_path.strip():
            return DeepSeekWriteObservation.from_text(
                text="file_path must be a non-empty string", is_error=True
            )

        executor = FileEditorExecutor(workspace_root=self.workspace_root)
        path = action.file_path
        exists = os.path.isfile(path)

        if exists:
            # DSH write overwrites — write new content directly.
            # We verify the file is readable before overwriting.
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(action.content)
            except OSError as e:
                return DeepSeekWriteObservation.from_text(text=str(e), is_error=True)

            return DeepSeekWriteObservation.from_text(
                text=_format_write_output(path, "update")
            )
        else:
            # New file — use FileEditor create.
            obs = executor(FileEditorAction(
                command="create",
                path=path,
                file_text=action.content,
            ))
            if obs.is_error:
                return DeepSeekWriteObservation.from_text(
                    text=obs.text or "", is_error=True
                )
            return DeepSeekWriteObservation.from_text(
                text=_format_write_output(path, "create")
            )


class DeepSeekWriteTool(ToolDefinition[DeepSeekWriteAction, DeepSeekWriteObservation]):
    """DeepSeek-harness-compatible ``write`` tool."""

    name: ClassVar[str] = "write"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        executor: ToolExecutor | None = None,
    ) -> Sequence["DeepSeekWriteTool"]:
        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        return [
            cls(
                action_type=DeepSeekWriteAction,
                observation_type=DeepSeekWriteObservation,
                description="Create or fully replace a UTF-8 text file.",
                annotations=ToolAnnotations(
                    title="write",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor or DeepSeekWriteExecutor(working_dir),
            )
        ]


register_tool(DeepSeekWriteTool.name, DeepSeekWriteTool)
