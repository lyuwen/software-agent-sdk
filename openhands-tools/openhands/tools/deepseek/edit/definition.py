"""DeepSeek-harness-compatible edit tool.

Model-facing name: ``edit``
Maps to FileEditorExecutor str_replace with the DSH schema:
  file_path, old_string, new_string, replace_all (optional, default false)

Output matches DSH's formatEditOutput:
  "The file <path> has been updated successfully."
  "The file <path> has been updated. All occurrences were successfully replaced."
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


class DeepSeekEditAction(Action):
    """Input schema for the DeepSeek-compatible ``edit`` tool."""

    file_path: str = Field(
        description="Path to edit, resolved by the filesystem backend.",
    )
    old_string: str = Field(
        description="Literal text to replace. Must match exactly.",
    )
    new_string: str = Field(
        description="Literal replacement text. Use an empty string to delete the match.",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace all matches. Defaults to false; when false, old_string must appear exactly once.",
    )


class DeepSeekEditObservation(Observation):
    """Output schema for the DeepSeek-compatible ``edit`` tool."""
    pass


class DeepSeekEditExecutor(ToolExecutor):
    """Executor for the DeepSeek-compatible edit tool."""

    def __init__(self, workspace_root: str) -> None:
        self.workspace_root = workspace_root

    def __call__(
        self,
        action: DeepSeekEditAction,
        conversation=None,  # noqa: ARG002
    ) -> DeepSeekEditObservation:
        from openhands.tools.file_editor.definition import FileEditorAction
        from openhands.tools.file_editor.impl import FileEditorExecutor

        if not action.file_path.strip():
            return DeepSeekEditObservation.from_text(
                text="file_path must be a non-empty string", is_error=True
            )
        if not action.old_string:
            return DeepSeekEditObservation.from_text(
                text="old_string must be a non-empty string", is_error=True
            )
        if action.old_string == action.new_string:
            return DeepSeekEditObservation.from_text(
                text="old_string and new_string must differ", is_error=True
            )

        executor = FileEditorExecutor(workspace_root=self.workspace_root)

        if action.replace_all:
            # FileEditor str_replace only replaces one occurrence and requires uniqueness.
            # For replace_all we do the replacement directly on the file.
            try:
                with open(action.file_path, encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except OSError as e:
                return DeepSeekEditObservation.from_text(text=str(e), is_error=True)

            if action.old_string not in content:
                return DeepSeekEditObservation.from_text(
                    text=f"No replacement was performed, old_string `{action.old_string[:50]}` was not found in {action.file_path}.",
                    is_error=True,
                )

            new_content = content.replace(action.old_string, action.new_string)
            try:
                with open(action.file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            except OSError as e:
                return DeepSeekEditObservation.from_text(text=str(e), is_error=True)

            return DeepSeekEditObservation.from_text(
                text=f"The file {action.file_path} has been updated. All occurrences were successfully replaced."
            )
        else:
            obs = executor(FileEditorAction(
                command="str_replace",
                path=action.file_path,
                old_str=action.old_string,
                new_str=action.new_string,
            ))
            if obs.is_error:
                return DeepSeekEditObservation.from_text(
                    text=obs.text or "", is_error=True
                )
            return DeepSeekEditObservation.from_text(
                text=f"The file {action.file_path} has been updated successfully."
            )


class DeepSeekEditTool(ToolDefinition[DeepSeekEditAction, DeepSeekEditObservation]):
    """DeepSeek-harness-compatible ``edit`` tool."""

    name: ClassVar[str] = "edit"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        executor: ToolExecutor | None = None,
    ) -> Sequence["DeepSeekEditTool"]:
        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        return [
            cls(
                action_type=DeepSeekEditAction,
                observation_type=DeepSeekEditObservation,
                description="Edit an existing UTF-8 text file by replacing literal text.",
                annotations=ToolAnnotations(
                    title="edit",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=executor or DeepSeekEditExecutor(working_dir),
            )
        ]


register_tool(DeepSeekEditTool.name, DeepSeekEditTool)
