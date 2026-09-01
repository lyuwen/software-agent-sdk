"""DeepSeek-harness-compatible str_replace_editor tool.

Model-facing name: ``str_replace_editor``
Commands: view, create, str_replace, insert  (undo_edit is intentionally absent)

Thin adapter over the existing OpenHands FileEditorExecutor / FileEditor backend.
The backend logic (matching, uniqueness checks, line numbering, truncation) is
reused unchanged; only the model-facing schema is adjusted to match DSH.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

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

# DSH exposes exactly these four commands; undo_edit is not part of the schema.
DeepSeekCommandLiteral = Literal["view", "create", "str_replace", "insert"]


class DeepSeekStrReplaceEditorAction(Action):
    """Input schema for the DeepSeek-compatible ``str_replace_editor`` tool."""

    command: DeepSeekCommandLiteral = Field(
        description=(
            "The commands to run. Allowed options are: `view`, `create`, "
            "`str_replace`, `insert`."
        ),
    )
    path: str = Field(description="Absolute path to file or directory.")
    file_text: str | None = Field(
        default=None,
        description=(
            "Required parameter of `create` command, with the content of the file "
            "to be created."
        ),
    )
    old_str: str | None = Field(
        default=None,
        description=(
            "Required parameter of `str_replace` command containing the string in "
            "`path` to replace."
        ),
    )
    new_str: str | None = Field(
        default=None,
        description=(
            "Optional parameter of `str_replace` command containing the new string "
            "(if not given, no string will be added). Required parameter of `insert` "
            "command containing the string to insert."
        ),
    )
    insert_line: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Required parameter of `insert` command. The `new_str` will be inserted "
            "AFTER the line `insert_line` of `path`."
        ),
    )
    view_range: list[int] | None = Field(
        default=None,
        description=(
            "Optional parameter of `view` command when `path` points to a file. "
            "If none is given, the full file is shown. If provided, the file will be "
            "shown in the indicated line number range, e.g. [11, 12] will show lines "
            "11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all "
            "lines from `start_line` to the end of the file."
        ),
    )


class DeepSeekStrReplaceEditorObservation(Observation):
    """Output schema for the DeepSeek-compatible ``str_replace_editor`` tool."""

    pass


class DeepSeekStrReplaceEditorExecutor(ToolExecutor):
    """Adapter that drives FileEditorExecutor with DeepSeek-harness calling conventions."""

    def __init__(self, file_editor_executor) -> None:  # type: ignore[annotation]
        self._editor = file_editor_executor

    def __call__(
        self,
        action: DeepSeekStrReplaceEditorAction,
        conversation=None,
    ) -> DeepSeekStrReplaceEditorObservation:
        from openhands.tools.file_editor.definition import FileEditorAction

        editor_action = FileEditorAction(
            command=action.command,
            path=action.path,
            file_text=action.file_text,
            old_str=action.old_str,
            new_str=action.new_str,
            insert_line=action.insert_line,
            view_range=action.view_range,
        )

        obs = self._editor(editor_action, conversation)
        return DeepSeekStrReplaceEditorObservation.from_text(
            text=obs.text or "",
            is_error=obs.is_error,
        )


_TOOL_DESCRIPTION = (
    "Custom editing tool for viewing, creating and editing files\n"
    "* State is persistent across command calls and discussions with the user\n"
    "* If `path` is a file, `view` displays the result of applying `cat -n`. "
    "If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep\n"
    "* The `create` command cannot be used if the specified `path` already exists as a file\n"
    "* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`\n"
    "\n"
    "Notes for using the `str_replace` command:\n"
    "* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. "
    "Be mindful of whitespaces!\n"
    "* If the `old_str` parameter is not unique in the file, the replacement will not be performed. "
    "Make sure to include enough context in `old_str` to make it unique\n"
    "* The `new_str` parameter should contain the edited lines that should replace the `old_str`"
)


class DeepSeekStrReplaceEditorTool(
    ToolDefinition[
        DeepSeekStrReplaceEditorAction,
        DeepSeekStrReplaceEditorObservation,
    ]
):
    """DeepSeek-harness-compatible ``str_replace_editor`` tool backed by FileEditorExecutor."""

    # Override automatic name derivation so the model sees "str_replace_editor".
    name: ClassVar[str] = "str_replace_editor"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        executor: ToolExecutor | None = None,
    ) -> Sequence["DeepSeekStrReplaceEditorTool"]:
        from openhands.tools.file_editor.impl import FileEditorExecutor

        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        if executor is None:
            file_editor_executor = FileEditorExecutor(workspace_root=working_dir)
        else:
            file_editor_executor = executor

        deepseek_executor = DeepSeekStrReplaceEditorExecutor(file_editor_executor)

        description = (
            f"{_TOOL_DESCRIPTION}\n\n"
            f"Your current working directory is: {working_dir}\n"
            f"When exploring project structure, start with this directory "
            f"instead of the root filesystem."
        )

        return [
            cls(
                action_type=DeepSeekStrReplaceEditorAction,
                observation_type=DeepSeekStrReplaceEditorObservation,
                description=description,
                annotations=ToolAnnotations(
                    title="str_replace_editor",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
                executor=deepseek_executor,
            )
        ]


register_tool(DeepSeekStrReplaceEditorTool.name, DeepSeekStrReplaceEditorTool)
