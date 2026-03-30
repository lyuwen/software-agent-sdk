"""LSP Tool — native function-calling wrapper for the Pyright-based LSP CLI.

The agent calls lsp_tool(command="get_definition", file_path=..., symbol=..., line=...)
which is translated into:
    python /tmp/lsp_tool.py get_definition --file_path ... --symbol ... --line ...
and executed inside the container via the TerminalExecutor.

Set LSP_NAMING=camelCase to expose camelCase command names to the LLM.
"""

import os
import shlex
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import Field
from rich.text import Text

from openhands.sdk.llm import ImageContent, TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

# ---------------------------------------------------------------------------
# Naming convention — single source of truth is lsp_tool.py
# ---------------------------------------------------------------------------
from openhands.tools.lsp.lsp_tool import (
    COMMAND_NAMES,
    CAMEL_TO_SNAKE,
    _ALIASES,
    normalize_command,
)


def _command_enum_schema(schema: dict) -> None:
    """Callable json_schema_extra: resolve the command enum at schema-generation time."""
    naming = os.environ.get("LSP_NAMING", "camelCase")
    if naming == "camelCase":
        schema["enum"] = list(COMMAND_NAMES.values())
    else:
        schema["enum"] = list(COMMAND_NAMES.keys())


# ---------------------------------------------------------------------------
# Action / Observation
# ---------------------------------------------------------------------------
class LSPAction(Action):
    """Parameters for an LSP tool call."""

    command: str = Field(
        description=(
            "The LSP command to run. "
            "Use get_definition/get_type_definition to jump to definitions with full source code. "
            "Use find_references to find all usages across the project. "
            "Use hover for type info and docstrings. "
            "Use get_implementation to find implementations of abstract methods/interfaces. "
            "Use get_call_hierarchy to see callers/callees (combined). "
            "Use prepare_call_hierarchy + incoming_calls/outgoing_calls for step-by-step. "
            "Use get_document_symbols for a file outline. "
            "Use get_workspace_symbols to search symbols project-wide."
        ),
        json_schema_extra=_command_enum_schema,
    )
    file_path: str | None = Field(
        default=None,
        description=(
            "Absolute path to the file. Required for all commands except "
            "get_workspace_symbols."
        ),
    )
    symbol: str | None = Field(
        default=None,
        description=(
            "The name of a Python code entity (class, function, variable). "
            "Required for positional commands (get_definition, get_type_definition, "
            "find_references, hover, get_implementation, get_call_hierarchy, "
            "prepare_call_hierarchy, get_document_highlights)."
        ),
    )
    line: int | None = Field(
        default=None,
        description=(
            "Line number (1-indexed) where the symbol appears. "
            "Required for positional commands."
        ),
    )
    query: str | None = Field(
        default=None,
        description=(
            "Search query for get_workspace_symbols. Should be a symbol name "
            "(e.g., 'MyClass', not 'class MyClass')."
        ),
    )
    item: str | None = Field(
        default=None,
        description=(
            "JSON string of a call hierarchy item (returned by prepare_call_hierarchy). "
            "Required for incoming_calls and outgoing_calls."
        ),
    )

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("lsp ", style="bold cyan")
        content.append(self.command, style="white")
        if self.file_path:
            content.append(f" {self.file_path}", style="dim")
        if self.symbol:
            content.append(f":{self.symbol}", style="yellow")
        if self.line is not None:
            content.append(f"@L{self.line}", style="green")
        if self.query:
            content.append(f" ?{self.query}", style="magenta")
        return content


class LSPObservation(Observation):
    """Result from an LSP tool invocation."""

    @property
    def to_llm_content(self) -> Sequence[TextContent | ImageContent]:
        return [TextContent(text=self.text if self.text else "No output from LSP tool.")]

    @property
    def visualize(self) -> Text:
        text = Text()
        if self.is_error:
            text.append("LSP Error: ", style="red bold")
        content = self.text
        if content:
            for line in content.split("\n"):
                text.append(line + "\n", style="white")
        return text


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
class LSPToolExecutor(ToolExecutor[LSPAction, LSPObservation]):
    """Translates LSPAction into a CLI command and runs it via TerminalExecutor."""

    def __init__(self, terminal_executor: "ToolExecutor"):
        self.terminal = terminal_executor

    def __call__(self, action: LSPAction, conversation=None) -> LSPObservation:
        from openhands.tools.terminal import TerminalAction

        # Normalize to canonical snake_case for the CLI
        cmd = normalize_command(action.command)

        # Build the CLI command
        parts = ["python", "/tmp/lsp_tool.py", cmd]

        if action.file_path:
            parts.extend(["--file_path", shlex.quote(action.file_path)])
        if action.symbol:
            parts.extend(["--symbol", shlex.quote(action.symbol)])
        if action.line is not None:
            parts.extend(["--line", str(action.line)])
        if action.query:
            parts.extend(["--query", shlex.quote(action.query)])
        if action.item:
            parts.extend(["--item", shlex.quote(action.item)])

        cmd = " ".join(parts)
        result = self.terminal(TerminalAction(command=cmd, timeout=30.0))

        output = result.text if hasattr(result, "text") else str(result)
        is_error = False
        if hasattr(result, "metadata") and hasattr(result.metadata, "exit_code"):
            is_error = result.metadata.exit_code != 0

        return LSPObservation(
            content=[TextContent(text=output)],
            is_error=is_error,
        )

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Tool description (what the LLM sees)
# ---------------------------------------------------------------------------
LSP_TOOL_DESCRIPTION = """\
Code intelligence tool powered by Pyright Language Server. Provides semantic, \
AST-based understanding of Python code — far more accurate than grep for \
navigating and understanding code structure.

### Commands

| Command | What it does | Required args |
|---|---|---|
| `get_definition` | Jump to where a symbol is defined; returns full source code | file_path, line, symbol |
| `get_type_definition` | Jump to the type's definition with source code | file_path, line, symbol |
| `find_references` | Find all usages of a symbol across the project | file_path, line, symbol |
| `hover` | Get docstring, inferred type, and signature | file_path, line, symbol |
| `get_implementation` | Find implementations of an interface/abstract method | file_path, line, symbol |
| `get_call_hierarchy` | Who calls this function, and what does it call? (combined) | file_path, line, symbol |
| `prepare_call_hierarchy` | Get a call hierarchy item for a symbol (step 1) | file_path, line, symbol |
| `incoming_calls` | Who calls this function? (requires item from prepare) | item |
| `outgoing_calls` | What does this function call? (requires item from prepare) | item |
| `get_document_symbols` | List all symbols in a file (classes, functions, vars) | file_path |
| `get_workspace_symbols` | Search for a symbol name across the entire project | query |
| `get_document_highlights` | Find all usages of a symbol within one file | file_path, line, symbol |

### Tips
- Use `get_document_symbols` to understand a file's structure before reading it
- Use `get_workspace_symbols` to find where something is defined across the project
- Use `get_definition` to jump to and read the actual source code of any symbol
- Use `find_references` to understand how widely a function or class is used
- Use `get_call_hierarchy` for a quick combined view of callers + callees
- For fine-grained control, use `prepare_call_hierarchy` then `incoming_calls`/`outgoing_calls`
- `line` is 1-indexed (matches what you see in file editors)
- `symbol` is just the identifier name (e.g., `MyClass`, `my_function`, `my_var`)
"""


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------
class LSPTool(ToolDefinition[LSPAction, LSPObservation]):
    """Native function-calling wrapper for the Pyright-based LSP CLI tool."""

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        terminal_executor: "ToolExecutor | None" = None,
        lsp_naming: str | None = None,
    ) -> Sequence["LSPTool"]:
        # Allow explicit naming override via Tool(params={"lsp_naming": "camelCase"})
        if lsp_naming is not None:
            os.environ["LSP_NAMING"] = lsp_naming

        if terminal_executor is None:
            from openhands.tools.terminal.impl import TerminalExecutor

            terminal_executor = TerminalExecutor(
                working_dir=conv_state.workspace.working_dir,
            )

        executor = LSPToolExecutor(terminal_executor)

        return [
            cls(
                description=LSP_TOOL_DESCRIPTION,
                action_type=LSPAction,
                observation_type=LSPObservation,
                executor=executor,
                annotations=ToolAnnotations(
                    title="LSP Code Intelligence",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
            )
        ]


# Auto-register when this module is imported
register_tool(LSPTool.name, LSPTool)
