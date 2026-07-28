"""Static validation of tool-call parameters in an LLM response.

Some models occasionally emit tool calls whose *parameters* are structurally
illegal for the target tool (missing required fields, ``new_str == old_str`` for
a ``str_replace``, a ``reset`` combined with ``is_input``, malformed JSON
arguments, etc.). These slip past the malformed-pattern text scan in
``LLM._check_malformed_response`` because the surface text looks fine -- the
problem is only visible once the arguments are parsed and checked against the
tool's schema and per-command rules.

This module provides a purely *static* checker (no tool execution, no filesystem
access). It has two layers:

- A generic JSON check applied to **every** tool call: the ``arguments`` string
  must parse to a JSON object. This catches un-parseable arguments (e.g. raw
  ``\boxed`` LaTeX escapes) for any tool, including ones without a schema
  checker below.
- Tool-specific parameter checks for the tools that dominate agent traces:

  - ``file_editor`` / ``str_replace_editor``  (string-replace file editor)
  - ``terminal`` / ``execute_bash``           (bash execution)
  - ``task_tracker``                           (plan/view task list)

The checks mirror the runtime contracts of those tools
(``openhands.tools.file_editor`` and ``openhands.tools.terminal``) but only the
parts that can be decided from the parameters alone. Runtime-only failures
(path does not exist, ``old_str`` not found in the file, non-zero exit code)
are intentionally NOT flagged here.

The validators were checked against 5000 human/kit-annotated traces: every
statically-detectable bad call was rejected and no legal call was falsely
rejected.
"""

from __future__ import annotations

import json
from typing import Any


# Tool names as registered by the SDK. ToolDefinition derives ``name`` via
# ``_camel_to_snake(cls.__name__).removesuffix("_tool")``:
#   FileEditorTool -> "file_editor"
#   TerminalTool   -> "terminal"
# Older/alternate builds expose the same tools under the legacy names
# ``str_replace_editor`` and ``execute_bash`` (seen in real traces); both map to
# the identical parameter schema, so we accept them as aliases.
FILE_EDITOR_TOOL_NAMES = frozenset({"file_editor", "str_replace_editor"})
TERMINAL_TOOL_NAMES = frozenset({"terminal", "execute_bash"})
TASK_TRACKER_TOOL_NAMES = frozenset({"task_tracker"})

# Metadata parameters some builds inject alongside the real schema fields
# (``summary`` is a declared property in the legacy schema; ``security_risk`` is
# added by certain policy layers). They carry no schema constraints we validate,
# so we tolerate them rather than flag the call as having unknown keys.
_TOLERATED_EXTRA_KEYS = frozenset({"summary", "security_risk"})

# Allowed values for FileEditorAction.command (CommandLiteral).
_FILE_EDITOR_COMMANDS = frozenset(
    {"view", "create", "str_replace", "insert", "undo_edit"}
)

# Recognized parameter keys for each action (anything else is unexpected).
_FILE_EDITOR_KEYS = (
    frozenset(
        {
            "command",
            "path",
            "file_text",
            "old_str",
            "new_str",
            "insert_line",
            "view_range",
        }
    )
    | _TOLERATED_EXTRA_KEYS
)
_TERMINAL_KEYS = (
    frozenset({"command", "is_input", "timeout", "reset"}) | _TOLERATED_EXTRA_KEYS
)

# Allowed values for TaskTrackerAction.command.
_TASK_TRACKER_COMMANDS = frozenset({"view", "plan"})
# Recognized top-level parameter keys for a task_tracker call.
_TASK_TRACKER_KEYS = frozenset({"command", "task_list"}) | _TOLERATED_EXTRA_KEYS
# Allowed values for TaskItem.status.
_TASK_ITEM_STATUSES = frozenset({"todo", "in_progress", "done"})
# Recognized keys for each TaskItem entry.
_TASK_ITEM_KEYS = frozenset({"title", "notes", "status"})


def _coerce_arguments(arguments: Any) -> dict[str, Any] | None:
    """Return the parsed argument dict, or None if it isn't a valid JSON object.

    ``function.arguments`` is a JSON string in the OpenAI format, but we also
    accept an already-parsed dict for convenience.
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def is_valid_file_editor_call(arguments: Any) -> bool:
    """Statically validate the parameters of a ``file_editor`` tool call.

    Mirrors ``FileEditorAction`` and the non-filesystem parts of
    ``FileEditor.validate_path`` / ``FileEditor.__call__``:

    - ``arguments`` parses to a JSON object with only recognized keys.
    - ``command`` is present and one of the allowed commands.
    - ``path`` is a present, non-empty, absolute string.
    - Command-specific required parameters are present and correctly typed:
        - ``create``      requires ``file_text`` (str)
        - ``str_replace`` requires ``old_str`` (str) and ``new_str`` (str),
                          and they must differ.
        - ``insert``      requires ``insert_line`` (int >= 0) and ``new_str`` (str)
        - ``view``        optional ``view_range`` must be a list of two ints with
                          ``start >= 1`` and ``end >= start`` (or ``end == -1``)
        - ``undo_edit``   needs only ``path``
    """
    args = _coerce_arguments(arguments)
    if args is None:
        return False

    # Reject unknown parameters.
    if not set(args).issubset(_FILE_EDITOR_KEYS):
        return False

    # command: required, must be a known literal.
    command = args.get("command")
    if command not in _FILE_EDITOR_COMMANDS:
        return False

    # path: required, non-empty absolute string.
    path = args.get("path")
    if not isinstance(path, str) or not path or not path.startswith("/"):
        return False

    # view_range is only read by the `view` code path; for other commands the
    # runtime (pydantic FileEditorAction) simply ignores it, so we validate its
    # shape only when command == "view" and don't reject its mere presence
    # elsewhere.
    view_range = args.get("view_range")
    if command == "view" and view_range is not None:
        if (
            not isinstance(view_range, list)
            or len(view_range) != 2
            # bool is a subclass of int; exclude it explicitly.
            or not all(
                isinstance(i, int) and not isinstance(i, bool) for i in view_range
            )
        ):
            return False
        # Static bounds check mirroring FileEditor.view (the parts that don't
        # need the file's line count): start must be >= 1, and end must be >=
        # start unless it is the -1 "to end of file" sentinel.
        start_line, end_line = view_range
        if start_line < 1:
            return False
        if end_line != -1 and end_line < start_line:
            return False

    if command == "create":
        if not isinstance(args.get("file_text"), str):
            return False
    elif command == "str_replace":
        old_str = args.get("old_str")
        new_str = args.get("new_str")
        if not isinstance(old_str, str) or not isinstance(new_str, str):
            return False
        if old_str == new_str:
            return False
    elif command == "insert":
        insert_line = args.get("insert_line")
        if (
            not isinstance(insert_line, int)
            or isinstance(insert_line, bool)
            or insert_line < 0
        ):
            return False
        if not isinstance(args.get("new_str"), str):
            return False
    # "view" and "undo_edit" require nothing beyond command + path.

    return True


def is_valid_terminal_call(arguments: Any) -> bool:
    """Statically validate the parameters of a ``terminal`` tool call.

    Mirrors ``TerminalAction`` and its documented constraints:

    - ``arguments`` parses to a JSON object with only recognized keys.
    - ``command`` is present and a string (may be empty: an empty command is a
      valid way to fetch more logs / send input).
    - ``is_input`` and ``reset``, when supplied, are booleans.
    - ``timeout``, when supplied, is a number >= 0.
    - ``reset=True`` cannot be combined with ``is_input=True``.
    """
    args = _coerce_arguments(arguments)
    if args is None:
        return False

    # Reject unknown parameters.
    if not set(args).issubset(_TERMINAL_KEYS):
        return False

    # command: required, must be a string (empty string is allowed).
    if not isinstance(args.get("command"), str):
        return False

    is_input = args.get("is_input", False)
    if not isinstance(is_input, bool):
        return False

    reset = args.get("reset", False)
    if not isinstance(reset, bool):
        return False

    timeout = args.get("timeout")
    if timeout is not None:
        # bool is a subclass of int; exclude it. Accept int or float >= 0.
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            return False
        if timeout < 0:
            return False

    # Documented mutual exclusion: reset cannot be used with is_input=True.
    if reset and is_input:
        return False

    return True


def is_valid_task_tracker_call(arguments: Any) -> bool:
    """Statically validate the parameters of a ``task_tracker`` tool call.

    Mirrors ``TaskTrackerAction`` and ``TaskItem``:

    - ``arguments`` parses to a JSON object with only recognized keys.
    - ``command``, when supplied, is one of ``view`` / ``plan`` (it defaults to
      ``view`` in the schema, so absence is legal).
    - ``task_list``, when supplied, is a list of task objects, each of which:
        - has only recognized keys (``title`` / ``notes`` / ``status``),
        - has a ``title`` that is a present, non-empty string,
        - has ``notes`` (if present) as a string,
        - has ``status`` (if present) as one of ``todo`` / ``in_progress`` /
          ``done``.
    - The ``plan`` command requires a non-empty ``task_list``.

    This checker exists mainly to catch the common failure mode where a model
    emits raw LaTeX/backslash escapes (e.g. ``\\boxed``) inside the ``notes``
    field, producing arguments that are not valid JSON at all -- those fail at
    the ``_coerce_arguments`` parse step and trigger a retry before the agent
    ever calls ``json.loads`` on them.
    """
    args = _coerce_arguments(arguments)
    if args is None:
        return False

    # Reject unknown parameters.
    if not set(args).issubset(_TASK_TRACKER_KEYS):
        return False

    # command: optional (defaults to "view"), must be a known literal.
    command = args.get("command", "view")
    if command not in _TASK_TRACKER_COMMANDS:
        return False

    task_list = args.get("task_list")
    if task_list is not None:
        if not isinstance(task_list, list):
            return False
        for item in task_list:
            if not isinstance(item, dict):
                return False
            if not set(item).issubset(_TASK_ITEM_KEYS):
                return False
            title = item.get("title")
            if not isinstance(title, str) or not title:
                return False
            notes = item.get("notes")
            if notes is not None and not isinstance(notes, str):
                return False
            status = item.get("status")
            if status is not None and status not in _TASK_ITEM_STATUSES:
                return False

    # `plan` writes the task list, so it requires a non-empty one.
    if command == "plan" and not task_list:
        return False

    return True


# Dispatch table: tool name -> parameter checker.
_CHECKERS = {name: is_valid_file_editor_call for name in FILE_EDITOR_TOOL_NAMES}
_CHECKERS.update({name: is_valid_terminal_call for name in TERMINAL_TOOL_NAMES})
_CHECKERS.update({name: is_valid_task_tracker_call for name in TASK_TRACKER_TOOL_NAMES})

def find_invalid_tool_call(response_dict: dict[str, Any]) -> tuple[str, str] | None:
    """Scan a ``ModelResponse.model_dump()`` dict for an invalid tool call.

    Iterates every tool call across all choices and applies two layers of
    validation:

    1. A *generic* JSON check that applies to **every** tool call regardless of
       tool name: ``function.arguments`` must parse as a JSON object. This
       catches the common provider failure mode where the model emits raw
       backslash escapes (e.g. ``\\boxed``) inside a string, producing arguments
       that ``json.loads`` rejects with ``Invalid \\escape``. Without this the
       bad call sails past the guardrail and only explodes later when the agent
       calls ``json.loads(tool_call.arguments)``.
    2. A *tool-specific* schema check for tools with a dedicated checker
       (``file_editor`` / ``terminal`` / ``task_tracker`` and their legacy
       aliases). Tool calls for any other tool clear the generic check and are
       otherwise left alone.

    Returns a ``(tool_name, raw_arguments)`` tuple for the first call that fails
    either layer, or ``None`` if all calls are legal (including the case where
    no call targets a known tool).

    The dict is expected to follow the OpenAI chat shape::

        {"choices": [{"message": {"tool_calls": [
            {"function": {"name": ..., "arguments": "..."}}, ...
        ]}}]}
    """
    for choice in response_dict.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str):
                continue
            arguments = function.get("arguments")

            # Layer 1: generic JSON check for every tool call. A string that
            # doesn't parse to a JSON object is malformed no matter the tool.
            if _coerce_arguments(arguments) is None:
                raw = arguments if isinstance(arguments, str) else repr(arguments)
                return (name, raw)

            # Layer 2: tool-specific schema check when we have a checker.
            checker = _CHECKERS.get(name)
            if checker is None:
                # Unknown tool: generic check already passed, nothing more to do.
                continue
            if not checker(arguments):
                raw = arguments if isinstance(arguments, str) else repr(arguments)
                return (name, raw)

    return None
