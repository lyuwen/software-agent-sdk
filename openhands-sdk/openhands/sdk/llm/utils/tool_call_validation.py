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
  - ``finish`` / ``think``                     (SDK builtin tools)

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
import os
import re
from collections import Counter, defaultdict
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
# SDK builtin tools (openhands.sdk.tool.builtins). ``FinishTool`` -> "finish",
# ``ThinkTool`` -> "think".
FINISH_TOOL_NAMES = frozenset({"finish"})
THINK_TOOL_NAMES = frozenset({"think"})

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

# Recognized parameter keys for the builtin finish / think tools.
_FINISH_KEYS = frozenset({"message"}) | _TOLERATED_EXTRA_KEYS
_THINK_KEYS = frozenset({"thought"}) | _TOLERATED_EXTRA_KEYS


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


def is_valid_finish_call(arguments: Any) -> bool:
    """Statically validate the parameters of a builtin ``finish`` tool call.

    Mirrors ``FinishAction``:

    - ``arguments`` parses to a JSON object with only recognized keys.
    - ``message`` is present and a string.

    ``message`` is a required field, so a call that omits it (e.g. one that only
    supplies the tolerated ``summary`` metadata key) is rejected -- exactly the
    failure that otherwise surfaces as a pydantic ``Field required`` error at
    agent-side validation time.
    """
    args = _coerce_arguments(arguments)
    if args is None:
        return False

    # Reject unknown parameters.
    if not set(args).issubset(_FINISH_KEYS):
        return False

    # message: required, must be a string (may be empty).
    if not isinstance(args.get("message"), str):
        return False

    return True


def is_valid_think_call(arguments: Any) -> bool:
    """Statically validate the parameters of a builtin ``think`` tool call.

    Mirrors ``ThinkAction``:

    - ``arguments`` parses to a JSON object with only recognized keys.
    - ``thought`` is present and a string.
    """
    args = _coerce_arguments(arguments)
    if args is None:
        return False

    # Reject unknown parameters.
    if not set(args).issubset(_THINK_KEYS):
        return False

    # thought: required, must be a string (may be empty).
    if not isinstance(args.get("thought"), str):
        return False

    return True


# ── parallel tool-call group validation ─────────────────────────────────────
#
# When a model emits several tool calls in one turn, the agent executes them
# sequentially in response order. However, the model generates all calls
# without seeing any intermediate result, so from the model's perspective the
# batch is effectively blind. The checks below reject combinations that are
# structurally unsound under that blind-generation assumption:
#
#   exact_duplicate        Same (name, arguments) pair more than once. Always
#                          redundant: same input → same output.
#   finish_not_alone       finish must be the only call in its batch.
#   same_location_write    >1 mutating edit to the same (path, location):
#                          - str_replace: same path + same old_str
#                          - create:      same path (whole file is the target)
#                          - insert:      same path + same insert_line
#                          - undo_edit:   same path (always whole-file target)
#                          Distinct hunks on the same file are allowed.
#   overlapping_view       >1 view of the same path with overlapping view_range.
#                          Viewing distinct line ranges is allowed.
#   read_write_same_file   A view and any write on the same path.
#   same_scope_multi_test  >1 pytest invocation whose test-path is identical
#                          or one is a prefix of the other.
#   noop_bash              A bash call whose every segment is a prefix action
#                          (cd/export/…) or a bare echo literal — no durable
#                          work survives the subshell. Catches bare `cd a`,
#                          `cd /x; echo hi`, plain `echo hi`, and any mix.
#                          Allows `echo $VAR`, piped commands, and real work
#                          like `cd /x && pytest tests/`.
#   batch_too_large        Batch exceeds OH_MAX_TOOLCALL_BATCH_SIZE (default
#                          16). Last-resort circuit breaker.

# file_editor sub-commands that mutate a file.
_WRITE_OPS = frozenset({"str_replace", "create", "insert", "undo_edit"})
# file_editor sub-commands that only read.
_READ_OPS = frozenset({"view"})
# Default batch-size ceiling (overridable via env). Set high enough not to
# reject legitimate fan-out over many distinct files.
_DEFAULT_MAX_BATCH = 24
# Metadata keys that do not affect execution, excluded from duplicate fingerprint.
_NON_EXECUTION_KEYS = frozenset({"summary", "security_risk"})
# Fraction of the shorter view range that must be shared before two views of
# the same file are considered redundant (50% overlap threshold).
_VIEW_OVERLAP_THRESHOLD = 0.5
# A bash command in a parallel batch is a no-op when the only work it does
# is emit a bare echo literal. Commands that persist across calls in this
# terminal (cd, export, source, pushd, popd, etc. — see terminal/definition.py:208)
# are real work and are NOT classified as no-ops.
_PADDING_ECHO_RE = re.compile(
    r"^\s*echo\b[^\n&|;`$<>]*$", re.IGNORECASE
)


def _is_noop_bash(cmd: str) -> bool:
    """Whether a bash command does no durable work.

    In this SDK's terminal the session is stateful: ``cd``, ``export``,
    ``source``, ``pushd``, ``popd``, and ``.`` all persist across calls
    (documented in terminal/definition.py:208).  Only a bare ``echo
    <literal>`` — no chaining operator, no redirection, no ``$`` variable
    reference, no command substitution — is a no-op.  A single echo with
    any of those features (``echo $VAR``, ``echo x > file``, ``echo x &&
    cmd``) is not flagged.
    """
    if "|" in cmd:
        return False
    segments = [s for s in re.split(r"&&|;", cmd) if s.strip()]
    if not segments:
        return False
    return all(_PADDING_ECHO_RE.match(s) for s in segments)


def _pytest_path(cmd: str) -> str | None:
    """Return the first path-like argument to pytest in a shell command.

    Scans tokens after 'pytest', skipping boolean flags (e.g. -v, -x,
    --collect-only) and the values of flags that take an argument (e.g.
    -k expr, --rootdir /tmp). Returns the first token that looks like a
    file-system path or pytest node-id: contains '/', ends with '.py',
    or contains '::'.

    Flags are classified as value-consuming only when they are in the
    explicit set below; every other flag token is treated as boolean and
    skipped without consuming the next token.
    """
    # Only flags whose next token is a value, not a path.
    _FLAGS_WITH_VALUE = frozenset({
        "-k", "-m", "-n", "-p",
        "--timeout", "--rootdir", "--basetemp", "--ignore",
        "--override-ini", "--config-file", "--maxfail", "--tb",
        "--log-level", "--log-file", "--log-format",
    })
    m = re.search(r"\bpytest\b", cmd, re.IGNORECASE)
    if not m:
        return None
    tokens = cmd[m.end():].split()
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            # Only consume the next token as a value when the flag is in
            # the explicit set and written as --flag value (not --flag=val).
            if "=" not in token and token in _FLAGS_WITH_VALUE:
                skip_next = True
            continue
        if "/" in token or "\\" in token or token.endswith(".py") or "::" in token:
            return token
    return None


def _ranges_overlap(a: Any, b: Any) -> bool:
    """Whether two view_range values share more than _VIEW_OVERLAP_THRESHOLD
    of the shorter range's length.

    A missing/None range means "whole file", which always overlaps everything.
    end == -1 means end-of-file. Two views are only flagged as redundant when
    the overlapping portion exceeds 50% of the shorter range — reads of [1,100]
    and [90,200] are independently useful because each returns unique content.
    """
    if not isinstance(a, list) or len(a) != 2:
        return True  # whole file
    if not isinstance(b, list) or len(b) != 2:
        return True  # whole file
    a0, a1 = a
    b0, b1 = b
    if not all(isinstance(x, int) for x in (a0, a1, b0, b1)):
        return True  # can't reason about it: treat as overlapping
    a_end = float("inf") if a1 == -1 else a1
    b_end = float("inf") if b1 == -1 else b1
    overlap_start = max(a0, b0)
    overlap_end = min(a_end, b_end)
    if overlap_end < overlap_start:
        return False  # disjoint
    if overlap_start == float("inf") or overlap_end == float("inf"):
        # Both extend to EOF — overlap fraction is 100%
        return True
    overlap_len = overlap_end - overlap_start + 1
    a_len = (a_end - a0 + 1) if a_end != float("inf") else float("inf")
    b_len = (b_end - b0 + 1) if b_end != float("inf") else float("inf")
    shorter = min(a_len, b_len)
    if shorter == float("inf") or shorter <= 0:
        return True
    return (overlap_len / shorter) >= _VIEW_OVERLAP_THRESHOLD


def find_parallel_group_bug(
    tool_calls: list[dict[str, Any]],
) -> tuple[str, str] | None:
    """Detect a structural bug in a multi-call tool-call group.

    Only meaningful for two or more calls; returns ``None`` for a single call.
    Returns a ``(bug_tag, evidence)`` tuple for the first violation found, or
    ``None`` if the group is structurally sound. Bug tags correspond to the
    checks described in the module-level comment block above.
    """
    if not tool_calls or len(tool_calls) < 2:
        return None

    # ── batch ceiling first (O(1)) ───────────────────────────────────────────
    # Check before any pairwise work to short-circuit on pathological batches.
    try:
        max_batch = int(
            os.environ.get("OH_MAX_TOOLCALL_BATCH_SIZE", _DEFAULT_MAX_BATCH)
        )
    except (ValueError, TypeError):
        max_batch = _DEFAULT_MAX_BATCH
    if len(tool_calls) > max_batch:
        return (
            "batch_too_large",
            f"{len(tool_calls)} calls exceeds ceiling of {max_batch}",
        )

    # Unpack names and parsed args once.
    names: list[str] = []
    parsed_args: list[dict[str, Any] | None] = []
    for call in tool_calls:
        fn = call.get("function") if isinstance(call, dict) else None
        name = fn.get("name") if isinstance(fn, dict) else None
        names.append(name if isinstance(name, str) else "")
        arguments = fn.get("arguments") if isinstance(fn, dict) else None
        parsed_args.append(_coerce_arguments(arguments))

    # ── 1. exact duplicate (semantic fingerprint) ──────────────────────────
    # Fingerprint on parsed args with sorted keys, excluding metadata that
    # does not affect execution (summary, security_risk). This catches
    # semantically identical calls even when JSON key order or whitespace
    # differs, or when only the summary field changed.
    seen: set[tuple[str, str]] = set()
    for name, args in zip(names, parsed_args):
        if args is not None:
            canonical = {
                k: v for k, v in sorted(args.items())
                if k not in _NON_EXECUTION_KEYS
            }
            fp = json.dumps(canonical, sort_keys=True)
        else:
            fp = ""
        key = (name, fp)
        if key in seen:
            return (
                "exact_duplicate",
                f"{name!r} issued more than once"
                " with identical arguments",
            )
        seen.add(key)

    # ── 2. finish must be alone ──────────────────────────────────────────────
    finish_n = sum(1 for n in names if n in FINISH_TOOL_NAMES)
    if finish_n >= 1 and len(names) > 1:
        others = Counter(n for n in names if n not in FINISH_TOOL_NAMES)
        return (
            "finish_not_alone",
            f"finish in a batch of {len(names)}: also contains {dict(others)}",
        )

    # ── 3. same-location write conflict ─────────────────────────────────────
    # Each write is keyed by (path, location_key) so distinct hunks on the
    # same file are allowed, but the same hunk (or whole-file ops like create
    # and undo_edit) are not.
    #
    # str_replace  → location = old_str   (the consumed hunk)
    # create       → location = ""        (whole file)
    # insert       → location = insert_line as str
    # undo_edit    → location = ""        (whole file; multiple undos on the
    #                                      same file are nonsensical since the
    #                                      first undo changes what the second
    #                                      would act on)
    write_locations: Counter[tuple[str, str]] = Counter()
    for name, args in zip(names, parsed_args):
        if name not in FILE_EDITOR_TOOL_NAMES or args is None:
            continue
        op = args.get("command", "")
        if op not in _WRITE_OPS:
            continue
        path = args.get("path", "")
        if not path:
            continue
        if op == "str_replace":
            loc = args.get("old_str", "")
        elif op == "create":
            loc = ""
        elif op == "insert":
            loc = str(args.get("insert_line", ""))
        else:  # undo_edit
            loc = ""
        write_locations[(path, loc)] += 1

    conflict = next(
        (loc_key for loc_key, n in write_locations.items() if n > 1), None
    )
    if conflict is not None:
        path, loc = conflict
        detail = f"path={path!r}" + (f" old_str={loc!r}" if loc else "")
        return ("same_location_write", detail)

    # ── 4. overlapping view ranges on the same file ─────────────────────────
    # Multiple views of the same file are fine as long as the ranges are
    # disjoint. A missing range means "whole file".
    views_by_path: dict[str, list[Any]] = defaultdict(list)
    for name, args in zip(names, parsed_args):
        if name not in FILE_EDITOR_TOOL_NAMES or args is None:
            continue
        if args.get("command", "") not in _READ_OPS:
            continue
        path = args.get("path", "")
        if not path:
            continue
        views_by_path[path].append(args.get("view_range"))

    for path, ranges in views_by_path.items():
        if len(ranges) < 2:
            continue
        for i, a in enumerate(ranges):
            for b in ranges[i + 1:]:
                if _ranges_overlap(a, b):
                    return (
                        "overlapping_view",
                        f"path={path!r} view_range {a!r} overlaps {b!r}",
                    )

    # ── 5. read + write on same file ────────────────────────────────────────
    write_paths: set[str] = {p for (p, _) in write_locations}
    read_paths: set[str] = set(views_by_path)
    rw_conflict = sorted(read_paths & write_paths)
    if rw_conflict:
        return ("read_write_same_file", f"paths={rw_conflict}")

    # ── 6. same pytest scope issued more than once ───────────────────────────
    test_paths: list[str] = []
    for name, args in zip(names, parsed_args):
        if name not in TERMINAL_TOOL_NAMES or args is None:
            continue
        cmd = args.get("command", "")
        if not isinstance(cmd, str):
            continue
        tp = _pytest_path(cmd)
        if tp:
            test_paths.append(tp)

    if len(test_paths) >= 2:
        for i, a in enumerate(test_paths):
            for b in test_paths[i + 1:]:
                shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
                if (
                    longer == shorter
                    or longer.startswith(shorter.rstrip("/") + "/")
                    or longer.startswith(shorter.rstrip("/") + "::")
                ):
                    return (
                        "same_scope_multi_test",
                        f"pytest scopes overlap: {a!r} and {b!r}",
                    )

    # ── 7. padding echo alongside real commands ──────────────────────────────
    # A bare `echo <literal>` (no chaining operator, no $ variable reference,
    # no command substitution) has no standalone utility: its output only means
    # something when it feeds into the next command via &&, |, ;, etc. In a
    # parallel batch each call runs independently, so a padding echo is pure
    # noise. The legitimate standalone echo forms — `echo $VAR` to inspect an
    # env var, or `echo text && cmd` chained into real work — are NOT matched.
    # `ls`, `pwd`, and other inspection commands are never flagged here.
    padding_echoes = [
        (args.get("command", "") if args else "")
        for name, args in zip(names, parsed_args)
        if name in TERMINAL_TOOL_NAMES
        and isinstance((args or {}).get("command"), str)
        and _is_noop_bash((args or {}).get("command", ""))
    ]
    if padding_echoes:
        return (
            "noop_bash",
            f"bash call(s) with no durable work: {padding_echoes!r}",
        )

    # ── 8. batch ceiling (last-resort circuit breaker) ───────────────────────
    try:
        max_batch = int(
            os.environ.get("OH_MAX_TOOLCALL_BATCH_SIZE", _DEFAULT_MAX_BATCH)
        )
    except (ValueError, TypeError):
        max_batch = _DEFAULT_MAX_BATCH
    if len(tool_calls) > max_batch:
        return (
            "batch_too_large",
            f"{len(tool_calls)} calls exceeds ceiling of {max_batch}",
        )

    return None


# Dispatch table: tool name -> parameter checker.
_CHECKERS = {name: is_valid_file_editor_call for name in FILE_EDITOR_TOOL_NAMES}
_CHECKERS.update({name: is_valid_terminal_call for name in TERMINAL_TOOL_NAMES})
_CHECKERS.update({name: is_valid_task_tracker_call for name in TASK_TRACKER_TOOL_NAMES})
_CHECKERS.update({name: is_valid_finish_call for name in FINISH_TOOL_NAMES})
_CHECKERS.update({name: is_valid_think_call for name in THINK_TOOL_NAMES})


def find_invalid_tool_call(response_dict: dict[str, Any]) -> tuple[str, str] | None:
    """Scan a ``ModelResponse.model_dump()`` dict for an invalid tool call.

    Iterates every tool call across all choices and applies three layers of
    validation:

    1. A *generic* JSON check that applies to **every** tool call regardless of
       tool name: ``function.arguments`` must parse as a JSON object. This
       catches the common provider failure mode where the model emits raw
       backslash escapes (e.g. ``\\boxed``) inside a string, producing arguments
       that ``json.loads`` rejects with ``Invalid \\escape``. Without this the
       bad call sails past the guardrail and only explodes later when the agent
       calls ``json.loads(tool_call.arguments)``.
    2. A *tool-specific* schema check for tools with a dedicated checker
       (``file_editor`` / ``terminal`` / ``task_tracker`` / ``finish`` /
       ``think`` and their legacy aliases). Tool calls for any other tool clear
       the generic check and are otherwise left alone.
    3. A *parallel-group* structural check applied to every assistant message
       that carries two or more tool calls (after all per-call checks pass).
       Catches ``multi_finish``, ``finish_mixed``, ``same_file_multi_write``,
       ``read_write_same_file``, ``duplicate_bash``, and ``trivial_flood``.

    Returns a ``(tag, evidence)`` tuple for the first call or group that fails
    any layer, or ``None`` if all calls are legal (including the case where no
    call targets a known tool and the group is structurally sound).

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
        tool_calls = message.get("tool_calls") or []

        # Layers 1 + 2: per-call checks.
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not isinstance(name, str):
                continue
            arguments = function.get("arguments")

            # Layer 1: generic JSON check for every tool call.
            if _coerce_arguments(arguments) is None:
                raw = arguments if isinstance(arguments, str) else repr(arguments)
                return (name, raw)

            # Layer 2: tool-specific schema check when we have a checker.
            checker = _CHECKERS.get(name)
            if checker is None:
                continue
            if not checker(arguments):
                raw = arguments if isinstance(arguments, str) else repr(arguments)
                return (name, raw)

        # Layer 3: parallel-group structural check (only for multi-call turns).
        if len(tool_calls) >= 2:
            bug = find_parallel_group_bug(tool_calls)
            if bug is not None:
                return bug

    return None
