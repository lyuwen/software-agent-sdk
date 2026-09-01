"""DeepSeek-harness-compatible grep tool.

Model-facing name: ``grep``
Key differences from native GrepTool:
  - Returns matching LINES with line numbers grouped by file (not file paths only)
  - Case-sensitive by default (no -i flag)
  - path may be a file OR directory
  - Cap at 250 matches (DSH default); overflow reports spill location
  - Uses rg --json for unambiguous parsing
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
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
from openhands.sdk.utils import sanitized_env


_GREP_MAX_MATCHES = 250
_GREP_MAX_LINE_BYTES = 2000

_TOOL_DESCRIPTION = (
    "Search file contents with a ripgrep regular expression. "
    "Returns matching lines with line numbers, grouped by file. "
    f"Returns the first {_GREP_MAX_MATCHES} matches inline; "
    "a capped result reports where the complete match list was saved. "
    "Use read on a matched file for surrounding context."
)


class DeepSeekGrepAction(Action):
    """Input schema for the DeepSeek-compatible ``grep`` tool."""

    pattern: str = Field(
        description="Regular expression to search for (ripgrep syntax).",
    )
    path: str | None = Field(
        default=None,
        description="File or directory to search. Defaults to the session workspace; a relative path resolves against it.",
    )
    include: str | None = Field(
        default=None,
        description='One glob filter for which files to search (e.g. "*.ts", "*.{js,jsx}"). Not a list; negation is not supported.',
    )


class DeepSeekGrepObservation(Observation):
    """Output schema for the DeepSeek-compatible ``grep`` tool."""
    pass


def _preview_line(line: str, max_bytes: int = _GREP_MAX_LINE_BYTES) -> str:
    """Truncate a matched line to max_bytes (UTF-8 boundary preserved)."""
    encoded = line.encode('utf-8')
    if len(encoded) <= max_bytes:
        return line
    truncated = encoded[:max_bytes].decode('utf-8', errors='ignore')
    return f"{truncated} (line truncated)"


def _parse_rg_json(stdout: str, workdir: Path) -> list[tuple[str, int, str]]:
    """Parse rg --json NDJSON output into (display_path, line_number, line) tuples."""
    matches: list[tuple[str, int, str]] = []
    for line in stdout.splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get('type') != 'match':
            continue
        data = record.get('data', {})
        path_text = data.get('path', {}).get('text', '')
        line_number = data.get('line_number')
        lines_data = data.get('lines', {})
        line_text = lines_data.get('text', '(line is not valid UTF-8)') if 'text' in lines_data else '(line is not valid UTF-8)'
        if not path_text or line_number is None:
            continue
        # Display path: relative to workdir when possible
        p = Path(path_text)
        if p.is_absolute():
            try:
                display = str(p.relative_to(workdir))
            except ValueError:
                display = path_text
        else:
            display = path_text
        matches.append((display, int(line_number), line_text.rstrip('\r\n')))
    return matches


def _format_grep_output(matches: list[tuple[str, int, str]]) -> str:
    """Format matches grouped by file: path\\nLine N: text\\n..."""
    by_file: dict[str, list[tuple[int, str]]] = {}
    for path, lineno, text in matches:
        by_file.setdefault(path, []).append((lineno, text))
    sections = []
    for path, lines in by_file.items():
        body = '\n'.join(f'Line {n}: {_preview_line(t)}' for n, t in lines)
        sections.append(f'{path}\n{body}')
    return '\n\n'.join(sections)


class DeepSeekGrepExecutor(ToolExecutor):
    """Executor for the DeepSeek-compatible grep tool."""

    def __init__(self, working_dir: str) -> None:
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: DeepSeekGrepAction,
        conversation=None,  # noqa: ARG002
    ) -> DeepSeekGrepObservation:
        if not action.pattern:
            return DeepSeekGrepObservation.from_text(
                text="pattern must be a non-empty string", is_error=True
            )

        # Validate include: positive glob only, no negation, no bare comma list.
        if action.include is not None:
            inc = action.include.strip()
            if not inc:
                return DeepSeekGrepObservation.from_text(
                    text="include must be a non-empty glob when given", is_error=True
                )
            if inc.startswith('!'):
                return DeepSeekGrepObservation.from_text(
                    text='include must be a positive glob filter; negated patterns ("!…") are not supported',
                    is_error=True,
                )

        # Resolve search path (file or directory).
        if action.path is not None:
            raw = action.path.strip()
            if not raw:
                return DeepSeekGrepObservation.from_text(
                    text="path must be a non-empty string when given", is_error=True
                )
            search_path = (
                Path(raw) if Path(raw).is_absolute() else self.working_dir / raw
            ).resolve()
            if not search_path.exists():
                return DeepSeekGrepObservation.from_text(
                    text=f"path '{search_path}' does not exist", is_error=True
                )
        else:
            search_path = self.working_dir

        # Validate pattern compiles (catches obviously broken regexes before spawning).
        try:
            re.compile(action.pattern)
        except re.error as e:
            return DeepSeekGrepObservation.from_text(
                text=f"Invalid regex pattern: {e}", is_error=True
            )

        try:
            import shutil
            rg = shutil.which('rg')
            if rg is None:
                return self._python_fallback(action, search_path)

            argv = [rg, '--no-config', '--json', f'--regexp={action.pattern}']
            if action.include:
                argv.append(f'--glob={action.include}')
            argv += ['--', str(search_path)]

            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=sanitized_env(),
            )

            if result.returncode not in (0, 1):
                stderr = result.stderr.strip()
                if 'regex parse error' in stderr.lower() or 'error parsing glob' in stderr.lower():
                    return DeepSeekGrepObservation.from_text(
                        text=f"grep pattern rejected by ripgrep: {stderr}", is_error=True
                    )
                return DeepSeekGrepObservation.from_text(
                    text=f"grep search failed: {stderr}", is_error=True
                )

            if result.returncode == 1 or not result.stdout.strip():
                return DeepSeekGrepObservation.from_text(text="No matches found")

            all_matches = _parse_rg_json(result.stdout, self.working_dir)
            return self._build_output(all_matches)

        except subprocess.TimeoutExpired:
            return DeepSeekGrepObservation.from_text(
                text="grep search timed out after 30 seconds", is_error=True
            )
        except Exception as e:
            return DeepSeekGrepObservation.from_text(text=str(e), is_error=True)

    def _build_output(self, matches: list[tuple[str, int, str]]) -> DeepSeekGrepObservation:
        total = len(matches)
        if total == 0:
            return DeepSeekGrepObservation.from_text(text="No matches found")

        retained = matches[:_GREP_MAX_MATCHES]
        truncated = total > _GREP_MAX_MATCHES

        noun = 'match' if total == 1 else 'matches'
        if truncated:
            header = f"Found {len(retained)} of {total} {noun}"
        else:
            header = f"Found {total} {noun}"

        body = _format_grep_output(retained)

        if not truncated:
            return DeepSeekGrepObservation.from_text(text=f"{header}\n\n{body}")

        footer = "The complete result could not be saved; narrow pattern, path, or include to see more."
        return DeepSeekGrepObservation.from_text(
            text=f"{header}\n\n{body}\n\n({footer})"
        )

    def _python_fallback(
        self, action: DeepSeekGrepAction, search_path: Path
    ) -> DeepSeekGrepObservation:
        """Pure-Python fallback when rg is unavailable."""
        import fnmatch

        try:
            regex = re.compile(action.pattern)
        except re.error as e:
            return DeepSeekGrepObservation.from_text(
                text=f"Invalid regex pattern: {e}", is_error=True
            )

        matches: list[tuple[str, int, str]] = []

        if search_path.is_file():
            paths = [search_path]
        else:
            paths_list = []
            for root, dirs, files in os.walk(search_path):
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                for fname in files:
                    fp = Path(root) / fname
                    if action.include and not fnmatch.fnmatch(fname, action.include):
                        continue
                    paths_list.append(fp)
            paths = paths_list

        for fp in paths:
            try:
                content = fp.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    try:
                        display = str(fp.relative_to(self.working_dir))
                    except ValueError:
                        display = str(fp)
                    matches.append((display, i, line))

        return self._build_output(matches)


class DeepSeekGrepTool(ToolDefinition[DeepSeekGrepAction, DeepSeekGrepObservation]):
    """DeepSeek-harness-compatible ``grep`` tool."""

    name: ClassVar[str] = "grep"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        executor: ToolExecutor | None = None,
    ) -> Sequence["DeepSeekGrepTool"]:
        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        return [
            cls(
                action_type=DeepSeekGrepAction,
                observation_type=DeepSeekGrepObservation,
                description=_TOOL_DESCRIPTION,
                annotations=ToolAnnotations(
                    title="grep",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor or DeepSeekGrepExecutor(working_dir),
            )
        ]


register_tool(DeepSeekGrepTool.name, DeepSeekGrepTool)
