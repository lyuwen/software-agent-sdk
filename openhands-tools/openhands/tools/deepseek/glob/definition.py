"""DeepSeek-harness-compatible glob tool.

Model-facing name: ``glob``
Wraps GlobExecutor but changes behavior to match DSH:
  - rg --no-ignore --hidden with VCS dir exclusions (includes hidden/ignored files)
  - Cap at 100 paths; overflow reports where complete list was saved (omitted here
    since we have no spill store; mirrors text that fits inline)
  - path resolves relative to working_dir, not process cwd
"""

from __future__ import annotations

import os
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


# VCS metadata directories ripgrep must never descend into.
_VCS_EXCLUDES: tuple[str, ...] = ('.git', '.svn', '.hg', '.bzr', '.jj', '.sl')

_GLOB_MAX_RESULTS = 100

# Description exactly as it appears in dsh-tools.json (dynamically formatted).
_TOOL_DESCRIPTION = (
    "Find files whose paths match a glob pattern. Returns matching file paths — never directories — "
    "including hidden and ignored files (VCS metadata directories are excluded). "
    f"Up to {_GLOB_MAX_RESULTS} paths come back in modification-time order; "
    f"a larger result returns the first {_GLOB_MAX_RESULTS} paths in modification-time order, "
    "says so, and reports where the complete sorted list was saved. "
    "This tool does not enumerate directory entries."
)


class DeepSeekGlobAction(Action):
    """Input schema for the DeepSeek-compatible ``glob`` tool."""

    pattern: str = Field(
        description=(
            'Glob pattern to match file paths against (e.g. "**/*.ts", "src/**/*.test.js"). '
            'A pattern with no "/" matches the basename at any depth, so "*" and "*.ts" both '
            'search the whole tree; include a separator to anchor the depth.'
        ),
    )
    path: str | None = Field(
        default=None,
        description="Directory to search in. Defaults to the session workspace; a relative path resolves against it.",
    )


class DeepSeekGlobObservation(Observation):
    """Output schema for the DeepSeek-compatible ``glob`` tool."""
    pass


def _build_glob_argv(pattern: str, search_path: str) -> list[str]:
    """Build rg --files argv matching DSH's buildGlobCommand."""
    parts = [
        '--files',
        f'--glob={pattern}',
        '--sort=modified',
        '--no-ignore',
        '--hidden',
        # Two negated globs per VCS dir: bare form prunes during traversal,
        # /**  form excludes contents when root is AT/INSIDE the dir.
        *[arg for name in _VCS_EXCLUDES
          for arg in (f'--glob=!**/{name}', f'--glob=!**/{name}/**')],
        '--',
        search_path,
    ]
    return parts


class DeepSeekGlobExecutor(ToolExecutor):
    """Executor for the DeepSeek-compatible glob tool."""

    def __init__(self, working_dir: str) -> None:
        self.working_dir = Path(working_dir).resolve()

    def __call__(
        self,
        action: DeepSeekGlobAction,
        conversation=None,  # noqa: ARG002
    ) -> DeepSeekGlobObservation:
        pattern = action.pattern.strip()
        if not pattern:
            return DeepSeekGlobObservation.from_text(
                text="pattern must be a non-empty string", is_error=True
            )

        # Resolve search path: relative paths are resolved against working_dir.
        if action.path is not None:
            raw = action.path.strip()
            if not raw:
                return DeepSeekGlobObservation.from_text(
                    text="path must be a non-empty string when given", is_error=True
                )
            search_path = (
                Path(raw) if Path(raw).is_absolute() else self.working_dir / raw
            ).resolve()
        else:
            search_path = self.working_dir

        if not search_path.is_dir():
            return DeepSeekGlobObservation.from_text(
                text=f"path '{search_path}' is not a valid directory", is_error=True
            )

        try:
            import shutil
            rg = shutil.which('rg')
            if rg is None:
                return self._python_fallback(pattern, search_path)

            argv = [rg, '--no-config'] + _build_glob_argv(pattern, str(search_path))
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                env=sanitized_env(),
            )

            if result.returncode not in (0, 1):
                # exit 1 = no matches; anything else = error
                return DeepSeekGlobObservation.from_text(
                    text=f"glob search failed: {result.stderr.strip()}", is_error=True
                )

            if result.returncode == 1 or not result.stdout.strip():
                return DeepSeekGlobObservation.from_text(text="No files found")

            paths = [
                self._display_path(line, search_path)
                for line in result.stdout.splitlines()
                if line
            ]

            if len(paths) <= _GLOB_MAX_RESULTS:
                return DeepSeekGlobObservation.from_text(text="\n".join(paths))

            inline = paths[:_GLOB_MAX_RESULTS]
            return DeepSeekGlobObservation.from_text(
                text=(
                    "\n".join(inline)
                    + f"\n\n(Showing {_GLOB_MAX_RESULTS} of {len(paths)} paths."
                    " The complete result could not be saved; narrow pattern or path to see more.)"
                )
            )

        except subprocess.TimeoutExpired:
            return DeepSeekGlobObservation.from_text(
                text="glob search timed out after 30 seconds", is_error=True
            )
        except Exception as e:
            return DeepSeekGlobObservation.from_text(text=str(e), is_error=True)

    def _display_path(self, raw: str, search_path: Path) -> str:
        """Return a path relative to working_dir when possible."""
        p = Path(raw)
        if not p.is_absolute():
            return raw
        try:
            return str(p.relative_to(self.working_dir))
        except ValueError:
            return raw

    def _python_fallback(self, pattern: str, search_path: Path) -> DeepSeekGlobObservation:
        """Pure-Python glob fallback when rg is unavailable."""
        import glob as glob_module

        original_cwd = os.getcwd()
        try:
            os.chdir(search_path)
            if '**' not in pattern:
                pattern = f'**/{pattern}'
            matches = glob_module.glob(pattern, recursive=True)
            file_paths = sorted(
                (
                    (str((search_path / m).absolute()), os.path.getmtime(str((search_path / m).absolute())))
                    for m in matches
                    if os.path.isfile(str((search_path / m).absolute()))
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            paths = [self._display_path(p, search_path) for p, _ in file_paths]
        finally:
            os.chdir(original_cwd)

        if not paths:
            return DeepSeekGlobObservation.from_text(text="No files found")
        if len(paths) <= _GLOB_MAX_RESULTS:
            return DeepSeekGlobObservation.from_text(text="\n".join(paths))
        inline = paths[:_GLOB_MAX_RESULTS]
        return DeepSeekGlobObservation.from_text(
            text=(
                "\n".join(inline)
                + f"\n\n(Showing {_GLOB_MAX_RESULTS} of {len(paths)} paths."
                " The complete result could not be saved; narrow pattern or path to see more.)"
            )
        )


class DeepSeekGlobTool(ToolDefinition[DeepSeekGlobAction, DeepSeekGlobObservation]):
    """DeepSeek-harness-compatible ``glob`` tool."""

    name: ClassVar[str] = "glob"

    @classmethod
    def create(
        cls,
        conv_state: "ConversationState",
        executor: ToolExecutor | None = None,
    ) -> Sequence["DeepSeekGlobTool"]:
        working_dir = conv_state.workspace.working_dir
        if not os.path.isdir(working_dir):
            raise ValueError(f"working_dir '{working_dir}' is not a valid directory")

        return [
            cls(
                action_type=DeepSeekGlobAction,
                observation_type=DeepSeekGlobObservation,
                description=_TOOL_DESCRIPTION,
                annotations=ToolAnnotations(
                    title="glob",
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=False,
                ),
                executor=executor or DeepSeekGlobExecutor(working_dir),
            )
        ]


register_tool(DeepSeekGlobTool.name, DeepSeekGlobTool)
