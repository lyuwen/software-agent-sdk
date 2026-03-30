"""LSP prompt & CLI helpers — reusable across benchmarks.

Provides:
- ``add_lsp_args(parser)`` — adds ``--lsp-naming`` to any argparser
- ``apply_lsp_naming(naming, forward_env)`` — sets env var + Docker forwarding
- ``get_lsp_command_names(naming)`` — builds the Jinja2 ``cmd`` dict
- ``get_default_lsp_prompt_path()`` — path to the bundled prompt template
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from openhands.tools.lsp.lsp_tool import COMMAND_NAMES


# ---------------------------------------------------------------------------
# CLI argument helper
# ---------------------------------------------------------------------------
def add_lsp_args(parser: argparse.ArgumentParser) -> None:
    """Add LSP-related CLI arguments to *parser*.

    Adds ``--lsp-naming`` with choices ``snake_case`` and ``camelCase``.
    """
    parser.add_argument(
        "--lsp-naming",
        type=str,
        default="camelCase",
        choices=["snake_case", "camelCase"],
        help="Naming convention for LSP tool commands (default: camelCase)",
    )


# ---------------------------------------------------------------------------
# Naming convention application
# ---------------------------------------------------------------------------
def apply_lsp_naming(
    lsp_naming: str,
    forward_env: list[str] | None = None,
) -> list[str]:
    """Set the ``LSP_NAMING`` env var and add it to *forward_env*.

    This controls three things in one call:

    1. **Tool definition** — ``definition.py`` reads ``LSP_NAMING`` at schema-
       generation time, so the command enum reflects the chosen convention.
    2. **Prompt template** — :func:`get_lsp_command_names` uses the same env
       var as its default, so Jinja2 templates render the right command names.
    3. **Docker forwarding** — the env var is added to *forward_env* so it is
       forwarded into the container.

    Must be called **before** workspace creation.

    Returns the (possibly modified) *forward_env* list.
    """
    os.environ["LSP_NAMING"] = lsp_naming
    forward_env = list(forward_env or [])
    if "LSP_NAMING" not in forward_env:
        forward_env.append("LSP_NAMING")
    return forward_env


# ---------------------------------------------------------------------------
# Jinja2 context helper
# ---------------------------------------------------------------------------
def get_lsp_command_names(naming: str | None = None) -> dict[str, str]:
    """Build a ``{snake_case: display_name}`` dict for Jinja2 templates.

    When *naming* is ``"camelCase"``, display names are camelCase.
    Otherwise, display names equal the snake_case keys.

    Usage in a Jinja2 template::

        context["cmd"] = get_lsp_command_names(naming)
        # then {{ cmd.get_definition }} renders as the display name
    """
    if naming is None:
        naming = os.environ.get("LSP_NAMING", "camelCase")
    if naming == "camelCase":
        return {k: v for k, v in COMMAND_NAMES.items()}
    return {k: k for k in COMMAND_NAMES}


# ---------------------------------------------------------------------------
# Prompt template path
# ---------------------------------------------------------------------------
def get_default_lsp_prompt_path() -> Path:
    """Return the path to the bundled ``default_lsp.j2`` prompt template."""
    return Path(__file__).parent / "prompts" / "default_lsp.j2"
