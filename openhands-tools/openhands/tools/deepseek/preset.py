"""DeepSeek-harness-compatible minimum preset.

Provides exactly: bash, str_replace_editor
"""

from openhands.sdk.tool import Tool


def get_deepseek_minimal_tools() -> list[Tool]:
    """Return the DeepSeek-harness minimum tool set.

    Produces exactly ``bash`` and ``str_replace_editor`` — no finish tool,
    no task tracker, no browser.  The returned list is ordered to match the
    DSH minimum preset: bash first, then str_replace_editor.

    These are registry-reference specs (``Tool(name=…)``), not instantiated
    tools.  Executors are created later by the conversation framework when it
    calls ``ToolDefinition.create(conv_state)`` for each spec.

    Returns:
        List of two :class:`Tool` spec objects ready to pass to :class:`Agent`.
    """
    # Import triggers registration side effects.
    from openhands.tools.deepseek.bash.definition import DeepSeekBashTool
    from openhands.tools.deepseek.str_replace_editor.definition import (
        DeepSeekStrReplaceEditorTool,
    )

    return [
        Tool(name=DeepSeekBashTool.name),
        Tool(name=DeepSeekStrReplaceEditorTool.name),
    ]
