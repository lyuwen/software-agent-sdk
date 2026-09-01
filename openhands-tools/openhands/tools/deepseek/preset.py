"""DeepSeek-harness-compatible presets.

Minimum preset:  bash, str_replace_editor
Standard preset: bash, read, write, edit, glob, grep
"""

from openhands.sdk.tool import Tool


def get_deepseek_minimal_tools() -> list[Tool]:
    """Return the DSH minimum tool set: bash + str_replace_editor."""
    from openhands.tools.deepseek.bash.definition import DeepSeekBashTool
    from openhands.tools.deepseek.str_replace_editor.definition import (
        DeepSeekStrReplaceEditorTool,
    )

    return [
        Tool(name=DeepSeekBashTool.name),
        Tool(name=DeepSeekStrReplaceEditorTool.name),
    ]


def get_deepseek_standard_tools() -> list[Tool]:
    """Return the DSH standard tool set: bash, read, write, edit, glob, grep."""
    from openhands.tools.deepseek.bash.definition import DeepSeekBashTool
    from openhands.tools.deepseek.edit.definition import DeepSeekEditTool
    from openhands.tools.deepseek.glob.definition import DeepSeekGlobTool
    from openhands.tools.deepseek.grep.definition import DeepSeekGrepTool
    from openhands.tools.deepseek.read.definition import DeepSeekReadTool
    from openhands.tools.deepseek.write.definition import DeepSeekWriteTool

    return [
        Tool(name=DeepSeekBashTool.name),
        Tool(name=DeepSeekReadTool.name),
        Tool(name=DeepSeekWriteTool.name),
        Tool(name=DeepSeekEditTool.name),
        Tool(name=DeepSeekGlobTool.name),
        Tool(name=DeepSeekGrepTool.name),
    ]

