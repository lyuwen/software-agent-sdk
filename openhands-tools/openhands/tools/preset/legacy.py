"""Default preset configuration for OpenHands agents."""

from openhands.sdk import Agent
from openhands.sdk.context.condenser import (
    LLMSummarizingCondenser,
)
from openhands.sdk.context.condenser.base import CondenserBase
from openhands.sdk.llm.llm import LLM
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import Tool, register_tool

from openhands.tools.file_editor import FileEditorTool
from openhands.tools.terminal import TerminalTool
from openhands.tools.task_tracker import TaskTrackerTool


logger = get_logger(__name__)


class StrReplaceEditorTool(FileEditorTool):
  pass


class ExecuteBashTool(TerminalTool):
  pass


def register_legacy_tools(enable_browser: bool = True) -> None:
    """Register the legacy set of tools."""
    # Tools are now automatically registered when imported

    logger.debug(f"Tool: {TerminalTool.name} registered.")
    logger.debug(f"Tool: {FileEditorTool.name} registered.")
    logger.debug(f"Tool: {TaskTrackerTool.name} registered.")

    register_tool("str_replace_editor", StrReplaceEditorTool)
    register_tool("execute_bash", ExecuteBashTool)

    if enable_browser:
        raise NotImplementedError("Legacy brower use not supported")


def get_legacy_tools(
    enable_browser: bool = False,
) -> list[Tool]:
    """Get the legacy set of tool specifications for the standard experience.

    Args:
        enable_browser: Whether to include browser tools.
    """
    register_legacy_tools(enable_browser=enable_browser)

    tools = [
        Tool(name="execute_bash"),
        Tool(name="str_replace_editor"),
        Tool(name=TaskTrackerTool.name),
    ]
    if enable_browser:
        raise NotImplementedError("Legacy brower use not supported")
    return tools


def get_legacy_condenser(llm: LLM) -> CondenserBase:
    # Create a condenser to manage the context. The condenser will automatically
    # truncate conversation history when it exceeds max_size, and replaces the dropped
    # events with an LLM-generated summary.
    condenser = LLMSummarizingCondenser(llm=llm, max_size=80, keep_first=4)

    return condenser


def get_legacy_agent(
    llm: LLM,
    cli_mode: bool = True,
) -> Agent:
    tools = get_legacy_tools(
        # Disable browser tools in CLI mode
        enable_browser=not cli_mode,
    )
    agent = Agent(
        llm=llm,
        tools=tools,
        system_prompt_kwargs={"cli_mode": cli_mode},
        condenser=get_legacy_condenser(
            llm=llm.model_copy(update={"usage_id": "condenser"})
        ),
    )
    return agent
