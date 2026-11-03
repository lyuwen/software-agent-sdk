import json
from typing import Any

from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import N_CHAR_PREVIEW, LLMConvertibleEvent
from openhands.sdk.event.types import SourceType
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.tool import ToolDefinition


class SystemPromptEvent(LLMConvertibleEvent):
    """System prompt added by the agent."""

    source: SourceType = "agent"
    system_prompt: TextContent = Field(..., description="The system prompt text")
    tools: list[ToolDefinition] = Field(
        ..., description="List of tools as ToolDefinition objects"
    )
    tool_result_pruning: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Tool-result pruning policy that was active when the conversation "
            "started: {'enabled': False} or {'enabled': True, 'threshold_chars': "
            "int, 'head_chars': int, 'tail_chars': int}. Recorded so exported "
            "trajectories reproduce the pruning the model actually saw instead "
            "of re-reading the exporter's environment. None on events created "
            "before the field existed."
        ),
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this system prompt event."""
        content = Text()
        content.append("System Prompt:\n", style="bold")
        content.append(self.system_prompt.text)
        content.append(f"\n\nTools Available: {len(self.tools)}")
        for tool in self.tools:
            # Use ToolDefinition properties directly
            description = tool.description.split("\n")[0][:100]
            if len(description) < len(tool.description):
                description += "..."

            content.append(f"\n  - {tool.name}: {description}\n")

            # Get parameters from the action type schema
            try:
                params_dict = tool.action_type.to_mcp_schema()
                params_str = json.dumps(params_dict)
                if len(params_str) > 200:
                    params_str = params_str[:197] + "..."
                content.append(f"  Parameters: {params_str}")
            except Exception:
                content.append("  Parameters: <unavailable>")
        return content

    def to_llm_message(self) -> Message:
        return Message(role="system", content=[self.system_prompt])

    def __str__(self) -> str:
        """Plain text string representation for SystemPromptEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        prompt_preview = (
            self.system_prompt.text[:N_CHAR_PREVIEW] + "..."
            if len(self.system_prompt.text) > N_CHAR_PREVIEW
            else self.system_prompt.text
        )
        tool_count = len(self.tools)
        return (
            f"{base_str}\n  System: {prompt_preview}\n  Tools: {tool_count} available"
        )
