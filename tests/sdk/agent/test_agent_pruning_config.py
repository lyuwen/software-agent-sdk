"""Tests for the tool-result pruning policy recorded on the system prompt event."""

import uuid
from pathlib import Path

import pytest

from openhands.sdk.agent import Agent
from openhands.sdk.context.tool_result_pruner import (
    ENV_ENABLED,
    ENV_HEAD_CHARS,
    ENV_TAIL_CHARS,
    ENV_THRESHOLD_CHARS,
)
from openhands.sdk.conversation import ConversationState
from openhands.sdk.event import SystemPromptEvent
from openhands.sdk.workspace import LocalWorkspace


@pytest.fixture(autouse=True)
def _clear_pruner_env(monkeypatch: pytest.MonkeyPatch):
    for name in (ENV_ENABLED, ENV_THRESHOLD_CHARS, ENV_HEAD_CHARS, ENV_TAIL_CHARS):
        monkeypatch.delenv(name, raising=False)


def _new_state(agent: Agent, tmp_path: Path) -> ConversationState:
    return ConversationState.create(
        id=uuid.uuid4(),
        agent=agent,
        workspace=LocalWorkspace(working_dir=str(tmp_path)),
    )


def _system_prompt_event(agent: Agent, state: ConversationState) -> SystemPromptEvent:
    events: list = []
    agent.init_state(state, on_event=events.append)
    system_events = [event for event in events if isinstance(event, SystemPromptEvent)]
    assert len(system_events) == 1
    return system_events[0]


def test_init_state_records_disabled_pruning(mock_llm, tmp_path: Path):
    agent = Agent(llm=mock_llm)

    event = _system_prompt_event(agent, _new_state(agent, tmp_path))

    assert event.tool_result_pruning == {"enabled": False}


def test_init_state_records_enabled_pruning(mock_llm, tmp_path: Path, monkeypatch):
    monkeypatch.setenv(ENV_ENABLED, "1")
    monkeypatch.setenv(ENV_THRESHOLD_CHARS, "1000")
    monkeypatch.setenv(ENV_HEAD_CHARS, "100")
    monkeypatch.setenv(ENV_TAIL_CHARS, "50")
    agent = Agent(llm=mock_llm)

    event = _system_prompt_event(agent, _new_state(agent, tmp_path))

    assert event.tool_result_pruning == {
        "enabled": True,
        "threshold_chars": 1000,
        "head_chars": 100,
        "tail_chars": 50,
    }
