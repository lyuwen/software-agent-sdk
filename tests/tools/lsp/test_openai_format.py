"""
Tests for the OpenAI-format tool definition dict in lsp_tool.py.

This is the JSON blob the LLM sees in the ``tools`` array of an OpenAI
chat-completion request.
"""
import json

import pytest

import openhands.tools.lsp.lsp_tool as _mod

# The OpenAI-format tool dict
TOOL_DEF: dict = _mod.lsp_tool  # type: ignore[attr-defined]

# The commands exposed to the LLM (from the tool definition enum)
LLM_COMMANDS = TOOL_DEF["function"]["parameters"]["properties"]["command"]["enum"]

# All 12 user-facing commands in snake_case
EXPECTED_SNAKE = {
    "get_definition", "get_type_definition", "find_references",
    "hover", "get_implementation", "get_call_hierarchy",
    "prepare_call_hierarchy", "incoming_calls", "outgoing_calls",
    "get_document_symbols", "get_workspace_symbols", "get_document_highlights",
}

# All 12 user-facing commands in camelCase
EXPECTED_CAMEL = {
    "goToDefinition", "goToTypeDefinition", "findReferences",
    "hover", "goToImplementation", "getCallHierarchy",
    "prepareCallHierarchy", "incomingCalls", "outgoingCalls",
    "getDocumentSymbols", "getWorkspaceSymbols", "getDocumentHighlights",
}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
class TestToolDefinitionStructure:
    """Verify the dict matches OpenAI function-calling spec."""

    def test_top_level_keys(self):
        assert TOOL_DEF["type"] == "function"
        assert "function" in TOOL_DEF

    def test_function_keys(self):
        func = TOOL_DEF["function"]
        assert func["name"] == "lsp_tool"
        assert isinstance(func["description"], str)
        assert len(func["description"]) > 50
        assert "parameters" in func

    def test_parameters_is_json_schema(self):
        params = TOOL_DEF["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

    def test_command_is_required(self):
        assert "command" in TOOL_DEF["function"]["parameters"]["required"]


# ---------------------------------------------------------------------------
# Parameter definitions
# ---------------------------------------------------------------------------
class TestParameterDefinitions:
    @pytest.fixture()
    def props(self):
        return TOOL_DEF["function"]["parameters"]["properties"]

    def test_command_enum(self, props):
        cmd = props["command"]
        assert cmd["type"] == "string"
        assert "enum" in cmd
        # Default naming is snake_case
        assert set(cmd["enum"]) == EXPECTED_SNAKE

    def test_command_count(self, props):
        assert len(props["command"]["enum"]) == 12

    def test_file_path(self, props):
        assert props["file_path"]["type"] == "string"

    def test_symbol(self, props):
        assert props["symbol"]["type"] == "string"

    def test_line(self, props):
        assert props["line"]["type"] == "integer"

    def test_query(self, props):
        assert props["query"]["type"] == "string"

    def test_all_have_descriptions(self, props):
        for name, schema in props.items():
            assert "description" in schema, f"{name} lacks description"
            assert len(schema["description"]) > 10

    def test_optional_params_not_required(self):
        required = TOOL_DEF["function"]["parameters"]["required"]
        for name in ("file_path", "symbol", "line", "query", "item"):
            assert name not in required

    def test_item_param(self, props):
        assert props["item"]["type"] == "string"


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------
class TestSerialisation:
    def test_json_round_trip(self):
        s = json.dumps(TOOL_DEF)
        parsed = json.loads(s)
        assert parsed["function"]["name"] == "lsp_tool"

    def test_in_completion_request(self):
        """Tool def can sit in a realistic completion request."""
        request = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [TOOL_DEF],
            "tool_choice": "auto",
        }
        s = json.dumps(request)
        parsed = json.loads(s)
        assert len(parsed["tools"]) == 1

    def test_alongside_bash_tool(self):
        bash = {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Run a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
        request = {"tools": [TOOL_DEF, bash]}
        parsed = json.loads(json.dumps(request))
        names = [t["function"]["name"] for t in parsed["tools"]]
        assert "lsp_tool" in names
        assert "execute_bash" in names


# ---------------------------------------------------------------------------
# Simulated tool-call / tool-result round-trip
# ---------------------------------------------------------------------------
class TestToolCallFormat:
    def test_tool_call_response(self):
        """Simulated assistant message with a tool_call."""
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "lsp_tool",
                        "arguments": json.dumps({
                            "command": "get_definition",
                            "file_path": "/workspace/repo/foo.py",
                            "symbol": "Bar",
                            "line": 42,
                        }),
                    },
                }
            ],
        }
        s = json.dumps(msg)
        parsed = json.loads(s)
        args = json.loads(parsed["tool_calls"][0]["function"]["arguments"])
        assert args["command"] == "get_definition"
        assert args["line"] == 42

    def test_tool_result_message(self):
        """Tool result fed back into the conversation."""
        msg = {
            "role": "tool",
            "tool_call_id": "call_abc",
            "content": "[status_code]: success\n[Result]: Found definition ...",
        }
        s = json.dumps(msg)
        parsed = json.loads(s)
        assert parsed["role"] == "tool"
        assert "success" in parsed["content"]


# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------
class TestNamingConvention:
    def test_normalize_snake_passthrough(self):
        assert _mod.normalize_command("get_definition") == "get_definition"
        assert _mod.normalize_command("find_references") == "find_references"
        assert _mod.normalize_command("hover") == "hover"

    def test_normalize_camel_to_snake(self):
        assert _mod.normalize_command("goToDefinition") == "get_definition"
        assert _mod.normalize_command("findReferences") == "find_references"
        assert _mod.normalize_command("goToImplementation") == "get_implementation"
        assert _mod.normalize_command("prepareCallHierarchy") == "prepare_call_hierarchy"
        assert _mod.normalize_command("incomingCalls") == "incoming_calls"
        assert _mod.normalize_command("outgoingCalls") == "outgoing_calls"

    def test_normalize_old_aliases(self):
        assert _mod.normalize_command("get_hover") == "hover"
        assert _mod.normalize_command("get_references") == "find_references"

    def test_get_user_facing_commands_snake(self):
        cmds = _mod.get_user_facing_commands("snake_case")
        assert set(cmds) == EXPECTED_SNAKE

    def test_get_user_facing_commands_camel(self):
        cmds = _mod.get_user_facing_commands("camelCase")
        assert set(cmds) == EXPECTED_CAMEL

    def test_bidirectional_mapping_complete(self):
        """Every snake_case key has a camelCase value and vice versa."""
        assert len(_mod.COMMAND_NAMES) == 12
        assert len(_mod.CAMEL_TO_SNAKE) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
