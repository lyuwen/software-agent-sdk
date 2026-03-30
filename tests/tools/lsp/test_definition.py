"""
Tests for the native SDK tool definition (definition.py).
"""
from pathlib import Path

import pytest


# Path to the SDK LSP module
LSP_DIR = Path(__file__).parent.parent.parent.parent / "openhands-tools" / "openhands" / "tools" / "lsp"


class TestNativeToolDefinition:
    """Verify the native SDK tool definition exists and has correct structure."""

    def test_definition_module_exists(self):
        defn = LSP_DIR / "definition.py"
        assert defn.is_file(), "LSP tool definition module missing"

    def test_definition_has_action_class(self):
        src = (LSP_DIR / "definition.py").read_text()
        assert "class LSPAction" in src

    def test_definition_has_observation_class(self):
        src = (LSP_DIR / "definition.py").read_text()
        assert "class LSPObservation" in src

    def test_definition_has_tool_class(self):
        src = (LSP_DIR / "definition.py").read_text()
        assert "class LSPTool" in src

    def test_definition_registers_tool(self):
        src = (LSP_DIR / "definition.py").read_text()
        assert "register_tool" in src

    def test_definition_has_all_commands(self):
        src = (LSP_DIR / "definition.py").read_text()
        for cmd in [
            "get_definition", "get_type_definition", "find_references",
            "hover", "get_implementation", "get_call_hierarchy",
            "prepare_call_hierarchy", "incoming_calls", "outgoing_calls",
            "get_document_symbols", "get_workspace_symbols", "get_document_highlights",
        ]:
            assert cmd in src, f"Command {cmd} not in definition"

    def test_definition_has_item_field(self):
        src = (LSP_DIR / "definition.py").read_text()
        assert "item:" in src or "item :" in src

    def test_definition_has_naming_config(self):
        src = (LSP_DIR / "definition.py").read_text()
        assert "LSP_NAMING" in src
        assert "COMMAND_NAMES" in src

    def test_definition_imports_from_lsp_tool(self):
        """definition.py should import naming code from lsp_tool.py, not duplicate it."""
        src = (LSP_DIR / "definition.py").read_text()
        assert "from openhands.tools.lsp.lsp_tool import" in src

    def test_lsp_tool_exists(self):
        assert (LSP_DIR / "lsp_tool.py").is_file()

    def test_lsp_daemon_exists(self):
        assert (LSP_DIR / "lsp_daemon.py").is_file()

    def test_workspace_setup_exists(self):
        assert (LSP_DIR / "workspace_setup.py").is_file()

    def test_prompt_helpers_exists(self):
        assert (LSP_DIR / "prompt_helpers.py").is_file()

    def test_prompt_template_exists(self):
        assert (LSP_DIR / "prompts" / "default_lsp.j2").is_file()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
