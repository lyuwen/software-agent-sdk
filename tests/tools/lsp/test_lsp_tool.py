"""
Tests for lsp_tool.py internals: result formatting, response parsing,
command list, and the OpenAI-format tool definition dict from R2E-Gym's __init__.py.
"""
import pytest

from openhands.tools.lsp.lsp_tool import (
    LSPResult,
    LSPResponseParser,
    uri_to_path,
    _format_range,
    _format_location,
    ALLOWED_LSP_COMMANDS,
)


# ---------------------------------------------------------------------------
# LSPResult
# ---------------------------------------------------------------------------
class TestLSPResult:
    def test_success(self):
        r = LSPResult(result={"summary": "Found 3 references"})
        assert r.status_code == "success"
        assert r.error is None
        assert "Found 3 references" in r.to_text()

    def test_error(self):
        r = LSPResult(error="Connection refused")
        assert r.status_code == "error"
        assert "Connection refused" in r.to_text()

    def test_empty(self):
        r = LSPResult(result=None)
        text = r.to_text()
        assert "No output" in text or "Succeeded" in text

    def test_to_dict_success(self):
        r = LSPResult(result={"summary": "ok"})
        d = r.to_dict()
        assert d["status_code"] == "success"
        assert d["result"]["summary"] == "ok"

    def test_to_dict_error(self):
        r = LSPResult(error="boom")
        d = r.to_dict()
        assert d["status_code"] == "error"
        assert d["error"] == "boom"


# ---------------------------------------------------------------------------
# LSPResponseParser
# ---------------------------------------------------------------------------
class TestLSPResponseParser:
    def test_parse_references(self):
        data = [
            {"uri": "file:///a.py", "range": {"start": {"line": 1, "character": 0}, "end": {"line": 1, "character": 5}}},
            {"uri": "file:///b.py", "range": {"start": {"line": 10, "character": 0}, "end": {"line": 10, "character": 5}}},
        ]
        result = LSPResponseParser.parse_references(data)
        assert "2 reference" in result["summary"]
        assert "2 file" in result["summary"]

    def test_parse_empty_references(self):
        result = LSPResponseParser.parse_references([])
        assert "No references" in result["summary"]

    def test_parse_hover_markup_content(self):
        data = {"contents": {"kind": "markdown", "value": "```python\ndef foo() -> int\n```"}}
        result = LSPResponseParser.parse_hover(data)
        assert "foo" in result["summary"]

    def test_parse_hover_plain_string(self):
        data = {"contents": "some_type_info"}
        result = LSPResponseParser.parse_hover(data)
        assert "Hover Info" in result["summary"]

    def test_render_enhanced_definition(self):
        data = {
            "summary": "Found definition",
            "source_code": "class Foo:\n    pass",
            "locations": [],
        }
        result = LSPResponseParser.render_enhanced_definition(data)
        assert "SOURCE CODE" in result["summary"]
        assert "class Foo" in result["summary"]

    def test_render_call_hierarchy(self):
        data = {
            "target": {"name": "do_stuff", "detail": "", "uri": ""},
            "incoming_calls": [],
            "outgoing_calls": [],
        }
        result = LSPResponseParser.render_full_call_hierarchy(data)
        assert "do_stuff" in result["summary"]
        assert "None found" in result["summary"]

    def test_format_kind(self):
        assert LSPResponseParser._format_kind(5) == "Class"
        assert LSPResponseParser._format_kind(12) == "Function"
        assert "Kind" in LSPResponseParser._format_kind(999)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
class TestHelpers:
    def test_uri_to_path(self):
        p = uri_to_path("file:///testbed/module.py")
        assert str(p) == "/testbed/module.py"

    def test_uri_to_path_invalid(self):
        with pytest.raises(ValueError):
            uri_to_path("https://example.com")

    def test_format_range(self):
        r = {"start": {"line": 9, "character": 0}, "end": {"line": 15, "character": 0}}
        text = _format_range(r)
        assert "10" in text  # 1-indexed
        assert "16" in text

    def test_format_range_empty(self):
        assert "Unknown" in _format_range({})  # type: ignore[arg-type]

    def test_format_location(self):
        loc = {
            "uri": "file:///testbed/foo.py",
            "range": {"start": {"line": 4, "character": 0}, "end": {"line": 10, "character": 0}},
        }
        text = _format_location(loc)
        assert "/testbed/foo.py" in text


# ---------------------------------------------------------------------------
# CLI argument list (the ALLOWED_LSP_COMMANDS at module scope)
# ---------------------------------------------------------------------------
class TestCLICommands:
    def test_cli_commands_include_core(self):
        """The CLI-level ALLOWED_LSP_COMMANDS list includes
        the commands the agent will actually call."""
        for cmd in [
            "get_definition", "get_type_definition", "get_references",
            "get_hover", "get_document_symbols", "get_workspace_symbols",
            "get_call_hierarchy",
        ]:
            assert cmd in ALLOWED_LSP_COMMANDS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
