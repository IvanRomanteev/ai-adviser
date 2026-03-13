"""
Tests for the prompt loader to ensure UTF‑8 encoding is respected.
"""

from ai_adviser.prompts.loader import load_prompt


def test_load_prompt_no_encoding_errors() -> None:
    """The prompt should load without introducing mojibake characters."""
    content = load_prompt("rag_system.md")
    # Ensure typical problematic characters are not present
    assert "â" not in content, "Prompt contains unexpected encoding artefacts"
    # Sanity check that the prompt includes the expected header
    assert "You are a banking assistant MVP." in content