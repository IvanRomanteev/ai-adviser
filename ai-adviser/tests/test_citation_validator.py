# tests/test_citation_validator.py
"""
Tests for the citation extraction and validation helpers.
"""

from ai_adviser.rag.citations import extract_citation_indices, validate_answer_citations
from ai_adviser.config import settings


def test_extract_citations() -> None:
    answer = "This is a fact [S1] and another fact [S2]."
    assert extract_citation_indices(answer) == [1, 2]


def test_validate_answer_citations_valid(monkeypatch) -> None:
    """Ensure that an answer with inline citations passes baseline validation when structural enforcement is disabled."""
    monkeypatch.setattr(settings, "CITATION_ENFORCE_STRUCTURE", False, raising=False)
    answer = "Answer with citations [S1] [S2]"
    sources = [{"id": 1}, {"id": 2}]
    assert validate_answer_citations(answer, sources) is True


def test_validate_answer_citations_invalid_missing(monkeypatch) -> None:
    """An answer without any citations should fail baseline validation."""
    monkeypatch.setattr(settings, "CITATION_ENFORCE_STRUCTURE", False, raising=False)
    answer = "Answer without citations."
    sources = [{"id": 1}, {"id": 2}]
    assert validate_answer_citations(answer, sources) is False


def test_validate_answer_citations_invalid_out_of_range(monkeypatch) -> None:
    """A citation referencing a non-existent source should fail baseline validation."""
    monkeypatch.setattr(settings, "CITATION_ENFORCE_STRUCTURE", False, raising=False)
    answer = "Answer with bad citation [S3]"
    sources = [{"id": 1}, {"id": 2}]
    assert validate_answer_citations(answer, sources) is False


def test_validate_answer_citations_enforce_structure_missing_sources(monkeypatch) -> None:
    """When structural enforcement is enabled, missing a Sources section is invalid."""
    monkeypatch.setattr(settings, "CITATION_ENFORCE_STRUCTURE", True, raising=False)
    answer = "Just text with citation [S1]"
    sources = [{"id": 1}, {"id": 2}]
    assert validate_answer_citations(answer, sources) is False


def test_validate_answer_citations_enforce_structure_valid(monkeypatch) -> None:
    """A properly structured answer with a Sources section should pass when enforcement is enabled."""
    monkeypatch.setattr(settings, "CITATION_ENFORCE_STRUCTURE", True, raising=False)
    answer = "Body sentence [S1]\n\nSources\n[S1] http://example.com"
    sources = [{"id": 1, "blob_url": "http://example.com"}]
    assert validate_answer_citations(answer, sources) is True


def test_validate_answer_citations_sources_only(monkeypatch) -> None:
    """An answer consisting only of a Sources section is invalid even under enforcement."""
    monkeypatch.setattr(settings, "CITATION_ENFORCE_STRUCTURE", True, raising=False)
    answer = "Sources\n[S1] http://example.com"
    sources = [{"id": 1, "blob_url": "http://example.com"}]
    assert validate_answer_citations(answer, sources) is False
