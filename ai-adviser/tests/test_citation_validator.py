"""
Tests for the citation extraction and validation helpers.
"""

from ai_adviser.rag.citations import extract_citation_indices, validate_answer_citations


def test_extract_citations() -> None:
    answer = "This is a fact [S1] and another fact [S2]."
    assert extract_citation_indices(answer) == [1, 2]


def test_validate_answer_citations_valid() -> None:
    answer = "Answer with citations [S1] [S2]"
    sources = [{"id": 1}, {"id": 2}]
    assert validate_answer_citations(answer, sources) is True


def test_validate_answer_citations_invalid_missing() -> None:
    answer = "Answer without citations."
    sources = [{"id": 1}, {"id": 2}]
    assert validate_answer_citations(answer, sources) is False


def test_validate_answer_citations_invalid_out_of_range() -> None:
    answer = "Answer with bad citation [S3]"
    sources = [{"id": 1}, {"id": 2}]
    # S3 references a non‑existent source
    assert validate_answer_citations(answer, sources) is False