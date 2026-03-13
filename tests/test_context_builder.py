"""
Unit tests for the context builder used in the RAG pipeline.
"""

from ai_adviser.config import settings
from ai_adviser.rag.context import build_context_from_hits, default_token_length


def test_build_context_filters_and_limits(monkeypatch) -> None:
    """Hits below the score threshold should be filtered and token limits enforced."""
    # Override configuration for this test
    monkeypatch.setattr(settings, "MAX_CONTEXT_TOKENS", 6, raising=False)
    monkeypatch.setattr(settings, "SCORE_THRESHOLD", 0.4, raising=False)
    # Create three hits with varying scores
    hits = [
        {settings.TEXT_FIELD: "First snippet", "score": 0.9, "blob_url": "url1"},
        {settings.TEXT_FIELD: "Second snippet with more words", "score": 0.5, "blob_url": "url2"},
        {settings.TEXT_FIELD: "Third snippet", "score": 0.1, "blob_url": "url3"},
    ]
    context, sources = build_context_from_hits(
        hits,
        max_tokens=settings.MAX_CONTEXT_TOKENS,
        token_fn=default_token_length,
        score_threshold=settings.SCORE_THRESHOLD,
    )
    # Only the first two snippets should be included (third filtered by threshold)
    assert "[S1]" in context and "[S2]" in context
    assert "Third snippet" not in context
    assert len(sources) == 2
    # Ensure the token budget was respected (approximate word count <= 6)
    assert len(context.split()) <= settings.MAX_CONTEXT_TOKENS