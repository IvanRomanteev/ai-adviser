"""
Integration tests for the `/rag_chat` endpoint. These tests simulate
responses from the embedding, search and chat clients by monkeypatching
their implementations. The FastAPI TestClient is used to exercise the
endpoint end to end.
"""

from fastapi.testclient import TestClient

import ai_adviser.api.main as main


def setup_function() -> None:
    """Reset the memory store between tests to avoid cross‑contamination."""
    # Reinitialise the in‑memory SQLite store by recreating tables
    main.memory_store._init_db()


def test_rag_chat_no_hits_returns_not_found(monkeypatch) -> None:
    """When retrieval returns no hits the service should return the fallback."""
    # Stub external clients
    monkeypatch.setattr(main, "embed_text", lambda text: [0.0], raising=False)
    monkeypatch.setattr(main, "hybrid_search", lambda q, v, top_k: [], raising=False)
    monkeypatch.setattr(main, "llm_chat", lambda msgs, max_tokens=900, temperature=0.2: "irrelevant", raising=False)
    client = TestClient(main.app)
    resp = client.post("/rag_chat", json={"question": "What is my balance?", "top_k": 3})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Not found in the knowledge base."
    assert data["chunks"] == []


def test_rag_chat_missing_citations_returns_not_found(monkeypatch) -> None:
    """If the model does not produce citations the service should fall back."""
    # Provide a single search hit
    hit = {
        main.settings.TEXT_FIELD: "Your account earns interest",  # snippet
        "score": 0.9,
        "blob_url": "http://example.com/doc1",
        "uid": "1",
        "snippet_parent_id": "1",
    }
    monkeypatch.setattr(main, "embed_text", lambda text: [0.1], raising=False)
    monkeypatch.setattr(main, "hybrid_search", lambda q, v, top_k: [hit], raising=False)
    # Stub chat to return an answer without citations
    monkeypatch.setattr(
        main,
        "llm_chat",
        lambda msgs, max_tokens=900, temperature=0.2: "This answer has no citations.",
        raising=False,
    )
    client = TestClient(main.app)
    resp = client.post("/rag_chat", json={"question": "How does interest work?", "top_k": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Not found in the knowledge base."
    # Even though we had a hit, since citations are missing we expect no chunks
    assert data["chunks"] == []


def test_rag_chat_valid_citations(monkeypatch) -> None:
    """A valid answer with citations should be returned without modification."""
    hit = {
        main.settings.TEXT_FIELD: "Rebuilding credit requires timely payments.",
        "score": 0.8,
        "blob_url": "http://example.com/credit",
        "uid": "2",
        "snippet_parent_id": "2",
    }
    monkeypatch.setattr(main, "embed_text", lambda text: [0.2], raising=False)
    monkeypatch.setattr(main, "hybrid_search", lambda q, v, top_k: [hit], raising=False)
    # The model returns a citation pointing to the first snippet
    monkeypatch.setattr(
        main,
        "llm_chat",
        lambda msgs, max_tokens=900, temperature=0.2: "You should pay on time. [S1]",
        raising=False,
    )
    client = TestClient(main.app)
    resp = client.post("/rag_chat", json={"question": "How do I rebuild credit?", "top_k": 1})
    assert resp.status_code == 200
    data = resp.json()
    # Since the model produced a valid citation the answer is returned as is
    assert data["answer"].startswith("You should pay on time")
    # One chunk should be returned
    assert len(data["chunks"]) == 1