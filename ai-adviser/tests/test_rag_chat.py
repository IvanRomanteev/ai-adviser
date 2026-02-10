# tests/test_rag_chat.py
"""
Integration tests for the `/rag_chat` endpoint. These tests simulate
responses from the embedding, search and chat clients by monkeypatching
their implementations. The FastAPI TestClient is used to exercise the
endpoint end to end.
"""

from fastapi.testclient import TestClient

import ai_adviser.api.main as main


def setup_function() -> None:
    """Reset the memory store between tests to avoid cross-contamination."""
    main.memory_store._init_db()


def test_rag_chat_no_hits_returns_not_found(monkeypatch) -> None:
    """When retrieval returns no hits the service should return the fallback."""
    monkeypatch.setattr(main.settings, "CITATION_STRICT", True, raising=False)
    monkeypatch.setattr(main.settings, "CITATION_ENFORCE_STRUCTURE", False, raising=False)
    monkeypatch.setattr(main.settings, "SUMMARY_ENABLED", False, raising=False)
    monkeypatch.setattr(main.settings, "REWRITE_ENABLED", False, raising=False)

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
    monkeypatch.setattr(main.settings, "CITATION_STRICT", True, raising=False)
    monkeypatch.setattr(main.settings, "CITATION_ENFORCE_STRUCTURE", False, raising=False)
    monkeypatch.setattr(main.settings, "SUMMARY_ENABLED", False, raising=False)
    monkeypatch.setattr(main.settings, "REWRITE_ENABLED", False, raising=False)

    hit = {
        main.settings.TEXT_FIELD: "Your account earns interest",
        "score": 0.9,
        "blob_url": "http://example.com/doc1",
        "uid": "1",
        "snippet_parent_id": "1",
    }
    monkeypatch.setattr(main, "embed_text", lambda text: [0.1], raising=False)
    monkeypatch.setattr(main, "hybrid_search", lambda q, v, top_k: [hit], raising=False)
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
    # New behavior: structured fallback that preserves the model answer
    assert "This answer has no citations." in data["answer"]
    assert "Grounding note" in data["answer"]
    assert "Closest sources:" in data["answer"]
    assert "http://example.com/doc1" in data["answer"]


    # Keep existing strict behavior: no chunks are returned on citation failure
    # New behavior: chunks are still returned (useful for inspection / UI)
    assert len(data["chunks"]) == 1

    chunk0 = data["chunks"][0]
    assert "interest" in chunk0["content"]
    # URL может быть не на верхнем уровне, поэтому ищем в разных местах
    blob_url = (
            chunk0.get("blob_url")
            or chunk0.get("source_file")  # <-- ВОТ ЭТОГО НЕ ХВАТАЛО
            or (chunk0.get("metadata") or {}).get("blob_url")
            or (chunk0.get("raw") or {}).get("blob_url")
            or (chunk0.get("raw") or {}).get("source_file")
            or (chunk0.get("source") or "")
            or ""
    )
    assert "http://example.com/doc1" in blob_url


def test_rag_chat_valid_citations(monkeypatch) -> None:
    """A valid answer with citations should be returned without modification."""
    monkeypatch.setattr(main.settings, "CITATION_STRICT", True, raising=False)
    monkeypatch.setattr(main.settings, "CITATION_ENFORCE_STRUCTURE", False, raising=False)
    monkeypatch.setattr(main.settings, "SUMMARY_ENABLED", False, raising=False)
    monkeypatch.setattr(main.settings, "REWRITE_ENABLED", False, raising=False)

    hit = {
        main.settings.TEXT_FIELD: "Rebuilding credit requires timely payments.",
        "score": 0.8,
        "blob_url": "http://example.com/credit",
        "uid": "2",
        "snippet_parent_id": "2",
    }
    monkeypatch.setattr(main, "embed_text", lambda text: [0.2], raising=False)
    monkeypatch.setattr(main, "hybrid_search", lambda q, v, top_k: [hit], raising=False)
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
    assert data["answer"].startswith("You should pay on time")
    assert len(data["chunks"]) == 1
