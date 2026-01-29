"""Integration tests for RAG chat using domain questions from the banking knowledge base.

This test module exercises the /rag_chat endpoint end‑to‑end using
representative questions drawn from the underlying knowledge base.  It
mimics the user's PowerShell script by issuing a series of queries
against the API and asserting that answers are returned with proper
citations when content is available or that a deterministic fallback
message is returned when the query falls outside of the knowledge
base.  A separate test verifies that the conversation memory is
maintained across turns and that the retrieval query incorporates
previous user messages when query rewriting is disabled.

The external embedding, search and language model functions are
monkeypatched to deterministic mocks so that no network calls occur.
Citation strictness and structural enforcement are enabled to ensure
fully grounded answers.
"""

from fastapi.testclient import TestClient
import ai_adviser.api.main as main


def setup_function() -> None:
    """Reset the in‑memory conversation store before each test."""
    main.memory_store._init_db()


def test_kb_questions(monkeypatch) -> None:
    """Run a sequence of banking domain questions through the RAG endpoint.

    The first nine questions correspond to topics expected to be found in
    the knowledge base and should return answers containing inline
    citations plus a Sources section.  The final question asks about
    purchasing bitcoin, which lies outside of the domain and should
    trigger a fallback response.  Citation strictness and structure
    enforcement are enabled so that answers without proper citations
    would be rejected, but all provided LLM responses include an
    inline citation.
    """
    # Configure deterministic settings
    monkeypatch.setattr(main.settings, "CITATION_STRICT", True, raising=False)
    monkeypatch.setattr(main.settings, "CITATION_ENFORCE_STRUCTURE", True, raising=False)
    monkeypatch.setattr(main.settings, "SUMMARY_ENABLED", False, raising=False)
    monkeypatch.setattr(main.settings, "REWRITE_ENABLED", False, raising=False)

    # Stub embed_text to return a constant vector.  The returned vector
    # value is unused by the hybrid_search mock.
    monkeypatch.setattr(main, "embed_text", lambda text: [0.1], raising=False)

    # Define a set of banking queries and indicate whether they are
    # expected to yield hits in the knowledge base.
    queries = [
        "How can I rebuild my credit?",
        "What is a secured credit card and how does it help?",
        "How much should I spend on it each month?",
        "What happens if I miss a payment?",
        "What is overdraft protection and how does it work?",
        "How do I dispute an unauthorized card transaction?",
        "What is the difference between ACH and a wire transfer?",
        "Can I withdraw early from a certificate of deposit (CD)? What are penalties?",
        "Are my deposits insured and what are the limits?",
        "How do I buy bitcoin inside this bank app?",
    ]

    # Create a retrieval mapping keyed by the raw question.  Each entry
    # contains a single search hit with the configured TEXT_FIELD,
    # blob_url, uid and snippet_parent_id.  For the final bitcoin
    # question no entries are provided to simulate a miss.
    retrieval_mapping: dict[str, list[dict[str, object]]] = {}
    for idx, q in enumerate(queries[:-1], start=1):  # all but the last
        retrieval_mapping[q] = [
            {
                main.settings.TEXT_FIELD: f"Snippet for '{q}'",
                "score": 0.9,
                "blob_url": f"http://example.com/doc{idx}",
                "uid": str(idx),
                "snippet_parent_id": str(idx),
            }
        ]
    # Last query has no hits
    retrieval_mapping[queries[-1]] = []

    def hybrid_search_mock(query: str, vec: list[float], top_k: int) -> list[dict[str, object]]:
        # In this test rewriting is disabled so the retrieval_query
        # passed into hybrid_search is exactly the user's question.  Use
        # retrieval_mapping to return the corresponding hits or an empty
        # list for unknown queries.
        return retrieval_mapping.get(query, [])

    monkeypatch.setattr(main, "hybrid_search", hybrid_search_mock, raising=False)

    # Provide a canned answer for each query with an inline citation.
    answer_map: dict[str, str] = {}
    for idx, q in enumerate(queries[:-1], start=1):
        answer_map[q] = f"Answer for '{q}' [S1]"
    # The bitcoin question deliberately lacks a canned answer since no
    # hits means the endpoint should return the fallback directly.

    def llm_chat_mock(messages: list[dict[str, str]], max_tokens: int = 900, temperature: float = 0.2) -> str:
        # Extract the question from the last user message.  The prompt
        # sent to the LLM always ends with "QUESTION:\n<question>".
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                # Split on the QUESTION marker.  If missing fall back to
                # the whole content.
                if "QUESTION:\n" in content:
                    q = content.split("QUESTION:\n", 1)[1].strip()
                else:
                    q = content.strip()
                break
        else:
            # No user message; return empty string
            return ""
        return answer_map.get(q, "")

    monkeypatch.setattr(main, "llm_chat", llm_chat_mock, raising=False)

    client = TestClient(main.app)
    # Iterate through the queries, preserving the index for assertions.
    for i, query in enumerate(queries):
        resp = client.post("/rag_chat", json={"question": query, "top_k": 1})
        assert resp.status_code == 200
        data = resp.json()
        if i == len(queries) - 1:
            # The bitcoin question should produce a fallback response
            assert data["answer"] == main.FALLBACK_MESSAGE
            assert data["chunks"] == []
        else:
            # For found questions an answer with citations and sources
            # should be returned.  The answer's body should start with
            # the canned answer, and one chunk should be returned.
            expected_body = answer_map[query]
            assert data["answer"].startswith(expected_body)
            assert len(data["chunks"]) == 1


def test_kb_memory_history(monkeypatch) -> None:
    """Verify that conversation history is preserved across turns and
    retrieval queries include previous user messages when rewriting
    is disabled.

    Two related questions are issued in the same thread.  The first
    should result in a single embedding call with the original
    question.  The second call should embed the concatenation of the
    previous user question and the current question.  After both
    calls the memory store should contain four entries (two user
    messages and two assistant responses).
    """
    # Reset the memory
    main.memory_store._init_db()
    # Enable strict citations but disable rewriting (so retrieval
    # queries are concatenated with history).
    monkeypatch.setattr(main.settings, "CITATION_STRICT", True, raising=False)
    monkeypatch.setattr(main.settings, "CITATION_ENFORCE_STRUCTURE", True, raising=False)
    monkeypatch.setattr(main.settings, "SUMMARY_ENABLED", False, raising=False)
    monkeypatch.setattr(main.settings, "REWRITE_ENABLED", False, raising=False)
    # Capture the text passed into embed_text for later inspection.
    embed_calls: list[str] = []

    def embed_text_capture(text: str) -> list[float]:
        embed_calls.append(text)
        return [0.0]

    monkeypatch.setattr(main, "embed_text", embed_text_capture, raising=False)

    # Define two related questions
    q1 = "What is a secured credit card and how does it help?"
    q2 = "How much should I spend on it each month?"
    # Provide retrieval hits and answers for both
    retrieval_mapping = {
        q1: [
            {
                main.settings.TEXT_FIELD: "A secured card helps rebuild credit.",
                "score": 0.9,
                "blob_url": "http://example.com/secured",
                "uid": "10",
                "snippet_parent_id": "10",
            }
        ],
        q2: [
            {
                main.settings.TEXT_FIELD: "Keep utilization under 30%.",
                "score": 0.9,
                "blob_url": "http://example.com/utilization",
                "uid": "11",
                "snippet_parent_id": "11",
            }
        ],
    }

    def hybrid_search_mock(query: str, vec: list[float], top_k: int) -> list[dict[str, object]]:
        return retrieval_mapping.get(query, [])

    monkeypatch.setattr(main, "hybrid_search", hybrid_search_mock, raising=False)

    answer_map = {
        q1: "A secured credit card requires a cash deposit and can build credit history. [S1]",
        q2: "Experts recommend keeping your monthly spending below 30% of your credit limit. [S1]",
    }

    def llm_chat_mock(messages: list[dict[str, str]], max_tokens: int = 900, temperature: float = 0.2) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if "QUESTION:\n" in content:
                    q = content.split("QUESTION:\n", 1)[1].strip()
                else:
                    q = content.strip()
                break
        else:
            return ""
        return answer_map.get(q, "")

    monkeypatch.setattr(main, "llm_chat", llm_chat_mock, raising=False)
    client = TestClient(main.app)
    # Use a dedicated thread_id for this conversation
    thread_id = "t-memory"
    # First call
    resp1 = client.post(
        "/rag_chat",
        json={"question": q1, "top_k": 1, "thread_id": thread_id},
    )
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["answer"].startswith(answer_map[q1])
    assert len(data1["chunks"]) == 1
    # Second call
    resp2 = client.post(
        "/rag_chat",
        json={"question": q2, "top_k": 1, "thread_id": thread_id},
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["answer"].startswith(answer_map[q2])
    assert len(data2["chunks"]) == 1
    # The embed_text function should have been called twice.  The
    # second call should contain both the previous user message and
    # the current question separated by a space because rewriting is
    # disabled and history is non-empty.
    assert len(embed_calls) == 2
    assert embed_calls[0] == q1
    assert q1 in embed_calls[1] and q2 in embed_calls[1]
    # Verify that memory_store persisted the conversation
    history = main.memory_store.get_history(thread_id)
    # Expect 4 entries: user and assistant for each turn
    assert len(history) == 4