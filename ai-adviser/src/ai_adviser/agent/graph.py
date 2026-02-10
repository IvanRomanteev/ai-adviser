"""
LangGraph orchestration for the Retrieval-Augmented Generation pipeline.

This module defines a stateful execution graph that chains together the
individual steps of the RAG workflow: input normalisation, memory
loading, embedding, retrieval, context construction, generation,
citation validation and memory persistence.

NOTE: This uses a minimal local stub of LangGraph runtime provided in
`langgraph.graph` (NOT the PyPI langgraph package unless you added it).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from ai_adviser.config import settings
from ai_adviser.memory.store import MemoryStore
from ai_adviser.clients.azure_models import embed_text, chat as llm_chat
from ai_adviser.clients.azure_search import hybrid_search
from ai_adviser.prompts.loader import load_prompt
from ai_adviser.rag.context import build_context_from_hits, default_token_length
from ai_adviser.rag.citations import validate_answer_citations


class State(TypedDict, total=False):
    thread_id: Optional[str]
    user_id: Optional[str]

    question: str
    history: List[Dict[str, str]]

    query_vec: List[float]
    hits: List[Dict[str, Any]]

    context: str
    sources: List[Dict[str, Any]]

    answer: str

    # Internal helper flag to prevent infinite retries
    _grounding_retry_done: bool


_graph_memory = MemoryStore(settings.SQLITE_DB_PATH)
_system_prompt = load_prompt("rag_system.md")


def normalize_input(state: State) -> Dict[str, Any]:
    """Trim whitespace from the question."""
    q = (state.get("question") or "").strip()
    return {"question": q}


def load_memory(state: State) -> Dict[str, Any]:
    """Retrieve previous messages for this thread."""
    thread_id = state.get("thread_id")
    if not thread_id:
        return {"history": []}

    hist = _graph_memory.get_history(thread_id) or []
    # Ensure history is a list[{"role": str, "content": str}]
    cleaned: List[Dict[str, str]] = []
    for m in hist:
        try:
            role = m.get("role")
            content = m.get("content")
            if isinstance(role, str) and isinstance(content, str):
                cleaned.append({"role": role, "content": content})
        except Exception:
            # Skip malformed items
            continue
    return {"history": cleaned}


def embed_query(state: State) -> Dict[str, Any]:
    """Compute an embedding for the user's question."""
    if not state.get("question"):
        return {"query_vec": []}
    vec = embed_text(state["question"])
    return {"query_vec": vec}


def retrieve(state: State) -> Dict[str, Any]:
    """Retrieve relevant snippets using hybrid search."""
    if not state.get("question"):
        return {"hits": []}
    hits = hybrid_search(state["question"], state.get("query_vec", []), settings.TOP_K)
    return {"hits": hits or []}


def build_context(state: State) -> Dict[str, Any]:
    """Construct the context string and source metadata from hits."""
    context, sources = build_context_from_hits(
        state.get("hits", []),
        max_tokens=settings.MAX_CONTEXT_TOKENS,
        token_fn=default_token_length if settings.MAX_CONTEXT_TOKENS else None,
        score_threshold=settings.SCORE_THRESHOLD,
    )
    return {"context": context or "", "sources": sources or []}


def _build_messages(state: State, *, extra_user_instruction: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Correct message order:
      1) system
      2) history (user/assistant)
      3) current user prompt (with context + question)
    """
    messages: List[Dict[str, str]] = []

    # 1) system first
    messages.append({"role": "system", "content": _system_prompt})

    # 2) history next
    for m in state.get("history", []):
        role = m.get("role")
        content = m.get("content")
        if isinstance(role, str) and isinstance(content, str) and role in {"user", "assistant"}:
            messages.append({"role": role, "content": content})

    # 3) current user message
    user_parts = [
        "CONTEXT:",
        state.get("context", "") or "",
        "",
        "QUESTION:",
        state.get("question", "") or "",
    ]
    if extra_user_instruction:
        user_parts.extend(["", "INSTRUCTIONS:", extra_user_instruction])

    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages


def generate(state: State) -> Dict[str, Any]:
    """Generate an answer from the language model."""
    if not state.get("question"):
        return {"answer": "Please provide a question."}

    messages = _build_messages(state)
    answer = llm_chat(messages, max_tokens=900, temperature=0.2)
    return {"answer": answer or ""}


def validate_grounding(state: State) -> Dict[str, Any]:
    """
    Ensure the answer contains valid citations.

    Improvement:
    - If strict mode is on and citations are invalid, try ONE re-generation with
      an explicit instruction to include citations [S1], [S2], ...
    - If still invalid, fall back.
    """
    if not settings.CITATION_STRICT:
        return {}

    answer = state.get("answer", "") or ""
    sources = state.get("sources", []) or []

    # If there are no sources, strict citations cannot be satisfied
    if not sources:
        return {"answer": "Not found in the knowledge base."}

    if validate_answer_citations(answer, sources):
        return {}

    # One retry only
    if not state.get("_grounding_retry_done", False):
        retry_instruction = (
            "You MUST answer using ONLY the provided CONTEXT. "
            "Add inline citations in the format [S1], [S2], ... "
            "Each factual claim must be supported by a citation. "
            "If the context does not contain the answer, say: "
            "\"Not found in the knowledge base.\""
        )
        retry_messages = _build_messages(state, extra_user_instruction=retry_instruction)
        retry_answer = llm_chat(retry_messages, max_tokens=900, temperature=0.2) or ""

        if validate_answer_citations(retry_answer, sources):
            return {"answer": retry_answer, "_grounding_retry_done": True}

        return {"answer": "Not found in the knowledge base.", "_grounding_retry_done": True}

    return {"answer": "Not found in the knowledge base."}


def save_memory(state: State) -> Dict[str, Any]:
    """Persist the question and answer to memory."""
    thread_id = state.get("thread_id")
    user_id = state.get("user_id")

    # Don't persist empty interactions
    q = (state.get("question") or "").strip()
    a = (state.get("answer") or "").strip()
    if not q and not a:
        return {}

    if thread_id:
        _graph_memory.append(thread_id, user_id, "user", q)
        _graph_memory.append(thread_id, user_id, "assistant", a)
    return {}


# Define the directed graph structure
graph = StateGraph(State)

graph.add_node("normalize_input", normalize_input)
graph.add_node("load_memory", load_memory)
graph.add_node("embed_query", embed_query)
graph.add_node("retrieve", retrieve)
graph.add_node("build_context", build_context)
graph.add_node("generate", generate)
graph.add_node("validate_grounding", validate_grounding)
graph.add_node("save_memory", save_memory)

graph.set_entry_point("normalize_input")
graph.add_edge("normalize_input", "load_memory")
graph.add_edge("load_memory", "embed_query")
graph.add_edge("embed_query", "retrieve")
graph.add_edge("retrieve", "build_context")
graph.add_edge("build_context", "generate")
graph.add_edge("generate", "validate_grounding")
graph.add_edge("validate_grounding", "save_memory")
graph.add_edge("save_memory", END)

app = graph.compile()

__all__ = ["State", "graph", "app"]
