"""FastAPI application implementing a simplified RAG pipeline.

This module defines a small FastAPI app with an endpoint ``/rag_chat``
that performs retrieval augmented generation.  The implementation is
adapted from the production ai_adviser project but stripped down for
testing.  In particular, we avoid external network calls and rely on
monkeypatching to simulate embedding, search and language model
behaviour.  Citation handling is implemented in a strict and
deterministic manner according to the requirements described in the
problem statement.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from ai_adviser.api.schemas import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    RagChatRequest,
    RagChatResponse,
    RagChunk,
)
from ai_adviser.clients import azure_models as _azure_models
from ai_adviser.clients.azure_search import hybrid_search
from ai_adviser.config import settings
from ai_adviser.memory.store import MemoryStore
from ai_adviser.observability.langfuse import span
from ai_adviser.observability.metrics import record_metric, collector, setup_metrics
from ai_adviser.observability.tracing import init_tracing
from ai_adviser.prompts.loader import load_prompt
from ai_adviser.rag.citations import (
    validate_answer_citations,
    build_sources_mapping,
    upsert_sources_section,
    enforce_citation_structure,
    FALLBACK_MESSAGE,
)
from ai_adviser.rag.rewrite import rewrite_query
from ai_adviser.rag.context import build_context_from_hits, default_token_length
from ai_adviser.rag.guard import is_relevant_to_sources

logger = logging.getLogger(__name__)


class UTF8JSONResponse(JSONResponse):
    """Force UTF-8 charset in JSON responses.

    PowerShell's Invoke-RestMethod can mis-decode UTF-8 as cp1252/latin-1 when
    the charset is missing. Setting it explicitly prevents "â€™" artifacts.
    """

    media_type = "application/json; charset=utf-8"



app = FastAPI(
    title="ai-adviser",
    version="0.1.0",
    default_response_class=UTF8JSONResponse,
)

# Подключаем роутер поиска ПОСЛЕ создания app
from .search_endpoint import router as search_router  # noqa: E402

app.include_router(search_router)

# Determine the correct names for the language model and embedding functions.
# The production code defines `chat` whereas the simplified test harness uses
# `llm_chat`.  Support both by introspecting the azure_models module.
if hasattr(_azure_models, "llm_chat"):
    llm_chat = _azure_models.llm_chat  # type: ignore[assignment]
else:
    # In production azure_models the function is named `chat`.  Alias it.
    llm_chat = _azure_models.chat  # type: ignore[assignment]
embed_text = _azure_models.embed_text  # type: ignore[assignment]

# Load the system prompt for RAG behaviour.  In tests this returns an empty string.
BANKING_RAG_SYSTEM_PROMPT = load_prompt("rag_system.md")

# Initialise the conversation memory store.  The backend parameter is
# ignored by the simplified MemoryStore and conversations are kept in
# memory only.
memory_store = MemoryStore(settings.SQLITE_DB_PATH)


def structured_grounding_fallback(
    original_answer: str, sources: list[dict[str, Any]], limit: int = 5
) -> str:
    """Preserve the best available answer and add a grounding note + closest sources.

    Used when CITATION_STRICT is enabled but we cannot produce valid inline citations.
    """
    note = (
        "⚠️ Grounding note: I couldn't produce a response with valid inline citations "
        "in the required [S#] format. The answer below may be partially unsupported "
        "by the retrieved knowledge base.\n"
    )

    if not sources:
        closest = "- (no sources available)\n"
    else:
        lines: list[str] = []
        for i, src in enumerate(sources[: max(limit, 1)], start=1):
            raw = (src or {}).get("raw") or {}
            url = src.get("blob_url") or raw.get("blob_url") or ""
            uid = raw.get("uid")
            score = raw.get("score") or raw.get("@search.score") or src.get("score")

            score_str = ""
            try:
                if score is not None:
                    score_str = f" (score={float(score):.4f})"
            except Exception:
                score_str = ""

            if url:
                lines.append(f"- [S{i}] {url}{score_str}")
            elif uid is not None:
                lines.append(f"- [S{i}] uid={uid}{score_str}")
            else:
                lines.append(f"- [S{i}] (source unavailable){score_str}")

        closest = "\n".join(lines) + "\n"

    original_answer = (original_answer or "").strip()
    return f"{original_answer}\n\n---\n\n{note}\nClosest sources:\n{closest}"


# Set up metrics and tracing instrumentation.  These functions are
# no-ops in the simplified implementation but remain for API parity.
setup_metrics(app)
init_tracing(app)


@app.get("/health")
def health() -> dict[str, bool]:
    """Liveness probe used by tests to check the server is running."""
    return {"ok": True}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest) -> EmbedResponse:
    """Embed a piece of text using the (stubbed) embedding model."""
    try:
        vec = embed_text(req.text)
        return EmbedResponse(dims=len(vec), embedding=vec)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Embedding request failed: {type(e).__name__}"
        ) from e


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Direct chat endpoint exposing the LLM.  Not used in current tests."""
    try:
        msgs = [m.model_dump() for m in req.messages]
        text = llm_chat(
            msgs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return ChatResponse(text=text)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Chat request failed: {type(e).__name__}"
        ) from e


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Middleware to record request latency and attach a request ID."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    start = time.perf_counter()
    response: Response
    error_occurred = False
    try:
        response = await call_next(request)
    except Exception:
        error_occurred = True
        raise
    finally:
        elapsed = time.perf_counter() - start
        try:
            path = request.url.path
            method = request.method
            status_code = response.status_code if "response" in locals() else 500
            collector.observe_request(path, method, status_code, elapsed)
        except Exception:
            pass
        if error_occurred:
            collector.increment("error_count", 1)
    response.headers["X-Request-ID"] = request_id
    return response


@app.post("/rag_chat", response_model=RagChatResponse)
def rag_chat(req: RagChatRequest) -> RagChatResponse:
    """Perform a retrieval-augmented chat with optional conversation memory.

    The pipeline roughly follows these steps:
    1. Validate and possibly rewrite the user's question.
    2. Embed the query and perform hybrid retrieval (monkeypatched in tests).
    3. Build a context string and sources list from the retrieval hits.
    4. Generate an answer from the LLM.
    5. When citations are required (CITATION_STRICT=True) validate the
       answer's citations.  If invalid, attempt one repair pass.  If
       still invalid, fall back to a generic "Not found" message.
    6. Insert a Sources section only after the answer has passed
       validation to avoid false positives during citation checking.
    7. Persist the conversation to memory and return the answer with
       associated chunks.
    """
    try:
        # Extract and validate the question
        question = (req.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question must not be empty")

        thread_id = req.thread_id or "default"
        user_id = req.user_id

        # Record that the rag_chat endpoint was called
        record_metric("rag_chat_calls", 1)

        # ---------------------------
        # Query rewriting for follow-up questions
        # ---------------------------
        embedding_query = question
        search_query = question

        history: list[dict[str, str]] = []
        if req.thread_id:
            history = memory_store.get_history(thread_id)

            new_query: str | None = None
            if settings.REWRITE_ENABLED:
                summary: str | None = None
                if settings.SUMMARY_ENABLED:
                    summary = memory_store.get_latest_summary(thread_id)

                recent_subset = history[-settings.REWRITE_LAST_N:] if history else []
                recent_contents = [m.get("content") or "" for m in recent_subset]

                new_query = rewrite_query(summary, recent_contents, question)

            if new_query:
                embedding_query = new_query
                search_query = new_query
            else:
                prev_user = next(
                    (
                        m.get("content")
                        for m in reversed(history)
                        if m.get("role") == "user" and m.get("content")
                    ),
                    None,
                )
                if prev_user:
                    embedding_query = f"{prev_user} {question}".strip()

                # ВАЖНО: hybrid_search должен получать только текущий вопрос
                search_query = question

        # ---------------------------
        # Embedding + retrieval
        # ---------------------------
        with span("embed"):
            query_vec = embed_text(embedding_query)
        record_metric("embed_calls", 1)

        with span("retrieve"):
            hits = hybrid_search(search_query, query_vec, top_k=req.top_k)
        record_metric("search_calls", 1)

        # Build context and sources from the hits
        with span("build_context"):
            context_str, sources = build_context_from_hits(
                hits,
                max_tokens=settings.MAX_CONTEXT_TOKENS,
                token_fn=default_token_length if settings.MAX_CONTEXT_TOKENS else None,
                score_threshold=settings.SCORE_THRESHOLD,
            )
            _, sources_mapping = build_sources_mapping(sources)

        # Strict fallback when no relevant chunks were retrieved
        if not sources or not context_str.strip():
            record_metric("fallback_no_hits", 1)
            answer = FALLBACK_MESSAGE
            if req.thread_id:
                memory_store.append(thread_id, user_id, "user", question)
                memory_store.append(thread_id, user_id, "assistant", answer)
            return RagChatResponse(answer=answer, chunks=[])

        # KB-only guard (strict mode)
        if settings.CITATION_STRICT:
            max_score = 0.0
            for src in sources:
                raw = (src or {}).get("raw") or {}
                score = (
                    raw.get("score")
                    or raw.get("@search.score")
                    or src.get("score")
                    or 0.0
                )
                try:
                    score_val = float(score)
                except Exception:
                    score_val = 0.0
                if score_val > max_score:
                    max_score = score_val

            base_threshold = float(getattr(settings, "SCORE_THRESHOLD", 0.0) or 0.0)
            guard_min_score = max(0.01, base_threshold * 1.1)

            if max_score < guard_min_score and not is_relevant_to_sources(
                search_query, sources
            ):
                record_metric("fallback_irrelevant_hits", 1)
                answer = FALLBACK_MESSAGE
                if req.thread_id:
                    memory_store.append(thread_id, user_id, "user", question)
                    memory_store.append(thread_id, user_id, "assistant", answer)
                return RagChatResponse(answer=answer, chunks=[])

        # ---------------------------
        # Build the prompt messages
        # ---------------------------
        messages: List[dict[str, str]] = []

        messages.append({"role": "system", "content": BANKING_RAG_SYSTEM_PROMPT})

        if settings.CITATION_STRICT:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Answer ONLY using the provided CONTEXT. "
                        "If the CONTEXT does not contain the answer, reply exactly: "
                        f"{FALLBACK_MESSAGE}\n\n"
                        "Citation rules:\n"
                        "- Use inline citations like [S1], [S2] that refer to CONTEXT snippets.\n"
                        "- Every sentence (and every bullet item) MUST include at least one citation.\n"
                        "- Do not cite anything outside the provided Sources mapping.\n"
                        "- Do not invent sources.\n"
                    ),
                }
            )

        if req.thread_id:
            summary_for_prompt: Optional[str] = None
            if settings.SUMMARY_ENABLED:
                summary_for_prompt = memory_store.get_latest_summary(thread_id)
            if summary_for_prompt:
                messages.append(
                    {"role": "system", "content": f"PREVIOUS SUMMARY:\n{summary_for_prompt}"}
                )

            keep_k = max(settings.SUMMARY_KEEP_LAST_K, 0)
            if history:
                recent = history[-keep_k:] if keep_k > 0 else []
                for m in recent:
                    role = m.get("role")
                    content = m.get("content")
                    if role in {"user", "assistant"} and isinstance(content, str):
                        messages.append({"role": role, "content": content})

        src_map = sources_mapping
        messages.append(
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context_str}\n\n"
                    f"SOURCES MAPPING (copy these blob URLs into Sources):\n{src_map}\n\n"
                    f"QUESTION:\n{question}"
                ),
            }
        )

        # ---------------------------
        # Generate the answer
        # ---------------------------
        with span("generate"):
            answer = llm_chat(messages, max_tokens=900, temperature=0.2)
        record_metric("generate_calls", 1)

        # ---------------------------
        # Citation validation, repair and structure enforcement
        # ---------------------------
        if settings.CITATION_STRICT:
            with span("validate_citations"):
                ok_base = validate_answer_citations(answer, sources, enforce_structure=False)
            if not ok_base:
                with span("repair_citations"):
                    repair_prompt: List[dict[str, str]] = [
                        {
                            "role": "system",
                            "content": (
                                "You are fixing citation formatting.\n"
                                "Rules:\n"
                                "- Do NOT add any new facts.\n"
                                "- Do NOT change meaning.\n"
                                "- Ensure the answer ends with a Sources section."
                                "- In Sources, include one line per citation exactly as: [S1] <url>\n"
                                "- Every sentence and every bullet MUST include at least one inline citation like [S1].\n"
                                "- You may ONLY cite from the provided Sources mapping.\n"
                                "- Keep the same overall structure as the original answer.\n"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"SOURCES MAPPING:\n{src_map}\n\n"
                                "ORIGINAL ANSWER (fix citations only):\n"
                                f"{answer}\n"
                            ),
                        },
                    ]
                    repaired = llm_chat(repair_prompt, max_tokens=900, temperature=0.0)

                with span("validate_citations_repair"):
                    ok2 = validate_answer_citations(repaired, sources, enforce_structure=False)
                if ok2:
                    answer = repaired
                else:
                    record_metric("fallback_citation_fail", 1)
                    answer = structured_grounding_fallback((repaired or "").strip() or answer, sources)

        if answer != FALLBACK_MESSAGE:
            if settings.CITATION_STRICT and settings.CITATION_ENFORCE_STRUCTURE:
                answer = enforce_citation_structure(answer, sources, prefer_bold_heading=False)
                ok_structure = validate_answer_citations(answer, sources, enforce_structure=True)
                if not ok_structure:
                    record_metric("fallback_citation_structure_fail", 1)
                    answer = structured_grounding_fallback(answer, sources)
            else:
                answer = upsert_sources_section(answer, sources, prefer_bold_heading=False)

        # Persist conversation to memory if a thread ID was provided
        if req.thread_id:
            memory_store.append(thread_id, user_id, "user", question)
            memory_store.append(thread_id, user_id, "assistant", answer)

        chunks: List[RagChunk] = []
        if answer != FALLBACK_MESSAGE:
            for src in sources:
                raw = src.get("raw") or {}
                snippet = (raw.get(settings.TEXT_FIELD) or "").strip()
                blob_url = src.get("blob_url")
                uid = raw.get("uid")
                parent = raw.get("snippet_parent_id")
                score = raw.get("score") or raw.get("@search.score")
                chunks.append(
                    RagChunk(
                        id=str(uid) if uid is not None else None,
                        content=snippet,
                        source_file=str(blob_url) if blob_url else None,
                        chunk_id=str(parent) if parent else None,
                        score=float(score) if score is not None else None,
                        raw=raw,
                    )
                )

        return RagChatResponse(answer=answer, chunks=chunks)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"RAG chat failed: {type(e).__name__}") from e


@app.get("/ready")
def ready() -> dict[str, Any]:
    checks = {
        "embed": {"ok": False, "error": None},
        "search": {"ok": False, "error": None},
    }

    query_vec = None

    # EMBED check
    try:
        query_vec = embed_text("ping")  # вернёт вектор нужной размерности
        checks["embed"]["ok"] = True
    except Exception as e:
        checks["embed"]["error"] = type(e).__name__

    # SEARCH check
    try:
        if query_vec is None:
            raise RuntimeError("Embedding failed; cannot validate search.")
        hybrid_search("ping", query_vec, top_k=1)
        checks["search"]["ok"] = True
    except Exception as e:
        checks["search"]["error"] = type(e).__name__

    all_ok = checks["embed"]["ok"] and checks["search"]["ok"]
    if not all_ok:
        raise HTTPException(status_code=503, detail={"ready": False, "checks": checks})

    return {"ready": True, "checks": checks}



@app.get("/checks/liveness")
def liveness() -> dict[str, bool]:
    return health()

@app.get("/checks/readiness")
def readiness() -> dict[str, Any]:
    return ready()
