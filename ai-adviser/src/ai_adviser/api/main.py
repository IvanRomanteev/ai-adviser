# src/ai_adviser/api/main.py
from __future__ import annotations

import logging

from azure.core.exceptions import ClientAuthenticationError
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import uuid
import time
from typing import Optional

from ai_adviser.api.schemas import (
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    RagChatRequest,
    RagChatResponse,
    RagChunk,
)
from ai_adviser.clients.azure_models import chat as llm_chat
from ai_adviser.clients.azure_models import embed_text
from ai_adviser.clients.azure_search import hybrid_search
from ai_adviser.config import settings
from ai_adviser.memory.store import MemoryStore
from ai_adviser.observability.langfuse import span
from ai_adviser.observability.metrics import record_metric, collector, setup_metrics
from ai_adviser.observability.tracing import init_tracing
from ai_adviser.prompts.loader import load_prompt
from ai_adviser.rag.citations import validate_answer_citations,build_sources_mapping, upsert_sources_section
from ai_adviser.rag.rewrite import rewrite_query
from ai_adviser.rag.context import build_context_from_hits, default_token_length
from ai_adviser.memory.summarizer import summarize_conversation, should_summarize


logger = logging.getLogger(__name__)


class UTF8JSONResponse(JSONResponse):
    """Force UTF-8 charset in JSON responses.

    PowerShell's Invoke-RestMethod can mis-decode UTF-8 as cp1252/latin-1 when
    the charset is missing. Setting it explicitly prevents "â€™" artifacts.
    """

    media_type = "application/json; charset=utf-8"


BANKING_RAG_SYSTEM_PROMPT = load_prompt("rag_system.md")

# Initialise the conversation memory store. The backend is selected via
# configuration. For now only an SQLite implementation is provided. The
# store persists data between requests and can be swapped for a Postgres
# implementation in production by adjusting the configuration.
if settings.CHECKPOINTER_BACKEND.lower() == "postgres" and settings.POSTGRES_DSN:
    # In a production environment we could implement a Postgres backed
    # memory store. Fallback to SQLite if Postgres DSN is not provided.
    memory_store = MemoryStore(settings.SQLITE_DB_PATH)
else:
    memory_store = MemoryStore(settings.SQLITE_DB_PATH)


app = FastAPI(
    title="ai-adviser",
    version="0.1.0",
    default_response_class=UTF8JSONResponse,
)

# Initialise metrics and tracing instrumentation.  These functions are
# idempotent and respect the METRICS_ENABLED and TRACING_ENABLED flags in
# configuration.  They must be called before defining any routes that rely on
# instrumentation.
setup_metrics(app)
init_tracing(app)

# Add a middleware to measure HTTP request latency, track error counts and
# propagate a request identifier.  This is executed for every request,
# including static routes and /metrics when metrics are enabled.
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    # Generate or reuse the X-Request-ID.  If the incoming request
    # already carries an ID we reuse it; otherwise we create a new one.
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    # Record the start time for latency calculation
    start = time.perf_counter()
    response: Response
    error_occurred = False
    try:
        response = await call_next(request)
    except Exception as exc:
        # Mark that an exception occurred so we can increment the error
        # counter.  We must re-raise after recording.
        error_occurred = True
        raise
    finally:
        # Compute elapsed time
        elapsed = time.perf_counter() - start
        # Record request latency metric if enabled
        try:
            path = request.url.path
            method = request.method
            status_code = response.status_code if 'response' in locals() else 500
            collector.observe_request(path, method, status_code, elapsed)
        except Exception:
            pass
        # Increment error counter if an exception was raised
        if error_occurred:
            collector.increment("error_count", 1)
    # Ensure the X-Request-ID is attached to the response for traceability
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    try:
        vec = embed_text(req.text)
        return EmbedResponse(dims=len(vec), embedding=vec)
    except ClientAuthenticationError as e:
        raise HTTPException(status_code=401, detail="Invalid Azure AI credentials") from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {type(e).__name__}") from e


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        msgs = [m.model_dump() for m in req.messages]
        text = llm_chat(
            msgs,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return ChatResponse(text=text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Chat request failed: {type(e).__name__}") from e


@app.post("/rag_chat", response_model=RagChatResponse)
def rag_chat(req: RagChatRequest):
    """Perform a retrieval-augmented chat with optional conversation memory.

    Pipeline:
    - (optional) expand retrieval query using previous user turn (Variant A)
    - embed(retrieval_query)
    - hybrid_search(retrieval_query)
    - build context with token budget
    - generate answer with LLM
    - validate citations; if invalid: try one repair pass; if still invalid -> fallback
    - persist memory when thread_id is provided
    """

    try:
        question = (req.question or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question must not be empty")

        thread_id = req.thread_id or "default"
        user_id = req.user_id

        record_metric("rag_chat_calls", 1)

        # ---------------------------
        # Query rewriting for follow‑up questions
        # ---------------------------
        # The retrieval query defaults to the raw question.  When a thread
        # identifier is provided we attempt to rewrite the question using
        # conversation context.  If rewriting is disabled or fails we fall
        # back to concatenating the last user message with the current
        # question, preserving the original behaviour.
        retrieval_query = question
        history: list[dict[str, str]] = []
        # Only load history when a thread ID was provided.  When no ID is
        # supplied we treat the request as a single‑turn interaction and
        # avoid incurring the cost of database access.
        if req.thread_id:
            history = memory_store.get_history(thread_id)
            # Determine the latest conversation summary (if any) to aid
            # rewriting.  Summaries are only used when summarisation is
            # enabled; otherwise ``summary`` remains None.
            summary: Optional[str] = None
            if settings.SUMMARY_ENABLED:
                summary = memory_store.get_latest_summary(thread_id)
            # Collect the contents of the last REWRITE_LAST_N messages
            recent_contents: list[str] = []
            if history:
                # Take up to the last N messages from the end of the history
                recent_subset = history[-settings.REWRITE_LAST_N :]
                recent_contents = [msg.get("content") or "" for msg in recent_subset]
            # Attempt query rewriting
            new_query: Optional[str] = rewrite_query(summary, recent_contents, question)
            if new_query:
                retrieval_query = new_query
            else:
                # Fallback to concatenating the last user message with the question
                prev_user_msgs = [m["content"] for m in history if m.get("role") == "user"]
                if prev_user_msgs:
                    retrieval_query = f"{prev_user_msgs[-1]} {question}"

        # Embed + retrieve
        with span("embed"):
            query_vec = embed_text(retrieval_query)
        # Count embedding calls for metrics
        record_metric("embed_calls", 1)

        with span("retrieve"):
            hits = hybrid_search(retrieval_query, query_vec, top_k=req.top_k)
        record_metric("search_calls", 1)

        # Build context
        with span("build_context"):

            context_str, sources = build_context_from_hits(
                hits,
                max_tokens=settings.MAX_CONTEXT_TOKENS,
                token_fn=default_token_length if settings.MAX_CONTEXT_TOKENS else None,
                score_threshold=settings.SCORE_THRESHOLD,
            )
            # Deterministic mapping of citations to blob URLs.
            # We pass this mapping to the model AND also use it to
            # post-process the final answer so the Sources section always
            # contains real blob URLs instead of placeholders or external URLs.
            _, sources_mapping = build_sources_mapping(sources)

        # No relevant chunks -> strict fallback
        if not sources or not context_str.strip():
            # Record fallback due to no hits
            record_metric("fallback_no_hits", 1)
            answer = "Not found in the knowledge base."
            if req.thread_id:
                memory_store.append(thread_id, user_id, "user", question)
                memory_store.append(thread_id, user_id, "assistant", answer)
                # After persisting the messages consider summarising
                try:
                    total_msgs = len(memory_store.get_history(thread_id))
                    if should_summarize(total_msgs):
                        hist = memory_store.get_history(thread_id)
                        summary_text = summarize_conversation(hist)
                        if summary_text:
                            memory_store.save_summary(thread_id, summary_text)
                except Exception:
                    pass
            return RagChatResponse(answer=answer, chunks=[])

        # Build message sequence for the generation step.  We include a
        # conversation summary and only the last few messages instead of
        # the entire history to keep the context window bounded.  The
        # summary is prepended as a system message so that the model can
        # reference prior turns without seeing all messages.  When there
        # is no summary we skip this part.
        messages: list[dict[str, str]] = []
        if req.thread_id:
            # Include the latest summary if summarisation is enabled
            summary_for_prompt: Optional[str] = None
            if settings.SUMMARY_ENABLED:
                try:
                    summary_for_prompt = memory_store.get_latest_summary(thread_id)
                except Exception:
                    summary_for_prompt = None
            if summary_for_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": f"PREVIOUS SUMMARY:\n{summary_for_prompt}",
                    }
                )
            # Append the last K messages from history to preserve recent context
            try:
                keep_k = max(settings.SUMMARY_KEEP_LAST_K, 0)
            except Exception:
                keep_k = 0
            if history:
                recent = history[-keep_k:] if keep_k > 0 else []
                for m in recent:
                    messages.append({"role": m["role"], "content": m["content"]})
        # Append the system prompt defining RAG behaviour
        messages.append({"role": "system", "content": BANKING_RAG_SYSTEM_PROMPT})
        # Append the user message containing the retrieved context and the question
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

        # Generate answer
        with span("generate"):
            answer = llm_chat(messages, max_tokens=900, temperature=0.2)
            # Make the Sources section deterministic and grounded in retrieved blob URLs.
            # This fixes cases where the model outputs placeholders or external site URLs.
            answer = upsert_sources_section(answer, sources, prefer_bold_heading=False)

        # Count generation calls for metrics
        record_metric("generate_calls", 1)

        # ---------------------------
        # Strict citations + repair loop
        # ---------------------------
        # Validate citations. If invalid in strict mode, attempt one repair pass, then fallback.
        fallback_due_to_citations = False

        if settings.CITATION_STRICT:
            with span("validate_citations"):
                ok = validate_answer_citations(answer, sources)

            if not ok:
                # --- Repair pass (1 retry) ---
                with span("repair_citations"):
                    # Build a compact mapping for the model
                    # Example: S1 -> blob_url, S2 -> blob_url ...
                    src_lines = []
                    for s in sources:
                        sid = s.get("id")
                        burl = s.get("blob_url")
                        src_lines.append(f"[S{sid}] {burl}")
                    src_map = sources_mapping

                    repair_prompt = [
                        {
                            "role": "system",
                            "content": (
                                "You are fixing citation formatting.\n"
                                "Rules:\n"
                                "- Do NOT add any new facts.\n"
                                "- Do NOT change meaning.\n"
                                "- Ensure the answer ends with a Sources section."
                                "- In Sources, include one line per citation exactly as: [S1] <url>"
                                "- Every sentence and every bullet MUST include at least one inline citation like [S1].\n"
                                "- You may ONLY cite from the provided Sources mapping.\n"
                                "- Keep the same overall structure as the original answer.\n"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "SOURCES MAPPING:\n"
                                f"{src_map}\n\n"
                                "ORIGINAL ANSWER (fix citations only):\n"
                                f"{answer}\n"
                            ),
                        },
                    ]

                    # lower temp for deterministic formatting
                    repaired = llm_chat(repair_prompt, max_tokens=900, temperature=0.0)
                    repaired = upsert_sources_section(repaired, sources, prefer_bold_heading=False)

                with span("validate_citations_repair"):
                    ok2 = validate_answer_citations(repaired, sources)

                if ok2:
                    answer = repaired
                else:
                    # Record fallback due to citation validation failure
                    record_metric("fallback_citation_fail", 1)
                    answer = "Not found in the knowledge base."
                    fallback_due_to_citations = True

        # Persist to memory
        if req.thread_id:
            memory_store.append(thread_id, user_id, "user", question)
            memory_store.append(thread_id, user_id, "assistant", answer)
            # After persisting the messages, determine whether a new summary
            # should be created and saved.  Summaries are generated after
            # every SUMMARY_EVERY_N messages.  We catch any exceptions
            # to avoid disrupting the main flow in case summarisation fails.
            try:
                total_msgs = len(memory_store.get_history(thread_id))
                if should_summarize(total_msgs):
                    hist = memory_store.get_history(thread_id)
                    summary_text = summarize_conversation(hist)
                    if summary_text:
                        memory_store.save_summary(thread_id, summary_text)
            except Exception:
                pass

        # Return chunks only when we have a grounded answer
        chunks: list[RagChunk] = []
        if not fallback_due_to_citations:
            for src in sources:
                raw = src.get("raw") or {}
                snippet = (raw.get(settings.TEXT_FIELD) or "").strip()
                blob_url = raw.get("blob_url")
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
def ready():
    """Readiness probe that checks embeddings + search client."""

    checks = {
        "embed": {"ok": False, "error": None},
        "search": {"ok": False, "error": None},
    }

    # EMBED check
    try:
        embed_text("ping")
        checks["embed"]["ok"] = True
    except RuntimeError as e:
        # SDK missing or similar - treat as dev-ok if you want
        checks["embed"]["ok"] = True
        checks["embed"]["error"] = type(e).__name__
    except Exception as e:
        checks["embed"]["error"] = type(e).__name__

    # SEARCH check
    try:
        from ai_adviser.clients.azure_search import get_search

        get_search()
        checks["search"]["ok"] = True
    except RuntimeError as e:
        checks["search"]["ok"] = True
        checks["search"]["error"] = type(e).__name__
    except Exception as e:
        checks["search"]["error"] = type(e).__name__

    all_ok = checks["embed"]["ok"] and checks["search"]["ok"]
    if not all_ok:
        raise HTTPException(status_code=503, detail={"ready": False, "checks": checks})

    return {"ready": True, "checks": checks}
