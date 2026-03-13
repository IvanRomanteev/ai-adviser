# src/ai_adviser/api/search_endpoint.py

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai_adviser.clients.azure_models import embed_text
from ai_adviser.clients.azure_search import hybrid_search
from ai_adviser.config import settings
from ai_adviser.rag.context import build_context_from_hits, default_token_length
from ai_adviser.rag.citations import build_sources_mapping

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    sources: str


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    query = (request.query or "").strip()


    if not query:
        return SearchResponse(sources="")

    # 1) Embed
    try:
        query_vec = embed_text(query)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Embedding failed: {type(e).__name__}"
        ) from e

    # 2) Search
    try:
        hits = hybrid_search(query, query_vec, top_k=settings.TOP_K)
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Search failed: {type(e).__name__}"
        ) from e


    try:
        context_str, sources_meta = build_context_from_hits(
            hits,
            max_tokens=settings.MAX_CONTEXT_TOKENS,
            token_fn=default_token_length if settings.MAX_CONTEXT_TOKENS else None,
            score_threshold=settings.SCORE_THRESHOLD,
        )
        _, sources_mapping = build_sources_mapping(sources_meta)

        sources_text = (
            f"CONTEXT:\n{context_str}\n\n"
            f"SOURCES:\n{sources_mapping}\n"
        )


        if getattr(settings, "MAX_CONTEXT_CHARS", None):
            max_chars = int(settings.MAX_CONTEXT_CHARS)
            if max_chars > 0 and len(sources_text) > max_chars:
                sources_text = sources_text[:max_chars]

        return SearchResponse(sources=sources_text)

    except Exception:

        return SearchResponse(sources=json.dumps(hits, ensure_ascii=False))
