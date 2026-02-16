# src/ai_adviser/api/search_endpoint.py

from fastapi import APIRouter
from pydantic import BaseModel

from ai_adviser.clients.azure_models import embed_text
from ai_adviser.clients.azure_search import hybrid_search
from ai_adviser.config import settings

router = APIRouter()


class SearchRequest(BaseModel):
    query: str


@router.post("/search")
def search(request: SearchRequest):
    query = (request.query or "").strip()
    if not query:
        return []

    query_vec = embed_text(query)
    hits = hybrid_search(query, query_vec, top_k=settings.TOP_K)
    return hits
