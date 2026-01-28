# src/ai_adviser/clients/azure_search.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

# Azure Search dependencies are optional. These imports may fail if the Azure
# SDK packages are not installed. When unavailable the hybrid_search
# function will raise a RuntimeError when invoked.
try:
    from azure.core.credentials import AzureKeyCredential  # type: ignore
    from azure.search.documents import SearchClient  # type: ignore
    from azure.search.documents.models import VectorizedQuery  # type: ignore
except Exception:
    AzureKeyCredential = None  # type: ignore
    SearchClient = None  # type: ignore
    # stubs for VectorizedQuery (may be imported below in try/except)
    VectorizedQuery = None  # type: ignore

# tenacity is an optional dependency. When unavailable the retry
# decorators become no‑ops. See azure_models for similar logic.
try:
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
except Exception:
    def retry(*dargs, **dkwargs):  # type: ignore[no-redef]
        def wrapper(fn):
            return fn

        return wrapper

    def stop_after_attempt(*args, **kwargs):  # type: ignore[no-redef]
        return None

    def wait_exponential(*args, **kwargs):  # type: ignore[no-redef]
        return None

try:
    from azure.search.documents.models import VectorizedQuery  # type: ignore
except Exception:  # stubs/version issues
    VectorizedQuery = None  # type: ignore

from ai_adviser.config import settings


def _search_enabled() -> bool:
    return bool(getattr(settings, "AZURE_SEARCH_ENDPOINT", "") and getattr(settings, "AZURE_SEARCH_API_KEY", "") and getattr(settings, "AZURE_SEARCH_INDEX", ""))


@lru_cache(maxsize=1)
def get_search() -> SearchClient:
    if not _search_enabled():
        raise RuntimeError("Azure Search is not configured (AZURE_SEARCH_* env vars missing).")
    if SearchClient is None or AzureKeyCredential is None:
        raise RuntimeError(
            "Azure Search SDK is not installed. Please install azure-search-documents to perform searches."
        )
    return SearchClient(
        endpoint=settings.AZURE_SEARCH_ENDPOINT,
        index_name=settings.AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY),
    )


@retry(
    stop=stop_after_attempt(settings.RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def hybrid_search(query_text: str, query_vec: list[float], top_k: int) -> list[dict[str, Any]]:
    """Perform a hybrid (keyword + vector) search against Azure Cognitive Search.

    The search combines the raw query text with a vector representation to
    retrieve the most relevant snippets. A retry mechanism handles
    transient failures such as HTTP 5xx responses or timeouts. The
    results are normalised to include a `score` field for convenience.

    Args:
        query_text: The raw user question.
        query_vec: A list of floats representing the query embedding.
        top_k: Maximum number of results to return.

    Returns:
        A list of dictionaries representing the search hits. Each
        dictionary will at least contain the snippet text, a unique
        identifier and the computed relevance score.
    """
    if VectorizedQuery is None:
        raise RuntimeError(
            "VectorizedQuery is unavailable in your azure-search-documents version."
        )
    search = get_search()
    vq = VectorizedQuery(
        vector=query_vec,
        k_nearest_neighbors=top_k,
        fields=settings.VECTOR_FIELD,
    )
    results = search.search(
        search_text=query_text,
        vector_queries=[vq],  # type: ignore[arg-type]
        top=top_k,
        select=["uid", settings.TEXT_FIELD, "blob_url", "snippet_parent_id"],
        # pass through request timeout from settings
        timeout=settings.TIMEOUT_SECONDS,
    )
    out: list[dict[str, Any]] = []
    for r in results:
        d = dict(r)
        score = d.get("@search.score")
        if score is not None:
            d["score"] = score
        out.append(d)
    return out
