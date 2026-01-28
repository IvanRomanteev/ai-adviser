"""
Utilities for constructing RAG contexts from retrieved search results.

The context builder is responsible for selecting a subset of retrieved
documents based on relevance scores, assigning citation identifiers and
producing a concatenated text block that can be passed to the language
model. A token budget can optionally be enforced via a tokenisation
function. When no token function is supplied the context is truncated by
character length using the configuration's `MAX_CONTEXT_CHARS`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

from ai_adviser.config import settings


def default_token_length(text: str) -> int:
    """Approximate the number of tokens in a string by counting whitespace‑
    separated words.

    This heuristic avoids pulling in heavy tokeniser dependencies such as
    `tiktoken` while still providing a rough control over context size.

    Args:
        text: The input string.

    Returns:
        An integer representing the approximate token count.
    """
    # Split on whitespace and count resulting segments
    return len(text.split())


def build_context_from_hits(
    hits: List[Dict[str, object]],
    *,
    max_tokens: int | None = None,
    token_fn: Callable[[str], int] | None = None,
    score_threshold: float = 0.0,
) -> Tuple[str, List[Dict[str, object]]]:
    """Construct a context string and associated source metadata from search hits.

    Args:
        hits: Search result dictionaries. Each entry should include at least
            `settings.TEXT_FIELD` for the snippet and may include `blob_url` and
            `score` keys.
        max_tokens: Optional maximum allowed number of tokens. If not
            provided, the default from settings (`MAX_CONTEXT_TOKENS`) is used.
        token_fn: A function to compute the token length of a string. If
            omitted the `default_token_length` heuristic is used. If set to
            None, token budget enforcement is disabled and the context is
            truncated based on character length (`MAX_CONTEXT_CHARS`).
        score_threshold: Only include hits with a score above this value.

    Returns:
        A tuple `(context, sources)` where `context` is a newline‑separated
        string with inline citation markers (e.g. [S1]) and `sources` is a
        list of dictionaries containing `id`, `blob_url` and `raw` keys.
    """
    # Determine maximum tokens from settings if not explicitly provided
    if max_tokens is None:
        max_tokens = getattr(settings, "MAX_CONTEXT_TOKENS", None)
    if token_fn is False:
        # explicitly disable token limits when token_fn is False
        max_tokens = None
    # Use default token length heuristic when a token budget is specified but
    # no custom token function was provided
    if max_tokens is not None and token_fn is None:
        token_fn = default_token_length

    context_lines: List[str] = []
    sources: List[Dict[str, object]] = []
    # Track the approximate number of tokens used so far. When a token function
    # is provided we will account for citation markers as a single token and
    # snippets based on the heuristic. This allows us to truncate the last
    # snippet rather than omitting it entirely when the budget would be
    # exceeded.
    tokens_used = 0
    # Build up context lines until token or char budget is exceeded
    for idx, hit in enumerate(hits):
        # Extract score and skip if below threshold
        score = hit.get("score") or hit.get("@search.score") or 0.0
        try:
            score_val = float(score)
        except Exception:
            score_val = 0.0
        if score_val < score_threshold:
            continue
        snippet = (hit.get(settings.TEXT_FIELD) or "").strip()
        if not snippet:
            continue
        blob_url = hit.get("blob_url") or hit.get("source_file")
        # Citation numbers are 1‑indexed
        citation_id = len(sources) + 1
        # Determine whether we need to enforce a token budget for this snippet
        if max_tokens is not None and token_fn is not None:
            # Calculate the token budget available for this snippet. A citation
            # marker (e.g. [S1]) is considered to consume exactly one token in
            # the context. The snippet itself is tokenised by the provided
            # `token_fn`.
            remaining_budget = max_tokens - tokens_used - 1  # reserve 1 token for the marker
            if remaining_budget <= 0:
                # No room even for the marker; stop building context
                break
            # Compute the tokenised words of the snippet
            snippet_tokens = snippet.split()
            # Truncate the snippet if it exceeds the remaining budget
            if token_fn(snippet) > remaining_budget:
                snippet_tokens = snippet_tokens[:remaining_budget]
                # If truncation leaves no tokens, skip this snippet entirely
                if not snippet_tokens:
                    break
                snippet = " ".join(snippet_tokens)
            # Append the citation and truncated snippet
            context_lines.append(f"[S{citation_id}] {snippet}")
            sources.append({"id": citation_id, "blob_url": blob_url, "raw": hit})
            # Update the token usage: marker + snippet
            tokens_used += 1 + len(snippet_tokens)
            # If token usage has reached the maximum, stop adding further snippets
            if tokens_used >= max_tokens:
                break
        else:
            # No token budget enforcement; append full snippet and check character budget
            context_lines.append(f"[S{citation_id}] {snippet}")
            sources.append({"id": citation_id, "blob_url": blob_url, "raw": hit})
            context_str = "\n\n".join(context_lines)
            if len(context_str) > settings.MAX_CONTEXT_CHARS:
                # Remove the last added line and stop
                context_lines.pop()
                sources.pop()
                break

    context = "\n\n".join(context_lines)
    return context, sources