"""Deterministic relevance guard for KB-only / strict RAG mode.

Why this exists
---------------
Hybrid retrieval can return *some* chunks even for unrelated questions
(because keyword + vector retrieval are merged and scored). If the model is
allowed to answer with those chunks, it may produce a fluent answer and
attach citations that are technically valid but semantically irrelevant.

This module provides a lightweight, deterministic guard used by the API
layer: when strict mode is enabled, we only proceed to generation if the
retrieved chunks share at least some meaningful lexical overlap with the
retrieval query. Otherwise we return the standard fallback message.

The heuristic is intentionally conservative and configurable only via code
(to keep settings minimal). It prioritizes blocking obvious out-of-domain
questions like "What is the capital of Paris?" while still allowing
follow-up questions when query rewriting/history concatenation is enabled.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set

from ai_adviser.config import settings

# Basic multilingual tokenization (Latin + Cyrillic + digits + apostrophe).
_WORD_RE = re.compile(r"[A-Za-zА-Яа-я0-9']+")

# Small stopword sets (not exhaustive; designed for the guard only).
_STOPWORDS_EN: Set[str] = {
    "a", "an", "the", "and", "or", "but",
    "what", "why", "how", "when", "where", "who",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "i", "you", "we", "they", "he", "she", "it", "my", "your", "our", "their",
    "this", "that", "these", "those",
    "in", "on", "at", "to", "from", "for", "of", "with", "as", "by", "about", "into", "inside",
    "each", "every", "per",
    "month", "months", "weekly", "daily", "year", "years",
    "much", "many", "more", "most", "some", "any",
}

_STOPWORDS_RU: Set[str] = {
    "и", "или", "а", "но", "да",
    "что", "как", "где", "когда", "почему", "зачем", "кто", "какой", "какая", "какие", "каково",
    "это", "этот", "эта", "эти", "тот", "та", "те",
    "я", "ты", "вы", "мы", "они", "он", "она", "оно",
    "в", "во", "на", "к", "ко", "из", "за", "для", "по", "о", "об", "от", "до",
    "у", "с", "со",
    "мне", "тебе", "вам", "нам", "им",
    "мой", "моя", "мои", "твой", "твоя", "ваш", "ваша",
}

# Domain-generic words that appear all over banking content and are not useful
# for detecting topical alignment. This list is intentionally short.
_DOMAIN_GENERIC: Set[str] = {
    "bank", "banking", "app", "apps", "account", "accounts",
}

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]

def _keywords(text: str) -> List[str]:
    tokens = _tokenize(text)
    out: List[str] = []
    for t in tokens:
        if t in _STOPWORDS_EN or t in _STOPWORDS_RU or t in _DOMAIN_GENERIC:
            continue
        # Keep reasonably informative tokens only.
        if len(t) < 4:
            continue
        out.append(t)
    # Deduplicate preserving order
    seen: Set[str] = set()
    dedup: List[str] = []
    for t in out:
        if t in seen:
            continue
        seen.add(t)
        dedup.append(t)
    return dedup

def is_relevant_to_sources(query_text: str, sources: Iterable[Dict[str, Any]]) -> bool:
    """Return True if retrieved sources look relevant to the query.

    Heuristic:
    - Build a lowercase "haystack" from all retrieved snippets.
    - Extract keywords from the query (stopwords removed).
    - If any *long* keyword (len>=6) exists, require at least one long keyword to appear.
      This blocks obvious off-topic questions like "bitcoin" even if generic words match.
    - Otherwise, require at least one keyword match.
    - If we can't extract any keywords, don't block (return True).

    This is not a semantic guarantee, but it prevents the most common failure
    mode: answering unrelated questions with valid-looking citations.
    """
    # Concatenate snippet text from raw hits.
    parts: List[str] = []
    for src in sources:
        raw = src.get("raw") or {}
        snippet = raw.get(settings.TEXT_FIELD) or raw.get("snippet") or ""
        if isinstance(snippet, str) and snippet.strip():
            parts.append(snippet)
    haystack = " ".join(parts).lower()

    kws = _keywords(query_text)
    if not kws:
        return True

    long_kws = [k for k in kws if len(k) >= 6]
    target = long_kws if long_kws else kws

    for k in target:
        if k in haystack:
            return True
    return False
