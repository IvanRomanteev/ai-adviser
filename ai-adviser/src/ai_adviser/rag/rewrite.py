"""
Query rewriting utilities for ai‑adviser.

This module provides helper functions to rewrite follow‑up user
questions into standalone search queries. Rewriting is important when
performing retrieval over a knowledge base because direct follow‑up
questions often lack context and therefore result in poor search recall.
By providing the conversation summary and recent messages to the
language model, the assistant can derive a complete query that
captures the relevant entities and constraints without adding any
external information. The rewriting behaviour can be toggled via
configuration (``settings.REWRITE_ENABLED``).

Example usage::

    from ai_adviser.rag.rewrite import rewrite_query
    summary = store.get_latest_summary(thread_id)
    recent_msgs = [m["content"] for m in history[-3:]]
    new_query = rewrite_query(summary, recent_msgs, question)
    if new_query:
        retrieval_query = new_query
    else:
        retrieval_query = f"{recent_msgs[-1]} {question}"

"""

from __future__ import annotations

from typing import List, Optional

from ai_adviser.clients.azure_models import chat as llm_chat
from ai_adviser.config import settings


def rewrite_query(
    summary: Optional[str], recent_messages: List[str], question: str
) -> Optional[str]:
    """Rewrite a follow‑up question into a standalone search query.

    Given a conversation summary, a list of recent message contents and
    the current question, this function asks the chat completion model
    to generate a fully specified query suitable for retrieval. The
    output should preserve all entities, numbers and constraints from
    the original question and must not introduce any information not
    present in the provided context. When rewriting is disabled via
    configuration the function returns ``None`` to signal that the
    caller should fall back to concatenation. If the model call
    fails or returns an empty string the function likewise returns
    ``None``.

    Args:
        summary: The latest conversation summary, if any.
        recent_messages: The most recent messages from the thread in
            chronological order. These messages provide additional
            context for understanding pronouns and implicit references
            in the question.
        question: The current user question to rewrite.

    Returns:
        A standalone query string or ``None`` if rewriting is disabled or
        unsuccessful.
    """
    if not settings.REWRITE_ENABLED:
        return None
    # Construct the context text by combining summary and recent messages.
    context_parts: List[str] = []
    if summary:
        context_parts.append(f"SUMMARY:\n{summary}")
    if recent_messages:
        # Only include non‑empty messages to avoid cluttering the prompt.
        filtered = [m for m in recent_messages if m.strip()]
        if filtered:
            context_parts.append("RECENT MESSAGES:\n" + "\n".join(filtered))
    context_text = "\n\n".join(context_parts).strip()
    # Compose the system prompt with clear instructions. We emphasise that
    # the model must not hallucinate new facts and should stick to the
    # original language of the question.
    system_prompt = (
        "You are an assistant helping to reformulate follow‑up questions "
        "into standalone search queries for a banking knowledge base.\n"
        "Rules:\n"
        "- Rewrite the user's question into a complete, standalone query.\n"
        "- Preserve all names, entities, numbers and constraints.\n"
        "- Do NOT add information that is not present in the conversation.\n"
        "- Keep the query concise and in the same language as the question."
    )
    user_prompt_lines: List[str] = []
    if context_text:
        user_prompt_lines.append(context_text)
    user_prompt_lines.append(f"QUESTION:\n{question}")
    user_prompt_lines.append(
        "Rewrite the question into a standalone search query without adding new information."
    )
    user_prompt = "\n\n".join(user_prompt_lines)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        result = llm_chat(
            messages,
            max_tokens=settings.REWRITE_MAX_TOKENS,
            temperature=settings.REWRITE_TEMPERATURE,
        )
        if result:
            return result.strip()
    except Exception:
        # On any error return None to indicate failure.
        return None
    return None
