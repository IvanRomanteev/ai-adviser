"""
Conversation summarisation utilities for ai‑adviser.

This module provides functions to generate a concise summary of a
conversation thread using the chat completion model. Summaries are used to
condense long conversation histories so that they fit within the
context window when performing retrieval‑augmented generation. The
summarisation logic is controlled via environment variables defined in
``ai_adviser.config.settings``. When summarisation is disabled the
functions in this module return empty strings.

The summariser takes the entire conversation history as input and asks
the model to produce a short description of the key points and decisions
without introducing new information. The summary is intentionally
generic so that it can be prepended as system context in subsequent
generation requests.

Example usage::

    from ai_adviser.memory.summarizer import summarize_conversation
    from ai_adviser.memory.store import MemoryStore

    store = MemoryStore()
    history = store.get_history("thread1")
    if should_summarize(len(history)):
        summary = summarize_conversation(history)
        if summary:
            store.save_summary("thread1", summary)

"""

from __future__ import annotations

from typing import List, Dict, Optional

from ai_adviser.clients.azure_models import chat as llm_chat
from ai_adviser.config import settings


def summarize_conversation(history: List[Dict[str, str]]) -> str:
    """Return a concise summary of a conversation.

    This function uses the chat model to generate a short summary from
    the provided conversation history. The summary should cover the key
    topics, user requests and assistant responses without adding any
    external information. If summarisation is disabled via configuration
    the function returns an empty string.

    Args:
        history: A list of message dictionaries with keys ``role`` and
            ``content`` describing the conversation in chronological order.

    Returns:
        A trimmed summary string or an empty string if summarisation is
        disabled or the model call fails.
    """
    if not settings.SUMMARY_ENABLED:
        return ""
    # Build a textual representation of the conversation.  We include the
    # speaker role to help the model distinguish between user and assistant
    # turns.  Roles are capitalised for readability.
    conv_lines: List[str] = []
    for msg in history:
        role = (msg.get("role") or "").capitalize()
        content = msg.get("content") or ""
        conv_lines.append(f"{role}: {content}")
    conversation = "\n".join(conv_lines)
    # Compose the system prompt instructing the model how to summarise.
    system_prompt = (
        "You are a helpful assistant tasked with summarising a banking "
        "customer support conversation.\n"
        "Rules:\n"
        "- Provide a concise summary of the key points, decisions and "
        "questions raised in the conversation.\n"
        "- Do not introduce any new facts that are not present in the "
        "conversation.\n"
        "- Keep the summary short (a few sentences) to conserve tokens."
    )
    user_prompt = (
        f"CONVERSATION:\n{conversation}\n\n"
        "Write a concise summary of the above conversation."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        summary = llm_chat(
            messages,
            max_tokens=settings.SUMMARY_MAX_TOKENS,
            temperature=settings.SUMMARY_TEMPERATURE,
        )
        return (summary or "").strip()
    except Exception:
        # On any error return an empty summary. Errors are handled upstream.
        return ""


def should_summarize(total_messages: int) -> bool:
    """Determine whether a new summary should be generated.

    Summaries are generated after every ``SUMMARY_EVERY_N`` messages in a
    conversation to keep the memory footprint bounded. This helper
    encapsulates the scheduling logic so that callers need only pass
    in the current message count. When summarisation is disabled this
    always returns False.

    Args:
        total_messages: The current number of messages in the thread.

    Returns:
        True if the message count has reached the summarisation interval,
        False otherwise.
    """
    if not settings.SUMMARY_ENABLED:
        return False
    # Generate a summary after every SUMMARY_EVERY_N messages (e.g. 6).
    interval = max(settings.SUMMARY_EVERY_N, 1)
    return total_messages > 0 and total_messages % interval == 0
