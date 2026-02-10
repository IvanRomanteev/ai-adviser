"""ADK orchestration for the banking adviser use case.

This module defines a `BankingOrchestrator` class which acts as the
top‑level agent in an Agent Development Kit (ADK) style workflow.  The
orchestrator dispatches user requests to one of three tool functions
depending on the request intent: a retrieval‑augmented chat tool, a
profile loader and a budget planner.  It also persists user and
conversation state between invocations using a simple JSON file based
store.

The orchestrator can be instantiated and invoked directly; see the
``README.md`` in this package for an example.  All configuration is
provided via environment variables:

``RAG_BASE_URL``
    The base URL of the RAG service.  Required by the
    ``banking_rag_chat_tool``.

``USER_PROFILE_SOURCE``
    Path or URL to a JSON document containing the user's profile.
    Used by ``get_user_basic_profile``.  When omitted, the default
    ``demo-uprof.json`` bundled with the repository is used.

``MEMORY_STORE_PATH``
    Directory where per‑user memory files are stored.  Defaults to
    ``.memory`` under the current working directory.  Each file is
    named ``user_<user_id>.json`` and contains the profile,
    conversation history and a rolling summary.

The orchestrator logs tool invocations and associates a unique
``request_id`` with each high level request.  OpenTelemetry tracing
support can be enabled by configuring the ``OTEL_TRACES_EXPORTER``
environment variable; if installed, spans will be created around tool
calls using the ``opentelemetry`` API.  Tracing support is optional
and gracefully degrades when the library is not available.

Note
----
This module does not import or run any web framework.  It is
intended to be used as part of a background worker or an API layer
that accepts user input and returns the orchestrated response.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import the tools module itself rather than the individual functions.  This
# indirection allows our tests to monkeypatch attributes on the tools
# module and have those changes observed by the orchestrator.  If we
# imported the functions directly they would be bound at import time and
# subsequent monkeypatches would not take effect.  See tests for an
# example.
from . import tools

# Attempt to import OpenTelemetry.  If not present tracing will be a no‑op.
try:
    from opentelemetry import trace  # type: ignore[import]
    from opentelemetry.trace import Tracer  # type: ignore[import]

    _tracer: Optional[Tracer] = trace.get_tracer(__name__)
except Exception:
    _tracer = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _ensure_dir(path: pathlib.Path) -> None:
    """Ensure that a directory exists."""
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def _strip_grounding_appendix(text: str) -> str:
    """Remove the structured grounding appendix from the assistant answer before persisting.

    This prevents memory/summary pollution with:
      - "Grounding note"
      - "Closest sources: ..."
    """
    if not isinstance(text, str):
        return str(text)

    if ("Grounding note" in text) and ("Closest sources" in text):
        # structured fallback uses '---' separator
        if "\n---\n" in text:
            return text.split("\n---\n", 1)[0].strip()
        if "---" in text:
            return text.split("---", 1)[0].strip()

    return text.strip()


@dataclass
class ConversationMemory:
    """Simple JSON file backed memory for conversation state.

    Each user gets a separate file containing their profile, a list of
    history messages (dicts with ``role`` and ``content``) and a
    summary string.  The file is read at the beginning of each
    interaction and written back once the orchestrator has updated the
    state.
    """

    memory_dir: pathlib.Path
    user_id: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _ensure_dir(self.memory_dir)
        self.file_path = self.memory_dir / f"user_{self.user_id}.json"
        # Load existing file if present
        if self.file_path.exists():
            try:
                with self.file_path.open("r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                logger.warning(
                    "Failed to load memory file %s; starting fresh", self.file_path
                )
                self.data = {}
        else:
            self.data = {}

    def get_profile(self) -> Dict[str, Any]:
        return self.data.get("profile", {})

    def set_profile(self, profile: Dict[str, Any]) -> None:
        self.data["profile"] = profile

    def get_history(self, thread_id: str) -> List[Dict[str, str]]:
        return self.data.get("history", {}).get(thread_id, [])

    def append_history(self, thread_id: str, role: str, content: str) -> None:
        history = self.data.setdefault("history", {}).setdefault(thread_id, [])
        history.append({"role": role, "content": content})

    def get_summary(self, thread_id: str) -> Optional[str]:
        return self.data.get("summary", {}).get(thread_id)

    def set_summary(self, thread_id: str, summary: str) -> None:
        summaries = self.data.setdefault("summary", {})
        summaries[thread_id] = summary

    def persist(self) -> None:
        tmp_path = self.file_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.file_path)


class BankingOrchestrator:
    """Root agent orchestrating banking queries.

    Parameters
    ----------
    user_id : str
        Unique identifier for the user.  Used to segregate memory and
        profile information.
    memory_store_path : Optional[str]
        Optional override for the directory in which memory files are
        stored.  Defaults to the value of the ``MEMORY_STORE_PATH``
        environment variable or ``.memory`` under the current working
        directory.

    Examples
    --------
    >>> orch = BankingOrchestrator(user_id="u123")
    >>> response = orch.run("What fees do you charge?", thread_id="t1")
    ... # dispatches to the RAG chat tool
    >>> response["answer"]  # doctest: +SKIP
    "..."
    """

    def __init__(
        self, user_id: str, *, memory_store_path: Optional[str] = None
    ) -> None:
        self.user_id = user_id
        mem_dir = pathlib.Path(
            memory_store_path or os.environ.get("MEMORY_STORE_PATH", ".memory")
        )
        self.memory = ConversationMemory(mem_dir, user_id)
        # Load the profile into memory at startup
        if not self.memory.get_profile():
            # Call through the tools module so that monkeypatches on
            # tools.get_user_basic_profile are respected.  See tests for
            # details.
            profile = tools.get_user_basic_profile()  # type: ignore[call-arg]
            self.memory.set_profile(profile)
            self.memory.persist()

    def _summarize_history(self, thread_id: str) -> Optional[str]:
        """Produce a short summary of the conversation so far.

        For the MVP we implement a very simple heuristic: if the
        conversation exceeds 10 messages, we keep only the last 3 and
        store a summary of the earlier messages consisting of their
        concatenated contents truncated to 500 characters.  This
        prevents unbounded context growth.  A production version could
        integrate a proper LLM summariser.
        """
        hist = self.memory.get_history(thread_id)
        if len(hist) <= 10:
            return None
        # Extract earlier messages excluding the last 3
        earlier = hist[:-3]
        text = "\n".join([m["content"] for m in earlier])
        summary = text[:500]
        return summary

    def run(self, message: str, *, thread_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user message and return the agent's response.

        The orchestrator inspects the message content to decide which
        tool to invoke.  Keywords like ``budget`` or ``plan`` trigger
        the budget planner; otherwise the retrieval tool is used.  The
        user's basic profile is loaded once and cached.  Conversation
        history and summaries are persisted between calls.

        Parameters
        ----------
        message : str
            The user's input.
        thread_id : Optional[str]
            Identifier grouping messages from the same conversation.  If
            omitted, a default thread is used and no history is
            recorded.
        Returns
        -------
        dict
            A dictionary containing the result of the invoked tool.
        """
        if not message or not message.strip():
            raise ValueError("message must not be empty")

        question = message.strip()
        tid = thread_id or "default"
        tool_thread_id = thread_id  # pass None to tools when caller is stateless

        # Generate a request ID for observability
        request_id = uuid.uuid4().hex
        logger.info(
            "request_id=%s user_id=%s thread_id=%s message=%s",
            request_id,
            self.user_id,
            tid,
            question,
        )

        # Retrieve profile for personalisation
        profile = self.memory.get_profile()

        # Determine intent based on simple keyword matching
        lower = question.lower()
        if any(kw in lower for kw in ["budget", "plan", "goal", "save", "buy", "purchase"]):
            tool_name = "budget_planner_tool"
            args = {
                "goal": question,
                "thread_id": tid,
                "user_id": self.user_id,
                "profile": profile,
                "memory": self.memory,
                "request_id": request_id,
            }
            result = self._call_tool(tool_name, args)

        elif any(kw in lower for kw in ["profile", "who am i", "about me"]):
            tool_name = "get_user_basic_profile"
            args = {"request_id": request_id}
            result = self._call_tool(tool_name, args)

        else:
            tool_name = "banking_rag_chat_tool"
            args = {
                "question": question,
                "top_k": 5,
                "thread_id": tool_thread_id,
                "user_id": self.user_id,
                "request_id": request_id,
            }
            result = self._call_tool(tool_name, args)

        # Append to history if we have a thread id
        if thread_id:
            self.memory.append_history(tid, "user", question)

            # Persist assistant content without structured grounding appendix
            assistant_content = result.get("answer")
            if isinstance(assistant_content, str) and assistant_content.strip():
                assistant_content = _strip_grounding_appendix(assistant_content)
            else:
                assistant_content = json.dumps(result, ensure_ascii=False)

            self.memory.append_history(tid, "assistant", assistant_content)

            # Possibly summarise conversation
            summary = self._summarize_history(tid)
            if summary:
                self.memory.set_summary(tid, summary)
            self.memory.persist()

        return result

    def _call_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool with optional tracing instrumentation."""
        span_ctx = None
        if _tracer is not None:
            span_ctx = _tracer.start_as_current_span(tool_name)

        try:
            if span_ctx:
                span_ctx.__enter__()

            logger.info(
                "Calling tool %s with args=%s",
                tool_name,
                {k: v for k, v in args.items() if k != "memory"},
            )

            if tool_name == "banking_rag_chat_tool":
                # Dispatch via the tools module.  Note: mypy may complain
                # about type arguments but at runtime this is resolved.
                return tools.banking_rag_chat_tool(**args)  # type: ignore[arg-type]

            if tool_name == "get_user_basic_profile":
                return tools.get_user_basic_profile()  # type: ignore[call-arg]

            if tool_name == "budget_planner_tool":
                # Inject profile and memory to planner via the tools module
                return tools.budget_planner_tool(**args)  # type: ignore[arg-type]

            raise ValueError(f"Unknown tool: {tool_name}")

        finally:
            if span_ctx:
                span_ctx.__exit__(None, None, None)
