"""
Configuration management for ai‑adviser.

This module defines a simple Settings class that reads values from
environment variables and provides sensible defaults when they are not
set. Using an explicit configuration class avoids relying on external
packages such as pydantic‑settings which may not be available at
runtime. All fields are typed for clarity but are resolved at runtime
from environment variables as strings unless converted explicitly.

To customise behaviour, set the corresponding environment variable
before starting the service. For example, to set a custom Azure AI
endpoint and API key:

```bash
export AZURE_AI_ENDPOINT="https://your-resource.services.ai.azure.com"
export AZURE_AI_API_KEY="<your-key>"
```
"""

from __future__ import annotations

import os
# Load .env once at startup (local/dev). In production prefer real env vars.
try:
    from dotenv import load_dotenv
    override = os.getenv("DOTENV_OVERRIDE", "false").lower() in {"1", "true", "yes"}
    load_dotenv(override=override)
except Exception:
    pass


from typing import Optional


def _get_env(key: str, default: str) -> str:
    v = os.getenv(key, default)
    v = v.strip().strip('"').strip("'")
    # защита от случая, когда в value кто-то засунул "KEY=..."
    if v.startswith(f"{key}="):
        v = v.split("=", 1)[1].strip()
    return v



class Settings:
    # Azure AI Inference / Azure OpenAI compatible endpoint
    AZURE_AI_ENDPOINT: str
    AZURE_AI_API_KEY: str
    AZURE_AI_API_VERSION: str

    # Model deployments
    CHAT_DEPLOYMENT: str
    EMBED_DEPLOYMENT: str

    # Azure Search
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_API_KEY: str
    AZURE_SEARCH_INDEX: str
    VECTOR_FIELD: str
    TEXT_FIELD: str

    # RAG controls
    TOP_K: int
    MAX_CONTEXT_CHARS: int
    MAX_CONTEXT_TOKENS: Optional[int]
    SCORE_THRESHOLD: float
    CITATION_STRICT: bool

    # Memory persistence
    CHECKPOINTER_BACKEND: str
    SQLITE_DB_PATH: str
    POSTGRES_DSN: Optional[str]

    # Retry and timeout
    RETRY_ATTEMPTS: int
    TIMEOUT_SECONDS: float

    # Observability
    METRICS_ENABLED: bool
    TRACING_ENABLED: bool
    OTLP_ENDPOINT: str

    # Query rewriting
    REWRITE_ENABLED: bool
    REWRITE_LAST_N: int
    REWRITE_MAX_TOKENS: int
    REWRITE_TEMPERATURE: float

    # Conversation summarisation
    SUMMARY_ENABLED: bool
    SUMMARY_EVERY_N: int
    SUMMARY_KEEP_LAST_K: int
    SUMMARY_MAX_TOKENS: int
    SUMMARY_TEMPERATURE: float

    # Citation enforcement
    CITATION_ENFORCE_STRUCTURE: bool

    def __init__(self) -> None:
        # Azure AI Inference / OpenAI
        self.AZURE_AI_ENDPOINT = _get_env(
            "AZURE_AI_ENDPOINT", "https://example.services.ai.azure.com"
        )
        self.AZURE_AI_API_KEY = _get_env("AZURE_AI_API_KEY", "")
        self.AZURE_AI_API_VERSION = _get_env("AZURE_AI_API_VERSION", "2024-10-21")
        # Deployments
        self.CHAT_DEPLOYMENT = _get_env("CHAT_DEPLOYMENT", "chat-deployment")
        self.EMBED_DEPLOYMENT = _get_env("EMBED_DEPLOYMENT", "embed-deployment")
        # Azure Search
        self.AZURE_SEARCH_ENDPOINT = _get_env(
            "AZURE_SEARCH_ENDPOINT", "https://example.search.windows.net"
        )
        self.AZURE_SEARCH_API_KEY = _get_env("AZURE_SEARCH_API_KEY", "")
        self.AZURE_SEARCH_INDEX = _get_env("AZURE_SEARCH_INDEX", "search-index")
        self.VECTOR_FIELD = _get_env("VECTOR_FIELD", "contentVector")
        self.TEXT_FIELD = _get_env("TEXT_FIELD", "content")
        # RAG controls
        self.TOP_K = int(_get_env("TOP_K", "8"))
        self.MAX_CONTEXT_CHARS = int(_get_env("MAX_CONTEXT_CHARS", "12000"))
        mct = _get_env("MAX_CONTEXT_TOKENS", "1500")
        # Accept 'none' to disable token budgeting
        self.MAX_CONTEXT_TOKENS = None if mct.lower() in {"none", ""} else int(mct)
        self.SCORE_THRESHOLD = float(_get_env("SCORE_THRESHOLD", "0.0"))
        self.CITATION_STRICT = _get_env("CITATION_STRICT", "true").lower() in (
            "1",
            "true",
            "yes",
        )
        # Memory persistence
        self.CHECKPOINTER_BACKEND = _get_env("CHECKPOINTER_BACKEND", "sqlite")
        self.SQLITE_DB_PATH = _get_env("SQLITE_DB_PATH", "ai_adviser.db")
        # Allow POSTGRES_DSN to be unset (treated as None)
        self.POSTGRES_DSN = os.getenv("POSTGRES_DSN")
        # Retry and timeouts
        self.RETRY_ATTEMPTS = int(_get_env("RETRY_ATTEMPTS", "3"))
        self.TIMEOUT_SECONDS = float(_get_env("TIMEOUT_SECONDS", "30"))

        # Observability
        self.METRICS_ENABLED = _get_env("METRICS_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.TRACING_ENABLED = _get_env("TRACING_ENABLED", "false").lower() in {"1", "true", "yes"}
        self.OTLP_ENDPOINT = _get_env("OTLP_ENDPOINT", "")

        # Query rewriting controls
        self.REWRITE_ENABLED = _get_env("REWRITE_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.REWRITE_LAST_N = int(_get_env("REWRITE_LAST_N", "3"))
        self.REWRITE_MAX_TOKENS = int(_get_env("REWRITE_MAX_TOKENS", "50"))
        self.REWRITE_TEMPERATURE = float(_get_env("REWRITE_TEMPERATURE", "0.0"))

        # Conversation summarisation controls
        self.SUMMARY_ENABLED = _get_env("SUMMARY_ENABLED", "true").lower() in {"1", "true", "yes"}
        self.SUMMARY_EVERY_N = int(_get_env("SUMMARY_EVERY_N", "6"))
        self.SUMMARY_KEEP_LAST_K = int(_get_env("SUMMARY_KEEP_LAST_K", "4"))
        self.SUMMARY_MAX_TOKENS = int(_get_env("SUMMARY_MAX_TOKENS", "120"))
        self.SUMMARY_TEMPERATURE = float(_get_env("SUMMARY_TEMPERATURE", "0.3"))

        # Citation enforcement controls
        self.CITATION_ENFORCE_STRUCTURE = _get_env("CITATION_ENFORCE_STRUCTURE", "false").lower() in {"1", "true", "yes"}


# Instantiate a singleton settings object
settings = Settings()

__all__ = ["settings", "Settings"]