# src/ai_adviser/clients/azure_models.py
from __future__ import annotations

from functools import lru_cache
from typing import Any, List

# Azure dependencies are optional. These imports may fail if the Azure
# SDK packages are not installed in the current environment. When
# unavailable the functions below will raise a RuntimeError when invoked.
try:
    from azure.core.credentials import AzureKeyCredential  # type: ignore
    from azure.ai.inference import ChatCompletionsClient, EmbeddingsClient  # type: ignore
    from azure.ai.inference.models import SystemMessage, UserMessage  # type: ignore
except Exception:
    AzureKeyCredential = None  # type: ignore
    ChatCompletionsClient = None  # type: ignore
    EmbeddingsClient = None  # type: ignore
    SystemMessage = None  # type: ignore
    UserMessage = None  # type: ignore
# tenacity is an optional dependency. When unavailable the retry
# decorators become no‑ops.
try:
    from tenacity import retry, stop_after_attempt, wait_exponential  # type: ignore
except Exception:
    def retry(*dargs: Any, **dkwargs: Any):  # type: ignore[no-redef]
        def wrapper(fn):
            return fn

        return wrapper

    def stop_after_attempt(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        return None

    def wait_exponential(*args: Any, **kwargs: Any):  # type: ignore[no-redef]
        return None

from ai_adviser.config import settings
from ai_adviser.clients.foundry_endpoints import to_inference_models_endpoint


@lru_cache(maxsize=1)
def _models_endpoint() -> str:
    return to_inference_models_endpoint(settings.AZURE_AI_ENDPOINT)


@lru_cache(maxsize=1)
def _chat_client() -> ChatCompletionsClient:
    if ChatCompletionsClient is None or AzureKeyCredential is None:
        raise RuntimeError(
            "Azure AI Inference SDK is not installed. Please install azure-ai-inference to use chat completions."
        )
    return ChatCompletionsClient(
        endpoint=_models_endpoint(),
        credential=AzureKeyCredential(settings.AZURE_AI_API_KEY),
        model=settings.CHAT_DEPLOYMENT,  # deployment name (Name из Foundry)
    )


@lru_cache(maxsize=1)
def _embed_client() -> EmbeddingsClient:
    if EmbeddingsClient is None or AzureKeyCredential is None:
        raise RuntimeError(
            "Azure AI Inference SDK is not installed. Please install azure-ai-inference to use embeddings."
        )
    return EmbeddingsClient(
        endpoint=_models_endpoint(),
        credential=AzureKeyCredential(settings.AZURE_AI_API_KEY),
        model=settings.EMBED_DEPLOYMENT,  # deployment name (Name из Foundry)
    )


@retry(
    stop=stop_after_attempt(settings.RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def embed_text(text: str) -> List[float]:
    """
    Compatibility-safe embedding call.

    Some versions of azure-ai-inference don't support request_timeout kwarg.
    We try with request_timeout first, then fall back.
    """
    client = _embed_client()

    try:
        resp = client.embed(input=[text], request_timeout=settings.TIMEOUT_SECONDS)
    except TypeError:
        # Fallback: no timeout kwarg (or SDK has different signature)
        resp = client.embed(input=[text])

    return resp.data[0].embedding


@retry(
    stop=stop_after_attempt(settings.RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
)
def chat(
    messages: list[dict[str, Any]],
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> str:
    """
    Compatibility-safe chat completion call.

    Some versions of azure-ai-inference don't support request_timeout kwarg.
    """
    client = _chat_client()

    typed_msgs = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role == "system":
            typed_msgs.append(SystemMessage(content=content))
        elif role == "user":
            typed_msgs.append(UserMessage(content=content))
        else:
            typed_msgs.append(UserMessage(content=content))

    try:
        resp = client.complete(
            messages=typed_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout=settings.TIMEOUT_SECONDS,
        )
    except TypeError:
        resp = client.complete(
            messages=typed_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return (resp.choices[0].message.content or "").strip()
