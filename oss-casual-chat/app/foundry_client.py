from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import urlparse


try:
    # Azure AI Inference SDK (Foundry)
    from azure.core.credentials import AzureKeyCredential  # type: ignore
    from azure.ai.inference import ChatCompletionsClient  # type: ignore
    from azure.ai.inference.models import SystemMessage, UserMessage  # type: ignore
except Exception:
    AzureKeyCredential = None  # type: ignore
    ChatCompletionsClient = None  # type: ignore
    SystemMessage = None  # type: ignore
    UserMessage = None  # type: ignore


class FoundryClientError(RuntimeError):
    pass


def to_inference_models_endpoint(project_or_base_endpoint: str) -> str:
    """Normalize AZURE_AI_ENDPOINT to the /models endpoint.

    Accepts either:
      - https://<res>.services.ai.azure.com/api/projects/<project>
      - https://<res>.services.ai.azure.com
      - https://<res>.services.ai.azure.com/models

    Returns:
      - https://<res>.services.ai.azure.com/models
    """

    s = (project_or_base_endpoint or "").strip().rstrip("/")
    if not s:
        raise ValueError("AZURE_AI_ENDPOINT is empty")

    # If already points to /models, keep it
    if s.endswith("/models"):
        return s

    p = urlparse(s)
    if not p.scheme or not p.netloc:
        raise ValueError(f"Invalid AZURE_AI_ENDPOINT: {project_or_base_endpoint!r}")

    base = f"{p.scheme}://{p.netloc}"
    return f"{base}/models"


def foundry_is_configured() -> bool:
    return bool(
        (os.getenv("AZURE_AI_ENDPOINT") or "").strip()
        and (os.getenv("AZURE_AI_API_KEY") or "").strip()
        and (os.getenv("CHAT_DEPLOYMENT") or "").strip()
    )




def foundry_ping(timeout_s: float = 5.0) -> dict[str, Any]:
    """Lightweight connectivity check for Azure AI Foundry chat deployment.

    Uses a minimal chat completion with max_tokens=1 to keep cost low.
    Returns a small dict with an optional text sample and token usage (if available).
    """

    if not foundry_is_configured():
        raise FoundryClientError(
            "Foundry is not configured (need AZURE_AI_ENDPOINT, AZURE_AI_API_KEY, CHAT_DEPLOYMENT)."
        )

    if UserMessage is None:
        raise FoundryClientError(
            "Azure AI Inference SDK is not installed. Add azure-ai-inference to requirements.txt."
        )

    client = _chat_client()
    messages = [UserMessage(content="ping")]

    try:
        try:
            resp = client.complete(
                messages=messages,
                max_tokens=1,
                temperature=0.0,
                request_timeout=timeout_s,
            )
        except TypeError:
            resp = client.complete(
                messages=messages,
                max_tokens=1,
                temperature=0.0,
            )
    except Exception as e:
        raise FoundryClientError(f"Foundry ping failed: {type(e).__name__}: {e}") from e

    out: dict[str, Any] = {}
    try:
        out["sample"] = (resp.choices[0].message.content or "").strip()[:200]
    except Exception:
        out["sample"] = ""

    # Usage fields exist in some SDK versions.
    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            out["usage"] = {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
    except Exception:
        pass

    return out

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


@lru_cache(maxsize=1)
def _chat_client() -> ChatCompletionsClient:
    if ChatCompletionsClient is None or AzureKeyCredential is None:
        raise FoundryClientError(
            "Azure AI Inference SDK is not installed. "
            "Add azure-ai-inference to requirements.txt to enable Foundry calls."
        )

    endpoint_raw = os.getenv("AZURE_AI_ENDPOINT") or ""
    api_key = os.getenv("AZURE_AI_API_KEY") or ""
    deployment = os.getenv("CHAT_DEPLOYMENT") or ""

    endpoint = to_inference_models_endpoint(endpoint_raw)

    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        model=deployment,  # deployment Name in Foundry
    )


def call_foundry_chat(
    user_input: str,
    *,
    system_prompt: Optional[str] = None,
    scenario: str = "chat",
    scenario_args: Optional[dict[str, Any]] = None,
    history: Optional[list[Any]] = None,
    verbosity: str = "normal",
) -> str:
    """Call a Foundry deployment (Azure AI Inference).

    Env vars used:
      - AZURE_AI_ENDPOINT (base or /models)
      - AZURE_AI_API_KEY
      - CHAT_DEPLOYMENT (deployment name in Foundry)
      - AZURE_TEMPERATURE (float, default 0.2)
      - AZURE_MAX_TOKENS (int, default depends on verbosity)
      - AZURE_TIMEOUT_SECONDS (float, default 60)
    """

    if not foundry_is_configured():
        raise FoundryClientError(
            "Foundry is not configured (need AZURE_AI_ENDPOINT, AZURE_AI_API_KEY, CHAT_DEPLOYMENT)."
        )

    client = _chat_client()

    # Defaults by verbosity (kept simple).
    vb = (verbosity or "normal").lower().strip()
    default_max = 256 if vb == "brief" else 512 if vb == "normal" else 1024

    temperature = _env_float("AZURE_TEMPERATURE", 0.2)
    max_tokens = _env_int("AZURE_MAX_TOKENS", default_max)
    timeout = _env_float("AZURE_TIMEOUT_SECONDS", 60.0)

    scenario_args = scenario_args or {}

    prompt = user_input
    if scenario and scenario != "chat":
        prompt = (
            f"SCENARIO: {scenario}\n"
            f"SCENARIO_ARGS: {scenario_args}\n\n"
            f"USER_INPUT:\n{user_input}"
        )

    # Build message list: system + history + current user prompt
    messages: list[Any] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))

    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = (item.get("role") or "").lower().strip()
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "system":
            messages.append(SystemMessage(content=content))
        else:
            # Keep compatibility with the production ai-adviser code which
            # maps unknown roles to UserMessage.
            messages.append(UserMessage(content=content))

    messages.append(UserMessage(content=prompt))

    try:
        # Some versions of azure-ai-inference support request_timeout kwarg.
        try:
            resp = client.complete(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                request_timeout=timeout,
            )
        except TypeError:
            resp = client.complete(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
    except Exception as e:
        raise FoundryClientError(f"Foundry request failed: {type(e).__name__}: {e}") from e

    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise FoundryClientError(
            f"Foundry response parsing failed: {type(e).__name__}: {e}"
        ) from e
