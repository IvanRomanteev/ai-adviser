from __future__ import annotations

import os
from typing import Any, Optional

import httpx


class OssClientError(RuntimeError):
    pass


def oss_is_configured() -> bool:
    """Return True if an OpenAI-compatible OSS backend is configured."""
    return bool((os.getenv("OSS_API_BASE") or "").strip())


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


def oss_ping(timeout_s: float = 5.0) -> dict[str, Any]:
    """Lightweight connectivity check for an OSS OpenAI-compatible endpoint.

    We try GET /v1/models first (cheap). If it's not available (404),
    fall back to a minimal /v1/chat/completions request with max_tokens=1.

    Returns a small dict with HTTP status and optional sample/model info.
    """

    base = (os.getenv("OSS_API_BASE") or "").rstrip("/")
    if not base:
        raise OssClientError("OSS is not configured (OSS_API_BASE is empty).")

    model = os.getenv("OSS_MODEL") or "oss-129b"
    api_key = os.getenv("OSS_API_KEY") or ""

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # 1) Try models endpoint (fast + cheap)
    url_models = f"{base}/v1/models"
    try:
        r = httpx.get(url_models, headers=headers, timeout=timeout_s)
    except Exception as e:
        raise OssClientError(f"OSS ping failed: {type(e).__name__}: {e}") from e

    if r.status_code == 404:
        # 2) Fallback: minimal chat completion (still cheap if max_tokens=1)
        url_chat = f"{base}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "temperature": 0.0,
            "max_tokens": 1,
        }
        headers2 = {"Content-Type": "application/json", **headers}
        try:
            r = httpx.post(url_chat, json=payload, headers=headers2, timeout=timeout_s)
        except Exception as e:
            raise OssClientError(f"OSS ping failed: {type(e).__name__}: {e}") from e

    if r.status_code >= 400:
        raise OssClientError(f"OSS ping failed: HTTP {r.status_code}: {r.text}")

    data: Any = None
    try:
        data = r.json()
    except Exception:
        data = None

    out: dict[str, Any] = {"http_status": r.status_code}

    # Parse /v1/models response when available
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        models = [m.get("id") for m in data["data"] if isinstance(m, dict) and m.get("id")]
        out["models_count"] = len(models)
        out["models_sample"] = models[:5]
        return out

    # Parse chat response
    if isinstance(data, dict):
        try:
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
            out["sample"] = (content or "").strip()[:200]
        except Exception:
            pass

    return out


def call_oss_chat(
    user_input: str,
    *,
    system_prompt: Optional[str] = None,
    scenario: str = "chat",
    scenario_args: Optional[dict[str, Any]] = None,
) -> str:
    """Call an OSS model through an OpenAI-compatible endpoint.

    The goal of this repo change is to provide a working API surface ASAP.
    Model calling is optional: if OSS_API_BASE isn't set, the caller will not
    attempt any external requests.

    Env vars:
      - OSS_API_BASE: base URL, e.g. http://vllm:8000
      - OSS_API_KEY: optional bearer token
      - OSS_MODEL: model id (default: oss-129b)
      - OSS_TEMPERATURE: float
      - OSS_MAX_TOKENS: int
      - OSS_TIMEOUT_SECONDS: float
    """

    base = (os.getenv("OSS_API_BASE") or "").rstrip("/")
    if not base:
        # No backend configured -> caller disabled.
        return ""

    model = os.getenv("OSS_MODEL") or "oss-129b"
    api_key = os.getenv("OSS_API_KEY") or ""

    timeout = _env_float("OSS_TIMEOUT_SECONDS", 60.0)
    temperature = _env_float("OSS_TEMPERATURE", 0.2)
    max_tokens = _env_int("OSS_MAX_TOKENS", 512)

    # Basic scenario formatting hook (kept intentionally simple).
    scenario_args = scenario_args or {}
    prompt = user_input
    if scenario and scenario != "chat":
        prompt = (
            f"SCENARIO: {scenario}\n"
            f"SCENARIO_ARGS: {scenario_args}\n\n"
            f"USER_INPUT:\n{user_input}"
        )

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    url = f"{base}/v1/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except Exception as e:
        raise OssClientError(f"OSS request failed: {type(e).__name__}: {e}") from e

    if resp.status_code >= 400:
        raise OssClientError(f"OSS request failed: HTTP {resp.status_code}: {resp.text}")

    data = resp.json()

    # OpenAI-compatible response parsing.
    try:
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    except Exception as e:
        raise OssClientError(f"OSS response parsing failed: {type(e).__name__}: {e}") from e
