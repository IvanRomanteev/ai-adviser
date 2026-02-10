"""Banking orchestrator tools.

This module provides three functions used by the ADK demo agent and the
``BankingOrchestrator`` wrapper:

* ``banking_rag_chat_tool``  – calls the FastAPI ``/rag_chat`` endpoint.
* ``get_user_basic_profile`` – loads and validates a user profile JSON.
* ``budget_planner_tool``    – creates a simple budget/savings plan.

This version is intentionally more defensive for ADK WebUI tool-calling:

* ``budget_planner_tool`` no longer requires callers to provide ``profile``,
  ``memory`` or ``request_id``. If missing, it will load/extract them.
* ``banking_rag_chat_tool`` supports passing HTTP headers and always sets
  ``X-Request-ID`` for tracing.
* ``banking_rag_chat_tool`` returns a ``not_found`` boolean to make it easier
  for the orchestrator/LLM to stop retry loops.
* Optional one-shot retry: if the call returns the fallback
  "Not found in the knowledge base.", the tool retries once (max) with:
  - top_k bumped to at least 10, and
  - thread_id removed (stateless) when thread_id was provided
  This mitigates \"topic poisoning\" when the user switches topics mid-thread.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate
from requests import Response
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_SCHEMA_PATH = _RESOURCE_DIR / "fdecl-userprofile-v1.json"
_DEFAULT_PROFILE_PATH = _RESOURCE_DIR / "demo-uprof.json"

_profile_cache: Optional[Dict[str, Any]] = None
_profile_mtime: Optional[float] = None

FALLBACK_MESSAGE = "Not found in the knowledge base."


def _load_profile_source() -> Path:
    env_src = os.environ.get("USER_PROFILE_SOURCE")
    if env_src:
        p = Path(env_src).expanduser()
        # если передали относительный путь — трактуем относительно текущего cwd
        return p if p.is_absolute() else (Path.cwd() / p).resolve()

    # 1) дефолт рядом с tools.py (самый правильный)
    if _DEFAULT_PROFILE_PATH.exists():
        return _DEFAULT_PROFILE_PATH

    # 2) если запускаешь из репо с src-layout
    src_layout = (
        Path.cwd()
        / "src"
        / "ai_adviser"
        / "adk"
        / "banking_orchestrator"
        / "resources"
        / "demo-uprof.json"
    )
    if src_layout.exists():
        return src_layout

    # 3) legacy layout без src/
    legacy = (
        Path.cwd()
        / "ai_adviser"
        / "adk"
        / "banking_orchestrator"
        / "resources"
        / "demo-uprof.json"
    )
    if legacy.exists():
        return legacy

    # ничего не нашли — вернём дефолт, чтобы сообщение об ошибке было предсказуемым
    return _DEFAULT_PROFILE_PATH



def _load_schema() -> Dict[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_user_basic_profile(*, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Load + validate the user profile JSON, with caching by file mtime."""
    _ = request_id

    global _profile_cache, _profile_mtime
    src_path = _load_profile_source()

    try:
        mtime = src_path.stat().st_mtime
    except FileNotFoundError as e:
        raise FileNotFoundError(f"User profile source not found: {src_path}") from e

    if _profile_cache is not None and _profile_mtime == mtime:
        return _profile_cache

    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    schema = _load_schema()
    try:
        jsonschema_validate(instance=data, schema=schema)
    except ValidationError as e:
        raise ValueError(f"User profile validation failed: {e.message}") from e

    annual_income = data.get("annual_income")
    if isinstance(annual_income, (int, float)):
        data["income_monthly_estimate"] = float(annual_income) / 12.0

    _profile_cache = data
    _profile_mtime = mtime
    return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(requests.RequestException),
)
def _post_with_retry(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 5.0,
) -> Response:
    return requests.post(url, json=payload, headers=headers, timeout=timeout)


def _is_fallback_answer(answer: Any) -> bool:
    return isinstance(answer, str) and answer.strip() == FALLBACK_MESSAGE

def _has_grounding_warning(answer: Any) -> bool:
    if not isinstance(answer, str):
        return False
    # достаточно устойчиво: оба маркера сразу
    return ("Grounding note" in answer) and ("Closest sources" in answer)


def _normalize_chunks_aliases(body: Dict[str, Any]) -> None:
    """Make chunk URL accessible via both `source_file` and `blob_url` keys."""
    chunks = body.get("chunks")
    if not isinstance(chunks, list):
        return
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        # API отдаёт source_file, некоторые клиенты ожидают blob_url
        if ch.get("blob_url") is None and ch.get("source_file"):
            ch["blob_url"] = ch["source_file"]
        if ch.get("source_file") is None and ch.get("blob_url"):
            ch["source_file"] = ch["blob_url"]


def banking_rag_chat_tool(
    *,
    question: str,
    top_k: int = 5,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 5.0,
    allow_stateless_retry: bool = True,
) -> Dict[str, Any]:
    """Call the external RAG service. Returns JSON + status_code + not_found."""
    base_url = os.environ.get("RAG_BASE_URL", "http://localhost:8000")
    url = base_url.rstrip("/") + "/rag_chat"

    if request_id is None or not str(request_id).strip():
        request_id = uuid.uuid4().hex

    payload: Dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "thread_id": thread_id,
        "user_id": user_id,
        "request_id": request_id,
    }

    req_headers: Dict[str, str] = {}
    if headers:
        req_headers.update({k: str(v) for k, v in headers.items()})
    req_headers.setdefault("X-Request-ID", request_id)

    try:
        resp = _post_with_retry(url, payload, headers=req_headers, timeout=timeout)
        try:
            body: Dict[str, Any] = resp.json()
        except Exception:
            body = {"error": "Invalid JSON response from RAG service", "raw": resp.text}

        body.setdefault("status_code", resp.status_code)
        body.setdefault("request_id", request_id)

        answer = body.get("answer")
        body["not_found"] = _is_fallback_answer(answer)
        body["grounding_warning"] = _has_grounding_warning(answer)
        _normalize_chunks_aliases(body)


        body["not_found"] = _is_fallback_answer(body.get("answer"))

        # One-shot retry (max one additional call):
        # - bump top_k to >= 10
        # - if thread_id was used, retry without thread_id (stateless) to avoid topic poisoning
        if (body["not_found"] or body.get("grounding_warning", False)) and allow_stateless_retry:

            retry_payload = dict(payload)
            changed = False

            retry_top_k = max(int(top_k), 10)
            if retry_top_k != int(top_k):
                retry_payload["top_k"] = retry_top_k
                changed = True

            if thread_id:
                retry_payload["thread_id"] = None
                changed = True

            if changed:
                resp2 = _post_with_retry(url, retry_payload, headers=req_headers, timeout=timeout)
                try:
                    body2: Dict[str, Any] = resp2.json()
                except Exception:
                    body2 = {"error": "Invalid JSON response from RAG service", "raw": resp2.text}

                body2.setdefault("status_code", resp2.status_code)
                body2.setdefault("request_id", request_id)
                body2["not_found"] = _is_fallback_answer(body2.get("answer"))

                answer2 = body2.get("answer")
                body2["not_found"] = _is_fallback_answer(answer2)
                body2["grounding_warning"] = _has_grounding_warning(answer2)
                _normalize_chunks_aliases(body2)


                if (
                    "error" not in body2
                    and not body2["not_found"]
                    and not body2.get("grounding_warning", False)
                ):
                    body2["retrieval_mode"] = "retry_top_k_or_stateless"
                    return body2


        return body

    except Exception as e:
        logger.exception("RAG chat HTTP call failed: %s", e)
        return {"error": str(e), "not_found": True, "request_id": request_id}


def budget_planner_tool(
    *,
    goal: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    profile: Optional[Dict[str, Any]] = None,
    memory: Optional[Any] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a simple budget plan for a given goal.

    Robust to ADK tool-call argument mistakes:
    - profile can be omitted or nested under memory.profile
    - request_id can be omitted/null
    """
    if request_id is None or not str(request_id).strip():
        request_id = uuid.uuid4().hex

    # ADK sometimes nests profile under memory.profile
    if profile is None and isinstance(memory, dict):
        maybe_profile = memory.get("profile")
        if isinstance(maybe_profile, dict):
            profile = maybe_profile

    if profile is None:
        profile = get_user_basic_profile(request_id=request_id)

    monthly_income = profile.get("income_monthly_estimate")
    annual_income = profile.get("annual_income")
    if monthly_income is None and isinstance(annual_income, (int, float)):
        monthly_income = float(annual_income) / 12.0

    if monthly_income is None:
        return {"error": "Cannot compute monthly income from profile", "request_id": request_id}

    monthly_income_f = float(monthly_income)

    essentials = monthly_income_f * 0.5
    savings = monthly_income_f * 0.2
    discretionary = monthly_income_f * 0.3

    return {
        "goal_summary": goal,
        "assumptions": {
            "monthly_income": round(monthly_income_f, 2),
            "essential_pct": 0.5,
            "savings_pct": 0.2,
            "discretionary_pct": 0.3,
        },
        "questions_needed": [
            "What is your target amount and timeframe for this goal?",
            "Do you have any existing debts or obligations that affect your budget?",
        ],
        "monthly_budget_plan": [
            {"category": "Essentials", "amount": round(essentials, 2)},
            {"category": "Savings", "amount": round(savings, 2)},
            {"category": "Discretionary", "amount": round(discretionary, 2)},
        ],
        "savings_plan": {
            "suggested_monthly_savings": round(savings, 2),
            "notes": "Consider automating savings transfers to reach your goal faster.",
        },
        "risks_and_tradeoffs": [
            "Unexpected expenses may require reducing discretionary spending.",
            "If essential costs exceed 50%, adjust allocations accordingly.",
        ],
        "next_actions": [
            "Track your spending for the next month to validate these assumptions.",
            "Adjust categories based on actual spending and goal requirements.",
        ],
        "request_id": request_id,
        "thread_id": thread_id,
        "user_id": user_id,
    }
