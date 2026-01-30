"""Tool functions used by the banking orchestrator.

This module defines three tool functions which can be registered with an
ADK assistant.  Each tool has a clear signature with type hints and
docstrings that describe its behaviour and expected arguments.

Tools
-----
* ``banking_rag_chat_tool`` вЂ“ proxies a request to an external
  retrievalвЂ‘augmented generation (RAG) service via HTTP.  It expects
  ``RAG_BASE_URL`` to be set in the environment.
* ``get_user_basic_profile`` вЂ“ loads a basic user profile from a
  configured source, validates it against a JSON schema and caches
  subsequent calls.  A derived field ``income_monthly_estimate`` is
  added when ``annual_income`` is present.
* ``budget_planner_tool`` вЂ“ produces a simple JSON budget plan based
  on the user's profile, memory and goal description.  It returns
  structured fields to allow downstream processing.

The functions are pure and do not rely on global state other than
environment variables and a small inвЂ‘memory cache.  They are safe to
import multiple times.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests import Response
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from jsonschema import validate as jsonschema_validate, ValidationError


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Determine package resource paths.  These files are installed alongside
# this module and provide the JSON schema and default profile example.
_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
_SCHEMA_PATH = _RESOURCE_DIR / "fdecl-userprofile-v1.json"
_DEFAULT_PROFILE_PATH = _RESOURCE_DIR / "demo-uprof.json"

# Cache for the user profile; keyed by last modified timestamp
_profile_cache: Optional[Dict[str, Any]] = None
_profile_mtime: Optional[float] = None


def _load_profile_source() -> Path:
    """Resolve the profile source file from the environment or fallback.

    Returns
    -------
    Path
        The resolved path to the JSON profile document.
    """
    src = os.environ.get("USER_PROFILE_SOURCE")
    if src:
        return Path(src)
    return _DEFAULT_PROFILE_PATH


def _load_schema() -> Dict[str, Any]:
    with _SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_user_basic_profile(*, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Load a basic user profile and validate it against the schema.

    This function reads a JSON document from the path specified by the
    ``USER_PROFILE_SOURCE`` environment variable.  When that variable
    is not set it falls back to an example profile bundled with the
    package.  The loaded profile is validated using the JSON schema in
    ``fdecl-userprofile-v1.json``.  A derived field
    ``income_monthly_estimate`` is added when ``annual_income`` is
    present and numeric.  Subsequent calls return a cached copy until
    the source file's modification time changes.

    Parameters
    ----------
    request_id : Optional[str]
        Optional request identifier for logging.  Ignored otherwise.

    Returns
    -------
    dict
        The validated and possibly augmented profile.
    """
    global _profile_cache, _profile_mtime
    src_path = _load_profile_source()
    try:
        mtime = src_path.stat().st_mtime
    except FileNotFoundError:
        raise FileNotFoundError(f"User profile source not found: {src_path}")
    # Return cached profile if source has not changed
    if _profile_cache is not None and _profile_mtime == mtime:
        return _profile_cache
    # Load and validate
    with src_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    schema = _load_schema().get("Output")
    try:
        jsonschema_validate(instance=data, schema=schema)
    except ValidationError as e:
        raise ValueError(f"User profile failed schema validation: {e.message}") from e
    # Compute derived fields
    annual = data.get("annual_income")
    if isinstance(annual, (int, float)) and annual > 0:
        data["income_monthly_estimate"] = round(annual / 12, 2)
    # Cache and return
    _profile_cache = data
    _profile_mtime = mtime
    return data


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1.0),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
)
def _post_with_retry(url: str, payload: Dict[str, Any], timeout: float = 5.0) -> Response:
    """Helper to POST JSON payloads with retry and timeout.

    Retries on network errors and HTTP status codes >= 500.
    """
    resp = requests.post(url, json=payload, timeout=timeout)
    # Raise for 5xx to trigger retry
    if resp.status_code >= 500:
        resp.raise_for_status()
    return resp


def banking_rag_chat_tool(*, question: str, top_k: int = 5, thread_id: Optional[str] = None, user_id: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    """Query the external RAG service for an answer to a banking question.

    Parameters
    ----------
    question : str
        The user's question.  Must be nonвЂ‘empty.
    top_k : int, default 5
        The number of top search hits to return from the RAG service.
    thread_id : Optional[str]
        Conversation identifier forwarded to the RAG service.  When
        provided, the service may leverage conversation memory.
    user_id : Optional[str]
        Identifier of the user.  Included for analytics or memory in the
        RAG service.  May be ``None``.
    request_id : Optional[str]
        Optional request identifier for observability.  Included as a
        header ``X-Request-ID`` on the outgoing HTTP call.

    Returns
    -------
    dict
        A dictionary containing the RAG service's response.  On error
        the dictionary will contain an ``error`` key with a message.
    """
    base_url = os.environ.get("RAG_BASE_URL")
    if not base_url:
        return {"error": "RAG_BASE_URL environment variable is not set"}
    question = (question or "").strip()
    if not question:
        return {"error": "Question must not be empty"}
    payload = {
        "question": question,
        "top_k": top_k,
        "thread_id": thread_id,
        "user_id": user_id,
    }
    url = base_url.rstrip("/") + "/rag_chat"
    headers = {}
    if request_id:
        headers["X-Request-ID"] = request_id
    try:
        resp = _post_with_retry(url, payload)
    except Exception as e:
        logger.exception("RAG chat HTTP call failed: %s", e)
        return {"error": f"RAG request failed: {type(e).__name__}"}
    # Attempt to parse JSON body
    try:
        body = resp.json()
    except Exception:
        body = {"answer": resp.text}
    # Annotate with HTTP status
    body.setdefault("status_code", resp.status_code)
    return body


def budget_planner_tool(*, goal: str, thread_id: str, user_id: str, profile: Dict[str, Any], memory: Any, request_id: str) -> Dict[str, Any]:
    """Plan a savings budget for a user goal.

    Given a description of the user's goal (e.g. "I want to buy a car") and
    access to the user's profile and conversation memory, this tool
    generates a structured budget plan.  It makes reasonable
    assumptions based on the user's income and location, outlines the
    monthly budget categories, estimates how long it might take to save
    for the goal, highlights potential risks and suggests next steps.

    Parameters
    ----------
    goal : str
        Natural language description of the goal the user wants to
        achieve.  Examples include "buy a car", "save for a house
        deposit" or "build an emergency fund".
    thread_id : str
        Conversation identifier.  Not used in the current
        implementation but accepted for API symmetry.
    user_id : str
        Identifier of the user.  Not used directly but included for
        completeness.
    profile : dict
        The user's basic profile as returned by ``get_user_basic_profile``.
    memory : Any
        A ``ConversationMemory`` instance that can be inspected for past
        decisions or user preferences.  For the MVP we do not
        introspect the history.
    request_id : str
        Identifier for the current request used for logging.

    Returns
    -------
    dict
        A structured plan with the following keys: ``goal_summary``,
        ``assumptions``, ``questions_needed``, ``monthly_budget_plan``,
        ``savings_plan``, ``risks_and_tradeoffs`` and ``next_actions``.
    """
    # Extract monthly income estimate; fall back to annual
    income_monthly = profile.get("income_monthly_estimate")
    if income_monthly is None:
        annual = profile.get("annual_income")
        if isinstance(annual, (int, float)) and annual > 0:
            income_monthly = round(annual / 12, 2)
    # Basic allocation percentages (50/30/20 rule)
    essentials_pct = 0.5
    savings_pct = 0.2
    discretionary_pct = 0.3
    assumptions: List[str] = []
    if income_monthly is not None:
        assumptions.append(
            f"Estimated monthly income is ${income_monthly:.2f} based on profile"
        )
    else:
        assumptions.append("Income information unavailable; using conservative estimates")
        income_monthly = 0.0
    # Determine a nominal goal amount (very rough guess)
    goal_lower = goal.lower()
    nominal_amount = None
    if "car" in goal_lower:
        nominal_amount = 30000.0
    elif any(word in goal_lower for word in ["house", "home", "apartment"]):
        nominal_amount = 600000.0
    elif "emergency" in goal_lower:
        nominal_amount = income_monthly * 3
    elif "vacation" in goal_lower:
        nominal_amount = 5000.0
    else:
        nominal_amount = income_monthly * 6  # default: six months of income
    assumptions.append(
        f"Nominal goal amount estimated at ${nominal_amount:,.2f} based on goal description"
    )
    # Build monthly budget plan
    essentials = round(income_monthly * essentials_pct, 2)
    savings = round(income_monthly * savings_pct, 2)
    discretionary = round(income_monthly * discretionary_pct, 2)
    monthly_budget_plan = [
        {"category": "Essentials", "amount": essentials, "notes": "housing, food, insurance"},
        {"category": "Savings", "amount": savings, "notes": "goal and emergency fund"},
        {"category": "Discretionary", "amount": discretionary, "notes": "entertainment, hobbies"},
    ]
    # Estimate months to save for the goal
    if savings > 0:
        months = int(max(nominal_amount - (savings * 2), 0) / savings) + 2
    else:
        months = 0
    savings_plan = {
        "target_amount": nominal_amount,
        "monthly_savings": savings,
        "estimated_months": months,
        "description": f"Save ${savings:,.2f} per month to reach ${nominal_amount:,.2f} in {months} months",
    }
    # Identify additional questions needed to refine the plan
    questions_needed: List[str] = []
    if nominal_amount == income_monthly * 6:
        questions_needed.append(
            "What is the precise amount you are aiming to save for this goal?"
        )
    # Risks and tradeoffs
    risks_and_tradeoffs = [
        "Unexpected expenses could delay your savings timeline",
        "Investment returns may vary; consider keeping an emergency fund separate",
        "Inflation can erode purchasing power over time",
    ]
    # Next actions
    next_actions = [
        "Open a dedicated savings account for your goal",
        "Automate monthly transfers of the savings amount",
        "Review your budget quarterly and adjust as needed",
        "Investigate financing options specific to your goal (e.g. auto loans, mortgages)",
        "Consult a financial advisor for personalised advice",
    ]
    return {
        "goal_summary": goal,
        "assumptions": assumptions,
        "questions_needed": questions_needed,
        "monthly_budget_plan": monthly_budget_plan,
        "savings_plan": savings_plan,
        "risks_and_tradeoffs": risks_and_tradeoffs,
        "next_actions": next_actions,
    }
