"""Unit tests for the banking orchestrator and its tools."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import requests
from unittest.mock import patch, MagicMock

from ai_adviser.adk.banking_orchestrator.agent import BankingOrchestrator, ConversationMemory
from ai_adviser.adk.banking_orchestrator import tools


def test_get_user_basic_profile_loads_and_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the profile loader reads and validates the profile file."""
    # Create a temporary profile file
    profile_data = {
        "modified_time": "2025-01-01T00:00:00",
        "user_id": "abc-123",
        "user_name": "Test User",
        "age_years": 30,
        "primary_residence": "Chicago, IL",
        "military_status": "none",
        "pension_or_retired": False,
        "nfcc_dmp": False,
        "annual_income": 120000,
        "employment": {"type": "full_time", "multi_job": False},
        "government_benefits": False,
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile_data))
    monkeypatch.setenv("USER_PROFILE_SOURCE", str(profile_path))
    # First call should load and compute derived field
    prof = tools.get_user_basic_profile()
    assert prof["user_id"] == "abc-123"
    assert prof.get("income_monthly_estimate") == pytest.approx(10000.0)
    # Modify file to ensure cache invalidation
    profile_data["annual_income"] = 60000
    profile_path.write_text(json.dumps(profile_data))
    prof2 = tools.get_user_basic_profile()
    assert prof2["income_monthly_estimate"] == pytest.approx(5000.0)


def test_banking_rag_chat_tool_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RAG chat tool should return the JSON body on success."""
    # Mock environment
    monkeypatch.setenv("RAG_BASE_URL", "https://api.example.com")
    # Mock requests.post
    def mock_post(url: str, json: Dict[str, Any], timeout: float) -> MagicMock:
        assert url == "https://api.example.com/rag_chat"
        assert json["question"] == "Test question"
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"answer": "42", "chunks": []}
        return mock_resp

    monkeypatch.setattr(tools, "_post_with_retry", lambda url, payload, timeout=5.0: mock_post(url, payload, timeout))
    result = tools.banking_rag_chat_tool(question="Test question", user_id="u1")
    assert result["answer"] == "42"
    assert result["status_code"] == 200


def test_banking_rag_chat_tool_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The RAG chat tool should return an error on failure."""
    monkeypatch.setenv("RAG_BASE_URL", "https://api.example.com")
    def mock_post(url: str, json: Dict[str, Any], timeout: float) -> MagicMock:
        raise requests.RequestException("boom")
    monkeypatch.setattr(tools, "_post_with_retry", lambda url, payload, timeout=5.0: mock_post(url, payload, timeout))
    result = tools.banking_rag_chat_tool(question="fail", user_id="u1")
    assert "error" in result


def test_budget_planner_tool_structure(tmp_path: Path) -> None:
    """Ensure the budget planner returns all required keys and sensible values."""
    profile = {
        "annual_income": 60000,
        "income_monthly_estimate": 5000,
    }
    dummy_memory = object()  # memory not used in current implementation
    result = tools.budget_planner_tool(
        goal="Buy a car", thread_id="t1", user_id="u1", profile=profile, memory=dummy_memory, request_id="req"
    )
    # Required top level keys
    for key in [
        "goal_summary",
        "assumptions",
        "questions_needed",
        "monthly_budget_plan",
        "savings_plan",
        "risks_and_tradeoffs",
        "next_actions",
    ]:
        assert key in result
    # Budget plan should contain three categories
    plan = result["monthly_budget_plan"]
    assert isinstance(plan, list) and len(plan) == 3
    categories = {item["category"] for item in plan}
    assert {"Essentials", "Savings", "Discretionary"} == categories


def test_orchestrator_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The orchestrator should dispatch based on message content."""
    # Stub tools to capture calls
    called = {}
    def fake_rag(**kwargs) -> Dict[str, Any]:
        called["rag"] = kwargs
        return {"answer": "rag"}
    def fake_profile(**kwargs) -> Dict[str, Any]:
        called["profile"] = kwargs
        return {"user_id": "u1"}
    def fake_budget(**kwargs) -> Dict[str, Any]:
        called["budget"] = kwargs
        return {"goal_summary": kwargs.get("goal")}
    monkeypatch.setattr(tools, "banking_rag_chat_tool", fake_rag)
    monkeypatch.setattr(tools, "get_user_basic_profile", fake_profile)
    monkeypatch.setattr(tools, "budget_planner_tool", fake_budget)
    # Ensure orchestrator uses our stubs
    orch = BankingOrchestrator(user_id="u1", memory_store_path=str(Path(".") / ".test_memory"))
    # Profile request
    resp = orch.run("Show my profile")
    assert "profile" in called
    # Budget request
    resp = orch.run("I want to save for a car", thread_id="t")
    assert "budget" in called
    # RAG request
    resp = orch.run("What is your fee?", thread_id="t")
    assert "rag" in called
