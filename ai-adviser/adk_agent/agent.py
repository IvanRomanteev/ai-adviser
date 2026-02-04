"""Banking orchestrator agent for the Google Agent Development Kit (ADK).

This module defines a single ``root_agent`` instance that wraps the existing
banking orchestrator tools so they can be invoked through the ADK runtime.

The agent uses the LiteLLM model adapter to connect to Azure OpenAI or
Foundry deployments.  Configure the model via the ``ORCH_MODEL``
environment variable; this should take the form ``azure/<deployment-name>``.
LiteLLM will automatically pick up ``AZURE_API_BASE``, ``AZURE_API_VERSION``
and ``AZURE_API_KEY`` from the environment.

To use this agent with the ADK web interface, run ``adk web`` from this
directory after ensuring that all required environment variables are set.
"""

from __future__ import annotations

import os

from google.adk.agents.llm_agent import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from ai_adviser.adk.banking_orchestrator.tools import (
    banking_rag_chat_tool,
    get_user_basic_profile,
    budget_planner_tool,
)


def _get_model() -> LiteLlm:
    """Initialise and return a LiteLLM model based on environment variables.

    The ``ORCH_MODEL`` variable should specify the provider and deployment
    identifier (e.g. ``azure/gpt-oss-120b``).  When unset a default
    Azure deployment is derived from the ``CHAT_DEPLOYMENT`` variable.
    """
    model_name = os.environ.get("ORCH_MODEL")
    if not model_name:
        deployment = os.environ.get("CHAT_DEPLOYMENT", "")
        # Prefix with ``azure/`` to instruct LiteLLM to call Azure endpoints
        model_name = f"azure/{deployment}" if deployment else "azure/"
    return LiteLlm(model=model_name)


root_agent: LlmAgent = LlmAgent(
    model=_get_model(),
    name="banking_orchestrator_agent",
    description="""
    An ADK agent that acts as a banking assistant.  It leverages an external
    Retrieval-Augmented Generation (RAG) service to answer factual
    questions about banking products and procedures, loads user profile
    information from a JSON document, and generates personalised budget plans.
    The agent routes user requests to one of three tools based on intent and
    persists conversation state via the underlying orchestrator's memory store.
    """,
    instruction="""
    You are a helpful banking assistant.  Use the provided tools to answer
    questions and perform tasks:

    1. For questions about banking products, fees or services, call
       ``banking_rag_chat_tool`` with the user's question.  This will return a
       JSON object containing an answer and supporting citations.
    2. To retrieve the current user's basic profile, call
       ``get_user_basic_profile``.  This returns a JSON document describing
       the user's income and other demographics.
    3. For budgeting or goal planning requests, call ``budget_planner_tool``.
       This tool returns a structured JSON plan with fields
       ``goal_summary``, ``assumptions``, ``questions_needed``,
       ``monthly_budget_plan``, ``savings_plan``, ``risks_and_tradeoffs`` and
       ``next_actions``.

    When invoking ``budget_planner_tool``, ensure that the response to the
    user is the JSON returned by the tool (do not alter its keys or
    structure).  Do not hallucinate answers; if information is not found
    in the knowledge base, rely on the fallback message provided by the tool.
    """,
    tools=[
        banking_rag_chat_tool,
        get_user_basic_profile,
        budget_planner_tool,
    ],
)

# Expose only ``root_agent`` to the ADK runtime
__all__ = ["root_agent"]