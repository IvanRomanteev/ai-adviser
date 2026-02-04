"""Banking orchestrator agent for the Google Agent Development Kit (ADK)."""

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
    model_name = os.environ.get("ORCH_MODEL")
    if not model_name:
        deployment = os.environ.get("CHAT_DEPLOYMENT", "")
        model_name = f"azure/{deployment}" if deployment else "azure/"
    return LiteLlm(model=model_name)


root_agent: LlmAgent = LlmAgent(
    model=_get_model(),
    name="banking_orchestrator_agent",
    description="""
    Banking assistant agent.

    It MUST use tools for factual banking questions (RAG), profile lookups, and
    budget planning.

    This prompt is intentionally strict to prevent the LLM from hallucinating or
    falling into tool retry loops in ADK WebUI.
    """,
    instruction="""
    You are a banking assistant.

    **Hard rules (must follow):**
    - Do NOT answer factual banking/credit/fees/procedures questions from your own knowledge.
      Always use the RAG tool.
    - Call **at most one tool per user message**. Never call the same tool repeatedly
      for the same user message.
    - If a tool returns an error or the fallback message "Not found in the knowledge base.",
      do NOT retry tools in a loop. Respond to the user and stop.

    **Routing rules:**
    1) **Profile**: If the user asks for their profile ("profile", "who am I", "about me"),
       call `get_user_basic_profile` and summarise the returned profile.

    2) **Budget planning**: If the user asks to plan a budget / savings goal / large purchase
       (e.g., "save", "budget", "plan", "buy a car", "vacation"), call `budget_planner_tool`
       with `goal` set to the user's request.
       - After the tool returns, respond with **exactly the JSON returned by the tool** (no edits).

    3) **Everything else banking-related**: call `banking_rag_chat_tool`.
       - If the question is a follow-up that depends on earlier context (e.g., contains
         "it/they/that/this"), rewrite it into a standalone question before calling the tool.
       - After the tool returns:
         * If `error` is present: tell the user the tool failed and include the error.
         * Otherwise: reply with `answer` (plain text). Do not add facts.
         * If `not_found` is true or `answer` equals the fallback message, ask 1-2 clarifying
           questions (e.g., which bank/product/country), but do not call another tool.
    """,
    tools=[
        banking_rag_chat_tool,
        get_user_basic_profile,
        budget_planner_tool,
    ],
)

__all__ = ["root_agent"]
