# Banking Orchestrator (ADK)

This module contains a lightweight agent built on top of a Google
Agent Development Kit (ADK) pattern.  The orchestrator can be used
locally to route user questions between a retrievalвЂ‘augmented
generation service, a profile loader and a simple budget planner.  It
persists conversation history and user profile information between
calls using plain JSON files.

## Installation

The package is part of the `ai_adviser` project.  It requires
PythonВ 3.11 or newer and the dependencies listed in the
`pyproject.toml`.  To install the project in editable mode run:

```bash
poetry install --with dev
```

Alternatively, use `pip` with the provided lock file.

## Configuration

The orchestrator is configured entirely via environment variables:

| Variable | Description | Required |
| --- | --- | --- |
| `RAG_BASE_URL` | Base URL of the existing RAG service.  The retrieval tool appends `/rag_chat` when making HTTP requests. | yes, for RAG queries |
| `USER_PROFILE_SOURCE` | Path to a JSON file containing a basic user profile.  When omitted, a demo profile bundled with the project is used. | no |
| `MEMORY_STORE_PATH` | Directory where perвЂ‘user memory files (profile, history and summary) are stored.  Defaults to `.memory` in the current working directory. | no |
| `OTEL_TRACES_EXPORTER` and other OpenTelemetry variables | When set, traces will be recorded around each tool invocation. | no |

## Usage

Instantiate the orchestrator with a `user_id` and call `run()` for
each user message.  Supply a `thread_id` to enable conversation
memory; omit it to perform stateless requests.

```python
from ai_adviser.adk.banking_orchestrator import BankingOrchestrator

orch = BankingOrchestrator(user_id="user123")

# Ask a factual question which will be handled by the RAG service
resp = orch.run("What are the fees for an international wire transfer?", thread_id="session1")
print(resp["answer"])

# Plan a budget for a goal; the budget planner is invoked automatically
resp = orch.run("I want to buy a car", thread_id="session1")
print(resp["monthly_budget_plan"])

# Retrieve the cached user profile
profile = orch.run("show my profile")
print(profile)
```

## Testing

Unit tests are provided under `tests/`.  They can be run with

```bash
pytest -q
```

The tests mock HTTP requests to the RAG service and validate that the
profile loader and budget planner produce the correct output structure
and handle invalid inputs gracefully.

## Implementation Notes

* **Tools** вЂ“ The three functions defined in `tools.py` are pure and
  stateless.  They rely only on inputs and environment variables and
  can be registered with an ADK assistant directly.
* **Memory** вЂ“ The `ConversationMemory` class manages perвЂ‘user state.
  It persists profile data, conversation history and rolling summaries
  to a JSON file.  The simple summarisation strategy truncates older
  messages to keep context bounded.
* **Observability** вЂ“ Each call to a tool logs its invocation along
  with a unique `request_id`.  When OpenTelemetry is available and
  configured, spans are emitted around tool calls for distributed
  tracing.

For more details see the source code in `agent.py` and `tools.py`.
