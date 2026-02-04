# ai-adviser

**ai-adviser** is a production-oriented Banking RAG (Retrieval-Augmented Generation) assistant MVP built for experimentation and real-world usage with **Azure AI services**.

The project provides a FastAPI-based RAG backend, optional agentic orchestration, persistent conversation memory, strict citation control, and observability hooks for production deployments.

---

## Overview

The service combines:

- **FastAPI** as the HTTP API layer
- **Azure AI Foundry Models** for chat completions and text embeddings
- **Azure AI Search** for hybrid (keyword + vector) retrieval over banking documents

The API exposes the following endpoints:

- `/health`
- `/ready`
- `/embed`
- `/chat`
- `/rag_chat`

Each RAG response returns:

- the final LLM-generated answer
- the retrieved context chunks (with metadata such as `blob_url` for traceability)

---

## Key Features

### Retrieval-Augmented Generation (RAG)

- Embeddings via Azure AI Foundry  
  Example: `text-embedding-3-small` (1536 dimensions)
- Chat completions via Azure AI Foundry  
  Example: `gpt-oss-120b`
- Hybrid retrieval using Azure AI Search (keyword + vector)

### Orchestration & Memory

- Multi-turn conversation memory with persistent storage
- LangGraph-style sequential orchestration:
  - input normalization
  - memory loading
  - embedding
  - retrieval
  - context building
  - generation
  - citation validation
  - memory persistence
- Conversation summarization to compress long histories while preserving key facts
- Follow-up question rewriting into stand-alone queries to improve retrieval quality

### Safety & Quality Controls

- **Strict citation validation**  
  Every factual claim must be grounded in retrieved context and referenced with inline citations (`[S1]`, `[S2]`, etc.)
- **Relevance guard**  
  Retrieved snippets are checked for keyword overlap with the question (stop-words removed). If no overlap is found, the system refuses to answer to prevent hallucinations.
- Context budgeting with configurable token, character, and score thresholds

### Reliability & Observability

- Retries and timeouts around external service calls with exponential backoff
- Optional Prometheus metrics (`/metrics`)
- Optional OpenTelemetry tracing with Jaeger / OTLP
- Example Docker Compose setup for local observability (Prometheus, Grafana, Jaeger)

### Agent Development Kit (ADK)

In addition to the HTTP API, the repository includes an **Agent Development Kit (ADK)** integration under:

```

src/ai_adviser/adk/banking_orchestrator

````

The ADK agent can route user requests between:

- Banking RAG chat tool (`/rag_chat`)
- User profile loader (JSON + schema validation)
- Budget planner (heuristic-based financial planning)

The agent persists conversation state and user profiles, summarizes long histories, and can be used independently from the HTTP API as part of a broader agentic workflow.

---

## High-Level Architecture

### `/rag_chat` Request Flow

1. User question received (optional `thread_id`, `user_id`)
2. Load conversation history and summary from persistent storage
3. Rewrite follow-up questions (optional)
4. Compute embeddings for the (rewritten) question
5. Retrieve top-K relevant snippets via hybrid search
6. Build context block with citation identifiers
7. Apply relevance guard
8. Generate answer using system prompt + context
9. Validate inline citations and Sources section
10. Persist conversation memory and summaries
11. Return final answer and retrieved chunks

---

## Prerequisites

- Python **3.12+**
- Azure resources:
  - Azure AI Foundry project with model deployments:
    - Chat model (e.g. `gpt-oss-120b`)
    - Embeddings model (e.g. `text-embedding-3-small`)
  - Azure AI Search service
  - An index populated with chunk-level document content

---

## Azure AI Search Index Expectations

Default index schema:

- `snippet` — chunk text (searchable)
- `snippet_vector` — vector field (1536 dims)
- `blob_url` — original document source
- `uid` — unique document or chunk identifier

Field names can be overridden via environment variables:

- `TEXT_FIELD`
- `VECTOR_FIELD`

---

## Installation

### 1. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
````

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Editable install (recommended for development):

```bash
pip install -U pip
pip install -e .
```

---

## Configuration

Create a `.env` file in the repository root (next to `pyproject.toml`).

Example:

```env
# --- Azure AI Foundry Models ---
AZURE_AI_ENDPOINT=https://<your-foundry-resource>.services.ai.azure.com/api/projects/<project-name>
AZURE_AI_API_KEY=your_key_here

CHAT_DEPLOYMENT=gpt-oss-120b
EMBED_DEPLOYMENT=text-embedding-3-small

# --- Azure AI Search ---
AZURE_SEARCH_ENDPOINT=https://<your-search-name>.search.windows.net
AZURE_SEARCH_API_KEY=your_search_key
AZURE_SEARCH_INDEX=kb-banking-v1-index

TEXT_FIELD=snippet
VECTOR_FIELD=snippet_vector

# --- Runtime controls ---
TOP_K=8
MAX_CONTEXT_CHARS=12000
MAX_CONTEXT_TOKENS=1500
SCORE_THRESHOLD=0.0
CITATION_STRICT=true

# --- Conversation memory ---
CHECKPOINTER_BACKEND=sqlite
SQLITE_DB_PATH=ai_adviser.db

# --- Retry behaviour ---
RETRY_ATTEMPTS=3
TIMEOUT_SECONDS=30
```

⚠️ **Do NOT commit `.env`**.
Use `.env.example` for placeholders.

---

## Running the API

### Development (auto-reload)

```bash
uvicorn ai_adviser.api.main:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## Readiness

The `/ready` endpoint performs lightweight dependency checks and returns:

* **200 OK** — service configured correctly
* **503** — missing or misconfigured dependencies

---

## API Examples

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Embeddings

```json
POST /embed
{
  "text": "hello"
}
```

### RAG Chat

```json
POST /rag_chat
{
  "question": "How can I rebuild my credit?",
  "top_k": 5,
  "thread_id": "abc123",
  "user_id": "user@example.com"
}
```

---

## Prompt Management

The system prompt used by `/rag_chat` is stored in:

```
src/ai_adviser/prompts/rag_system.md
```

It is loaded at runtime by:

```
src/ai_adviser/prompts/loader.py
```

---

## Observability

Optional features (disabled by default):

* Prometheus metrics exposed at `/metrics`
* OpenTelemetry tracing with Jaeger / OTLP exporters

A sample Docker Compose setup is provided for local monitoring.

---

## Testing

```bash
pytest -q
```

---

## Code Style

```bash
black .
isort .
```

---

## Production Notes

* Do **NOT** use `--reload` in production
* Run behind a reverse proxy (Nginx or managed ingress)
* Use multiple workers

Example:

```bash
gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 0.0.0.0:8000 \
  ai_adviser.api.main:app
```

---

## Roadmap

* LangGraph-based explicit state graphs
* Langfuse integration (tracing, offline/online evaluation)
* Azure-native authentication:

  * Managed Identity
  * Entra ID (keyless access)
