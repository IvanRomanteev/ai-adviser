# ai-adviser

**ai-adviser** is a Banking RAG (Retrieval-Augmented Generation) assistant MVP built for production-oriented experimentation with Azure AI services.

The service combines:

- **FastAPI** as an HTTP API layer  
- **Azure AI Foundry Models** for chat completions and text embeddings (via Azure AI Model Inference endpoint)  
- **Azure AI Search** for vector + hybrid retrieval over banking documents  

The API exposes `/health`, `/embed`, `/chat`, and `/rag_chat` endpoints and returns both:

- the final LLM answer  
- the retrieved chunks (with `blob_url` sources for traceability)

---

## Features

- **Embeddings via Azure AI Foundry deployment**  
  Example: `text-embedding-3-small` (1536 dims)

- **Chat completions via Azure AI Foundry deployment**  
  Example: `gpt-oss-120b`

- **Hybrid retrieval (keyword + vector)** via Azure AI Search

- **Conversation memory with persistent storage** (multi-turn threads)

- **LangGraph-style orchestration**  
  Sequential nodes perform:
  - input normalisation  
  - memory load  
  - embedding  
  - retrieval  
  - context building  
  - generation  
  - citation validation  
  - memory persistence  

- **Strict citation validation**  
  Every factual claim must be supported by retrieved context using `[S1]`, `[S2]` style inline references.

- **Context budgeting**  
  Configurable token and score limits control how much retrieved text is passed to the model.

- **Retries and timeouts**  
  External service calls are wrapped with exponential backoff.

- **Externalised system prompt**  
  Loaded at runtime from:  
  `src/ai_adviser/prompts/rag_system.md`

- **Swagger UI** available at `/docs`

- **Conversation summarisation**  
  Long conversation histories are periodically summarised to free space in the context window while preserving key information. Summaries are stored alongside original messages and reused for follow-up questions.

- **Question rewriting**  
  Follow-up questions can be rewritten into stand-alone queries using recent conversation history and summaries to improve retrieval quality.

- **Relevance guard**  
  Before a response is generated, retrieved snippets are checked for keyword overlap with the question (stop-words removed). If no overlap is found, the system refuses to answer to avoid hallucinations.

- **Metrics and tracing**  
  Optional Prometheus and OpenTelemetry integration exports request durations, RAG stage timings, and distributed traces.  
  A sample `docker-compose.yml` is provided to run Jaeger, Prometheus, and Grafana locally.

- **Agent Development Kit (ADK)**  
  A lightweight orchestration module under  
  `src/ai_adviser/adk/banking_orchestrator` routes requests between:
  - RAG chat  
  - user profile loader  
  - budget planner  

  Conversation state and user profiles are persisted in JSON files and can be invoked independently of the HTTP API.

---

## High-level Architecture

### `/rag_chat` Request Flow

1. **User question received**  
   The API accepts a question with optional `thread_id` and `user_id`.

2. **Load conversation history and summary**  
   If `thread_id` is provided, previous messages and summaries are loaded from persistent storage.

3. **Rewrite follow-up questions**  
   Recent messages and summaries may be used to rewrite clarifying questions into stand-alone queries.

4. **Compute embeddings**  
   The (possibly rewritten) question is embedded using the configured embedding model.

5. **Retrieve relevant snippets**  
   Hybrid (keyword + vector) search against Azure AI Search returns top-K snippets.

6. **Build context**  
   Snippets are truncated to fit token/character budgets and annotated with citation identifiers (`[S1]`, `[S2]`). Only snippets above the score threshold are included.

7. **Relevance guard**  
   Ensures retrieved snippets share at least one keyword with the question. Otherwise, the pipeline aborts with a fallback response.

8. **Generate answer**  
   The system prompt, conversation history, context block, and question are sent to the chat model.

9. **Validate citations**  
   Inline citations and the Sources section are validated. Invalid answers are replaced with a fallback message when strict mode is enabled.

10. **Persist memory**  
    Question, answer, and updated summary are stored by `thread_id` and `user_id`.

11. **Return response**  
    The API returns the final answer and retrieved chunks with metadata (`blob_url`, score, etc.).

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
- `uid` — unique document/chunk key

Field names can be overridden via environment variables:

- `TEXT_FIELD`
- `VECTOR_FIELD`

---

## Installation

### 1) Create and activate a virtual environment

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

### 2) Install dependencies

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
AZURE_SEARCH_API_KEY=your_search_admin_or_query_key
AZURE_SEARCH_INDEX=kb-banking-v1-index

TEXT_FIELD=snippet
VECTOR_FIELD=snippet_vector

TOP_K=8
MAX_CONTEXT_CHARS=12000
MAX_CONTEXT_TOKENS=1500
SCORE_THRESHOLD=0.0
CITATION_STRICT=true

CHECKPOINTER_BACKEND=sqlite
SQLITE_DB_PATH=ai_adviser.db

RETRY_ATTEMPTS=3
TIMEOUT_SECONDS=30
```

⚠️ **Do NOT commit `.env`.**
Use `.env.example` for placeholders.

---

## Running the API

### Development (auto-reload)

```bash
uvicorn ai_adviser.api.main:app --reload
```

Swagger UI:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Readiness

The `/ready` endpoint checks whether embedding and search clients can be constructed.

* **200** — service configured correctly
* **503** — missing or invalid dependencies

---

## API Endpoints

### `GET /health`

```bash
curl.exe http://127.0.0.1:8000/health
```

---

### `POST /embed`

Request:

```json
{
  "text": "hello"
}
```

Response:

* `dims`
* `embedding`

---

### `POST /chat`

Request:

```json
{
  "messages": [
    {"role": "system", "content": "You are a test assistant"},
    {"role": "user", "content": "Say hello in one word"}
  ]
}
```

Response:

* `text`

---

### `POST /rag_chat`

Request:

```json
{
  "question": "How can I rebuild my credit?",
  "top_k": 5,
  "thread_id": "abc123",
  "user_id": "user@example.com"
}
```

Response:

* `answer`
* `chunks[]`

---

## Prompt Management

System prompt location:

```
src/ai_adviser/prompts/rag_system.md
```

Loaded by:

```
src/ai_adviser/prompts/loader.py
```

---

## Observability and Monitoring

* **Prometheus metrics** via `/metrics`
* **OpenTelemetry tracing** with Jaeger / OTLP

Both are optional and disabled by default.

---

## Agent Development Kit (ADK)

A lightweight agent under:

```
src/ai_adviser/adk/banking_orchestrator
```

Routes requests between:

* banking RAG chat tool
* user profile loader
* budget planner

Maintains conversation memory, summarises long histories, and supports tracing.

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
* Use multiple workers or Gunicorn

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

* LangGraph orchestration
* Langfuse (tracing, evaluations)
* Azure native auth:

  * Managed Identity
  * Entra ID

```