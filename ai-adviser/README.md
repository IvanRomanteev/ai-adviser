# ai-adviser

**ai-adviser** is a **Banking RAG (Retrieval-Augmented Generation) assistant MVP** built for production-oriented experimentation with Azure AI services.

The service combines:

* **FastAPI** as an HTTP API layer
* **Azure AI Foundry Models** for **chat completions** and **text embeddings** (via Azure AI Model Inference endpoint)
* **Azure AI Search** for **vector + hybrid retrieval** over banking documents

The API exposes `/health`, `/embed`, `/chat`, and `/rag_chat` endpoints and returns both:

* the final LLM answer
* the retrieved chunks (with `blob_url` sources for traceability)

---

## Features

*  **Embeddings** via Azure AI Foundry deployment

  * Example: `text-embedding-3-small` (1536 dims)
*  **Chat completions** via Azure AI Foundry deployment

  * Example: `gpt-oss-120b`
*  **Hybrid retrieval** (keyword + vector) via Azure AI Search
*  **Conversation memory** with persistent storage (multi‑turn threads)
*  **LangGraph‑style orchestration**: sequential nodes perform input
  normalisation, memory load, embedding, retrieval, context building,
  generation, citation validation and memory persistence
*  **Strict citation validation** ensuring every factual claim is
  supported by retrieved context with `[S1]`, `[S2]` style inline
  references
*  **Context budgeting** with configurable token and score limits to
  manage the amount of retrieved text passed to the model
*  **Retries and timeouts** around external service calls using
  exponential backoff
*  **Externalised system prompt**

  * `src/ai_adviser/prompts/rag_system.md` (loaded at runtime)
*  **Swagger UI** available at `/docs`

---

## High-level architecture

### `/rag_chat` request flow

1. User sends a question
2. The question is embedded using an **Azure AI Foundry embeddings deployment**
3. Top-K relevant snippets are retrieved from **Azure AI Search** (vector or hybrid)
4. A context block is assembled from retrieved snippets (with sources)
5. `SYSTEM_PROMPT + CONTEXT + QUESTION` is sent to the **chat model**
6. The API returns:

   * `answer`
   * `chunks[]` containing snippet text and metadata (e.g. `blob_url`)

---

## Prerequisites

* **Python 3.12+**
* Azure resources:

  * **Azure AI Foundry project** with model deployments:

    * Chat model (e.g. `gpt-oss-120b`)
    * Embeddings model (e.g. `text-embedding-3-small`)
  * **Azure AI Search** service

    * An index populated with chunk-level document content

---

## Azure AI Search index expectations

By default, the project assumes the index schema includes:

* `snippet` — chunk text (searchable)
* `snippet_vector` — vector field (1536 dims)
* `blob_url` — original document source
* `uid` — unique document/chunk key

Field names can be overridden via environment variables:

* `TEXT_FIELD`
* `VECTOR_FIELD`

---

## Installation

### 1) Create and activate a virtual environment

**Windows (PowerShell):**

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

Editable install (recommended for development):

```
pip install -U pip
pip install -e .
```

---

## Configuration

Create a `.env` file in the repository root (next to `pyproject.toml`).

Example `.env`:

```
# --- Azure AI Foundry Models ---
# Foundry project endpoint (the code normalizes it to the inference endpoint)
AZURE_AI_ENDPOINT=https://<your-foundry-resource>.services.ai.azure.com/api/projects/<project-name>
AZURE_AI_API_KEY=your_key_here

# Deployment names (as configured in Foundry)
CHAT_DEPLOYMENT=gpt-oss-120b
EMBED_DEPLOYMENT=text-embedding-3-small

# --- Azure AI Search ---
AZURE_SEARCH_ENDPOINT=https://<your-search-name>.search.windows.net
AZURE_SEARCH_API_KEY=your_search_admin_or_query_key
AZURE_SEARCH_INDEX=kb-banking-v1-index

# Index field mapping
TEXT_FIELD=snippet
VECTOR_FIELD=snippet_vector

# Runtime controls
TOP_K=8
MAX_CONTEXT_CHARS=12000
MAX_CONTEXT_TOKENS=1500
SCORE_THRESHOLD=0.0
CITATION_STRICT=true

# Conversation memory
CHECKPOINTER_BACKEND=sqlite
SQLITE_DB_PATH=ai_adviser.db
POSTGRES_DSN=

# Retry behaviour
RETRY_ATTEMPTS=3
TIMEOUT_SECONDS=30
```

⚠️ Do **NOT** commit `.env`.
Use `.env.example` for placeholders.

---

## Running the API

### Development (auto-reload)

```
uvicorn ai_adviser.api.main:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

### Readiness

The `/ready` endpoint performs lightweight checks to ensure that the
embedding and search clients can be constructed. It returns HTTP 200
when the service is configured correctly or HTTP 503 if dependencies are
missing.

---

## API endpoints

### GET /health

Returns service status.

```
curl.exe http://127.0.0.1:8000/health
```

---

### POST /embed

Request body:

```
{
  "text": "hello"
}
```

Response:

* `dims` (e.g. 1536)
* `embedding` (float array)

---

### POST /chat

Request body:

```
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

### POST /rag_chat

Request body:

```
{
  "question": "How can I rebuild my credit?",
  "top_k": 5,
  "thread_id": "abc123",      // optional conversation identifier
  "user_id": "user@example.com"  // optional user identifier
}
```

Response:

* `answer` – the assistant’s reply. If no relevant snippets are found or
  the model fails to include valid citations the answer will be the
  fallback message `"Not found in the knowledge base."`
* `chunks[]` – list of retrieved snippets with metadata (`blob_url`, score,
  etc.)

---

## Quick smoke tests (Windows PowerShell)

Note: in PowerShell, `curl` is an alias for `Invoke-WebRequest`.
Use `Invoke-RestMethod` or `curl.exe`.

```
Invoke-RestMethod http://127.0.0.1:8000/health
```

```
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/embed `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"hello"}'
```

```
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/chat `
  -Method POST `
  -ContentType "application/json" `
  -Body '{
    "messages": [
      {"role":"system","content":"You are a test assistant"},
      {"role":"user","content":"Say hello in one word"}
    ]
  }'
```

```
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/rag_chat `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"question":"How can I rebuild my credit?","top_k":5}'
```

---

## Prompt management

The system prompt used by `/rag_chat` is stored in:

```
src/ai_adviser/prompts/rag_system.md
```

It is loaded at runtime by:

```
src/ai_adviser/prompts/loader.py
```

Edit the `.md` file to adjust:

* grounding rules
* answer format
* citation requirements
* refusal / safety constraints

---

## Testing

```
pytest -q
```

---

## Code style

```
black .
isort .
```

---

## Production notes

*  Do **NOT** use `--reload` in production
* Recommended setup:

  * multiple Uvicorn workers **or**
  * Gunicorn + UvicornWorker
* Always run behind a reverse proxy (Nginx / managed ingress)

Example (Linux container):

```
gunicorn \
  -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 0.0.0.0:8000 \
  ai_adviser.api.main:app
```

---

## Roadmap

* LangGraph orchestration:

  * conversation memory (`thread_id`)
  * explicit state graph
  * token / context budgeting
* Langfuse:

  * tracing
  * offline / online evaluations
* Azure native auth:

  * Managed Identity
  * Entra ID (keyless access)
