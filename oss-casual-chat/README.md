# oss-casual-chat

A minimal HTTP service that **implements the LiteLLM gateway envelope** (request/response) so the platform can call an OSS model (e.g. `gpt-oss-120b`) via a fixed API.

> Current goal: make the container respond correctly even with an empty/dummy answer, **and** provide standard health/diagnostics endpoints.

## API

### POST `/invoke` (aliases: `/`, `/v1/invoke`)

Request body (see `req/`):

- `id` (string) – request id (for logs)
- `user_input` (string) – user query
- `tools` (array[string]) – allowed tool names (currently ignored)
- `history` (array[Any]) – conversation history (kept permissive)
- `verbosity` (`brief|normal|verbose`)
- `scenario` (string, default `chat`)
- `scenario_args` (object)

Response body (see `resp/`):

- `id` (string)
- `user_id` (string|null)
- `status` (`success|tool_call|error|rate_limited`)
- `output_format` (string)
- `output` (object)
- `metrics` (object with token counters)

### Health & diagnostics

These endpoints follow the same conventions as the main `ai-adviser` service:

- `GET /health` – liveness (always returns `{"ok": true}`)
- `GET /checks/liveness` – alias to `/health`
- `GET /ready` – readiness (config-based, **no model call**; returns `503` if no upstream configured)
- `GET /checks/readiness` – alias to `/ready`
- `GET /diagnost` – pretty JSON with:
  - OSS connectivity status (Foundry preferred; or OpenAI-compatible backend)
  - configuration ok/bad
  - last errors ring buffer
  - query param: `ping=true|false` (default `true`) – whether to perform a lightweight connectivity check
- `GET /diagnostics` – alias to `/diagnost`
- `GET /version` – service identity + git commit + uptime (restarts visibility)

## Run locally (without Docker)

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port ${SERVICE_HTTP_PORT:-8093}
```

### Quick test (Linux/macOS)

```bash
curl -s http://localhost:8093/checks/liveness

curl -s http://localhost:8093/diagnost | cat

curl -s http://localhost:8093/version | cat

curl -s http://localhost:8093/invoke \
  -H 'Content-Type: application/json' \
  -d @req/litellm_gateway_request.example.json
```

### Quick test (Windows PowerShell)

```powershell
irm http://localhost:8093/checks/liveness

# Pretty-print JSON in console
irm http://localhost:8093/diagnost | ConvertTo-Json -Depth 20

irm http://localhost:8093/version | ConvertTo-Json -Depth 10

# Or get already-pretty JSON as text
curl.exe -s http://localhost:8093/diagnost

$body = Get-Content -Raw .\req\litellm_gateway_request.example.json
irm -Method Post -Uri "http://localhost:8093/invoke" -ContentType "application/json" -Body $body
```

## Docker

Default container port is `SERVICE_HTTP_PORT=8093`.

During CI builds, a `version.tmp` file (git commit SHA) may be placed next to the Dockerfile. The image copies it into `/app/version.tmp`, and `GET /version` exposes it. The image listens on 8093 by default and still respects an explicit override if you pass another value at runtime.


### Build

From repo root (Docker build context = repo root):

```bash
docker build -f oss-casual-chat/Dockerfile . -t oss-casual-chat:dev
```

Or (recommended) build with build context = service folder:

```bash
docker build -t oss-casual-chat:dev -f oss-casual-chat/Dockerfile oss-casual-chat
```

### Run

Minimal run (dummy mode will answer even without upstream):

```bash
docker run --rm -p 8093:8093 oss-casual-chat:dev
```

### Enable Azure AI Foundry calls (recommended for your setup)

If you already run OSS deployments via **Azure AI Foundry** (Azure AI Inference SDK), set:

- `AZURE_AI_ENDPOINT` – base endpoint or project endpoint (we normalize to `/models`)
- `AZURE_AI_API_KEY` – **secret**
- `CHAT_DEPLOYMENT` – deployment name (e.g. `gpt-oss-120b`)

Example:

```bash
docker run --rm -p 8093:8093 \
  -e AZURE_AI_ENDPOINT='https://<your>.services.ai.azure.com' \
  -e AZURE_AI_API_KEY='***' \
  -e CHAT_DEPLOYMENT='gpt-oss-120b' \
  oss-casual-chat:dev
```

### Enable OpenAI-compatible OSS backend (vLLM/TGI/etc.)

Set:

- `OSS_API_BASE` – e.g. `http://host.docker.internal:8000`
- `OSS_MODEL` – default `oss-129b`
- `OSS_API_KEY` – optional bearer token

Example:

```bash
docker run --rm -p 8093:8093 \
  -e OSS_API_BASE='http://host.docker.internal:8000' \
  -e OSS_MODEL='oss-129b' \
  oss-casual-chat:dev
```
