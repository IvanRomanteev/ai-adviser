# oldrag

A minimal stub for the "old RAG" tool API.

It exists because the senior engineer asked to create an `oldrag/` build context next to `oss-casual-chat/`.

Current goal: return a valid JSON envelope even if the chunks list is empty.

## API

### POST `/search` (aliases: `/`, `/v1/search`)

Request example: `req/rag_request.example.json`

Response example: `resp/rag_response.example.json`

### Health

- `GET /health` – liveness
- `GET /checks/liveness` – alias to `/health`
- `GET /checks/readiness` – alias to `/health` (no external deps in the stub)
- `GET /version` – service identity + git commit + uptime (restarts visibility)

## Docker

Default container port is `SERVICE_HTTP_PORT=8093`. The image listens on 8093 by default and still respects an explicit override if you pass another value at runtime.


Build from repo root:

```bash
docker build -f oldrag/Dockerfile . -t oldrag:dev
```

Or (recommended) build with build context = service folder:

```bash
docker build -t oldrag:dev -f oldrag/Dockerfile oldrag
```

Run:

```bash
docker run --rm -p 8093:8093 oldrag:dev
```

Test:

```bash
curl -s http://localhost:8093/checks/liveness

curl -s http://localhost:8093/version | cat
```
