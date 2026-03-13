FROM python:3.12-alpine

RUN apk add --no-cache \
    build-base gcc libffi-dev openssl-dev

WORKDIR /tmp
RUN python -m ensurepip \
 && python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
 && python -m pip install --no-cache-dir \
      fastapi \
      "uvicorn[standard]" \
      pydantic \
      python-dotenv \
      requests \
      tenacity \
      jsonschema \
      azure-search-documents \
      azure-ai-inference \
      prometheus-client \
      opentelemetry-sdk \
      google-generativeai \
      litellm


WORKDIR /app
COPY src/ /app/src/
COPY adk_agent/ /app/adk_agent/


ENV PYTHONPATH=/app/src:/app/adk_agent

# --- Non-secret runtime config baked into image ---
ENV AZURE_AI_ENDPOINT="https://banking-mvp-foundry.services.ai.azure.com" \
    AZURE_AI_API_VERSION="2024-10-21" \
    CHAT_DEPLOYMENT="gpt-oss-120b" \
    EMBED_DEPLOYMENT="text-embedding-3-small" \
    AZURE_SEARCH_ENDPOINT="https://banking-mvp-search.search.windows.net" \
    AZURE_SEARCH_INDEX="kb-banking-v1-index" \
    VECTOR_FIELD="snippet_vector" \
    TEXT_FIELD="snippet" \
    TOP_K="10" \
    SCORE_THRESHOLD="0.01"


ENV SERVICE_HTTP_PORT=8000

WORKDIR /app/src
CMD ["sh", "-c", "uvicorn ai_adviser.api.main:app --host 0.0.0.0 --port ${SERVICE_HTTP_PORT}"]
