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
      prometheus-client \
      opentelemetry-sdk \
      google-generativeai \
      litellm


WORKDIR /app
COPY src/ /app/src/
COPY adk_agent/ /app/adk_agent/


ENV PYTHONPATH=/app/src:/app/adk_agent


ENV SERVICE_HTTP_PORT=8000

WORKDIR /app/src
CMD ["sh", "-c", "uvicorn ai_adviser.api.main:app --host 0.0.0.0 --port ${SERVICE_HTTP_PORT}"]
