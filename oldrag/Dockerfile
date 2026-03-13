# oldrag
# Minimal stub that matches rag_api_schemas request/response envelope.
#
# NOTE: This Dockerfile is resilient to different build contexts:
#   - repo root:    docker build -t oldrag:dev -f oldrag/Dockerfile .
#   - service dir:  docker build -t oldrag:dev -f oldrag/Dockerfile oldrag

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICE_HTTP_PORT=8093

ENV AZURE_AI_ENDPOINT="https://banking-mvp-foundry.services.ai.azure.com" \
    CHAT_DEPLOYMENT="gpt-oss-120b"

WORKDIR /app

COPY . /src

RUN set -eux; \
    if [ -f /src/requirements.txt ]; then \
        cp /src/requirements.txt /app/requirements.txt; \
    elif [ -f /src/oldrag/requirements.txt ]; then \
        cp /src/oldrag/requirements.txt /app/requirements.txt; \
    else \
        echo "requirements.txt not found in build context"; \
        exit 1; \
    fi; \
    pip install --no-cache-dir -U pip; \
    pip install --no-cache-dir -r /app/requirements.txt

RUN set -eux; \
    mkdir -p /app/app; \
    if [ -d /src/app ]; then \
        cp -r /src/app/* /app/app/; \
    elif [ -d /src/oldrag/app ]; then \
        cp -r /src/oldrag/app/* /app/app/; \
    else \
        echo "app/ not found in build context"; \
        exit 1; \
    fi

# Capture git commit for /version endpoint.
# CI typically generates version.tmp next to this Dockerfile.
RUN set -eux; \
    if [ -f /src/version.tmp ]; then \
        cp /src/version.tmp /app/version.tmp; \
    elif [ -f /src/oldrag/version.tmp ]; then \
        cp /src/oldrag/version.tmp /app/version.tmp; \
    else \
        echo "unknown" > /app/version.tmp; \
    fi

EXPOSE 8093

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${SERVICE_HTTP_PORT:-8093}"]
