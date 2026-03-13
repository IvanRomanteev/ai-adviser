# oss-casual-chat
# Minimal API wrapper that matches the LiteLLM gateway request/response envelope.
#
# NOTE: This Dockerfile is resilient to different build contexts:
#   - repo root:    docker build -t oss-casual-chat:dev -f oss-casual-chat/Dockerfile .
#   - service dir:  docker build -t oss-casual-chat:dev -f oss-casual-chat/Dockerfile oss-casual-chat

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVICE_HTTP_PORT=8093

ENV AZURE_AI_ENDPOINT="https://banking-mvp-foundry.services.ai.azure.com" \
    CHAT_DEPLOYMENT="gpt-oss-120b"

WORKDIR /app

# Copy the build context (either repo root OR the service folder)
COPY . /src

# Install dependencies first (better layer caching)
RUN set -eux; \
    if [ -f /src/requirements.txt ]; then \
        cp /src/requirements.txt /app/requirements.txt; \
    elif [ -f /src/oss-casual-chat/requirements.txt ]; then \
        cp /src/oss-casual-chat/requirements.txt /app/requirements.txt; \
    else \
        echo "requirements.txt not found in build context"; \
        exit 1; \
    fi; \
    pip install --no-cache-dir -U pip; \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application
RUN set -eux; \
    mkdir -p /app/app; \
    if [ -d /src/app ]; then \
        cp -r /src/app/* /app/app/; \
    elif [ -d /src/oss-casual-chat/app ]; then \
        cp -r /src/oss-casual-chat/app/* /app/app/; \
    else \
        echo "app/ not found in build context"; \
        exit 1; \
    fi

# Capture git commit for /version endpoint.
# CI typically generates version.tmp next to this Dockerfile.
RUN set -eux; \
    if [ -f /src/version.tmp ]; then \
        cp /src/version.tmp /app/version.tmp; \
    elif [ -f /src/oss-casual-chat/version.tmp ]; then \
        cp /src/oss-casual-chat/version.tmp /app/version.tmp; \
    else \
        echo "unknown" > /app/version.tmp; \
    fi

EXPOSE 8093

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${SERVICE_HTTP_PORT:-8093}"]
