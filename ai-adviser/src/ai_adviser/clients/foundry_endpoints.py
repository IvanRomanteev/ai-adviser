from __future__ import annotations

from urllib.parse import urlparse


def to_inference_models_endpoint(project_or_base_endpoint: str) -> str:
    """
    Accepts either:
      - https://<res>.services.ai.azure.com/api/projects/<project>
      - https://<res>.services.ai.azure.com
      - https://<res>.services.ai.azure.com/models

    Returns:
      - https://<res>.services.ai.azure.com/models
    """
    s = project_or_base_endpoint.strip().rstrip("/")

    # If already points to /models, keep it
    if s.endswith("/models"):
        return s

    p = urlparse(s)
    if not p.scheme or not p.netloc:
        raise ValueError(f"Invalid AZURE_AI_ENDPOINT: {project_or_base_endpoint!r}")

    base = f"{p.scheme}://{p.netloc}"
    return f"{base}/models"
