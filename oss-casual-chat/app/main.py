from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .foundry_client import (
    FoundryClientError,
    call_foundry_chat,
    foundry_is_configured,
    foundry_ping,
)
from .oss_client import OssClientError, call_oss_chat, oss_is_configured, oss_ping
from .schemas import GatewayRequest, GatewayResponse, Metrics


logger = logging.getLogger("oss-casual-chat")


def _setup_logging() -> None:
    level = (os.getenv("LOG_LEVEL") or "INFO").upper().strip()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


_setup_logging()


class UTF8JSONResponse(JSONResponse):
    """Force UTF-8 charset in JSON responses (PowerShell friendly)."""

    media_type = "application/json; charset=utf-8"


app = FastAPI(title="oss-casual-chat", version="0.1.0", default_response_class=UTF8JSONResponse)


# Keep a small ring-buffer of recent errors for /diagnost
_LAST_ERRORS_MAX = int(os.getenv("DIAGNOST_LAST_ERRORS") or "20")
_LAST_ERRORS: deque[dict[str, Any]] = deque(maxlen=max(_LAST_ERRORS_MAX, 1))


# Process start markers for /version (uptime / restarts visibility)
_PROCESS_START_UTC = datetime.now(timezone.utc).replace(microsecond=0)
_PROCESS_START_MONO = time.monotonic()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _human_uptime(seconds: float) -> str:
    """Format uptime seconds as a human-friendly string (for quick restart checks)."""

    try:
        s = int(max(0.0, float(seconds)))
    except Exception:
        s = 0

    days, rem = divmod(s, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)

    if days > 0:
        return f"{days}d {hours:02d}:{mins:02d}:{secs:02d}"
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def _read_git_commit() -> Optional[str]:
    """Best-effort git commit resolver used by /version.

    CI typically writes the commit SHA into `version.tmp` next to the Dockerfile.
    The Dockerfile copies it into `/app/version.tmp`.
    """

    env_commit = (os.getenv("GIT_COMMIT") or os.getenv("SERVICE_GIT_COMMIT") or "").strip()
    if env_commit:
        return env_commit

    candidates = [
        (os.getenv("VERSION_FILE") or "").strip() or None,
        "/app/version.tmp",
        "/app/app/version.tmp",
        "/src/version.tmp",
        "/src/oss-casual-chat/version.tmp",
    ]

    for p in candidates:
        if not p:
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                v = (f.read() or "").strip()
            if v:
                return v
        except FileNotFoundError:
            continue
        except Exception:
            continue

    return None


def _push_error(scope: str, exc: BaseException | str, *, request_id: Optional[str] = None) -> None:
    try:
        if isinstance(exc, BaseException):
            err_type = type(exc).__name__
            err_msg = str(exc)
        else:
            err_type = "Error"
            err_msg = str(exc)

        _LAST_ERRORS.appendleft(
            {
                "ts": _utc_now_iso(),
                "scope": scope,
                "request_id": request_id,
                "type": err_type,
                "message": err_msg[:2000],
            }
        )
    except Exception:
        # Never fail the main flow because of diagnostics
        pass


def _estimate_tokens(text: str) -> int:
    """Very rough token estimator.

    For a connectivity stub it's enough to provide non-negative metrics.
    """

    if not text:
        return 0
    # A rough heuristic (works reasonably for English/Russian mixed text).
    return max(1, len(text) // 4)


def _resolve_user_id(req: GatewayRequest, header_user_id: Optional[str]) -> Optional[str]:
    if header_user_id:
        return header_user_id
    v = req.scenario_args.get("user_id")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _build_dummy_answer(req: GatewayRequest) -> str:
    """Fallback response when no OSS backend is configured."""

    mode = (os.getenv("DUMMY_MODE") or "echo").lower().strip()
    if mode == "empty":
        return ""
    if mode == "ok":
        return "ok"
    # default: echo
    return f"ok: {req.user_input}".strip()


def _build_output(req: GatewayRequest, answer_text: str) -> dict[str, Any]:
    # Keep output structure stable and simple.
    return {
        "text": answer_text,
        "scenario": req.scenario,
    }


def _build_metrics(req: GatewayRequest, answer_text: str, elapsed_s: float) -> Metrics:
    # The gateway example requires these fields.
    input_tokens = _estimate_tokens(req.user_input)
    output_tokens = _estimate_tokens(answer_text)
    # We keep cached_tokens=0 for now.
    total = input_tokens + output_tokens
    m = Metrics(
        input_tokens=input_tokens,
        cached_tokens=0,
        output_tokens=output_tokens,
        total_tokens=total,
    )
    return m


def _active_backend() -> str:
    if foundry_is_configured():
        return "foundry"
    if oss_is_configured():
        return "oss"
    return "dummy"


def _call_model_if_configured(req: GatewayRequest) -> tuple[str, Optional[str], str]:
    """Try calling OSS model if configured.

    Returns: (answer_text, error_message, backend_name)
    """

    system_prompt = os.getenv("SYSTEM_PROMPT")

    # Prefer Azure Foundry if configured (this matches how the main ai-adviser
    # project calls gpt-oss-* deployments via azure-ai-inference).
    if foundry_is_configured():
        try:
            answer_text = call_foundry_chat(
                req.user_input,
                system_prompt=system_prompt,
                scenario=req.scenario,
                scenario_args=req.scenario_args,
                history=req.history,
                verbosity=req.verbosity,
            )
            return answer_text, None, "foundry"
        except FoundryClientError as e:
            _push_error("foundry_chat", e, request_id=req.id)
            return "", str(e), "foundry"

    # OpenAI-compatible backend (e.g., vLLM/TGI) if configured.
    if oss_is_configured():
        try:
            answer_text = call_oss_chat(
                req.user_input,
                system_prompt=system_prompt,
                scenario=req.scenario,
                scenario_args=req.scenario_args,
            )
            return answer_text, None, "oss"
        except OssClientError as e:
            _push_error("oss_chat", e, request_id=req.id)
            return "", str(e), "oss"

    # No backend configured -> dummy response.
    return _build_dummy_answer(req), None, "dummy"


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.get("/checks/liveness")
def checks_liveness() -> dict[str, bool]:
    # Match ai-adviser main service conventions
    return health()


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness probe.

    IMPORTANT: this endpoint is **config-based** (no model call), so it is safe for frequent probes.
    """

    checks: dict[str, Any] = {
        "backend": _active_backend(),
        "foundry": {
            "configured": foundry_is_configured(),
            "endpoint": (os.getenv("AZURE_AI_ENDPOINT") or "").strip() or None,
            "deployment": (os.getenv("CHAT_DEPLOYMENT") or "").strip() or None,
            "api_key_present": bool((os.getenv("AZURE_AI_API_KEY") or "").strip()),
        },
        "oss": {
            "configured": oss_is_configured(),
            "api_base": (os.getenv("OSS_API_BASE") or "").strip() or None,
            "model": (os.getenv("OSS_MODEL") or "oss-129b").strip() or None,
            "api_key_present": bool((os.getenv("OSS_API_KEY") or "").strip()),
        },
        "dummy_mode": (os.getenv("DUMMY_MODE") or "echo").strip(),
    }

    # Consider ready if we can call a real backend (Foundry/OSS). Dummy mode is liveness-only.
    is_ready = checks["foundry"]["configured"] or checks["oss"]["configured"]
    if not is_ready:
        raise HTTPException(status_code=503, detail={"ready": False, "checks": checks})

    return {"ready": True, "checks": checks}


@app.get("/checks/readiness")
def checks_readiness() -> dict[str, Any]:
    return ready()


def _pretty_json(data: Any) -> Response:
    return Response(
        content=json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        media_type="application/json; charset=utf-8",
    )


@app.get("/version")
def version() -> Response:
    """Standard version endpoint (used in all daemons).

    Helps quickly identify:
      - which service responded
      - what git commit/version it runs
      - how long the process has been up (restarts visibility)
    """

    uptime_s = max(0.0, time.monotonic() - _PROCESS_START_MONO)
    git_commit = _read_git_commit() or "unknown"

    data: dict[str, Any] = {
        "service": {
            "name": app.title,
            "version": app.version,
            "git_commit": git_commit,
        },
        "time_utc": _utc_now_iso(),
        "process": {
            "pid": os.getpid(),
            "hostname": (os.getenv("HOSTNAME") or "").strip() or None,
            "started_at_utc": _PROCESS_START_UTC.isoformat(),
            "uptime_seconds": round(uptime_s, 3),
            "uptime_human": _human_uptime(uptime_s),
        },
        "http": {
            "service_http_port": (os.getenv("SERVICE_HTTP_PORT") or "").strip() or None
        },
    }

    return _pretty_json(data)


def _get_pkg_version(dist_name: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(dist_name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


@app.get("/diagnost")
def diagnost(ping: bool = True) -> Response:
    """Human-friendly diagnostics endpoint (pretty JSON).

    Includes:
      - OSS connectivity (Foundry preferred; or OpenAI-compatible backend)
      - configuration ok/bad
      - last errors ring buffer

    Query params:
      - ping (bool, default True): whether to perform a lightweight connectivity check
    """

    backend = _active_backend()

    # Configuration checks
    foundry_cfg_ok = foundry_is_configured()
    oss_cfg_ok = oss_is_configured()
    config_ok = foundry_cfg_ok or oss_cfg_ok

    problems: list[str] = []
    if not config_ok:
        problems.append(
            "No upstream backend configured. Set either (AZURE_AI_ENDPOINT, AZURE_AI_API_KEY, CHAT_DEPLOYMENT) "
            "or OSS_API_BASE (and optionally OSS_API_KEY/OSS_MODEL)."
        )

    diag: dict[str, Any] = {
        "service": {"name": "oss-casual-chat", "version": app.version},
        "time_utc": _utc_now_iso(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "azure_ai_inference_version": _get_pkg_version("azure-ai-inference"),
            "httpx_version": _get_pkg_version("httpx"),
        },
        "configuration": {
            "ok": config_ok,
            "backend_selected": backend,
            "problems": problems,
            "foundry": {
                "configured": foundry_cfg_ok,
                "endpoint": (os.getenv("AZURE_AI_ENDPOINT") or "").strip() or None,
                "deployment": (os.getenv("CHAT_DEPLOYMENT") or "").strip() or None,
                "api_key_present": bool((os.getenv("AZURE_AI_API_KEY") or "").strip()),
            },
            "oss_openai_compat": {
                "configured": oss_cfg_ok,
                "api_base": (os.getenv("OSS_API_BASE") or "").strip() or None,
                "model": (os.getenv("OSS_MODEL") or "oss-129b").strip() or None,
                "api_key_present": bool((os.getenv("OSS_API_KEY") or "").strip()),
            },
            "dummy": {"mode": (os.getenv("DUMMY_MODE") or "echo").strip()},
        },
        "oss_connection": {
            "backend": backend,
            "ping": {"attempted": bool(ping), "ok": None, "latency_ms": None, "details": None, "error": None},
        },
        "last_errors": list(_LAST_ERRORS),
    }

    # Connectivity check (optional, lightweight)
    if ping and backend in ("foundry", "oss"):
        timeout_s = float(os.getenv("DIAGNOST_TIMEOUT_SECONDS") or "5")
        t0 = time.perf_counter()
        try:
            if backend == "foundry":
                details = foundry_ping(timeout_s=timeout_s)
            else:
                details = oss_ping(timeout_s=timeout_s)
            dt_ms = int((time.perf_counter() - t0) * 1000)
            diag["oss_connection"]["ping"] = {
                "attempted": True,
                "ok": True,
                "latency_ms": dt_ms,
                "details": details,
                "error": None,
            }
        except Exception as e:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            _push_error("diagnost_ping", e)
            diag["oss_connection"]["ping"] = {
                "attempted": True,
                "ok": False,
                "latency_ms": dt_ms,
                "details": None,
                "error": f"{type(e).__name__}: {e}",
            }

    return _pretty_json(diag)


# Common misspelling / alias
@app.get("/diagnostics")
def diagnostics(ping: bool = True) -> Response:
    return diagnost(ping=ping)


@app.post("/")
@app.post("/invoke")
@app.post("/v1/invoke")
async def invoke(
    req: GatewayRequest,
    request: Request,
    x_user_id: Optional[str] = Header(default=None, alias="X-User-ID"),
) -> GatewayResponse:
    start = time.perf_counter()

    # Log the minimal envelope so we can confirm the request reaches the service.
    logger.info(
        "invoke id=%s scenario=%s verbosity=%s tools=%s",
        req.id,
        req.scenario,
        req.verbosity,
        req.tools,
    )

    user_id = _resolve_user_id(req, x_user_id)

    answer_text, model_error, backend = _call_model_if_configured(req)

    elapsed = time.perf_counter() - start

    if model_error:
        status = "error"
        output: Any = {
            "text": answer_text,
            "error": model_error,
            "scenario": req.scenario,
        }
    else:
        status = "success"
        output = _build_output(req, answer_text)

    metrics = _build_metrics(req, answer_text, elapsed)

    # output_format is scenario-dependent in the original doc.
    output_format = req.scenario or "chat"

    resp = GatewayResponse(
        id=req.id,
        user_id=user_id,
        status=status,  # type: ignore[arg-type]
        output_format=output_format,
        output=output,
        metrics=metrics,
    )

    # Small server-side latency marker for debugging; keeps API stable.
    request.state.elapsed_s = elapsed
    request.state.backend = backend

    return resp
