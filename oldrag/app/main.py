from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response

from .schemas import RagSearchRequest, RagSearchResponse


logger = logging.getLogger("oldrag")


def _setup_logging() -> None:
    level = (os.getenv("LOG_LEVEL") or "INFO").upper().strip()
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


_setup_logging()

class UTF8JSONResponse(JSONResponse):
    """Force UTF-8 charset in JSON responses (PowerShell friendly)."""

    media_type = "application/json; charset=utf-8"


app = FastAPI(title="oldrag", version="0.1.0", default_response_class=UTF8JSONResponse)


# Process start markers for /version (uptime / restarts visibility)
_PROCESS_START_UTC = datetime.now(timezone.utc).replace(microsecond=0)
_PROCESS_START_MONO = time.monotonic()



@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}



def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _human_uptime(seconds: float) -> str:
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
    env_commit = (os.getenv("GIT_COMMIT") or os.getenv("SERVICE_GIT_COMMIT") or "").strip()
    if env_commit:
        return env_commit

    candidates = [
        (os.getenv("VERSION_FILE") or "").strip() or None,
        "/app/version.tmp",
        "/src/version.tmp",
        "/src/oldrag/version.tmp",
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


def _pretty_json(data: object) -> Response:
    return Response(
        content=json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        media_type="application/json; charset=utf-8",
    )


@app.get("/version")
def version() -> Response:
    uptime_s = max(0.0, time.monotonic() - _PROCESS_START_MONO)
    data = {
        "service": {
            "name": app.title,
            "version": app.version,
            "git_commit": _read_git_commit() or "unknown",
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



@app.get("/checks/liveness")
def checks_liveness() -> dict[str, bool]:
    return health()


@app.get("/checks/readiness")
def checks_readiness() -> dict[str, bool]:
    # oldrag is a stub with no external deps; treat readiness as liveness
    return health()

@app.post("/")
@app.post("/search")
@app.post("/v1/search")
async def search(req: RagSearchRequest) -> RagSearchResponse:
    start = time.perf_counter()
    logger.info(
        "search id=%s max_sources=%s min_score=%s queries=%d",
        req.id,
        req.max_sources,
        req.min_score,
        len(req.queries),
    )

    # Stub implementation: return empty chunks, but keep a valid envelope.
    # Later: replace this with a call to the real RAG search backend.
    elapsed = time.perf_counter() - start

    details: Optional[str] = None
    if os.getenv("INCLUDE_DEBUG_DETAILS") == "1":
        details = f"stub; elapsed_s={elapsed:.6f}"

    return RagSearchResponse(id=req.id, status="ok", details=details, chunks=[])
