import logging
import re
import time
import uuid

from fastapi import FastAPI, Request

from app.api.routes import router
from app.config import get_settings
from app.observability import configure_logging, request_id_context

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Citation-aware multimodal RAG for engineering PDFs and scientific figures.",
)


@app.middleware("http")
async def request_trace(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = supplied_id if re.fullmatch(r"[A-Za-z0-9._-]{1,100}", supplied_id) else uuid.uuid4().hex
    token = request_id_context.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "event=request_completed method=%s path=%s status=%s duration_ms=%.2f",
            request.method, request.url.path, response.status_code,
            (time.perf_counter() - started) * 1000,
        )
        return response
    except Exception:
        logger.exception(
            "event=request_failed method=%s path=%s duration_ms=%.2f",
            request.method, request.url.path, (time.perf_counter() - started) * 1000,
        )
        raise
    finally:
        request_id_context.reset(token)


app.include_router(router)

