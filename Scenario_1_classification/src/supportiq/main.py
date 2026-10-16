"""FastAPI factory and centralized public error handling."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from supportiq import __version__
from supportiq.api.routes import router
from supportiq.config import Settings
from supportiq.exceptions import ArtifactError, BatchSizeError, InferenceError, SupportIQError
from supportiq.models.artifacts import load_artifact
from supportiq.models.classifier import Classifier
from supportiq.services.classification import ClassificationService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, service: ClassificationService | None = None) -> FastAPI:
    """Build an application; load the artifact once unless a service is injected."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        if app.state.service is None:
            try:
                model, metadata = load_artifact(settings.model_path)
                app.state.service = ClassificationService(
                    Classifier(model),
                    settings.confidence_threshold,
                    metadata["model_version"],
                    {
                        key: value
                        for key, value in metadata.items()
                        if key
                        in {
                            "trained_at",
                            "dataset_sha256",
                            "train_samples",
                            "validation_samples",
                            "test_samples",
                            "sklearn_version",
                        }
                    },
                )
                logger.info("Trained model loaded")
            except (ArtifactError, KeyError) as exc:
                logger.error("Model is unavailable: %s", exc)
        yield

    app = FastAPI(title="SupportIQ", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.service = service
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in exc.errors())
        return JSONResponse(
            status_code=422, content={"error": {"code": "invalid_request", "message": details}}
        )

    @app.exception_handler(SupportIQError)
    async def domain_error(_request: Request, exc: SupportIQError) -> JSONResponse:
        status = 503 if isinstance(exc, ArtifactError) else 500 if isinstance(exc, InferenceError) else 422
        message = str(exc) if not isinstance(exc, InferenceError) else "Model inference failed."
        if isinstance(exc, BatchSizeError):
            message = f"Batch exceeds the maximum of {settings.max_batch_size} messages."
        return JSONResponse(status_code=status, content={"error": {"code": exc.code, "message": message}})

    @app.exception_handler(Exception)
    async def unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected request failure", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error."}},
        )

    return app


app = create_app()
