"""Thin FastAPI endpoints using the application-injected service."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from supportiq.config import CATEGORIES
from supportiq.exceptions import ArtifactError, BatchSizeError
from supportiq.schemas.classification import (
    BatchItemResponse,
    BatchRequest,
    BatchResponse,
    ClassificationRequest,
    ClassificationResponse,
    HealthResponse,
    ModelInfoResponse,
)
from supportiq.services.classification import ClassificationService

router = APIRouter()


def get_service(request: Request) -> ClassificationService:
    """Resolve the loaded service or fail readiness clearly."""
    service = request.app.state.service
    if service is None:
        raise ArtifactError("Trained model is unavailable. Run the offline training command first.")
    return service


ServiceDependency = Annotated[ClassificationService, Depends(get_service)]


@router.get("/health", response_model=HealthResponse, summary="Process and model readiness")
def health(request: Request) -> HealthResponse:
    """Report process state; return 503 when model inference is unavailable."""
    service = request.app.state.service
    if service is None:
        raise ArtifactError("Trained model is unavailable. Run the offline training command first.")
    return HealthResponse(status="ready", model_loaded=True, model_version=service.model_version)


@router.get("/v1/model/info", response_model=ModelInfoResponse)
def model_info(service: ServiceDependency) -> ModelInfoResponse:
    """Return safe, public model metadata."""
    return ModelInfoResponse(
        model_name="TF-IDF + Logistic Regression",
        model_version=service.model_version,
        supported_categories=list(CATEGORIES),
        confidence_threshold=service.threshold,
        training_metadata=service.training_metadata,
    )


@router.post("/v1/classify", response_model=ClassificationResponse)
def classify(payload: ClassificationRequest, service: ServiceDependency) -> ClassificationResponse:
    """Classify one customer support message."""
    return ClassificationResponse(**service.classify(payload.message))


@router.post("/v1/classify/batch", response_model=BatchResponse)
def classify_batch(payload: BatchRequest, request: Request, service: ServiceDependency) -> BatchResponse:
    """Classify a bounded batch; one invalid item invalidates the whole request."""
    if len(payload.messages) > request.app.state.settings.max_batch_size:
        raise BatchSizeError("Batch exceeds the configured maximum size.")
    results = service.classify_batch([item.message for item in payload.messages])
    return BatchResponse(
        results=[BatchItemResponse(index=index, **item) for index, item in enumerate(results)]
    )
