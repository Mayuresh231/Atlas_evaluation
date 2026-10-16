"""Pydantic API contracts."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from supportiq.preprocessing.text import normalize_message


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassificationRequest(StrictModel):
    message: str = Field(..., max_length=10000, description="Customer support message")

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        """Apply the same normalization used by the trained pipeline."""
        return normalize_message(value)


class BatchRequest(StrictModel):
    messages: list[ClassificationRequest] = Field(..., min_length=1)


class ClassificationResponse(StrictModel):
    predicted_category: str
    confidence: float = Field(ge=0, le=1)
    needs_human_review: bool
    category_probabilities: dict[str, float]
    model_version: str


class BatchItemResponse(ClassificationResponse):
    index: int


class BatchResponse(StrictModel):
    results: list[BatchItemResponse]


class HealthResponse(StrictModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(StrictModel):
    model_name: str
    model_version: str
    supported_categories: list[str]
    confidence_threshold: float
    training_metadata: dict[str, str | int | float]


class ErrorResponse(StrictModel):
    error: dict[str, str]
