from enum import StrEnum
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class AnswerStatus(StrEnum):
    answered = "answered"
    insufficient_evidence = "insufficient_evidence"


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Citation(BaseModel):
    source_id: str
    evidence_id: UUID
    document_id: UUID
    document_title: str
    page_number: int | None = None
    section: str | None = None
    figure_label: str | None = None
    evidence_type: str
    quote: str | None = None
    source_url: str
    asset_url: str | None = None


class ChatResponse(BaseModel):
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    database: str
    openai: str


class IngestResponse(BaseModel):
    document_id: UUID
    title: str
    chunks_created: int
    figures_created: int
    unchanged: bool = False


class DocumentSummary(BaseModel):
    id: UUID
    title: str
    media_type: str
    page_count: int | None
    evidence_count: int
    created_at: datetime


class DocumentDetail(DocumentSummary):
    original_filename: str | None = None
    updated_at: datetime
    file_url: str


class EvidenceDetail(BaseModel):
    id: UUID
    document_id: UUID
    document_title: str
    kind: str
    content: str
    page_number: int | None
    section: str | None
    figure_label: str | None
    bbox: dict[str, float] | None
    asset_url: str | None = None

