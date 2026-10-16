import re
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from openai import OpenAIError
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_session
from app.dependencies import get_chat_service, get_provider
from app.models import Document, EvidenceChunk
from app.schemas import (
    ChatRequest, ChatResponse, DocumentDetail, DocumentSummary, EvidenceDetail,
    HealthResponse, IngestResponse,
)
from app.services.chat import ChatService
from app.services.ingestion import IngestionService

router = APIRouter()


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    session: Session = Depends(get_session),
) -> list[DocumentSummary]:
    rows = session.execute(
        select(Document, func.count(EvidenceChunk.id))
        .outerjoin(EvidenceChunk)
        .group_by(Document.id)
        .order_by(Document.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return [
        DocumentSummary(
            id=document.id, title=document.title, media_type=document.media_type,
            page_count=document.page_count, evidence_count=count,
            created_at=document.created_at,
        )
        for document, count in rows
    ]


@router.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: uuid.UUID, session: Session = Depends(get_session)) -> DocumentDetail:
    row = session.execute(
        select(Document, func.count(EvidenceChunk.id))
        .outerjoin(EvidenceChunk)
        .where(Document.id == document_id)
        .group_by(Document.id)
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    document, count = row
    return DocumentDetail(
        id=document.id, title=document.title, media_type=document.media_type,
        page_count=document.page_count, evidence_count=count,
        original_filename=document.document_metadata.get("original_filename"),
        created_at=document.created_at, updated_at=document.updated_at,
        file_url=f"/documents/{document.id}/file",
    )


@router.get("/documents/{document_id}/file", response_class=FileResponse)
def get_document_file(document_id: uuid.UUID, session: Session = Depends(get_session)) -> FileResponse:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(document.source_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Source file is no longer available")
    filename = document.document_metadata.get("original_filename") or path.name
    return FileResponse(path, media_type=document.media_type, filename=filename)


def _get_evidence(document_id: uuid.UUID, evidence_id: uuid.UUID, session: Session) -> EvidenceChunk:
    evidence = session.scalar(
        select(EvidenceChunk).where(
            EvidenceChunk.id == evidence_id,
            EvidenceChunk.document_id == document_id,
        )
    )
    if evidence is None:
        raise HTTPException(status_code=404, detail="Evidence not found")
    return evidence


@router.get("/documents/{document_id}/evidence/{evidence_id}", response_model=EvidenceDetail)
def get_evidence(
    document_id: uuid.UUID,
    evidence_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> EvidenceDetail:
    evidence = _get_evidence(document_id, evidence_id, session)
    return EvidenceDetail(
        id=evidence.id, document_id=evidence.document_id,
        document_title=evidence.document.title, kind=evidence.kind,
        content=evidence.content, page_number=evidence.page_number,
        section=evidence.section, figure_label=evidence.figure_label, bbox=evidence.bbox,
        asset_url=(
            f"/documents/{document_id}/evidence/{evidence_id}/asset"
            if evidence.asset_path else None
        ),
    )


@router.get("/documents/{document_id}/evidence/{evidence_id}/asset", response_class=FileResponse)
def get_evidence_asset(
    document_id: uuid.UUID,
    evidence_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> FileResponse:
    evidence = _get_evidence(document_id, evidence_id, session)
    if not evidence.asset_path:
        raise HTTPException(status_code=404, detail="Evidence has no visual asset")
    path = Path(evidence.asset_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Visual asset is no longer available")
    return FileResponse(path)


@router.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    database_status = "ok"
    openai_status = "ok"
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError:
        database_status = "unavailable"
    try:
        get_provider().ping()
    except (OpenAIError, RuntimeError):
        openai_status = "unavailable"
    status = "ok" if database_status == openai_status == "ok" else "degraded"
    return HealthResponse(status=status, database=database_status, openai=openai_status)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, service: ChatService = Depends(get_chat_service)) -> ChatResponse:
    try:
        return service.answer(payload.question)
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail="OpenAI request failed") from exc


@router.post("/documents", response_model=IngestResponse, status_code=201)
def ingest_document(
    file: UploadFile = File(...),
    describe_images: bool = True,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=415, detail="Supported types: PDF, PNG, JPG, JPEG, WEBP")
    content = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "document").stem)[:100]
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{safe_stem}-{uuid.uuid4().hex[:8]}{suffix}"
    path.write_bytes(content)
    try:
        return IngestionService(session, get_provider(), settings).ingest(path, describe_images)
    except (ValueError, RuntimeError, OpenAIError) as exc:
        session.rollback()
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

