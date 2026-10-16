import hashlib
import logging
import mimetypes
import time
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Document, EvidenceChunk
from app.schemas import IngestResponse
from app.services.extraction import DocumentExtractor
from app.services.openai_provider import AIProvider

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(self, session: Session, provider: AIProvider, settings: Settings):
        self.session = session
        self.provider = provider
        self.settings = settings
        self.extractor = DocumentExtractor(
            settings.data_dir / "assets", settings.chunk_size, settings.chunk_overlap
        )

    def ingest(self, path: Path, describe_images: bool = True) -> IngestResponse:
        started = time.perf_counter()
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        existing_hash = self.session.scalar(select(Document).where(Document.content_sha256 == digest))
        if existing_hash:
            logger.info(
                "event=ingestion_skipped reason=unchanged document_id=%s evidence_count=%d duration_ms=%.2f",
                existing_hash.id, len(existing_hash.chunks), (time.perf_counter() - started) * 1000,
            )
            return IngestResponse(
                document_id=existing_hash.id, title=existing_hash.title,
                chunks_created=len(existing_hash.chunks),
                figures_created=sum(c.kind != "text" for c in existing_hash.chunks), unchanged=True,
            )
        existing_path = self.session.scalar(select(Document).where(Document.source_path == str(path)))
        if existing_path:
            self.session.delete(existing_path)
            self.session.flush()

        extracted, page_count = self.extractor.extract(path, digest)
        if not extracted:
            raise ValueError("No text or visual evidence could be extracted")
        if describe_images:
            for item in extracted:
                if item.asset_path:
                    description = self.provider.describe_image(item.asset_path, item.content)
                    item.content = f"{item.content}\nVisual description: {description}".strip()
        embeddings = self.provider.embed([item.content for item in extracted])

        document = Document(
            title=path.stem, source_path=str(path), content_sha256=digest,
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            page_count=page_count, document_metadata={"original_filename": path.name},
        )
        self.session.add(document)
        self.session.flush()
        for item, embedding in zip(extracted, embeddings, strict=True):
            self.session.add(EvidenceChunk(
                document_id=document.id, kind=item.kind, content=item.content,
                page_number=item.page_number, section=item.section, figure_label=item.figure_label,
                bbox=item.bbox, asset_path=str(item.asset_path) if item.asset_path else None,
                embedding_model=self.settings.openai_embed_model, embedding=embedding,
            ))
        self.session.commit()
        figures = sum(item.kind != "text" for item in extracted)
        logger.info(
            "event=ingestion_completed document_id=%s evidence_count=%d figure_count=%d "
            "page_count=%s embedding_model=%s duration_ms=%.2f",
            document.id, len(extracted), figures, page_count,
            self.settings.openai_embed_model, (time.perf_counter() - started) * 1000,
        )
        return IngestResponse(
            document_id=document.id, title=document.title,
            chunks_created=len(extracted), figures_created=figures,
        )

