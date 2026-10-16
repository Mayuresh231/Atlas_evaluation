from dataclasses import dataclass
from pathlib import Path
from uuid import UUID


@dataclass(slots=True)
class ExtractedEvidence:
    kind: str
    content: str
    page_number: int | None = None
    section: str | None = None
    figure_label: str | None = None
    bbox: dict[str, float] | None = None
    asset_path: Path | None = None


@dataclass(slots=True)
class RetrievedEvidence:
    chunk_id: UUID
    document_id: UUID
    document_title: str
    kind: str
    content: str
    page_number: int | None
    section: str | None
    figure_label: str | None
    asset_path: str | None
    score: float
    fusion_score: float = 0.0
    source_id: str = ""

