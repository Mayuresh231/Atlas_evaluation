import re

from app.domain import RetrievedEvidence
from app.schemas import Citation
from app.services.openai_provider import GeneratedAnswer


def _normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def validate_citations(generated: GeneratedAnswer, evidence: list[RetrievedEvidence]) -> list[Citation]:
    by_id = {item.source_id: item for item in evidence}
    valid: list[Citation] = []
    seen: set[str] = set()
    for claimed in generated.citations:
        item = by_id.get(claimed.source_id)
        if item is None or claimed.source_id in seen:
            continue
        if f"[{claimed.source_id}]" not in generated.answer:
            continue
        if claimed.quote and _normalized(claimed.quote) not in _normalized(item.content):
            continue
        valid.append(Citation(
            source_id=item.source_id, evidence_id=item.chunk_id,
            document_id=item.document_id, document_title=item.document_title,
            page_number=item.page_number, section=item.section, figure_label=item.figure_label,
            evidence_type=item.kind, quote=claimed.quote,
            source_url=f"/documents/{item.document_id}/evidence/{item.chunk_id}",
            asset_url=(
                f"/documents/{item.document_id}/evidence/{item.chunk_id}/asset"
                if item.asset_path else None
            ),
        ))
        seen.add(claimed.source_id)
    return valid

