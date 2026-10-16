import uuid

from app.domain import RetrievedEvidence
from app.services.citations import validate_citations
from app.services.openai_provider import GeneratedAnswer, GeneratedCitation


def evidence(source_id: str = "S1") -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), document_title="Pump manual",
        kind="text", content="Stop the pump above 70 psi.", page_number=4,
        section="Safety", figure_label=None, asset_path=None, score=0.9, source_id=source_id,
    )


def test_accepts_citation_with_exact_support() -> None:
    generated = GeneratedAnswer(
        answer="Stop the pump above 70 psi [S1].",
        citations=[GeneratedCitation(source_id="S1", quote="above 70 psi")],
    )
    result = validate_citations(generated, [evidence()])
    assert len(result) == 1
    assert result[0].page_number == 4
    assert str(result[0].evidence_id) in result[0].source_url
    assert result[0].asset_url is None


def test_rejects_unknown_or_unsupported_citation() -> None:
    generated = GeneratedAnswer(
        answer="The limit is 90 psi [S2].",
        citations=[GeneratedCitation(source_id="S2", quote="90 psi")],
    )
    assert validate_citations(generated, [evidence()]) == []

