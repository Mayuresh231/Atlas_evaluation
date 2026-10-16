import logging
import time
from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.domain import RetrievedEvidence
from app.schemas import AnswerStatus, ChatResponse
from app.services.citations import validate_citations
from app.services.openai_provider import AIProvider

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[RetrievedEvidence]: ...


ABSTENTION = "I do not have enough evidence in the indexed documents to answer that question."


class ChatService:
    def __init__(self, retriever: Retriever, provider: AIProvider, settings: Settings):
        self.retriever = retriever
        self.provider = provider
        self.settings = settings

    def answer(self, question: str) -> ChatResponse:
        started = time.perf_counter()
        evidence = self.retriever.retrieve(question)
        if not evidence or evidence[0].score < self.settings.min_evidence_score:
            logger.info(
                "event=chat_abstained reason=weak_evidence evidence_count=%d duration_ms=%.2f",
                len(evidence), (time.perf_counter() - started) * 1000,
            )
            return self._abstain()
        generated = self.provider.answer(self._prompt(question, evidence), self._images(evidence))
        citations = validate_citations(generated, evidence)
        if not citations:
            logger.warning(
                "event=chat_abstained reason=invalid_citations evidence_count=%d duration_ms=%.2f",
                len(evidence), (time.perf_counter() - started) * 1000,
            )
            return self._abstain()
        logger.info(
            "event=chat_answered evidence_count=%d citation_count=%d cited_evidence_ids=%s duration_ms=%.2f",
            len(evidence), len(citations),
            ",".join(str(citation.evidence_id) for citation in citations),
            (time.perf_counter() - started) * 1000,
        )
        return ChatResponse(status=AnswerStatus.answered, answer=generated.answer, citations=citations)

    @staticmethod
    def _prompt(question: str, evidence: list[RetrievedEvidence]) -> str:
        sources = []
        for item in evidence:
            locator = f"page {item.page_number}" if item.page_number else "standalone image"
            if item.figure_label:
                locator += f", {item.figure_label}"
            sources.append(
                f"[{item.source_id}] {item.document_title} ({locator}; type={item.kind})\n{item.content[:5000]}"
            )
        return (
            f"Question: {question}\n\nEvidence:\n\n" + "\n\n".join(sources) +
            "\n\nWrite a concise engineering answer. Use only the evidence. Add [S#] after every "
            "factual claim. If sources conflict, explain the conflict. Do not estimate unreadable plot "
            "values. In the structured citations, quote exact supporting text when the source is textual; "
            "for a visual observation the quote may be null."
        )

    @staticmethod
    def _images(evidence: list[RetrievedEvidence]) -> list[Path]:
        return [Path(item.asset_path) for item in evidence if item.asset_path and item.kind in {"figure", "page_image", "image"}]

    @staticmethod
    def _abstain() -> ChatResponse:
        return ChatResponse(status=AnswerStatus.insufficient_evidence, answer=ABSTENTION)

