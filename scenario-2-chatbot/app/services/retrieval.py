import logging
import time
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.domain import RetrievedEvidence
from app.models import Document, EvidenceChunk
from app.services.openai_provider import AIProvider
from app.services.reranker import Reranker

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    rankings: list[list[UUID]],
    k: int = 60,
) -> dict[UUID, float]:
    """Combine independent rankings and normalize the theoretical best score to 1."""
    if not rankings:
        return {}
    raw_scores: dict[UUID, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            raw_scores[item_id] = raw_scores.get(item_id, 0.0) + 1.0 / (k + rank)
    maximum = len(rankings) / (k + 1)
    return {item_id: min(score / maximum, 1.0) for item_id, score in raw_scores.items()}


class RetrievalService:
    def __init__(self, session: Session, provider: AIProvider, reranker: Reranker, settings: Settings):
        self.session = session
        self.provider = provider
        self.reranker = reranker
        self.settings = settings

    def retrieve(self, question: str) -> list[RetrievedEvidence]:
        started = time.perf_counter()
        query_embedding = self.provider.embed([question])[0]
        limit = self.settings.retrieval_candidates
        distance = EvidenceChunk.embedding.cosine_distance(query_embedding)
        dense_rows = self.session.execute(
            select(EvidenceChunk, Document, (1 - distance).label("score"))
            .join(Document)
            .where(EvidenceChunk.embedding_model == self.settings.openai_embed_model)
            .order_by(distance)
            .limit(limit)
        ).all()

        query = func.websearch_to_tsquery("english", question)
        lexical_score = func.ts_rank_cd(EvidenceChunk.search_vector, query)
        lexical_rows = self.session.execute(
            select(EvidenceChunk, Document, lexical_score.label("score"))
            .join(Document)
            .where(EvidenceChunk.search_vector.op("@@")(query))
            .order_by(lexical_score.desc())
            .limit(limit)
        ).all()

        rows_by_id = {
            chunk.id: (chunk, document)
            for chunk, document, _ in [*dense_rows, *lexical_rows]
        }
        dense_scores = {
            chunk.id: max(0.0, float(score or 0.0))
            for chunk, _, score in dense_rows
        }
        fused_scores = reciprocal_rank_fusion(
            [
                [chunk.id for chunk, _, _ in dense_rows],
                [chunk.id for chunk, _, _ in lexical_rows],
            ],
            self.settings.rrf_k,
        )
        combined = {
            chunk_id: self._to_result(
                *rows_by_id[chunk_id],
                score=dense_scores.get(chunk_id, 0.0),
                fusion_score=fusion_score,
            )
            for chunk_id, fusion_score in fused_scores.items()
        }

        candidates = sorted(combined.values(), key=lambda item: item.fusion_score, reverse=True)
        ranked = self.reranker.rerank(question, candidates, self.settings.rerank_top_k)
        for index, item in enumerate(ranked, start=1):
            item.source_id = f"S{index}"
        logger.info(
            "event=retrieval_completed dense_count=%d lexical_count=%d fused_count=%d "
            "returned_count=%d evidence_ids=%s duration_ms=%.2f",
            len(dense_rows), len(lexical_rows), len(candidates), len(ranked),
            ",".join(str(item.chunk_id) for item in ranked),
            (time.perf_counter() - started) * 1000,
        )
        return ranked

    @staticmethod
    def _to_result(
        chunk: EvidenceChunk,
        document: Document,
        score: float,
        fusion_score: float,
    ) -> RetrievedEvidence:
        return RetrievedEvidence(
            chunk_id=chunk.id, document_id=document.id, document_title=document.title,
            kind=chunk.kind, content=chunk.content, page_number=chunk.page_number,
            section=chunk.section, figure_label=chunk.figure_label,
            asset_path=chunk.asset_path, score=score, fusion_score=fusion_score,
        )

