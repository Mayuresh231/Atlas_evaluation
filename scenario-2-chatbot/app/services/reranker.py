import logging
import math
import time

from app.domain import RetrievedEvidence

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str, enabled: bool = True):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def rerank(self, question: str, candidates: list[RetrievedEvidence], top_k: int) -> list[RetrievedEvidence]:
        started = time.perf_counter()
        if not candidates:
            return []
        if not self.enabled:
            logger.info("event=rerank_skipped reason=disabled candidate_count=%d", len(candidates))
            return candidates[:top_k]
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            logits = self._model.predict([(question, item.content) for item in candidates])
            for item, logit in zip(candidates, logits, strict=True):
                value = float(logit)
                item.score = (
                    1 / (1 + math.exp(-value))
                    if value >= 0
                    else math.exp(value) / (1 + math.exp(value))
                )
            ranked = sorted(candidates, key=lambda item: item.score, reverse=True)[:top_k]
            logger.info(
                "event=rerank_completed model=%s candidate_count=%d returned_count=%d duration_ms=%.2f",
                self.model_name, len(candidates), len(ranked), (time.perf_counter() - started) * 1000,
            )
            return ranked
        except (ImportError, OSError, RuntimeError) as exc:
            logger.warning(
                "event=rerank_fallback model=%s candidate_count=%d error_type=%s",
                self.model_name, len(candidates), type(exc).__name__,
            )
            return candidates[:top_k]

