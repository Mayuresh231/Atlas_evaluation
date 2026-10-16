"""Classification orchestration independent of HTTP."""

from collections.abc import Sequence

from supportiq.config import CATEGORIES
from supportiq.models.classifier import Classifier
from supportiq.preprocessing.text import normalize_messages


class ClassificationService:
    """Return model predictions and flag uncertain messages for review."""

    def __init__(
        self,
        classifier: Classifier,
        threshold: float,
        model_version: str,
        training_metadata: dict | None = None,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.classifier = classifier
        self.threshold = threshold
        self.model_version = model_version
        self.training_metadata = training_metadata or {}

    def classify(self, message: str) -> dict:
        """Classify one message."""
        return self.classify_batch([message])[0]

    def classify_batch(self, messages: Sequence[str]) -> list[dict]:
        """Classify messages in input order."""
        probabilities = self.classifier.predict_probabilities(normalize_messages(messages))
        output = []
        for scores in probabilities:
            category = max(CATEGORIES, key=lambda label: scores[label])
            confidence = scores[category]
            output.append(
                {
                    "predicted_category": category,
                    "confidence": confidence,
                    "needs_human_review": confidence < self.threshold,
                    "category_probabilities": scores,
                    "model_version": self.model_version,
                }
            )
        return output
