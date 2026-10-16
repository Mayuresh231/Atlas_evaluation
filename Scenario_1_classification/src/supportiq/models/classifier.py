"""Probability inference with strict output validation."""

import math
from collections.abc import Sequence

from sklearn.pipeline import Pipeline

from supportiq.config import CATEGORIES
from supportiq.exceptions import InferenceError
from supportiq.preprocessing.text import normalize_messages


class Classifier:
    """Reusable wrapper around one already fitted scikit-learn pipeline."""

    def __init__(self, model: Pipeline) -> None:
        self.model = model

    def predict_probabilities(self, messages: Sequence[str]) -> list[dict[str, float]]:
        """Return one complete, validated probability map per input message."""
        normalized = normalize_messages(messages)
        if not normalized:
            return []
        try:
            matrix = self.model.predict_proba(normalized)
            labels = list(self.model.classes_)
        except Exception as exc:
            raise InferenceError("Model inference failed.") from exc
        if set(labels) != set(CATEGORIES) or len(matrix) != len(normalized):
            raise InferenceError("Model returned an invalid probability shape.")
        results = []
        for row in matrix:
            if len(row) != len(labels):
                raise InferenceError("Model returned an invalid probability shape.")
            values = [float(value) for value in row]
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
                raise InferenceError("Model returned invalid probabilities.")
            if not math.isclose(sum(values), 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise InferenceError("Model probabilities do not sum to one.")
            results.append(dict(zip(labels, values, strict=True)))
        return results
