"""Real model probability and review behavior tests."""

import math

import pytest

from supportiq.config import CATEGORIES
from supportiq.exceptions import InferenceError
from supportiq.models.classifier import Classifier
from supportiq.services.classification import ClassificationService


def test_single_and_batch_probabilities(fitted) -> None:
    model, _ = fitted
    service = ClassificationService(Classifier(model), 0.6, "1.0.0")
    results = service.classify_batch(["I need an invoice", "The app crashes"])
    assert len(results) == 2
    assert service.classify("I need an invoice")["predicted_category"] in CATEGORIES
    for result in results:
        scores = result["category_probabilities"]
        assert set(scores) == set(CATEGORIES)
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in scores.values())
        assert sum(scores.values()) == pytest.approx(1)
        assert result["confidence"] == max(scores.values())


def test_review_threshold(fitted) -> None:
    model, _ = fitted
    classifier = Classifier(model)
    assert not ClassificationService(classifier, 0, "1").classify("Question")["needs_human_review"]
    assert ClassificationService(classifier, 1, "1").classify("Question")["needs_human_review"]


def test_prediction_failure(fitted, monkeypatch) -> None:
    model, _ = fitted

    def broken(_messages):
        raise RuntimeError("private internals")

    monkeypatch.setattr(model, "predict_proba", broken)
    with pytest.raises(InferenceError, match="inference failed"):
        Classifier(model).predict_probabilities(["hello"])
