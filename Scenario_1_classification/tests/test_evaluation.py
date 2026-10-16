"""Known metric values, absent classes, and JSON serialization."""

import json

import pytest

from supportiq.config import CATEGORIES
from supportiq.evaluation.metrics import error_analysis, evaluate, write_json_report


def test_known_metrics_and_order(settings) -> None:
    labels = list(CATEGORIES)
    report = evaluate([labels[0], labels[0], labels[1]], [labels[0], labels[1], labels[1]])
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["labels"] == labels
    assert report["per_category"][labels[0]]["recall"] == pytest.approx(0.5)
    assert report["per_category"][labels[2]]["support"] == 0
    assert len(report["confusion_matrix"]) == len(labels)
    assert report["confusion_matrix"][0][1] == 1
    write_json_report(settings.report_path, report)
    assert json.loads(settings.report_path.read_text())["sample_count"] == 3
    assert error_analysis(["1"], ["text"], [labels[0]], [labels[1]])[0]["message_id"] == "1"


def test_empty_evaluation() -> None:
    with pytest.raises(ValueError):
        evaluate([], [])
