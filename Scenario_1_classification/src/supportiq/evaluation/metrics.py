"""Machine-readable classification metrics with fixed label ordering."""

import json
from pathlib import Path

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from supportiq.config import CATEGORIES


def evaluate(y_true: list[str], y_pred: list[str]) -> dict:
    """Calculate overall and per-class metrics with zero_division=0."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Evaluation requires equally sized non-empty label sequences")
    labels = list(CATEGORIES)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "sample_count": len(y_true),
        "labels": labels,
        "class_distribution": dict(zip(labels, [int(n) for n in support], strict=True)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
        "per_category": {
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
            for label, p, r, f, n in zip(labels, precision, recall, f1, support, strict=True)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def error_analysis(
    ids: list[str], messages: list[str], y_true: list[str], y_pred: list[str]
) -> list[dict[str, str]]:
    """Return misclassified examples for an optional, locally stored report."""
    return [
        {"message_id": identifier, "message": message, "actual": actual, "predicted": predicted}
        for identifier, message, actual, predicted in zip(ids, messages, y_true, y_pred, strict=True)
        if actual != predicted
    ]


def write_json_report(path: Path, report: dict) -> None:
    """Write a UTF-8 JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
