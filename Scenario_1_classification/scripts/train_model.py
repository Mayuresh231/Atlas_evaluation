"""Train and evaluate the local support classifier."""

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import sklearn

from supportiq import __version__
from supportiq.config import Settings
from supportiq.evaluation.metrics import error_analysis, evaluate, write_json_report
from supportiq.models.artifacts import save_artifact
from supportiq.models.training import dataset_sha256, load_dataset, train_pipeline


def main() -> None:
    """Run one reproducible offline fit and held-out evaluation."""
    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=defaults.dataset_path)
    parser.add_argument("--model", type=Path, default=defaults.model_path)
    parser.add_argument("--report", type=Path, default=defaults.report_path)
    parser.add_argument("--max-features", type=int, default=defaults.max_features)
    parser.add_argument("--c", type=float, default=defaults.regularization_c)
    parser.add_argument("--seed", type=int, default=defaults.random_seed)
    parser.add_argument("--errors", type=Path, help="Optional local misclassification report")
    args = parser.parse_args()
    settings = Settings(
        model_path=args.model,
        dataset_path=args.dataset,
        report_path=args.report,
        max_features=args.max_features,
        regularization_c=args.c,
        random_seed=args.seed,
    )
    frame = load_dataset(settings.dataset_path)
    model, splits = train_pipeline(frame, settings)
    true = splits.test["category"].tolist()
    predicted = model.predict(splits.test["message"].tolist()).tolist()
    report = {
        "model_version": __version__,
        "dataset_sha256": dataset_sha256(settings.dataset_path),
        "split": {
            "train": len(splits.train),
            "validation": len(splits.validation),
            "test": len(splits.test),
            "seed": settings.random_seed,
        },
        "validation": evaluate(
            splits.validation["category"].tolist(),
            model.predict(splits.validation["message"].tolist()).tolist(),
        ),
        "test": evaluate(true, predicted),
    }
    metadata = {
        "model_version": __version__,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": report["dataset_sha256"],
        "train_samples": len(splits.train),
        "validation_samples": len(splits.validation),
        "test_samples": len(splits.test),
    }
    save_artifact(settings.model_path, model, metadata)
    write_json_report(settings.report_path, report)
    if args.errors:
        write_json_report(
            args.errors,
            {
                "misclassifications": error_analysis(
                    splits.test["message_id"].tolist(), splits.test["message"].tolist(), true, predicted
                )
            },
        )
    print(
        json.dumps(
            {
                "model": str(settings.model_path),
                "report": str(settings.report_path),
                "test_samples": len(true),
                "test_macro_f1": report["test"]["macro_f1"],
            }
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
