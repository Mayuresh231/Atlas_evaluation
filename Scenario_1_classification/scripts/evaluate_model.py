"""Evaluate an existing artifact on a supplied labeled CSV (not a new holdout)."""

import argparse
import json
from pathlib import Path

from supportiq.config import Settings
from supportiq.evaluation.metrics import evaluate, write_json_report
from supportiq.models.artifacts import load_artifact
from supportiq.models.training import load_dataset


def main() -> None:
    """Score an existing trusted artifact against a labeled CSV."""
    defaults = Settings.from_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=defaults.model_path)
    parser.add_argument("--report", type=Path, default=defaults.report_path.with_name("evaluation.json"))
    args = parser.parse_args()
    frame = load_dataset(args.dataset)
    model, metadata = load_artifact(args.model)
    report = {
        "model_version": metadata["model_version"],
        "evaluation": evaluate(frame["category"].tolist(), model.predict(frame["message"].tolist()).tolist()),
    }
    write_json_report(args.report, report)
    print(json.dumps({"report": str(args.report), "macro_f1": report["evaluation"]["macro_f1"]}))


if __name__ == "__main__":
    main()
