"""Versioned joblib persistence for trusted local model artifacts."""

import os
from pathlib import Path

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from supportiq import __version__
from supportiq.config import CATEGORIES
from supportiq.exceptions import ArtifactError

ARTIFACT_FORMAT = 1


def save_artifact(path: Path, model: Pipeline, metadata: dict) -> None:
    """Persist a trained pipeline and metadata using an atomic file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        joblib.dump({"format": ARTIFACT_FORMAT, "model": model, "metadata": metadata}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_artifact(path: Path) -> tuple[Pipeline, dict]:
    """Load and validate a controlled artifact; joblib files must be trusted."""
    if not path.is_file():
        raise ArtifactError("Trained model is missing. Run the offline training command first.")
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ArtifactError("Trained model artifact is unreadable or incompatible.") from exc
    if not isinstance(payload, dict) or payload.get("format") != ARTIFACT_FORMAT:
        raise ArtifactError("Trained model artifact has an unsupported format.")
    model, metadata = payload.get("model"), payload.get("metadata")
    if not isinstance(model, Pipeline) or not isinstance(metadata, dict):
        raise ArtifactError("Trained model artifact has an invalid structure.")
    if list(model.named_steps) != ["normalize", "tfidf", "classifier"]:
        raise ArtifactError("Trained model artifact has an unexpected pipeline.")
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise ArtifactError("Trained model artifact is not fitted for probability inference.")
    if set(model.classes_) != set(CATEGORIES):
        raise ArtifactError("Trained model categories do not match this service version.")
    if metadata.get("model_version") != __version__:
        raise ArtifactError("Trained model version does not match this service version.")
    if metadata.get("sklearn_version") != sklearn.__version__:
        raise ArtifactError("Trained model was built with a different scikit-learn version.")
    return model, metadata
