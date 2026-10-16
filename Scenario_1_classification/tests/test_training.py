"""Dataset validation, splitting, training, and artifact tests."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from supportiq.config import CATEGORIES, Settings
from supportiq.exceptions import ArtifactError, DatasetError
from supportiq.models.artifacts import load_artifact, save_artifact
from supportiq.models.training import load_dataset, split_dataset


def test_valid_dataset_and_reproducible_split(training_frame: pd.DataFrame, settings: Settings) -> None:
    training_frame.to_csv(settings.dataset_path, index=False)
    loaded = load_dataset(settings.dataset_path)
    first, second = split_dataset(loaded, 42), split_dataset(loaded, 42)
    assert len(loaded) == 60
    assert first.test["message_id"].tolist() == second.test["message_id"].tolist()
    assert set(first.train.message_id).isdisjoint(first.test.message_id)
    assert set(first.validation.message_id).isdisjoint(first.test.message_id)


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda f: f.drop(columns="message"), "missing required"),
        (lambda f: f.iloc[0:0], "empty"),
        (lambda f: f.assign(message=pd.NA), "blank message"),
        (lambda f: f.assign(category=pd.NA), "blank category"),
        (lambda f: f.assign(category="unknown"), "Unsupported categories"),
        (lambda f: f.assign(message_id="same"), "duplicate message_id"),
    ],
)
def test_invalid_dataset(training_frame: pd.DataFrame, settings: Settings, mutation, error: str) -> None:
    mutation(training_frame).to_csv(settings.dataset_path, index=False)
    with pytest.raises(DatasetError, match=error):
        load_dataset(settings.dataset_path)


def test_exact_duplicate_rejected(training_frame: pd.DataFrame, settings: Settings) -> None:
    training_frame.loc[1, "message"] = training_frame.loc[0, "message"]
    training_frame.to_csv(settings.dataset_path, index=False)
    with pytest.raises(DatasetError, match="duplicate messages"):
        load_dataset(settings.dataset_path)


def test_insufficient_groups(training_frame: pd.DataFrame) -> None:
    with pytest.raises(DatasetError, match="10 distinct"):
        split_dataset(training_frame.iloc[:9], 42)


def test_artifact_roundtrip_and_failures(fitted, settings: Settings, tmp_path: Path) -> None:
    model, _ = fitted
    import sklearn

    from supportiq import __version__

    metadata = {"model_version": __version__, "sklearn_version": sklearn.__version__}
    save_artifact(settings.model_path, model, metadata)
    restored, details = load_artifact(settings.model_path)
    assert set(restored.classes_) == set(CATEGORIES)
    assert details == metadata
    with pytest.raises(ArtifactError, match="missing"):
        load_artifact(tmp_path / "missing.joblib")
    settings.model_path.write_bytes(b"not a joblib artifact")
    with pytest.raises(ArtifactError, match="unreadable"):
        load_artifact(settings.model_path)
    joblib.dump({"format": 99, "model": model, "metadata": metadata}, settings.model_path)
    with pytest.raises(ArtifactError, match="unsupported format"):
        load_artifact(settings.model_path)
