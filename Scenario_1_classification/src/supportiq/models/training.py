"""Offline supervised training with group-aware held-out splits."""

import hashlib
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from supportiq.config import CATEGORIES, Settings
from supportiq.exceptions import DatasetError
from supportiq.preprocessing.text import normalize_message, normalize_messages

REQUIRED_COLUMNS = {"message_id", "message", "category"}


@dataclass(frozen=True)
class DataSplits:
    """Disjoint train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_dataset(path: Path) -> pd.DataFrame:
    """Read and validate a labeled CSV; reject exact duplicate messages."""
    try:
        frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError(f"Cannot read dataset: {exc}") from exc
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise DatasetError("Dataset is empty")
    for column in REQUIRED_COLUMNS:
        if frame[column].isna().any() or frame[column].str.strip().eq("").any():
            raise DatasetError(f"Dataset has missing or blank {column} values")
    if frame["message_id"].duplicated().any():
        raise DatasetError("Dataset has duplicate message_id values")
    unknown = set(frame["category"].unique()) - set(CATEGORIES)
    if unknown:
        raise DatasetError(f"Unsupported categories: {', '.join(sorted(unknown))}")
    frame = frame.copy()
    frame["message"] = frame["message"].map(normalize_message)
    keys = frame["message"].str.casefold()
    if keys.duplicated().any():
        raise DatasetError("Dataset contains exact duplicate messages after normalization")
    return frame.reset_index(drop=True)


def _near_duplicate_groups(frame: pd.DataFrame) -> list[int]:
    """Group highly similar text so related examples cannot cross partitions."""
    texts = frame["message"].str.casefold().tolist()
    labels = frame["category"].tolist()
    parent = list(range(len(texts)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            if abs(len(texts[left]) - len(texts[right])) > max(len(texts[left]), len(texts[right])) * 0.08:
                continue
            if SequenceMatcher(None, texts[left], texts[right]).ratio() >= 0.92:
                if labels[left] != labels[right]:
                    raise DatasetError("Near-duplicate messages have conflicting categories")
                parent[root(right)] = root(left)
    return [root(index) for index in range(len(texts))]


def split_dataset(frame: pd.DataFrame, seed: int = 42) -> DataSplits:
    """Stratify groups into 60/20/20 partitions; require ten groups per class."""
    groups = _near_duplicate_groups(frame)
    group_frame = pd.DataFrame({"group": groups, "category": frame["category"]}).drop_duplicates("group")
    counts = group_frame["category"].value_counts()
    insufficient = [label for label in CATEGORIES if counts.get(label, 0) < 10]
    if insufficient:
        raise DatasetError(
            "At least 10 distinct message groups per category are required: " + ", ".join(insufficient)
        )
    try:
        trainval, test = train_test_split(
            group_frame, test_size=0.2, stratify=group_frame["category"], random_state=seed
        )
        train, validation = train_test_split(
            trainval, test_size=0.25, stratify=trainval["category"], random_state=seed
        )
    except ValueError as exc:
        raise DatasetError(f"Cannot create stratified train/validation/test split: {exc}") from exc

    def select(part: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[[group in set(part["group"]) for group in groups]].reset_index(drop=True)

    return DataSplits(select(train), select(validation), select(test))


def build_pipeline(settings: Settings) -> Pipeline:
    """Create a fresh TF-IDF and multinomial logistic regression pipeline."""
    return Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_messages, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), max_features=settings.max_features, lowercase=True, strip_accents=None
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=settings.regularization_c, max_iter=1000, random_state=settings.random_seed
                ),
            ),
        ]
    )


def train_pipeline(frame: pd.DataFrame, settings: Settings) -> tuple[Pipeline, DataSplits]:
    """Split before fitting all learned features and train a single candidate model."""
    splits = split_dataset(frame, settings.random_seed)
    model = build_pipeline(settings)
    model.fit(splits.train["message"].tolist(), splits.train["category"].tolist())
    return model, splits


def dataset_sha256(path: Path) -> str:
    """Return a reproducibility fingerprint of the source CSV."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
