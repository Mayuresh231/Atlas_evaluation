"""Validated configuration and the centrally defined label taxonomy."""

import os
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = (
    "billing",
    "technical_issue",
    "account_access",
    "order_or_service_request",
    "general_inquiry",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path(value: str, root: Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class Settings:
    """Runtime and offline settings with safe local defaults."""

    model_path: Path = PROJECT_ROOT / "artifacts/model.joblib"
    dataset_path: Path = PROJECT_ROOT / "data/sample_support_messages.csv"
    report_path: Path = PROJECT_ROOT / "reports/metrics.json"
    confidence_threshold: float = 0.60
    max_batch_size: int = 50
    max_features: int = 20000
    regularization_c: float = 1.0
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 1 <= self.max_batch_size <= 1000:
            raise ValueError("max_batch_size must be between 1 and 1000")
        if self.max_features < 100:
            raise ValueError("max_features must be at least 100")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")

    @classmethod
    def from_env(cls) -> "Settings":
        """Read environment overrides without requiring a dotenv file."""
        return cls(
            model_path=_path(os.getenv("SUPPORTIQ_MODEL_PATH", "artifacts/model.joblib"), PROJECT_ROOT),
            dataset_path=_path(
                os.getenv("SUPPORTIQ_DATASET_PATH", "data/sample_support_messages.csv"),
                PROJECT_ROOT,
            ),
            report_path=_path(os.getenv("SUPPORTIQ_REPORT_PATH", "reports/metrics.json"), PROJECT_ROOT),
            confidence_threshold=float(os.getenv("SUPPORTIQ_CONFIDENCE_THRESHOLD", "0.60")),
            max_batch_size=int(os.getenv("SUPPORTIQ_MAX_BATCH_SIZE", "50")),
            max_features=int(os.getenv("SUPPORTIQ_MAX_FEATURES", "20000")),
            regularization_c=float(os.getenv("SUPPORTIQ_REGULARIZATION_C", "1.0")),
            random_seed=int(os.getenv("SUPPORTIQ_RANDOM_SEED", "42")),
        )
