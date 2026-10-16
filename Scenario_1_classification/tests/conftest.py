"""Small, label-balanced fixtures for fast tests."""

from pathlib import Path

import pandas as pd
import pytest

from supportiq.config import CATEGORIES, Settings
from supportiq.models.training import train_pipeline


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        model_path=tmp_path / "model.joblib",
        dataset_path=tmp_path / "data.csv",
        report_path=tmp_path / "metrics.json",
    )


@pytest.fixture
def training_frame() -> pd.DataFrame:
    subjects = {
        "billing": [
            "invoice fee",
            "payment card",
            "refund amount",
            "charged extra",
            "billing statement",
            "credit balance",
            "receipt tax",
            "renewal price",
            "discount missing",
            "overdue charge",
            "bank transaction",
            "monthly bill",
        ],
        "technical_issue": [
            "app crashes",
            "search fails",
            "export broken",
            "server error",
            "screen freezes",
            "webhook disconnects",
            "upload stalls",
            "form fails",
            "audio stops",
            "download breaks",
            "calendar bug",
            "browser hangs",
        ],
        "account_access": [
            "login rejected",
            "password forgotten",
            "account locked",
            "code missing",
            "sign in fails",
            "authenticator lost",
            "username invalid",
            "access denied",
            "verification fails",
            "session expires",
            "identity rejected",
            "profile locked",
        ],
        "order_or_service_request": [
            "cancel subscription",
            "order device",
            "upgrade plan",
            "add seats",
            "schedule visit",
            "replace item",
            "return purchase",
            "renew contract",
            "ship package",
            "pause service",
            "extend plan",
            "provision workspace",
        ],
        "general_inquiry": [
            "support hours",
            "documentation location",
            "service regions",
            "setup duration",
            "plan differences",
            "roadmap details",
            "language options",
            "office location",
            "accessibility features",
            "sales contact",
            "trial terms",
            "release schedule",
        ],
    }
    return pd.DataFrame(
        [
            {
                "message_id": f"{label[:2]}{index}",
                "message": f"Could you explain {subject} please?",
                "category": label,
            }
            for label in CATEGORIES
            for index, subject in enumerate(subjects[label])
        ]
    )


@pytest.fixture
def fitted(training_frame: pd.DataFrame, settings: Settings):
    model, splits = train_pipeline(training_frame, settings)
    return model, splits
