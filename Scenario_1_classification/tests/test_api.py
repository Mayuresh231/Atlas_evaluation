"""HTTP contract and readiness tests with an injected real model."""

import asyncio

import httpx

from supportiq.main import create_app
from supportiq.models.classifier import Classifier
from supportiq.services.classification import ClassificationService


def request(app, method: str, path: str, **kwargs):
    """Issue a request while running the application lifespan."""

    async def run():
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def startup():
            async with app.router.lifespan_context(app):
                yield

        async with (
            startup(),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
        ):
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def test_ready_health_info_and_classification(fitted, settings) -> None:
    model, _ = fitted
    service = ClassificationService(Classifier(model), 0.6, "1.0.0", {"train_samples": 36})
    app = create_app(settings, service)
    health = request(app, "GET", "/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ready", "model_loaded": True, "model_version": "1.0.0"}
    info = request(app, "GET", "/v1/model/info")
    assert info.json()["confidence_threshold"] == 0.6
    assert "model_path" not in info.text
    single = request(app, "POST", "/v1/classify", json={"message": "invoice question"})
    assert single.status_code == 200
    assert single.json()["confidence"] == max(single.json()["category_probabilities"].values())
    batch = request(
        app,
        "POST",
        "/v1/classify/batch",
        json={"messages": [{"message": "invoice question"}, {"message": "login issue"}]},
    )
    assert batch.status_code == 200
    assert [item["index"] for item in batch.json()["results"]] == [0, 1]


def test_validation_and_batch_limit(fitted, settings) -> None:
    model, _ = fitted
    from dataclasses import replace

    app = create_app(
        replace(settings, max_batch_size=1), ClassificationService(Classifier(model), 0.6, "1.0.0")
    )
    for payload in [{"message": "  "}, {"message": 123}, {}, {"message": "x", "extra": 1}]:
        response = request(app, "POST", "/v1/classify", json=payload)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"
    invalid_batch = request(
        app, "POST", "/v1/classify/batch", json={"messages": [{"message": "valid"}, {"message": " "}]}
    )
    assert invalid_batch.status_code == 422
    oversized = request(
        app, "POST", "/v1/classify/batch", json={"messages": [{"message": "valid"}, {"message": "another"}]}
    )
    assert oversized.status_code == 422
    assert oversized.json()["error"]["code"] == "invalid_batch_size"


def test_missing_model_readiness(settings) -> None:
    app = create_app(settings)
    for path in ["/health", "/v1/model/info"]:
        response = request(app, "GET", path)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert str(settings.model_path) not in response.text
    response = request(app, "POST", "/v1/classify", json={"message": "help"})
    assert response.status_code == 503
