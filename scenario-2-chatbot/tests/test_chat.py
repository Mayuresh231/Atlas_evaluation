import uuid
from pathlib import Path

from app.config import Settings
from app.domain import RetrievedEvidence
from app.schemas import AnswerStatus
from app.services.chat import ChatService
from app.services.openai_provider import GeneratedAnswer, GeneratedCitation


class FakeRetriever:
    def __init__(self, score: float):
        self.score = score

    def retrieve(self, question: str) -> list[RetrievedEvidence]:
        return [RetrievedEvidence(
            chunk_id=uuid.uuid4(), document_id=uuid.uuid4(), document_title="Manual",
            kind="text", content="Maximum pressure is 70 psi.", page_number=2,
            section=None, figure_label=None, asset_path=None, score=self.score, source_id="S1",
        )]


class FakeProvider:
    called = False

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1]]

    def describe_image(self, image_path: Path, context: str = "") -> str:
        return ""

    def answer(self, prompt: str, image_paths: list[Path]) -> GeneratedAnswer:
        self.called = True
        return GeneratedAnswer(
            answer="Maximum pressure is 70 psi [S1].",
            citations=[GeneratedCitation(source_id="S1", quote="Maximum pressure is 70 psi")],
        )

    def ping(self) -> None:
        return None


def settings() -> Settings:
    return Settings(_env_file=None, min_evidence_score=0.2)


def test_abstains_before_generation_when_evidence_is_weak() -> None:
    provider = FakeProvider()
    response = ChatService(FakeRetriever(0.1), provider, settings()).answer("What is the pressure?")
    assert response.status == AnswerStatus.insufficient_evidence
    assert provider.called is False


def test_returns_grounded_answer_and_citation() -> None:
    provider = FakeProvider()
    response = ChatService(FakeRetriever(0.9), provider, settings()).answer("What is the pressure?")
    assert response.status == AnswerStatus.answered
    assert response.citations[0].source_id == "S1"

