from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.database import get_session
from app.services.chat import ChatService
from app.services.openai_provider import OpenAIProvider
from app.services.reranker import Reranker
from app.services.retrieval import RetrievalService


@lru_cache
def get_provider() -> OpenAIProvider:
    return OpenAIProvider(get_settings())


@lru_cache
def get_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(settings.reranker_model, settings.reranker_enabled)


def get_chat_service(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> ChatService:
    provider = get_provider()
    retriever = RetrievalService(session, provider, get_reranker(), settings)
    return ChatService(retriever, provider, settings)

