"""Conservative normalization that retains punctuation, numbers, and negation."""

import unicodedata
from collections.abc import Iterable

from supportiq.exceptions import InvalidMessageError


def normalize_message(message: str) -> str:
    """Normalize Unicode and whitespace; reject blank or non-string input."""
    if not isinstance(message, str):
        raise InvalidMessageError("message must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", message).split())
    if not normalized:
        raise InvalidMessageError("message must not be blank")
    return normalized


def normalize_messages(messages: Iterable[str]) -> list[str]:
    """Top-level, serializable pipeline transformer for text sequences."""
    return [normalize_message(message) for message in messages]
