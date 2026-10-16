"""Normalization contract tests."""

import pytest

from supportiq.exceptions import InvalidMessageError
from supportiq.preprocessing.text import normalize_message


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" hello world ", "hello world"),
        ("one\n\t two   three", "one two three"),
        ("Ｃａｆｅ́ €12.50!", "Café €12.50!"),
        ("I cannot log in; do not close it.", "I cannot log in; do not close it."),
        ("Never refund $20?", "Never refund $20?"),
    ],
)
def test_normalize_preserves_meaning(raw: str, expected: str) -> None:
    assert normalize_message(raw) == expected


@pytest.mark.parametrize("value", ["", "  \n ", None, 12, ["message"]])
def test_rejects_blank_or_wrong_type(value) -> None:
    with pytest.raises(InvalidMessageError):
        normalize_message(value)
