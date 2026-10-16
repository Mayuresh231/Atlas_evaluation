from app.services.chunking import chunk_text


def test_chunk_text_preserves_content_and_overlap() -> None:
    text = "OPERATING LIMITS\n" + ("Pressure must remain stable. " * 80)
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 2
    assert chunks[0][0] == "OPERATING LIMITS"
    assert all(len(content) <= 310 for _, content in chunks)


def test_chunk_text_ignores_blank_input() -> None:
    assert chunk_text(" \n\t", size=500, overlap=50) == []

