import re


HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][^.!?]{2,100}$")


def chunk_text(text: str, size: int, overlap: int) -> list[tuple[str | None, str]]:
    """Split text into overlapping passages while retaining a likely section heading."""
    clean = re.sub(r"[ \t]+", " ", text.replace("\r", "")).strip()
    if not clean:
        return []
    chunks: list[tuple[str | None, str]] = []
    start = 0
    current_section: str | None = None
    while start < len(clean):
        target = min(start + size, len(clean))
        end = target
        if target < len(clean):
            boundary = max(clean.rfind("\n", start + size // 2, target), clean.rfind(". ", start + size // 2, target))
            if boundary > start:
                end = boundary + 1
        passage = clean[start:end].strip()
        first_line = passage.splitlines()[0].strip() if passage else ""
        if HEADING.match(first_line):
            current_section = first_line
        if passage:
            chunks.append((current_section, passage))
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks

