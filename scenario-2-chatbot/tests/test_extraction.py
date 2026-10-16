from pathlib import Path

import pymupdf

from app.services.extraction import DocumentExtractor


def test_extracts_text_with_page_number(tmp_path: Path) -> None:
    pdf_path = tmp_path / "manual.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Safety Limits\nMaximum pressure is 70 psi.")
    document.save(pdf_path)
    document.close()

    extractor = DocumentExtractor(tmp_path / "assets", chunk_size=500, chunk_overlap=50)
    evidence, pages = extractor.extract(pdf_path, "a" * 64)
    assert pages == 1
    assert any(item.page_number == 1 and "70 psi" in item.content for item in evidence)

