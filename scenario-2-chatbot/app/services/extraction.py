import io
import re
from pathlib import Path

import pymupdf

from app.domain import ExtractedEvidence
from app.services.chunking import chunk_text


FIGURE_RE = re.compile(r"\b(?:figure|fig\.)\s*([A-Za-z]?\d+[A-Za-z]?)", re.IGNORECASE)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


class DocumentExtractor:
    def __init__(self, asset_root: Path, chunk_size: int, chunk_overlap: int):
        self.asset_root = asset_root
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract(self, path: Path, content_hash: str) -> tuple[list[ExtractedEvidence], int | None]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path, content_hash)
        if suffix in IMAGE_SUFFIXES:
            asset = self._copy_image(path, content_hash)
            ocr = self._ocr(asset)
            content = f"Standalone scientific or engineering image: {path.name}. OCR text: {ocr}".strip()
            return [ExtractedEvidence(kind="image", content=content, asset_path=asset)], 1
        raise ValueError(f"Unsupported file type: {suffix}. Supported: PDF, PNG, JPG, JPEG, WEBP")

    def _extract_pdf(self, path: Path, content_hash: str) -> tuple[list[ExtractedEvidence], int]:
        output: list[ExtractedEvidence] = []
        asset_dir = self.asset_root / content_hash[:16]
        asset_dir.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(path) as document:
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                page_dict = page.get_text("dict", sort=True)
                text_blocks = [b for b in page_dict["blocks"] if b.get("type") == 0]
                page_text = "\n".join(
                    " ".join(span["text"] for line in block.get("lines", []) for span in line.get("spans", []))
                    for block in text_blocks
                ).strip()
                if not page_text:
                    page_text = self._ocr_page(page)
                for section, passage in chunk_text(page_text, self.chunk_size, self.chunk_overlap):
                    output.append(ExtractedEvidence(kind="text", content=passage, page_number=page_number, section=section))

                image_blocks = [b for b in page_dict["blocks"] if b.get("type") == 1]
                kept_image = False
                for image_index, block in enumerate(image_blocks, start=1):
                    if block.get("width", 0) * block.get("height", 0) < 20_000 or len(block.get("image", b"")) < 5_000:
                        continue
                    extension = block.get("ext", "png")
                    asset = asset_dir / f"page-{page_number}-image-{image_index}.{extension}"
                    asset.write_bytes(block["image"])
                    caption = self._nearest_caption(block["bbox"], text_blocks)
                    label = self._figure_label(caption)
                    content = f"{caption}\nOCR text: {self._ocr(asset)}".strip()
                    output.append(ExtractedEvidence(
                        kind="figure", content=content or f"Figure on page {page_number}", page_number=page_number,
                        figure_label=label, bbox=self._bbox(block["bbox"]), asset_path=asset,
                    ))
                    kept_image = True

                # Vector plots often are not PDF image objects. Render such pages for visual answering.
                if FIGURE_RE.search(page_text) and not kept_image:
                    asset = asset_dir / f"page-{page_number}-render.png"
                    page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(asset)
                    caption = next((line for line in page_text.splitlines() if FIGURE_RE.search(line)), "")
                    output.append(ExtractedEvidence(
                        kind="page_image", content=caption or f"Rendered figure page {page_number}",
                        page_number=page_number, figure_label=self._figure_label(caption), asset_path=asset,
                    ))
            return output, document.page_count

    def _copy_image(self, path: Path, content_hash: str) -> Path:
        asset_dir = self.asset_root / content_hash[:16]
        asset_dir.mkdir(parents=True, exist_ok=True)
        destination = asset_dir / f"source{path.suffix.lower()}"
        destination.write_bytes(path.read_bytes())
        return destination

    @staticmethod
    def _nearest_caption(image_bbox: tuple[float, ...], text_blocks: list[dict]) -> str:
        below = []
        for block in text_blocks:
            bbox = block["bbox"]
            if bbox[1] >= image_bbox[3] - 5 and bbox[1] - image_bbox[3] < 180:
                text = " ".join(span["text"] for line in block.get("lines", []) for span in line.get("spans", []))
                below.append((bbox[1] - image_bbox[3], text))
        below.sort(key=lambda item: item[0])
        return below[0][1] if below else ""

    @staticmethod
    def _figure_label(text: str) -> str | None:
        match = FIGURE_RE.search(text)
        return f"Figure {match.group(1)}" if match else None

    @staticmethod
    def _bbox(values: tuple[float, ...]) -> dict[str, float]:
        return dict(zip(("x0", "y0", "x1", "y1"), map(float, values), strict=True))

    @staticmethod
    def _ocr(path: Path) -> str:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(path)).strip()
        except (ImportError, OSError, RuntimeError):
            return ""

    @staticmethod
    def _ocr_page(page: pymupdf.Page) -> str:
        try:
            text_page = page.get_textpage_ocr(language="eng", dpi=200, full=True)
            return page.get_text("text", textpage=text_page, sort=True).strip()
        except RuntimeError:
            return ""

