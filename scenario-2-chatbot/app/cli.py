from pathlib import Path

import typer

from app.config import get_settings
from app.database import SessionLocal
from app.dependencies import get_provider
from app.services.ingestion import IngestionService

cli = typer.Typer(help="Ingest engineering PDFs and images into the knowledge base.")


@cli.callback()
def main() -> None:
    """Manage the engineering knowledge base."""


@cli.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True),
    describe_images: bool = typer.Option(True, "--describe-images/--skip-image-descriptions"),
) -> None:
    """Ingest one PDF or image, or every supported file in a directory."""
    settings = get_settings()
    files = [path] if path.is_file() else [
        item for item in path.rglob("*") if item.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
    ]
    if not files:
        raise typer.BadParameter("No supported files found")
    with SessionLocal() as session:
        service = IngestionService(session, get_provider(), settings)
        for item in files:
            result = service.ingest(item, describe_images)
            state = "unchanged" if result.unchanged else "indexed"
            typer.echo(f"{state}: {item} ({result.chunks_created} evidence records)")


if __name__ == "__main__":
    cli()

