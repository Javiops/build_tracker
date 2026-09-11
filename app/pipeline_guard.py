"""Operator hold for downstream work; ingestion never consults this file."""
from pathlib import Path
from app.config import DATA_DIR

HOLD_PATH = DATA_DIR / "ml" / "pipeline_hold.json"


def require_pipeline_clear(stage: str, hold_path: Path | None = None) -> None:
    path = HOLD_PATH if hold_path is None else hold_path
    if path.exists():
        raise SystemExit(
            f"PIPELINE HOLD: {stage} refused before work starts. See {path}. "
            "Ingestion may finish; repair and verify the corpus before clearing the hold."
        )
