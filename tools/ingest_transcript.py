from pydantic import BaseModel, Field
from typing import Optional
from ingestion.pipeline import ingest_file


class IngestTranscriptInput(BaseModel):
    file_path: str            = Field(description="Absolute or relative path to the .txt transcript file.")
    call_id:   Optional[str] = Field(default=None, description="Optional custom call ID. Auto-derived from filename if not provided.")
    overwrite: bool           = Field(default=False, description="Re-ingest if call already exists.")


def ingest_transcript(
    file_path: str,
    call_id:   Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Ingest a new transcript file end-to-end.
    Parses → chunks → classifies speaker roles (LLM) → embeds → stores.
    Use when user wants to add a new call transcript.
    """
    return ingest_file(
        filepath=file_path,
        call_id=call_id,
        overwrite=overwrite,
    )