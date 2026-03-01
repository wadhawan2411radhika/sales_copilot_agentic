from pydantic import BaseModel, Field
from typing import Optional
from storage.db import SessionLocal, Chunk


class GetChunksInput(BaseModel):
    call_ids:      Optional[list[str]] = Field(default=None, description="Filter by specific call IDs. None = all calls.")
    speakers:      Optional[list[str]] = Field(default=None, description="Filter by speaker names (partial match).")
    speaker_roles: Optional[list[str]] = Field(default=None, description="Filter by role: ae|prospect|se|ciso|cs|pricing|legal|procurement|other")
    start_after:   Optional[str]       = Field(default=None, description="Return chunks starting after this timestamp MM:SS")
    limit:         Optional[int]       = Field(default=None, description="Max chunks to return. None = all matching.")


def get_chunks(
    call_ids:      Optional[list[str]] = None,
    speakers:      Optional[list[str]] = None,
    speaker_roles: Optional[list[str]] = None,
    start_after:   Optional[str]       = None,
    limit:         Optional[int]       = None,
) -> dict:
    """
    Structured fetch of transcript chunks from SQLite.
    No semantic search — use this for:
    - Fetching ALL chunks from a specific call (for summarization)
    - Filtering by speaker or role
    - Getting full context without relevance ranking
    """
    session = SessionLocal()
    try:
        query = session.query(Chunk)

        if call_ids:
            query = query.filter(Chunk.call_id.in_(call_ids))

        if speaker_roles:
            query = query.filter(Chunk.speaker_role.in_(speaker_roles))

        if speakers:
            # Partial match — "Priya" matches "Priya (RevOps Director)"
            from sqlalchemy import or_
            conditions = [Chunk.speaker.ilike(f"%{s}%") for s in speakers]
            query = query.filter(or_(*conditions))

        # Order by call + time for coherent reading
        query = query.order_by(Chunk.call_id, Chunk.start_time)

        if limit:
            query = query.limit(limit)

        chunks = query.all()

        return {
            "total":  len(chunks),
            "chunks": [c.to_dict() for c in chunks],
        }

    finally:
        session.close()