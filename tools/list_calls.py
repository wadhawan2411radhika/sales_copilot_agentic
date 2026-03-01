from pydantic import BaseModel, Field
from typing import Optional
from storage.db import SessionLocal, Call


class ListCallsInput(BaseModel):
    sort_by: str = Field(default="ingested_at", description="Field to sort by: ingested_at | duration_sec | call_id")
    order:   str = Field(default="desc",        description="Sort order: asc | desc")
    limit:   int = Field(default=20,            description="Max number of calls to return")


def list_calls(
    sort_by: str = "ingested_at",
    order:   str = "desc",
    limit:   int = 20,
) -> dict:
    """
    Returns metadata for all ingested calls.
    Use this when user wants to see available calls, call IDs, 
    or needs to resolve 'last call' / 'recent call' references.
    """
    session = SessionLocal()
    try:
        query = session.query(Call)

        # Sorting
        sort_col = getattr(Call, sort_by, Call.ingested_at)
        if order == "desc":
            query = query.order_by(sort_col.desc(), Call.call_id.desc())
        else:
            query = query.order_by(sort_col.asc(), Call.call_id.asc())

        calls = query.limit(limit).all()

        return {
            "total": len(calls),
            "calls": [c.to_dict() for c in calls],
        }

    finally:
        session.close()