from pydantic import BaseModel, Field
from typing import Optional
from storage.db import SessionLocal, Chunk
from storage.vector_store import VectorStore
from generation.llm_client import llm_client
from config import config


class SearchTranscriptsInput(BaseModel):
    query:         str                     = Field(description="Natural language search query")
    call_ids:      Optional[list[str]]     = Field(default=None, description="Scope search to specific calls. None = search all calls.")
    speaker_roles: Optional[list[str]]     = Field(default=None, description="Filter results by speaker role: prospect|ae|se|etc.")
    speakers:      Optional[list[str]]     = Field(default=None, description="Filter results by speaker name (partial match).")
    top_k:         int                     = Field(default=10,   description="Number of results to return.")


def search_transcripts(
    query:         str,
    call_ids:      Optional[list[str]] = None,
    speaker_roles: Optional[list[str]] = None,
    speakers:      Optional[list[str]] = None,
    top_k:         int = config.DEFAULT_TOP_K,
) -> dict:
    """
    Semantic search over all transcript chunks using FAISS.
    Use this when user asks about topics, themes, or content across calls.
    Returns ranked chunks with citations.
    """
    # Step 1: Embed the query
    query_embedding = llm_client.embed(query)

    # Step 2: FAISS search → (chunk_id, score) pairs
    vector_store = VectorStore()
    raw_results  = vector_store.search(query_embedding, top_k=top_k * 3)  # over-fetch for post-filtering

    if not raw_results:
        return {"total": 0, "query": query, "chunks": []}

    # Step 3: Fetch chunk metadata from SQLite + apply filters
    session = SessionLocal()
    try:
        chunk_id_to_score = {chunk_id: score for chunk_id, score in raw_results}
        chunk_ids_ordered = [r[0] for r in raw_results]

        query_db = session.query(Chunk).filter(Chunk.chunk_id.in_(chunk_ids_ordered))

        # Apply post-filters
        if call_ids:
            query_db = query_db.filter(Chunk.call_id.in_(call_ids))

        if speaker_roles:
            query_db = query_db.filter(Chunk.speaker_role.in_(speaker_roles))

        if speakers:
            from sqlalchemy import or_
            conditions = [Chunk.speaker.ilike(f"%{s}%") for s in speakers]
            query_db = query_db.filter(or_(*conditions))

        chunks = query_db.all()

        # Step 4: Re-rank by FAISS score + enforce top_k
        scored_chunks = sorted(
            chunks,
            key=lambda c: chunk_id_to_score.get(c.chunk_id, 0),
            reverse=True
        )[:top_k]

        results = []
        for chunk in scored_chunks:
            d = chunk.to_dict()
            d["similarity_score"] = round(chunk_id_to_score.get(chunk.chunk_id, 0), 4)
            results.append(d)

        return {
            "total":  len(results),
            "query":  query,
            "chunks": results,
        }

    finally:
        session.close()