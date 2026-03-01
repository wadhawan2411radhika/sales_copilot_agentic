import os
from pathlib import Path

from ingestion.parser import TranscriptParser
from ingestion.chunker import Chunker
from storage.db import SessionLocal, Call, Chunk as ChunkModel, init_db
from storage.vector_store import VectorStore
from generation.llm_client import llm_client


def derive_call_id(filepath: str) -> str:
    """'data/transcripts/2_pricing_call.txt' → '2_pricing_call'"""
    return Path(filepath).stem


def ingest_file(
    filepath: str,
    call_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    """
    Full ingestion pipeline for one transcript file.
    Returns summary dict.
    """
    init_db()

    filepath = os.path.abspath(filepath)
    call_id  = call_id or derive_call_id(filepath)

    # ── Read raw text ──────────────────────────────────────────────────────
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    session = SessionLocal()

    try:
        # ── Check for existing call ────────────────────────────────────────
        existing = session.query(Call).filter_by(call_id=call_id).first()
        if existing and not overwrite:
            return {
                "call_id": call_id,
                "status":  "skipped",
                "reason":  "already ingested. Use overwrite=True to re-ingest.",
            }
        if existing and overwrite:
            session.delete(existing)
            session.commit()

        # ── Step 1: Parse ─────────────────────────────────────────────────
        parser     = TranscriptParser()
        transcript = parser.parse(raw_text, call_id=call_id, filename=os.path.basename(filepath))

        # ── Step 2: Chunk + LLM speaker role classification ───────────────
        chunker = Chunker()
        chunked = chunker.chunk(transcript)

        # ── Step 3: Batch embed all chunks (single API call) ──────────────
        vector_store     = VectorStore()
        embedding_texts  = [c.to_embedding_text() for c in chunked.chunks]
        embeddings       = llm_client.embed_batch(embedding_texts)

        # ── Step 4: Persist Call to SQLite ────────────────────────────────
        call_record = Call(
            call_id      = call_id,
            filename     = os.path.basename(filepath),
            duration_sec = transcript.duration_sec,
            participants = transcript.participants,
            raw_text     = raw_text,
        )
        session.add(call_record)
        session.flush()   # get call_id into DB before adding chunks

        # ── Step 5: Persist Chunks + Embeddings ───────────────────────────
        for chunk, embedding in zip(chunked.chunks, embeddings):
            # Store in FAISS → get position back
            position = vector_store.add(chunk.chunk_id, embedding)

            chunk_record = ChunkModel(
                chunk_id     = chunk.chunk_id,
                call_id      = call_id,
                speaker      = chunk.speaker,
                speaker_role = chunk.speaker_role,
                start_time   = chunk.start_time,
                end_time     = chunk.end_time,
                text         = chunk.text,
                embedding_id = position,
            )
            session.add(chunk_record)

        session.commit()

        return {
            "call_id":       call_id,
            "status":        "success",
            "chunks_created": len(chunked.chunks),
            "participants":  transcript.participants,
            "speaker_roles": chunked.speaker_roles,
            "duration_sec":  transcript.duration_sec,
        }

    except Exception as e:
        session.rollback()
        raise e

    finally:
        session.close()


def ingest_directory(directory: str = "data/transcripts") -> list[dict]:
    """Ingest all .txt files in a directory."""
    results = []
    for filepath in sorted(Path(directory).glob("*.txt")):
        print(f"Ingesting: {filepath.name} ...")
        result = ingest_file(str(filepath))
        print(f"  → {result['status']} | {result.get('chunks_created', 0)} chunks")
        results.append(result)
    return results