"""
tests/test_ingestion.py
Tests for transcript parsing, chunking, and ingestion pipeline.
LLM-dependent tests marked @pytest.mark.llm — skip with: pytest -m "not llm"
"""

import os
import uuid
import pytest
import tempfile

from ingestion.parser import TranscriptParser
from ingestion.chunker import Chunker
from ingestion.pipeline import ingest_file, derive_call_id
from storage.db import init_db, SessionLocal, Call, Chunk


MINIMAL_TRANSCRIPT = """[00:00] AE (Alex):  Good morning Sarah, thanks for joining.

[00:08] Prospect (Sarah – VP Sales):  Hi Alex. Ready to go.

[00:12] AE (Alex):  Let me show you the dashboard first.

[00:20] Prospect (Sarah – VP Sales):  Looks interesting. What's the pricing?

[00:25] AE (Alex):  We're at 2000 per user per month, billed annually.

[00:32] Prospect (Sarah – VP Sales):  That's higher than what we pay now. Can you do better?

[00:38] AE (Alex):  We can offer 15% for an annual commit.

[00:45] Prospect (Sarah – VP Sales):  Send it across and we'll review.

[00:50] AE (Alex):  Perfect. Talk soon!
"""


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def tmp_transcript_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(MINIMAL_TRANSCRIPT)
        path = f.name
    yield path
    os.unlink(path)


# ── Parser ────────────────────────────────────────────────────────────────────

class TestTranscriptParser:

    def setup_method(self):
        self.parser = TranscriptParser()

    def test_correct_turn_count(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        assert len(result.turns) == 9

    def test_extracts_speakers_and_timestamps(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        assert result.turns[0].speaker == "AE (Alex)"
        assert result.turns[0].start_time == "00:00"
        assert result.turns[1].start_time == "00:08"

    def test_fills_end_times(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        assert result.turns[0].end_time == result.turns[1].start_time

    def test_unique_participants_and_duration(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        assert len(result.participants) == 2
        assert result.duration_sec == 50

    def test_empty_input(self):
        result = self.parser.parse("", call_id="empty", filename="empty.txt")
        assert result.turns == []
        assert result.duration_sec is None

    def test_multiline_turn_concatenated(self):
        text = "[00:00] AE (Alex):  First line.\nContinuation line.\n\n[00:10] Prospect (Sarah):  Reply."
        result = self.parser.parse(text, call_id="test", filename="test.txt")
        assert "Continuation line" in result.turns[0].text


# ── Chunker ───────────────────────────────────────────────────────────────────

class TestChunker:

    def setup_method(self):
        self.parser     = TranscriptParser()
        self.chunker    = Chunker()
        self.transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        self.mock_roles = {
            "AE (Alex)":                  "ae",
            "Prospect (Sarah - VP Sales)": "prospect",
        }

    def test_no_chunk_crosses_speaker_boundary(self):
        chunks = self.chunker._build_chunks(self.transcript.turns, "test", self.mock_roles)
        valid  = {"AE (Alex)", "Prospect (Sarah – VP Sales)"}
        for chunk in chunks:
            assert chunk.speaker in valid

    def test_chunk_ids_unique_and_call_id_correct(self):
        chunks = self.chunker._build_chunks(self.transcript.turns, "test_call", self.mock_roles)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))
        for chunk in chunks:
            assert chunk.call_id == "test_call"

    def test_citation_contains_required_fields(self):
        chunks = self.chunker._build_chunks(self.transcript.turns, "test_call", self.mock_roles)
        for chunk in chunks:
            assert chunk.call_id in chunk.citation
            assert chunk.start_time in chunk.citation
            assert chunk.speaker in chunk.citation

    def test_unknown_speaker_falls_back_to_other(self):
        chunks = self.chunker._build_chunks(self.transcript.turns, "test", {"AE (Alex)": "ae"})
        prospect_chunks = [c for c in chunks if c.speaker == "Prospect (Sarah – VP Sales)"]
        for chunk in prospect_chunks:
            assert chunk.speaker_role == "other"

    @pytest.mark.llm
    def test_llm_classifies_both_roles_correctly(self):
        """Single LLM call — validates ae and prospect are identified."""
        roles = self.chunker._classify_speakers(
            call_id="test",
            participants=self.transcript.participants,
            sample_turns=self.chunker._get_sample_turns(self.transcript.turns),
        )
        valid_roles = {"ae", "prospect", "se", "ciso", "cs", "pricing", "legal", "procurement", "other"}
        assert len(roles) == 2
        assert "ae"       in roles.values(), f"No ae found. Got: {roles}"
        assert "prospect" in roles.values(), f"No prospect found. Got: {roles}"
        for role in roles.values():
            assert role in valid_roles


# ── Pipeline ──────────────────────────────────────────────────────────────────

class TestIngestionPipeline:

    def test_derive_call_id(self):
        assert derive_call_id("data/transcripts/2_pricing_call.txt") == "2_pricing_call"
        assert derive_call_id("/deep/path/my_call.txt") == "my_call"

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ingest_file("/nonexistent/path/call.txt")

    @pytest.mark.llm
    def test_full_pipeline_success(self, tmp_transcript_file):
        call_id = f"test_{uuid.uuid4().hex[:8]}"
        result  = ingest_file(tmp_transcript_file, call_id=call_id)

        assert result["status"] == "success"
        assert result["chunks_created"] > 0

        session = SessionLocal()
        try:
            call   = session.query(Call).filter_by(call_id=call_id).first()
            chunks = session.query(Chunk).filter_by(call_id=call_id).all()
            assert call is not None
            assert len(chunks) == result["chunks_created"]
        finally:
            session.query(Chunk).filter_by(call_id=call_id).delete()
            session.query(Call).filter_by(call_id=call_id).delete()
            session.commit()
            session.close()

    @pytest.mark.llm
    def test_skip_existing_and_overwrite(self, tmp_transcript_file):
        call_id = f"test_{uuid.uuid4().hex[:8]}"
        ingest_file(tmp_transcript_file, call_id=call_id)

        assert ingest_file(tmp_transcript_file, call_id=call_id)["status"]                    == "skipped"
        assert ingest_file(tmp_transcript_file, call_id=call_id, overwrite=True)["status"]    == "success"

        session = SessionLocal()
        try:
            session.query(Chunk).filter_by(call_id=call_id).delete()
            session.query(Call).filter_by(call_id=call_id).delete()
            session.commit()
        finally:
            session.close()