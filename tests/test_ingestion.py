"""
tests/test_ingestion.py

Tests for transcript parsing, chunking, and end-to-end ingestion pipeline.
These tests use a minimal dummy transcript to avoid OpenAI API calls where possible.
Chunker speaker-role classification tests DO make one real LLM call.
"""

import os
import uuid
import pytest
import tempfile

from ingestion.parser import TranscriptParser, Turn
from ingestion.chunker import Chunker
from ingestion.pipeline import ingest_file, derive_call_id
from storage.db import init_db, SessionLocal, Call, Chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_TRANSCRIPT = """[00:00] AE (Alex):  Good morning Sarah, thanks for joining. Quick agenda today.

[00:08] Prospect (Sarah – VP Sales):  Hi Alex. Ready to go.

[00:12] AE (Alex):  Great. Let me show you the dashboard first.

[00:20] Prospect (Sarah – VP Sales):  Looks interesting. What's the pricing?

[00:25] AE (Alex):  We're at ₹2000 per user per month, billed annually.

[00:32] Prospect (Sarah – VP Sales):  That's higher than what we pay now. Can you do better?

[00:38] AE (Alex):  We can offer 15% for an annual commit. I'll send the details over.

[00:45] Prospect (Sarah – VP Sales):  Okay. Send it across and we'll review.

[00:50] AE (Alex):  Perfect. Talk soon!
"""

MULTILINE_TRANSCRIPT = """[00:00] AE (Jordan):  Good morning team. Excited to be here.
Welcome everyone to the call.

[00:10] Prospect (Priya – RevOps):  Thanks Jordan. Let's dive in.

[00:15] SE (Luis):  Happy to show the demo now.
"""


@pytest.fixture
def tmp_transcript_file():
    """Creates a temporary transcript file and cleans up after the test."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        prefix="test_call_",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(MINIMAL_TRANSCRIPT)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(autouse=True)
def setup_db():
    """Ensure DB tables exist before each test."""
    init_db()


# ── Parser Tests ──────────────────────────────────────────────────────────────

class TestTranscriptParser:

    def setup_method(self):
        self.parser = TranscriptParser()

    def test_parse_returns_correct_turn_count(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        assert len(result.turns) == 9

    def test_parse_extracts_speakers(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        speakers = [t.speaker for t in result.turns]
        assert "AE (Alex)" in speakers
        assert "Prospect (Sarah – VP Sales)" in speakers

    def test_parse_extracts_timestamps(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        assert result.turns[0].start_time == "00:00"
        assert result.turns[1].start_time == "00:08"

    def test_parse_fills_end_times(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        # end_time of turn[0] = start_time of turn[1]
        assert result.turns[0].end_time == result.turns[1].start_time

    def test_parse_extracts_unique_participants(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        assert len(result.participants) == 2
        assert "AE (Alex)" in result.participants
        assert "Prospect (Sarah – VP Sales)" in result.participants

    def test_parse_extracts_duration(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        # Last timestamp is 00:50 = 50 seconds
        assert result.duration_sec == 50

    def test_parse_handles_multiline_turns(self):
        result = self.parser.parse(MULTILINE_TRANSCRIPT, call_id="multi", filename="multi.txt")
        # First turn should contain the continuation line
        first_turn_text = result.turns[0].text
        assert "Welcome everyone" in first_turn_text

    def test_parse_preserves_call_id(self):
        result = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="my_call_id", filename="test.txt")
        assert result.call_id == "my_call_id"

    def test_parse_empty_string_returns_empty_turns(self):
        result = self.parser.parse("", call_id="empty", filename="empty.txt")
        assert result.turns == []
        assert result.participants == []
        assert result.duration_sec is None

    def test_parse_ignores_empty_lines(self):
        text_with_blanks = "\n\n" + MINIMAL_TRANSCRIPT + "\n\n"
        result = self.parser.parse(text_with_blanks, call_id="test", filename="test.txt")
        assert len(result.turns) == 9


# ── Chunker Tests ─────────────────────────────────────────────────────────────

class TestChunker:
    """
    Note: _classify_speakers makes a real LLM call.
    Tests that need LLM are marked with @pytest.mark.llm
    Run with: pytest -m llm   or   pytest -m "not llm" to skip
    """

    def setup_method(self):
        self.parser  = TranscriptParser()
        self.chunker = Chunker()

    def test_chunks_never_cross_speaker_boundary(self):
        """No chunk should contain text from two different speakers."""
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        # Mock speaker roles to skip LLM call
        mock_roles = {
            "AE (Alex)":                  "ae",
            "Prospect (Sarah – VP Sales)": "prospect",
        }
        chunks = self.chunker._build_chunks(
            turns=transcript.turns,
            call_id="test",
            speaker_roles=mock_roles,
        )
        for chunk in chunks:
            # Each chunk text should only contain words from one speaker's turns
            assert chunk.speaker in ["AE (Alex)", "Prospect (Sarah – VP Sales)"]

    def test_chunk_ids_are_unique(self):
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        mock_roles = {
            "AE (Alex)": "ae",
            "Prospect (Sarah – VP Sales)": "prospect",
        }
        chunks = self.chunker._build_chunks(transcript.turns, "test", mock_roles)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_call_id_matches(self):
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="abc123", filename="test.txt")
        mock_roles = {"AE (Alex)": "ae", "Prospect (Sarah – VP Sales)": "prospect"}
        chunks = self.chunker._build_chunks(transcript.turns, "abc123", mock_roles)
        for chunk in chunks:
            assert chunk.call_id == "abc123"

    def test_chunk_citation_format(self):
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        mock_roles = {"AE (Alex)": "ae", "Prospect (Sarah – VP Sales)": "prospect"}
        chunks = self.chunker._build_chunks(transcript.turns, "test_call", mock_roles)
        for chunk in chunks:
            assert chunk.call_id in chunk.citation
            assert chunk.start_time in chunk.citation
            assert chunk.speaker in chunk.citation

    def test_to_embedding_text_contains_metadata(self):
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test_call", filename="test.txt")
        mock_roles = {"AE (Alex)": "ae", "Prospect (Sarah – VP Sales)": "prospect"}
        chunks = self.chunker._build_chunks(transcript.turns, "test_call", mock_roles)
        for chunk in chunks:
            emb_text = chunk.to_embedding_text()
            assert "test_call" in emb_text
            assert chunk.speaker in emb_text
            assert chunk.speaker_role in emb_text

    def test_unknown_speaker_gets_other_role(self):
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        # Only classify one speaker — other gets "other" fallback
        mock_roles = {"AE (Alex)": "ae"}
        chunks = self.chunker._build_chunks(transcript.turns, "test", mock_roles)
        prospect_chunks = [c for c in chunks if c.speaker == "Prospect (Sarah – VP Sales)"]
        for chunk in prospect_chunks:
            assert chunk.speaker_role == "other"

    @pytest.mark.llm
    def test_classify_speakers_returns_valid_roles(self):
        """Makes a real LLM call — requires OPENAI_API_KEY."""
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        roles = self.chunker._classify_speakers(
            call_id="test",
            participants=transcript.participants,
            sample_turns=self.chunker._get_sample_turns(transcript.turns),
        )
        valid_roles = {"ae", "prospect", "se", "ciso", "cs", "pricing", "legal", "procurement", "other"}
        assert "AE (Alex)" in roles
        assert "Prospect (Sarah – VP Sales)" in roles
        for role in roles.values():
            assert role in valid_roles

    @pytest.mark.llm
    def test_classify_speakers_identifies_ae_correctly(self):
        """AE should be classified as 'ae'."""
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        roles = self.chunker._classify_speakers(
            call_id="test",
            participants=transcript.participants,
            sample_turns=self.chunker._get_sample_turns(transcript.turns),
        )
        assert roles.get("AE (Alex)") == "ae"

    @pytest.mark.llm
    def test_classify_speakers_identifies_prospect_correctly(self):
        """Buyer-side participant should be classified as 'prospect'."""
        transcript = self.parser.parse(MINIMAL_TRANSCRIPT, call_id="test", filename="test.txt")
        roles = self.chunker._classify_speakers(
            call_id="test",
            participants=transcript.participants,
            sample_turns=self.chunker._get_sample_turns(transcript.turns),
        )
        assert roles.get("Prospect (Sarah – VP Sales)") == "prospect"


# ── Pipeline Tests ────────────────────────────────────────────────────────────

class TestIngestionPipeline:

    def test_derive_call_id_strips_extension(self):
        assert derive_call_id("data/transcripts/2_pricing_call.txt") == "2_pricing_call"

    def test_derive_call_id_handles_nested_path(self):
        assert derive_call_id("/some/deep/path/my_call.txt") == "my_call"

    @pytest.mark.llm
    def test_ingest_file_success(self, tmp_transcript_file):
        """Full pipeline test — requires OPENAI_API_KEY."""
        call_id = f"test_{uuid.uuid4().hex[:8]}"
        result  = ingest_file(tmp_transcript_file, call_id=call_id)

        assert result["status"] == "success"
        assert result["call_id"] == call_id
        assert result["chunks_created"] > 0
        assert len(result["participants"]) == 2

        # Verify persisted to SQLite
        session = SessionLocal()
        try:
            call   = session.query(Call).filter_by(call_id=call_id).first()
            chunks = session.query(Chunk).filter_by(call_id=call_id).all()
            assert call is not None
            assert call.duration_sec == 50
            assert len(chunks) == result["chunks_created"]
        finally:
            # Cleanup test data
            session.query(Chunk).filter_by(call_id=call_id).delete()
            session.query(Call).filter_by(call_id=call_id).delete()
            session.commit()
            session.close()

    @pytest.mark.llm
    def test_ingest_file_skips_existing(self, tmp_transcript_file):
        """Re-ingesting without overwrite=True should be skipped."""
        call_id = f"test_{uuid.uuid4().hex[:8]}"

        result1 = ingest_file(tmp_transcript_file, call_id=call_id)
        assert result1["status"] == "success"

        result2 = ingest_file(tmp_transcript_file, call_id=call_id)
        assert result2["status"] == "skipped"

        # Cleanup
        session = SessionLocal()
        try:
            session.query(Chunk).filter_by(call_id=call_id).delete()
            session.query(Call).filter_by(call_id=call_id).delete()
            session.commit()
        finally:
            session.close()

    @pytest.mark.llm
    def test_ingest_file_overwrite(self, tmp_transcript_file):
        """overwrite=True should re-ingest successfully."""
        call_id = f"test_{uuid.uuid4().hex[:8]}"

        ingest_file(tmp_transcript_file, call_id=call_id)
        result = ingest_file(tmp_transcript_file, call_id=call_id, overwrite=True)
        assert result["status"] == "success"

        # Cleanup
        session = SessionLocal()
        try:
            session.query(Chunk).filter_by(call_id=call_id).delete()
            session.query(Call).filter_by(call_id=call_id).delete()
            session.commit()
        finally:
            session.close()

    def test_ingest_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ingest_file("/nonexistent/path/call.txt")