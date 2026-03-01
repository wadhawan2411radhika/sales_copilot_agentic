"""
tests/test_retrieval.py
Tests for all retrieval and analysis tools.
Requires ingested data: python -c "from ingestion.pipeline import ingest_directory; ingest_directory()"
LLM tests marked @pytest.mark.llm — skip with: pytest -m "not llm"
"""

import pytest
from storage.db import init_db, SessionLocal, Call
from agent.result_store import ResultStore


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture(scope="module")
def requires_ingested_data():
    session = SessionLocal()
    count   = session.query(Call).count()
    session.close()
    if count == 0:
        pytest.skip("No ingested data. Run ingestion first.")


# ── list_calls ────────────────────────────────────────────────────────────────

class TestListCalls:

    def test_returns_calls_with_required_fields(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls()
        assert result["total"] == len(result["calls"])
        required = {"call_id", "filename", "ingested_at", "participants", "chunk_count"}
        for call in result["calls"]:
            assert required.issubset(call.keys())
            assert call["chunk_count"] > 0
            assert isinstance(call["participants"], list)

    def test_limit_and_sort(self, requires_ingested_data):
        from tools.list_calls import list_calls
        assert len(list_calls(limit=2)["calls"]) <= 2
        dates = [c["ingested_at"] for c in list_calls(order="desc")["calls"]]
        assert dates == sorted(dates, reverse=True)


# ── get_chunks ────────────────────────────────────────────────────────────────

class TestGetChunks:

    def test_scopes_by_call_id(self, requires_ingested_data):
        from tools.list_calls import list_calls
        from tools.get_chunks import get_chunks
        call_id = list_calls(limit=1)["calls"][0]["call_id"]
        result  = get_chunks(call_ids=[call_id])
        assert result["total"] > 0
        for chunk in result["chunks"]:
            assert chunk["call_id"] == call_id

    def test_filters_by_role_and_speaker(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        for chunk in get_chunks(speaker_roles=["prospect"])["chunks"]:
            assert chunk["speaker_role"] == "prospect"
        result = get_chunks(speakers=["Priya"])
        assert result["total"] > 0
        for chunk in result["chunks"]:
            assert "priya" in chunk["speaker"].lower()

    def test_citation_format_and_empty_result(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        for chunk in get_chunks(limit=5)["chunks"]:
            assert chunk["call_id"] in chunk["citation"]
            assert chunk["start_time"] in chunk["citation"]
        assert get_chunks(call_ids=["nonexistent_xyz"])["total"] == 0


# ── search_transcripts ────────────────────────────────────────────────────────

class TestSearchTranscripts:

    @pytest.mark.llm
    def test_returns_scored_results(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(query="pricing and discount", top_k=5)
        assert result["total"] > 0
        scores = [c["similarity_score"] for c in result["chunks"]]
        assert scores == sorted(scores, reverse=True)
        assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.llm
    def test_scoped_search_and_role_filter(self, requires_ingested_data):
        from tools.list_calls import list_calls
        from tools.search_transcripts import search_transcripts
        call_id = list_calls(limit=1)["calls"][0]["call_id"]
        for chunk in search_transcripts(query="pricing", call_ids=[call_id])["chunks"]:
            assert chunk["call_id"] == call_id
        for chunk in search_transcripts(query="pricing concerns", speaker_roles=["prospect"], top_k=5)["chunks"]:
            assert chunk["speaker_role"] == "prospect"

    @pytest.mark.llm
    def test_specific_fact_retrievable(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(query="20 percent pilot discount")
        texts  = " ".join(c["text"] for c in result["chunks"]).lower()
        assert "20" in texts or "discount" in texts


# ── analyze_chunks ────────────────────────────────────────────────────────────

class TestAnalyzeChunks:

    def test_empty_chunks_graceful(self):
        from tools.analyze_chunks import analyze_chunks
        result = analyze_chunks(chunks=[], task="summarize", criteria="anything")
        assert result["result"] != ""

    @pytest.mark.llm
    def test_summarize_with_citations(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        from tools.analyze_chunks import analyze_chunks
        chunks = get_chunks(call_ids=["2_pricing_call"], limit=15)["chunks"]
        result = analyze_chunks(chunks=chunks, task="summarize", criteria="pricing and discounts")
        assert len(result["result"]) > 50
        assert result["chunks_used"] == 15
        assert "[" in result["result"] and "|" in result["result"]

    @pytest.mark.llm
    def test_sentiment_filter(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        from tools.analyze_chunks import analyze_chunks
        chunks = search_transcripts(query="pricing", top_k=8)["chunks"]
        result = analyze_chunks(chunks=chunks, task="sentiment_filter", criteria="negative")
        assert result["task"] == "sentiment_filter"
        assert result["result"]

    @pytest.mark.llm
    def test_compare_across_calls(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        from tools.analyze_chunks import analyze_chunks
        chunks = search_transcripts(
            query="pricing discussion",
            call_ids=["2_pricing_call", "4_negotiation_call"],
            top_k=10
        )["chunks"]
        result = analyze_chunks(chunks=chunks, task="compare", criteria="pricing strategy")
        assert len(result["result"]) > 100


# ── ResultStore ───────────────────────────────────────────────────────────────

class TestResultStore:

    def test_save_get_reset(self):
        store  = ResultStore()
        chunks = [{"chunk_id": "1", "text": "test"}]

        # Test get_chunks caching
        store.save("get_chunks", {"chunks": chunks})
        assert store.get("get_chunks")["chunks"] == chunks
        assert store.get_latest_chunks() == chunks

        # get_chunks takes priority over search_transcripts (fixed order in implementation)
        store.save("search_transcripts", {"chunks": [{"chunk_id": "2"}]})
        assert store.get_latest_chunks() == chunks  # get_chunks still wins

        # Test reset clears everything
        store.reset()
        assert store.get("get_chunks") is None
        assert store.get_latest_chunks() == []

        # Now search_transcripts works when get_chunks is absent
        store.save("search_transcripts", {"chunks": [{"chunk_id": "2"}]})
        assert store.get_latest_chunks() == [{"chunk_id": "2"}]

    def test_missing_key_returns_none(self):
        from agent.result_store import ResultStore
        assert ResultStore().get("nonexistent") is None