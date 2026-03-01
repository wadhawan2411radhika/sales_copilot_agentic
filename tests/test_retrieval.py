"""
tests/test_retrieval.py

Tests for the 4 retrieval and analysis tools.
These tests assume transcripts have already been ingested via:
    python -c "from ingestion.pipeline import ingest_directory; ingest_directory()"

Semantic search and analyze_chunks tests are marked @pytest.mark.llm
as they require OPENAI_API_KEY.

Run all:              pytest tests/test_retrieval.py -v
Skip LLM tests:       pytest tests/test_retrieval.py -v -m "not llm"
Run only LLM tests:   pytest tests/test_retrieval.py -v -m llm
"""

import pytest
from storage.db import init_db, SessionLocal, Call, Chunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture(scope="module")
def requires_ingested_data():
    """
    Skip entire module if no calls have been ingested.
    Run ingestion first: python -c "from ingestion.pipeline import ingest_directory; ingest_directory()"
    """
    session = SessionLocal()
    count   = session.query(Call).count()
    session.close()
    if count == 0:
        pytest.skip("No ingested data found. Run ingestion first.")


# ── list_calls Tests ──────────────────────────────────────────────────────────

class TestListCalls:

    def test_list_calls_returns_dict_with_calls_key(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls()
        assert "calls" in result
        assert "total" in result

    def test_list_calls_total_matches_calls_length(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls()
        assert result["total"] == len(result["calls"])

    def test_list_calls_each_has_required_fields(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls()
        required = {"call_id", "filename", "ingested_at", "participants", "chunk_count"}
        for call in result["calls"]:
            assert required.issubset(call.keys()), f"Missing fields in: {call.keys()}"

    def test_list_calls_limit_respected(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls(limit=2)
        assert len(result["calls"]) <= 2

    def test_list_calls_sort_desc_most_recent_first(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls(sort_by="ingested_at", order="desc")
        dates  = [c["ingested_at"] for c in result["calls"]]
        assert dates == sorted(dates, reverse=True)

    def test_list_calls_sort_asc(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls(sort_by="ingested_at", order="asc")
        dates  = [c["ingested_at"] for c in result["calls"]]
        assert dates == sorted(dates)

    def test_list_calls_participants_is_list(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls()
        for call in result["calls"]:
            assert isinstance(call["participants"], list)
            assert len(call["participants"]) > 0

    def test_list_calls_chunk_count_positive(self, requires_ingested_data):
        from tools.list_calls import list_calls
        result = list_calls()
        for call in result["calls"]:
            assert call["chunk_count"] > 0


# ── get_chunks Tests ──────────────────────────────────────────────────────────

class TestGetChunks:

    def test_get_chunks_all_returns_results(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        result = get_chunks()
        assert result["total"] > 0
        assert len(result["chunks"]) == result["total"]

    def test_get_chunks_by_call_id_scopes_correctly(self, requires_ingested_data):
        from tools.list_calls import list_calls
        from tools.get_chunks import get_chunks

        call_id = list_calls(limit=1)["calls"][0]["call_id"]
        result  = get_chunks(call_ids=[call_id])
        for chunk in result["chunks"]:
            assert chunk["call_id"] == call_id

    def test_get_chunks_by_speaker_role_filters(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        result = get_chunks(speaker_roles=["prospect"])
        for chunk in result["chunks"]:
            assert chunk["speaker_role"] == "prospect"

    def test_get_chunks_by_multiple_roles(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        result = get_chunks(speaker_roles=["ae", "se"])
        for chunk in result["chunks"]:
            assert chunk["speaker_role"] in ("ae", "se")

    def test_get_chunks_by_speaker_name_partial_match(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        result = get_chunks(speakers=["Priya"])
        assert result["total"] > 0
        for chunk in result["chunks"]:
            assert "priya" in chunk["speaker"].lower()

    def test_get_chunks_limit_respected(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        result = get_chunks(limit=5)
        assert len(result["chunks"]) <= 5

    def test_get_chunks_each_has_citation(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        result = get_chunks(limit=10)
        for chunk in result["chunks"]:
            assert "citation" in chunk
            assert chunk["call_id"] in chunk["citation"]
            assert chunk["start_time"] in chunk["citation"]

    def test_get_chunks_empty_call_id_returns_empty(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        result = get_chunks(call_ids=["nonexistent_call_xyz"])
        assert result["total"] == 0
        assert result["chunks"] == []

    def test_get_chunks_ordered_by_time_within_call(self, requires_ingested_data):
        from tools.list_calls import list_calls
        from tools.get_chunks import get_chunks

        call_id = list_calls(limit=1)["calls"][0]["call_id"]
        result  = get_chunks(call_ids=[call_id])
        times   = [c["start_time"] for c in result["chunks"]]
        assert times == sorted(times)


# ── search_transcripts Tests ──────────────────────────────────────────────────

class TestSearchTranscripts:

    @pytest.mark.llm
    def test_search_returns_results_for_valid_query(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(query="pricing and discount")
        assert result["total"] > 0
        assert len(result["chunks"]) > 0

    @pytest.mark.llm
    def test_search_results_have_similarity_score(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(query="security compliance")
        for chunk in result["chunks"]:
            assert "similarity_score" in chunk
            assert 0 <= chunk["similarity_score"] <= 1

    @pytest.mark.llm
    def test_search_results_sorted_by_score_descending(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(query="competitor comparison", top_k=5)
        scores = [c["similarity_score"] for c in result["chunks"]]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.llm
    def test_search_scoped_to_call_id(self, requires_ingested_data):
        from tools.list_calls import list_calls
        from tools.search_transcripts import search_transcripts

        call_id = list_calls(limit=1)["calls"][0]["call_id"]
        result  = search_transcripts(query="pricing", call_ids=[call_id])
        for chunk in result["chunks"]:
            assert chunk["call_id"] == call_id

    @pytest.mark.llm
    def test_search_filtered_by_speaker_role(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(
            query="pricing concerns",
            speaker_roles=["prospect"],
            top_k=5
        )
        for chunk in result["chunks"]:
            assert chunk["speaker_role"] == "prospect"

    @pytest.mark.llm
    def test_search_top_k_respected(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(query="product features", top_k=3)
        assert len(result["chunks"]) <= 3

    @pytest.mark.llm
    def test_search_returns_empty_for_nonsense_query(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        # Nonsense query should return results but with low scores
        result = search_transcripts(
            query="xkqzplm flubberwock nonsense xyz123",
            top_k=3
        )
        # Should still return (FAISS always returns top_k) but scores will be low
        assert "chunks" in result
        assert "total" in result

    @pytest.mark.llm
    def test_search_specific_fact_retrieval(self, requires_ingested_data):
        """Test that specific factual content is retrievable."""
        from tools.search_transcripts import search_transcripts
        result = search_transcripts(query="20 percent pilot discount pricing")
        texts  = " ".join(c["text"] for c in result["chunks"]).lower()
        # The demo call mentions 20% discount — should surface
        assert "20" in texts or "discount" in texts


# ── analyze_chunks Tests ──────────────────────────────────────────────────────

class TestAnalyzeChunks:

    @pytest.mark.llm
    def test_analyze_summarize_returns_result(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        from tools.analyze_chunks import analyze_chunks

        chunks = get_chunks(call_ids=["2_pricing_call"], limit=10)["chunks"]
        result = analyze_chunks(
            chunks=chunks,
            task="summarize",
            criteria="pricing and discounts"
        )
        assert "result" in result
        assert len(result["result"]) > 50
        assert result["task"] == "summarize"

    @pytest.mark.llm
    def test_analyze_extract_returns_result(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        from tools.analyze_chunks import analyze_chunks

        chunks = search_transcripts(query="security concerns", top_k=5)["chunks"]
        result = analyze_chunks(
            chunks=chunks,
            task="extract",
            criteria="security objections and compliance questions"
        )
        assert result["result"]
        assert result["cited_chunks"]

    @pytest.mark.llm
    def test_analyze_sentiment_filter_returns_result(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        from tools.analyze_chunks import analyze_chunks

        chunks = search_transcripts(query="pricing", top_k=8)["chunks"]
        result = analyze_chunks(
            chunks=chunks,
            task="sentiment_filter",
            criteria="negative"
        )
        assert "result" in result
        assert result["task"] == "sentiment_filter"

    @pytest.mark.llm
    def test_analyze_compare_across_calls(self, requires_ingested_data):
        from tools.search_transcripts import search_transcripts
        from tools.analyze_chunks import analyze_chunks

        chunks = search_transcripts(
            query="pricing discussion",
            call_ids=["2_pricing_call", "4_negotiation_call"],
            top_k=10
        )["chunks"]
        result = analyze_chunks(
            chunks=chunks,
            task="compare",
            criteria="how pricing strategy and discounts differed"
        )
        assert "result" in result
        assert len(result["result"]) > 100

    @pytest.mark.llm
    def test_analyze_cites_sources(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        from tools.analyze_chunks import analyze_chunks

        chunks = get_chunks(call_ids=["1_demo_call"], limit=5)["chunks"]
        result = analyze_chunks(
            chunks=chunks,
            task="qa",
            criteria="what product features were demonstrated?"
        )
        # Response should contain citation format
        assert "[" in result["result"] and "|" in result["result"]

    def test_analyze_empty_chunks_returns_graceful_message(self):
        from tools.analyze_chunks import analyze_chunks
        result = analyze_chunks(
            chunks=[],
            task="summarize",
            criteria="anything"
        )
        assert "result" in result
        assert result["result"] != ""     # should have a helpful message, not crash

    @pytest.mark.llm
    def test_analyze_chunks_used_count_matches(self, requires_ingested_data):
        from tools.get_chunks import get_chunks
        from tools.analyze_chunks import analyze_chunks

        chunks = get_chunks(limit=7)["chunks"]
        result = analyze_chunks(
            chunks=chunks,
            task="extract",
            criteria="action items"
        )
        assert result["chunks_used"] == 7


# ── Result Store Tests ────────────────────────────────────────────────────────

class TestResultStore:

    def test_save_and_retrieve(self):
        from agent.result_store import ResultStore
        store = ResultStore()
        store.save("get_chunks", {"chunks": [{"chunk_id": "abc", "text": "hello"}]})
        result = store.get("get_chunks")
        assert result["chunks"][0]["chunk_id"] == "abc"

    def test_get_latest_chunks_from_get_chunks(self):
        from agent.result_store import ResultStore
        store  = ResultStore()
        chunks = [{"chunk_id": "1", "text": "test"}]
        store.save("get_chunks", {"chunks": chunks})
        assert store.get_latest_chunks() == chunks

    def test_get_latest_chunks_from_search(self):
        from agent.result_store import ResultStore
        store  = ResultStore()
        chunks = [{"chunk_id": "2", "text": "search result"}]
        store.save("search_transcripts", {"chunks": chunks})
        assert store.get_latest_chunks() == chunks

    def test_get_latest_chunks_returns_empty_when_no_data(self):
        from agent.result_store import ResultStore
        store = ResultStore()
        assert store.get_latest_chunks() == []

    def test_reset_clears_store(self):
        from agent.result_store import ResultStore
        store = ResultStore()
        store.save("get_chunks", {"chunks": [{"chunk_id": "x"}]})
        store.reset()
        assert store.get("get_chunks") is None
        assert store.get_latest_chunks() == []

    def test_get_nonexistent_key_returns_none(self):
        from agent.result_store import ResultStore
        store = ResultStore()
        assert store.get("nonexistent_tool") is None