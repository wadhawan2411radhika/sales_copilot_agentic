# Sales Call Copilot 🎯

A CLI-based agentic chatbot that helps sales teams query, summarise, and extract insights from past sales call transcripts — with **source citations on every answer**.

---

## 1. Problem Statement

Sales teams record hundreds of calls but lack structured tools to extract insight from them. This system ingests call transcripts, stores them with embeddings, and lets users ask natural language questions via a CLI chatbot.

Every response must indicate **which conversation segment(s) contributed to the answer** — not just a summary, but a traceable, cited answer grounded in transcript evidence.

Example queries the system handles:

```
list my call ids
summarise the last call
give me all negative comments when pricing was mentioned
ingest a new call transcript from <path>
what objections did prospects raise across all calls?
compare how pricing was handled in call 2 vs call 4
```

---

## 2. Observations

### This is not a direct RAG problem

A naive RAG pipeline (embed → retrieve top-k → generate) would fail several of these queries:

| Query | Why naive RAG fails |
|---|---|
| `list my call ids` | No retrieval needed — pure metadata lookup |
| `summarise the last call` | Needs ALL chunks from one call, not top-k semantic results |
| `give me negative comments on pricing` | Requires semantic search + sentiment classification as a second step |
| `compare pricing across call 2 and call 4` | Requires scoped retrieval + comparative reasoning — two distinct operations |
| `ingest a new transcript` | A command, not a question — triggers a pipeline |

### This is not a Text2SQL problem either

Queries like *"what objections did prospects raise?"* have no structured SQL representation — they require semantic understanding of free-form dialogue.

### The real problem is intent and tool diversity

Different queries require entirely different execution paths. The system needs to:
- Understand query **intent** (lookup vs. retrieval vs. analysis vs. ingestion)
- Route to the right **data access pattern** (SQL filter vs. vector search vs. full fetch)
- **Chain operations** when needed (retrieve → then reason)
- Return answers **with citations grounded in the source transcript**

This calls for an **agentic design** where an LLM orchestrates purpose-built tools.

---

## 3. Solution Design

### 3.1 Assumptions

1. **Transcript format** — Files follow `[MM:SS] Speaker Name (Role): dialogue text`. The parser handles multi-line turns gracefully.

2. **Speaker role classification** — One LLM call per transcript classifies all participants into roles (`ae`, `prospect`, `se`, `ciso`, `cs`, `pricing`, `legal`, `procurement`, `other`). Done at ingestion time, not query time.

3. **Embedding dimensions** — FAISS index initialised at 1536 dimensions (`text-embedding-3-small`). Switching embedding models requires re-ingestion.

4. **FAISS deletion** — `IndexFlatIP` does not support in-place deletion. Re-ingesting with `overwrite=True` removes the SQLite record but leaves the old vector as a stale entry. Acceptable for MVP.

5. **Single-user CLI** — Session history and result cache are in-memory per run. Multi-user support would require session IDs and persistent storage.

6. **Audio not in scope** — System ingests `.txt` transcript files only. Speech-to-text is an upstream concern.

7. **LLM defaults** — `gpt-4o-mini` for cost efficiency. Swappable to `gpt-4o` via `.env` for higher reasoning quality.

---

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI  (cli.py)                      │
│            Rich terminal · verbose mode · history       │
└───────────────────────┬─────────────────────────────────┘
                        │ user message
┌───────────────────────▼─────────────────────────────────┐
│                  Agent  (agent/agent.py)                 │
│                                                         │
│  1. Sends query + tool definitions to LLM               │
│  2. LLM decides which tools to call and in what order   │
│  3. Tools execute; results cached in ResultStore        │
│  4. LLM synthesizes final answer with citations         │
│  5. Conversation history persists across turns          │
└──┬──────────┬───────────┬──────────┬────────────────────┘
   │          │           │          │
┌──▼───┐ ┌───▼────┐ ┌────▼────┐ ┌───▼────────────────┐
│list  │ │  get   │ │ search  │ │     analyze        │
│calls │ │ chunks │ │ trans-  │ │     chunks         │
│      │ │        │ │ cripts  │ │  (LLM reasoning +  │
│      │ │        │ │         │ │   citation output) │
└──┬───┘ └───┬────┘ └────┬────┘ └────────────────────┘
   │          │           │
┌──▼──────────▼───┐  ┌───▼──────────────┐
│    SQLite        │  │     FAISS         │
│  calls + chunks  │  │  vector index     │
│  full text       │  │  cosine sim       │
│  speaker roles   │  │  chunk_id map     │
└──────────────────┘  └──────────────────┘
```

**Ingestion pipeline (offline):**
```
.txt file
   → Parser        (timestamp + speaker extraction, turn detection)
   → Chunker       (speaker-turn merging + LLM role classification)
   → LLM Embedder  (batch embed enriched chunk text)
   → SQLite        (Call + Chunk records)
   → FAISS         (vector index + chunk_id position map)
```

---

### 3.3 Approach

#### How Citations Are Guaranteed

The problem statement requires every answer to indicate which segment(s) contributed. This is enforced through three reinforcing layers — the LLM does not *generate* citations, it *copies* them from pre-formatted evidence:

**Layer 1 — Storage:** Every `Chunk` row in SQLite carries a `citation` field built at write time:
```
[2_pricing_call | 01:24 | Dan (Finance VP)]
```

**Layer 2 — Prompt formatting:** Before any LLM reasoning call, `_format_chunks_block()` renders each chunk with its citation inline so the model sees the source alongside the content:
```
[1] [2_pricing_call | 01:24 | Dan (Finance VP)]
    Competitor X quoted us a flat ₹1500 per seat with unlimited minutes...

[2] [2_pricing_call | 07:11 | Dan (Finance VP)]
    CFO hates ramp discounts; they mess with accrual...
```

**Layer 3 — Prompt instruction:** Every task prompt (summarize, extract, compare, sentiment_filter, qa) contains an explicit instruction:
> *"Cite every specific claim with [call_id | timestamp | speaker]. Do not make claims without a citation."*

This makes citations factually grounded rather than hallucinated — the model references `[1]`, `[2]` etc. from the block it was given.

#### Agentic Tool Loop

The agent runs an iterative loop (max 8 rounds):
1. LLM receives user message + 5 tool definitions
2. LLM returns one or more tool calls with arguments
3. Tools execute; results stored in `ResultStore`
4. Results appended to message context as tool responses
5. Loop repeats until LLM returns a final text answer (no more tool calls)

#### Intent-Driven Tool Chaining

The system prompt encodes explicit chaining patterns per intent type, so the LLM plans correctly without trial-and-error:

```
SUMMARIZE:    list_calls → get_chunks (all) → analyze_chunks(task=summarize)
TOPIC QUERY:  search_transcripts → analyze_chunks(task=extract/qa)
SENTIMENT:    search_transcripts → analyze_chunks(task=sentiment_filter)
COMPARISON:   search_transcripts(scoped) → analyze_chunks(task=compare)
INGEST:       ingest_transcript
LIST:         list_calls  (no further tools needed)
```

#### ResultStore Pattern

`analyze_chunks` operates on potentially 60+ chunks — too large to pass as a JSON tool argument reliably. The `ResultStore` caches the last `get_chunks` / `search_transcripts` output in session memory. `analyze_chunks` auto-loads from it. The agent also strips `chunks` from LLM-generated args defensively:

```python
tool_args.pop("chunks", None)   # never let LLM pass chunks as argument
return analyze_chunks(**tool_args, _result_store=self.result_store)
```

This keeps tool arguments small while enabling reasoning over large full-call contexts.

#### LLM at Ingestion: Speaker Role Classification

Rather than regex-matching speaker names to roles, a single LLM call at ingestion classifies every participant in one shot:

```
Input:  ["Jordan (AE)", "Priya (RevOps Director)", "Dan (Finance VP)"]
Output: {"Jordan (AE)": "ae", "Priya (RevOps Director)": "prospect", "Dan (Finance VP)": "prospect"}
```

This costs one API call per transcript and enables role-based filtering at query time (`speaker_roles=["prospect"]`) without any hardcoded name matching or regex rules.

#### Metadata-Enriched Embeddings

Each chunk is embedded with metadata prepended to the content text:
```
Call: 2_pricing_call
Speaker: Dan (Finance VP) (Role: prospect)
Time: 01:24
Content: Competitor X quoted us a flat ₹1500...
```

Including call ID, speaker, and role in the embedding improves retrieval precision for queries scoped by speaker or call — the vector encodes context, not just content.

---

### 3.4 Design Decisions

#### Two-Layer Storage: SQLite + FAISS

| Query type | Data path | Reasoning |
|---|---|---|
| `list calls`, `filter by speaker/role` | SQLite only | Structured filters — no vector overhead |
| `summarise this call` (full context) | SQLite only | Needs ALL chunks, not top-k ranked results |
| `what did prospects say about pricing?` | FAISS → SQLite join | Semantic relevance ranking + metadata enrichment |
| Metadata for citations | Always SQLite | Single source of truth — no duplication in FAISS |

FAISS stores only vectors + an integer position → `chunk_id` map. All metadata (speaker, timestamp, text, citation) lives in SQLite and is joined at query time. This avoids duplicating data and keeps FAISS payloads minimal.

**Why not a managed vector DB (Qdrant/Weaviate)?** At this scale, FAISS + SQLite is simpler, faster to set up, and has zero network dependency. The interface is abstracted in `storage/vector_store.py` — migrating to Qdrant is a single file change.

#### Speaker-Turn Chunking over Fixed Token Windows

Fixed 256-token windows split mid-sentence and destroy speaker attribution. Turn-based chunking:
- Preserves who said what — critical for `"what did Priya say about X?"` queries
- Makes citations meaningful — `[MM:SS | Speaker]` maps to a coherent thought, not a mid-sentence fragment
- Merges short consecutive same-speaker turns (up to 4 turns / 300 tokens) to avoid micro-chunks with poor embedding signal

#### 5 Generic Tools over Intent-Specific Handlers

Instead of `SummarizeHandler`, `ObjectionHandler`, `PricingHandler`, the system has 5 composable tools. New question types need zero new tools — just new `task` + `criteria` values in `analyze_chunks`. This was validated against 10 representative queries during design, all covered by combinations of the 5 tools.

---

## 4. Project Structure

```
sales_copilot/
├── agent/
│   ├── agent.py              # Agentic loop, tool definitions, system prompt
│   └── result_store.py       # Inter-tool data cache (ResultStore pattern)
│
├── generation/
│   ├── llm_client.py         # OpenAI wrapper: chat + batch embeddings + retry
│   └── prompts.py            # Task-specific prompt templates (5 tasks)
│
├── ingestion/
│   ├── parser.py             # [MM:SS] Speaker: text → structured turns
│   ├── chunker.py            # Turn merging + LLM speaker role classification
│   └── pipeline.py           # Orchestrates parse → chunk → embed → store
│
├── storage/
│   ├── db.py                 # SQLAlchemy models: Call, Chunk + session utils
│   └── vector_store.py       # FAISS wrapper: add, search, save/load
│
├── tools/
│   ├── list_calls.py         # Metadata fetch with sort/filter
│   ├── get_chunks.py         # Structured chunk fetch (SQLite only)
│   ├── search_transcripts.py # Semantic search: FAISS → SQLite join
│   ├── analyze_chunks.py     # LLM reasoning: 5 task types + citation enforcement
│   └── ingest_transcript.py  # Pipeline trigger tool
│
├── tests/
│   ├── test_ingestion.py     # Parser, chunker, pipeline tests
│   └── test_retrieval.py     # Tool-level tests + ResultStore unit tests
│
├── data/
│   ├── transcripts/          # Raw .txt transcript files
│   ├── sales_copilot.db      # SQLite (auto-created on first run)
│   └── faiss_index.*         # FAISS index + id map (auto-created)
│
├── cli.py                    # Entry point — Rich terminal interface
├── config.py                 # Centralised config loaded from .env
├── requirements.txt
├── .env.example
└── README.md
```

---

## 5. Setup

### Prerequisites

- Python 3.10+
- OpenAI API key (`text-embedding-3-small` + `gpt-4o-mini` access)

### Installation

```bash
# 1. Clone
git clone <repo-url>
cd sales_copilot

# 2. Virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

### Ingest Sample Transcripts

```bash
# Ingest all provided transcripts
python -c "from ingestion.pipeline import ingest_directory; ingest_directory()"

# Verify ingestion
python -c "
from storage.db import SessionLocal, Call, Chunk
s = SessionLocal()
print(f'Calls:  {s.query(Call).count()}')
print(f'Chunks: {s.query(Chunk).count()}')
for c in s.query(Call).all():
    print(f'  {c.call_id}: {len(c.chunks)} chunks')
s.close()
"
```

---

## 6. Usage

### Run

```bash
python cli.py
```

### CLI Commands

| Input | Behaviour |
|---|---|
| Any natural language | Agent plans tool chain, returns cited answer |
| `verbose` | Toggle tool call visibility — shows which tools fired and with what args |
| `reset` | Clear conversation history and result cache |
| `ingest <path>` | Shortcut to ingest a new transcript |
| `exit` / `quit` | End session |

### Example Queries

```bash
# Metadata
list my call ids

# Summarisation
summarise the last call
give me a summary of the pricing call

# Cross-call extraction
what objections did prospects raise across all calls?
which calls mentioned Competitor X?
what were the action items from the negotiation call?

# Sentiment
give me all negative comments when pricing was mentioned

# Speaker-scoped
what did Priya say about onboarding?
what security concerns did Arjun raise?

# Comparison
compare how pricing was handled in the pricing call vs the negotiation call

# Ingest
ingest data/transcripts/6_enterprise_demo_call.txt

# Multi-turn (resolves references from history)
You: which call had the most security objections?
You: summarise that call
```

### Sample Cited Output

```
You: give me all negative comments when pricing was mentioned

Copilot:

**1. Hidden overage costs**
[2_pricing_call | 01:24 | Dan (Finance VP)]
Dan raised concern that Competitor X's "unlimited minutes" claim masks a soft
cap — framing our limited-minutes model as the more transparent option.

**2. CFO resistance to ramp discounts**
[2_pricing_call | 07:11 | Dan (Finance VP)]
"CFO hates ramp discounts; they mess with accrual." Dan rejected the
3-month ramp structure Maya proposed.

**3. Headline number gap**
[2_pricing_call | 06:41 | Dan (Finance VP)]
Competitor X quoted ₹60L for 3 years vs our ₹69L — a 15% gap Dan
flagged explicitly for CFO review before sign-off.
```

---

## 7. Tests

```bash
# No API cost — skips all LLM calls
pytest tests/ -v -m "not llm"

# Full suite — requires API key + ingested data
pytest tests/ -v -m llm

# Individual modules
pytest tests/test_ingestion.py -v
pytest tests/test_retrieval.py -v
```

### Coverage Summary

| Module | What's tested |
|---|---|
| `test_ingestion.py` | Turn count, timestamp extraction, multiline handling, end-time filling, speaker boundary enforcement, chunk ID uniqueness, citation format, LLM role classification (`llm`), full pipeline ingestion, overwrite/skip behaviour |
| `test_retrieval.py` | `list_calls` sort/filter/field validation, `get_chunks` call scoping/role filter/speaker partial match, `search_transcripts` score ordering/scoping/role filter, `analyze_chunks` all 5 tasks + citation presence, ResultStore save/retrieve/reset |

---

## 8. Limitations

1. **FAISS deletion not supported** — `IndexFlatIP` has no in-place delete. Re-ingesting cleans SQLite but leaves stale vectors in FAISS. Mitigate with `IndexIDMap` or a vector DB with native deletion.

2. **Context window on large calls** — Very long calls (200+ chunks) may approach LLM context limits. A proper fix is hierarchical summarisation (map chunks → reduce summaries).

3. **In-memory session only** — `ResultStore` and conversation history reset on CLI exit. No cross-session persistence.

4. **Speaker classification errors** — Unusual titles or ambiguous roles may be misclassified. No correction feedback loop post-ingestion.

5. **Citation at chunk granularity** — Citations point to a merged chunk (up to 4 turns). Sub-sentence attribution would require storing individual turns as separate embeddings.

6. **No streaming** — CLI waits for the full agent loop before displaying output. Adds perceived latency on multi-tool queries.

7. **English-primary** — Embedding and prompt quality degrades on heavily code-switched speech (Hinglish, etc.).

---

## 9. Future Extensions

| Extension | Approach |
|---|---|
| **Streaming output** | OpenAI streaming API + Rich live display |
| **Persistent sessions** | Store history in SQLite keyed by `session_id` |
| **Audio ingestion** | Add Whisper transcription step before `parser.py` in pipeline |
| **Hierarchical summarisation** | Map-reduce in `analyze_chunks` for calls exceeding context window |
| **New analysis task** | Add prompt template to `generation/prompts.py` TASK_PROMPTS dict |
| **New tool** | Add function + Pydantic schema in `tools/`, register in `agent.py` |
| **Web UI** | Wrap agent in FastAPI; replace CLI with React frontend |
| **Deal health scoring** | Add `score_deal` tool running regression over extracted signals |
| **Auto follow-up drafting** | Add `draft_followup` tool using objections + action items as context |