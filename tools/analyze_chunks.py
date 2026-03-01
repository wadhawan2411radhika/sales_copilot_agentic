from pydantic import BaseModel, Field
from typing import Optional
from generation.llm_client import llm_client
from generation.prompts import build_analysis_prompt


class AnalyzeChunksInput(BaseModel):
    chunks:           list[dict] = Field(default=[], description="List of chunk dicts from get_chunks or search_transcripts. Can be empty if use_cached_chunks=true.")
    task:             str        = Field(description="Task: summarize | sentiment_filter | extract | compare | qa")
    criteria:         str        = Field(description="Natural language instruction. E.g. 'focus on objections', 'negative sentiment only'.")
    output_format:    str        = Field(default="text", description="Output format: text | bullets | json")
    use_cached_chunks: bool      = Field(default=True,  description="If true and chunks is empty, automatically use chunks from the previous get_chunks or search_transcripts call.")


def analyze_chunks(
    chunks:            list[dict] = [],
    task:              str = "qa",
    criteria:          str = "",
    output_format:     str = "text",
    use_cached_chunks: bool = True,
    # Injected by agent — not visible to LLM
    _result_store=None,
) -> dict:
    """
    LLM-powered analysis on chunks.
    If chunks is empty and use_cached_chunks=True, pulls from previous tool result automatically.
    """
    # Auto-inject chunks from result store if not passed
    if not chunks and use_cached_chunks and _result_store is not None:
        chunks = _result_store.get_latest_chunks()
        if chunks:
            source = "auto-loaded from previous tool result"
        else:
            return {
                "task":         task,
                "criteria":     criteria,
                "result":       "No chunks available. Please run get_chunks or search_transcripts first.",
                "cited_chunks": [],
            }
    
    if not chunks:
        return {
            "task":         task,
            "criteria":     criteria,
            "result":       "No chunks provided to analyze.",
            "cited_chunks": [],
        }

    prompt = build_analysis_prompt(
        chunks=chunks,
        task=task,
        criteria=criteria,
        output_format=output_format,
    )

    result = llm_client.chat(
        prompt=prompt,
        system=(
            "You are a senior sales intelligence analyst. "
            "You analyze B2B sales call transcripts and extract structured insights. "
            "Always cite sources using the citation field: [call_id | timestamp | speaker]."
        ),
        temperature=0.1,
    )

    cited_chunks = [
        {
            "chunk_id":   c.get("chunk_id"),
            "call_id":    c.get("call_id"),
            "citation":   c.get("citation"),
            "speaker":    c.get("speaker"),
            "start_time": c.get("start_time"),
        }
        for c in chunks
    ]

    return {
        "task":         task,
        "criteria":     criteria,
        "result":       result,
        "cited_chunks": cited_chunks,
        "chunks_used":  len(chunks),
    }