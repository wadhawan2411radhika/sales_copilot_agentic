def _format_chunks_block(chunks: list[dict]) -> str:
    """Format chunks into a readable block for LLM prompts."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(
            f"[{i}] {chunk.get('citation', 'unknown')}\n"
            f"    {chunk.get('text', '')}"
        )
    return "\n\n".join(lines)


def build_analysis_prompt(
    chunks:        list[dict],
    task:          str,
    criteria:      str,
    output_format: str = "text",
) -> str:

    chunks_block = _format_chunks_block(chunks)
    n_calls      = len(set(c.get("call_id") for c in chunks))
    format_note  = {
        "text":    "Write in clear prose paragraphs.",
        "bullets": "Use bullet points. Each bullet must include a citation.",
        "json":    "Return a valid JSON object only. No markdown.",
    }.get(output_format, "Write in clear prose.")

    TASK_PROMPTS = {

        "summarize": f"""You are summarizing content from {n_calls} sales call(s).

Criteria: {criteria}

Transcript segments:
{chunks_block}

Provide a structured summary with these sections:
1. **TL;DR** (one sentence)
2. **Key Topics Discussed**
3. **Objections / Concerns Raised** (cite each with [call_id | timestamp | speaker])
4. **Action Items** (with owner if mentioned)
5. **Deal Risk Signals** (budget freeze, competitor mentions, hesitation)

{format_note}
Cite every specific claim with its source.""",


        "sentiment_filter": f"""You are a sentiment analyst for B2B sales conversations.

Filter criteria: {criteria}

Transcript segments to analyze:
{chunks_block}

Instructions:
- Analyze each segment for sentiment (positive / negative / neutral)
- Return ONLY segments that match the criteria: "{criteria}"
- For each matching segment, state: the sentiment label, why it matches, and the citation
- If no segments match, say so clearly

{format_note}""",


        "extract": f"""You are extracting specific information from sales call transcripts.

What to extract: {criteria}

Transcript segments:
{chunks_block}

Instructions:
- Extract only what matches: "{criteria}"  
- Group by call if multiple calls are present
- Cite every extracted item with [call_id | timestamp | speaker]
- Do not infer or hallucinate — only extract what is explicitly stated

{format_note}""",


        "compare": f"""You are comparing information across {n_calls} sales calls.

What to compare: {criteria}

Transcript segments (from multiple calls):
{chunks_block}

Instructions:
- Organize your comparison by call
- Highlight similarities and differences for: "{criteria}"
- Use citations for every specific claim [call_id | timestamp | speaker]
- End with a brief conclusion on the key differences

{format_note}""",


        "qa": f"""You are answering a question based solely on sales call transcript evidence.

Question / Task: {criteria}

Available transcript segments:
{chunks_block}

Instructions:
- Answer based ONLY on the provided segments
- Cite every claim with [call_id | timestamp | speaker]
- If the answer is not in the segments, say "Not found in provided transcripts"
- Do not speculate beyond what is stated

{format_note}""",
    }

    return TASK_PROMPTS.get(task, TASK_PROMPTS["qa"])