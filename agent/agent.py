import json
from openai import OpenAI
from config import config
from agent.result_store import ResultStore
from tools.list_calls import list_calls, ListCallsInput
from tools.get_chunks import get_chunks, GetChunksInput
from tools.search_transcripts import search_transcripts, SearchTranscriptsInput
from tools.analyze_chunks import analyze_chunks, AnalyzeChunksInput
from tools.ingest_transcript import ingest_transcript, IngestTranscriptInput


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_calls",
            "description": (
                "Returns metadata for all ingested calls including call IDs, participants, "
                "duration, and chunk counts. Use this when the user asks to list calls, "
                "see available transcripts, or when you need to resolve references like "
                "'last call', 'most recent call', or 'call from last week'."
            ),
            "parameters": ListCallsInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_chunks",
            "description": (
                "Fetches ALL transcript chunks using structured filters (call ID, speaker, role). "
                "No semantic search — use when you need full content from a specific call "
                "for summarization, or filtering by speaker/role is sufficient. "
                "Results are automatically available to analyze_chunks via cache."
            ),
            "parameters": GetChunksInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_transcripts",
            "description": (
                "Semantic vector search over all transcript chunks. Use when the user "
                "asks about topics or themes — e.g. 'what objections came up', "
                "'competitor mentions', 'pricing discussions'. "
                "Results are automatically available to analyze_chunks via cache."
            ),
            "parameters": SearchTranscriptsInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_chunks",
            "description": (
                "LLM-powered analysis on chunks from get_chunks or search_transcripts. "
                "IMPORTANT: You do NOT need to pass chunks — they are loaded automatically "
                "from the previous tool call. Just specify task and criteria. "
                "Tasks: summarize | sentiment_filter | extract | compare | qa. "
                "Always call this after get_chunks or search_transcripts to reason over results."
            ),
            "parameters": AnalyzeChunksInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_transcript",
            "description": (
                "Ingests a new transcript file end-to-end. Use when user says "
                "'ingest', 'add', 'load', or 'import' a transcript from a file path."
            ),
            "parameters": IngestTranscriptInput.model_json_schema(),
        },
    },
]


SYSTEM_PROMPT = """You are a Sales Intelligence Copilot helping sales teams understand their calls.

You have 5 tools. Key workflow rules:

1. SUMMARIZE A CALL:
   - list_calls (to find call ID if needed)
   - get_chunks (call_ids=[...]) 
   - analyze_chunks (task="summarize", criteria="...") ← chunks auto-loaded, don't pass them

2. TOPIC/THEME QUESTIONS across calls:
   - search_transcripts (query="...")
   - analyze_chunks (task="extract" or "qa", criteria="...") ← chunks auto-loaded

3. SENTIMENT FILTERING:
   - search_transcripts (query="...")
   - analyze_chunks (task="sentiment_filter", criteria="negative") ← chunks auto-loaded

4. CROSS-CALL COMPARISON:
   - search_transcripts (query="...", call_ids=[...])
   - analyze_chunks (task="compare", criteria="...") ← chunks auto-loaded

5. INGEST: ingest_transcript (file_path="...")

CRITICAL: When calling analyze_chunks, NEVER try to pass chunks as an argument.
They are automatically loaded from the previous get_chunks or search_transcripts result.
Just pass task and criteria.

Always include citations in format [call_id | timestamp | speaker] in your final answer."""


class SalesAgent:

    MAX_TOOL_ROUNDS = 8

    def __init__(self):
        self.client       = OpenAI(api_key=config.OPENAI_API_KEY)
        self.history      = []
        self.result_store = ResultStore()

    def chat(self, user_message: str, verbose: bool = False) -> str:
        self.history.append({"role": "user", "content": user_message})
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history

        for round_num in range(self.MAX_TOOL_ROUNDS):

            response = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
            )

            message = response.choices[0].message
            finish  = response.choices[0].finish_reason
            messages.append(message)

            if finish == "stop" or not message.tool_calls:
                final = message.content or ""
                self.history.append({"role": "assistant", "content": final})
                return final

            if verbose:
                for tc in message.tool_calls:
                    args_preview = tc.function.arguments[:120]
                    print(f"\n  [tool] {tc.function.name}({args_preview}...)")

            tool_results = []
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                result = self._execute_tool(tool_name, tool_args)

                # Save to result store so analyze_chunks can auto-load
                self.result_store.save(tool_name, result)

                result_str = json.dumps(result, default=str)

                if verbose:
                    print(f"  [result] {result_str[:200]}...")

                tool_results.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      result_str,
                })

            messages.extend(tool_results)

        fallback = "I wasn't able to complete the request within the tool call limit. Please try rephrasing."
        self.history.append({"role": "assistant", "content": fallback})
        return fallback

    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Execute tool — injects result_store into analyze_chunks."""
        try:
            if tool_name == "list_calls":
                return list_calls(**tool_args)
            elif tool_name == "get_chunks":
                return get_chunks(**tool_args)
            elif tool_name == "search_transcripts":
                return search_transcripts(**tool_args)
            elif tool_name == "analyze_chunks":
                # Strip chunks if LLM mistakenly passed them (too large)
                tool_args.pop("chunks", None)
                return analyze_chunks(**tool_args, _result_store=self.result_store)
            elif tool_name == "ingest_transcript":
                return ingest_transcript(**tool_args)
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e), "tool": tool_name, "args": tool_args}

    def reset(self):
        self.history = []
        self.result_store.reset()