class ResultStore:
    """
    In-memory scratchpad for the agent session.
    Stores the last output of each tool so subsequent tools
    can reference previous results without passing large data as arguments.
    """

    def __init__(self):
        self._store: dict[str, any] = {}

    def save(self, tool_name: str, result: any):
        self._store[tool_name] = result

    def get(self, tool_name: str) -> any:
        return self._store.get(tool_name)

    def get_latest_chunks(self) -> list[dict]:
        """
        Returns chunks from the most recent get_chunks or search_transcripts call.
        analyze_chunks uses this when the LLM forgets to pass chunks explicitly.
        """
        for tool_name in ("get_chunks", "search_transcripts"):
            result = self._store.get(tool_name)
            if result and isinstance(result, dict):
                chunks = result.get("chunks", [])
                if chunks:
                    return chunks
        return []

    def reset(self):
        self._store = {}