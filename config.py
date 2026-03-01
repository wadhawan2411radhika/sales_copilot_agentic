from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    OPENAI_API_KEY: str     = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL: str    = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL: str          = os.getenv("LLM_MODEL", "gpt-4o-mini")
    DB_PATH: str            = os.getenv("DB_PATH", "./data/sales_copilot.db")
    FAISS_INDEX_PATH: str   = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

    # Chunking
    MAX_CHUNK_TURNS: int    = 4     # merge up to N consecutive same-speaker turns
    MAX_CHUNK_TOKENS: int   = 300   # soft cap before forcing a new chunk

    # Retrieval
    DEFAULT_TOP_K: int      = 10

config = Config()