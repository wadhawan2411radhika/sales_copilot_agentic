import os
import pickle
import numpy as np
import faiss

from config import config


class VectorStore:
    """
    Thin FAISS wrapper.
    
    Design:
    - FAISS stores vectors indexed by integer position (0, 1, 2, ...)
    - We maintain a parallel list `chunk_ids` so position → chunk_id
    - chunk_id → SQLite for all metadata (no duplication in FAISS)
    """

    FAISS_PATH   = config.FAISS_INDEX_PATH + ".index"
    ID_MAP_PATH  = config.FAISS_INDEX_PATH + ".pkl"
    DIMENSION    = 1536   # text-embedding-3-small output dim

    def __init__(self):
        self.index: faiss.Index = None
        self.chunk_ids: list[str] = []    # position i → chunk_id
        self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.FAISS_PATH) and os.path.exists(self.ID_MAP_PATH):
            self.index = faiss.read_index(self.FAISS_PATH)
            with open(self.ID_MAP_PATH, "rb") as f:
                self.chunk_ids = pickle.load(f)
        else:
            # Inner product on L2-normalised vectors = cosine similarity
            self.index = faiss.IndexFlatIP(self.DIMENSION)
            self.chunk_ids = []

    def add(self, chunk_id: str, embedding: list[float]) -> int:
        """
        Add one embedding. Returns its position (embedding_id stored in SQLite).
        """
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)                  # normalize for cosine sim

        position = self.index.ntotal
        self.index.add(vec)
        self.chunk_ids.append(chunk_id)

        self.save()
        return position

    def search(self, query_embedding: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """
        Returns list of (chunk_id, score) sorted by relevance.
        """
        if self.index.ntotal == 0:
            return []

        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        top_k   = min(top_k, self.index.ntotal)
        scores, positions = self.index.search(vec, top_k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:       # FAISS returns -1 for empty slots
                continue
            results.append((self.chunk_ids[pos], float(score)))

        return results          # [(chunk_id, score), ...] descending

    def delete_by_call(self, chunk_ids_to_remove: list[str]):
        """
        FAISS IndexFlatIP doesn't support deletion natively.
        Rebuild index excluding removed chunk_ids — called on re-ingest.
        """
        # Not implemented for MVP — overwrite=False prevents duplicates
        pass

    def save(self):
        os.makedirs(os.path.dirname(self.FAISS_PATH), exist_ok=True)
        faiss.write_index(self.index, self.FAISS_PATH)
        with open(self.ID_MAP_PATH, "wb") as f:
            pickle.dump(self.chunk_ids, f)

    @property
    def total_vectors(self) -> int:
        return self.index.ntotal