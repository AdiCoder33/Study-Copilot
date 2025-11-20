"""In-memory FAISS vector store backed by Hugging Face embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

try:  # Defensive import: some envs lazy-load AutoConfig which sentence-transformers expects
    from transformers import AutoConfig as _  # noqa: F401
except ImportError:
    import transformers
    from transformers.models.auto.configuration_auto import AutoConfig

    setattr(transformers, "AutoConfig", AutoConfig)

from sentence_transformers import SentenceTransformer


@dataclass
class VectorStoreManager:
    """Build and query an in-memory FAISS index."""

    embedding_model: str
    device: Optional[str] = None
    batch_size: int = 32
    text_chunks: List[str] = field(default_factory=list)
    metadatas: List[Dict[str, Any]] = field(default_factory=list)
    full_text: str = ""

    def __post_init__(self):
        self._embedder: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatL2] = None
        self._dimension: Optional[int] = None

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            self._embedder = SentenceTransformer(self.embedding_model, device=self.device or "cpu")
        return self._embedder

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        vectors = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.astype("float32")

    def build(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Create a FAISS index from prepared chunk texts."""
        if not texts:
            raise ValueError("No texts supplied for vector store construction.")

        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.append(self._embed_batch(batch))

        matrix = np.vstack(embeddings).astype("float32")
        self._dimension = matrix.shape[1]
        self._index = faiss.IndexFlatL2(self._dimension)
        self._index.add(matrix)

        self.text_chunks = texts
        self.metadatas = metadatas or [{} for _ in texts]
        self.full_text = " ".join(texts)

    def is_ready(self) -> bool:
        return self._index is not None and len(self.text_chunks) > 0

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.is_ready():
            raise RuntimeError("Vector store has not been built yet.")
        query_vec = self._embed_batch([query])
        scores, indices = self._index.search(query_vec, top_k)
        hits = []
        for idx, distance in zip(indices[0], scores[0]):
            if idx == -1 or idx >= len(self.text_chunks):
                continue
            hits.append(
                {
                    "text": self.text_chunks[idx],
                    "metadata": self.metadatas[idx],
                    "distance": float(distance),
                }
            )
        return hits
