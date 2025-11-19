"""Utilities for preparing cleaned text for embeddings."""

from __future__ import annotations

import re
from typing import List, Tuple

from .config import ChunkConfig


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> List[str]:
    """Split text into overlapping windows with soft sentence boundaries."""
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    chunk_size = max(chunk_size, 200)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    chunks: List[str] = []
    start = 0
    length = len(normalized)

    while start < length:
        end = min(start + chunk_size, length)
        window = normalized[start:end]

        # Try to end the chunk on a natural boundary.
        adjusted_end = end
        for sep in separators:
            if not sep:
                continue
            idx = window.rfind(sep)
            if idx == -1:
                continue
            absolute_idx = start + idx + len(sep.strip())
            if absolute_idx - start >= chunk_size * 0.6:
                adjusted_end = absolute_idx
                break

        chunk = normalized[start:adjusted_end].strip()
        if chunk:
            chunks.append(chunk)

        if adjusted_end >= length:
            break

        start = max(adjusted_end - chunk_overlap, 0)
        if start >= length:
            break

    return chunks


def chunk_text_with_metadata(text: str, chunk_config: ChunkConfig, source: str | None = None):
    """Return chunk texts alongside lightweight metadata for tracing."""
    chunks = chunk_text(
        text,
        chunk_size=chunk_config.chunk_size,
        chunk_overlap=chunk_config.chunk_overlap,
        separators=chunk_config.separators,
    )
    metadata = [
        {
            "source": source or "document",
            "chunk_index": index,
            "char_length": len(chunk),
        }
        for index, chunk in enumerate(chunks)
    ]
    return chunks, metadata
