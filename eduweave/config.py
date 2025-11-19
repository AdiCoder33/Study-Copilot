"""Central configuration objects for EduWeave."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ChunkConfig:
    """Controls how raw text is split before embedding."""

    chunk_size: int = 800
    chunk_overlap: int = 150
    min_chunk_size: int = 200
    separators: Tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass
class RetrievalConfig:
    """Controls the retrieval step for RAG Q&A."""

    top_k: int = 5
    max_k: int = 8
    distance_cutoff: float = 1.5


@dataclass
class GenerationConfig:
    """Temperature and token budgeting for downstream generations."""

    answer_temperature: float = 0.2
    summary_temperature: float = 0.25
    mcq_temperature: float = 0.6
    max_context_chars: int = 6000
    max_summary_chars: int = 4500
    max_mcq_chars: int = 5000
    answer_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    summary_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    mcq_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    answer_max_tokens: int = 300
    summary_max_tokens: int = 350
    mcq_max_tokens: int = 400


@dataclass
class AppConfig:
    """Bundle config sections used across the Streamlit app."""

    project_title: str = "EduWeave - AI Study Copilot for PDFs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
