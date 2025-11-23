"""Retrieval-augmented generation helpers."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .config import GenerationConfig, RetrievalConfig
from .local_llm import get_text_generator
from .vector_store import VectorStoreManager
import re


@dataclass
class AnswerResponse:
    answer: str
    sources: List[Dict[str, Any]]
    mode: str


SYSTEM_PROMPT = """You are EduWeave, an academic AI assistant.
Answer questions strictly with the provided context. Avoid speculation.
Keep responses concise (one or two sentences) and directly address the question.
If the answer is missing, say you cannot find it in the supplied notes."""


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", f"chunk-{idx}")
        formatted.append(f"[{source}] {chunk['text']}")
    return "\n\n".join(formatted)


def answer_question(
    question: str,
    vector_store: Optional[VectorStoreManager],
    retrieval_cfg: RetrievalConfig,
    generation_cfg: GenerationConfig,
    use_rag: bool = True,
    temperature: Optional[float] = None,
) -> AnswerResponse:
    """Answer a question either by RAG or by querying the raw corpus."""
    if not vector_store or not vector_store.is_ready():
        raise RuntimeError("Vector store is not ready. Upload and process PDFs first.")

    if use_rag:
        hits = vector_store.search(question, top_k=retrieval_cfg.top_k)
        context = _format_context(hits)
        mode = "rag"
    else:
        hits = []
        context = textwrap.shorten(vector_store.full_text, width=generation_cfg.max_context_chars, placeholder=" ...")
        mode = "baseline"

    if not context:
        context = "Context is empty."

    prompt = f"""Context:
{context}

Question: {question}
Answer in a single short paragraph that directly addresses the question:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    generator = get_text_generator(generation_cfg.answer_model)
    raw_answer = generator.generate(
        prompt="\n".join([m["content"] for m in messages]),
        temperature=temperature or generation_cfg.answer_temperature,
        max_new_tokens=generation_cfg.answer_max_tokens,
    )
    # Prefer the portion after the last "Answer:" marker if the model echoed the prompt.
    parts = re.split(r"(?i)\banswer\s*:\s*", raw_answer)
    answer_body = parts[-1].strip() if len(parts) > 1 else raw_answer.strip()

    # Drop lines that look like echoed prompt/context.
    cleaned_lines = []
    for line in answer_body.splitlines():
        if re.match(r"(?i)\s*(context|question)\s*:", line):
            continue
        if re.search(r"EduWeave|AI assistant|Avoid speculation|Keep responses concise", line, flags=re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    cleaned = " ".join(cleaned_lines).strip()

    # Keep up to the first seven sentences to stay concise but informative.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    answer = " ".join(sentences[:7]).strip() if sentences else cleaned

    # Fallback if the model echoed instructions instead of answering.
    if not answer or "if the answer is missing" in answer.lower():
        answer = "I cannot find the answer in the supplied notes."

    return AnswerResponse(answer=answer, sources=hits, mode=mode)
