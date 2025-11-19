"""Retrieval-augmented generation helpers."""

from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from huggingface_hub import InferenceClient

from .config import GenerationConfig, RetrievalConfig
from .vector_store import VectorStoreManager


@dataclass
class AnswerResponse:
    answer: str
    sources: List[Dict[str, Any]]
    mode: str


SYSTEM_PROMPT = """You are EduWeave, an academic AI assistant.
Answer questions strictly with the provided context. Avoid speculation.
If the answer is missing, say you cannot find it in the supplied notes."""


def _chat_completion(messages, model: str, temperature: float, max_tokens: int) -> str:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    client = InferenceClient(model=model, token=token)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content_blocks = response.choices[0].message.get("content", [])
        text = "".join(block.get("text", "") for block in content_blocks)
        if text:
            return text.strip()
    except Exception:
        pass  # Fallback to text-generation style prompt below.

    prompt = ""
    for msg in messages:
        role = msg["role"]
        prompt += f"<|{role}|>\n{msg['content'].strip()}\n"
    prompt += "<|assistant|>\n"

    response = client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False,
    )
    return response.strip()


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
Helpful answer:"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    answer = _chat_completion(
        messages,
        generation_cfg.answer_model,
        temperature or generation_cfg.answer_temperature,
        generation_cfg.answer_max_tokens,
    )

    return AnswerResponse(answer=answer, sources=hits, mode=mode)
