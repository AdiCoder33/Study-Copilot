"""LLM-based generation helpers for summaries and MCQs."""

from __future__ import annotations

import json
import re
import os
from typing import Any, Dict, List, Tuple

from huggingface_hub import InferenceClient

from .config import GenerationConfig


def _build_client(model: str) -> InferenceClient:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    return InferenceClient(model=model, token=token)


def _chat_with_hf(model: str, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int) -> str:
    client = _build_client(model)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        blocks = response.choices[0].message.get("content", [])
        text = "".join(block.get("text", "") for block in blocks)
        if text:
            return text.strip()
    except Exception:
        pass

    prompt = f"<|system|>\n{system_prompt.strip()}\n<|user|>\n{user_prompt.strip()}\n<|assistant|>\n"
    text = client.text_generation(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        return_full_text=False,
    )
    return text.strip()


def _clean_context(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def generate_summary(
    corpus_text: str,
    config: GenerationConfig,
    temperature: float | None = None,
) -> str:
    """Generate a structured summary suitable for engineering students."""
    context = _clean_context(corpus_text, config.max_summary_chars)
    if not context:
        raise ValueError("No text available for summarization.")

    prompt = f"""Source material:
{context}

Produce a concise study summary with:
- a short overview paragraph,
- 2-4 bullet lists grouped by major themes,
- highlight critical formulas or definitions if present.
Keep the tone friendly and academic."""

    return _chat_with_hf(
        config.summary_model,
        "You write concise, structured study notes.",
        prompt,
        temperature or config.summary_temperature,
        config.summary_max_tokens,
    )


def generate_mcqs(
    corpus_text: str,
    num_questions: int,
    config: GenerationConfig,
    temperature: float | None = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """Generate MCQs plus a fallback raw response."""
    context = _clean_context(corpus_text, config.max_mcq_chars)
    if not context:
        raise ValueError("No text available for MCQ generation.")

    prompt = f"""Use the material below to create {num_questions} engineering-style MCQs.

Material:
{context}

Return valid JSON with this schema:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "A",
    "explanation": "1 sentence justification"
  }}
]
Include varied concepts and avoid trivia."""

    raw_output = _chat_with_hf(
        config.mcq_model,
        "You design rigorous but fair multiple-choice questions.",
        prompt,
        temperature or config.mcq_temperature,
        config.mcq_max_tokens,
    )

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, list):
            return parsed, raw_output
    except json.JSONDecodeError:
        pass

    return [], raw_output
