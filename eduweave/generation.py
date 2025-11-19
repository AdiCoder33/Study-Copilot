"""LLM-based generation helpers for summaries and MCQs."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from .config import GenerationConfig
from .local_llm import get_text_generator


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

    generator = get_text_generator(config.summary_model)
    return generator.generate(
        prompt="You write concise, structured study notes.\n" + prompt,
        temperature=temperature or config.summary_temperature,
        max_new_tokens=config.summary_max_tokens,
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

    generator = get_text_generator(config.mcq_model)
    raw_output = generator.generate(
        prompt="You design rigorous but fair multiple-choice questions.\n" + prompt,
        temperature=temperature or config.mcq_temperature,
        max_new_tokens=config.mcq_max_tokens,
    )

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, list):
            return parsed, raw_output
    except json.JSONDecodeError:
        pass

    return [], raw_output
