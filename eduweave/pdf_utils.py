"""Utilities for working with uploaded PDF files."""

from __future__ import annotations

import io
import re
from typing import List

from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Remove null bytes, collapse whitespace, and trim noise from a page."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(file_obj: io.BytesIO, filename: str | None = None) -> str:
    """Extract readable text from each page of the uploaded PDF stream."""
    reader = PdfReader(file_obj)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - pypdf raises various errors
            label = f"{filename or 'PDF'} page {number}"
            raise RuntimeError(f"Failed to extract text from {label}: {exc}") from exc
        cleaned = clean_text(raw)
        if cleaned:
            pages.append(cleaned)
    return "\n\n".join(pages)


def combine_texts(texts: List[str]) -> str:
    """Join multiple PDF texts into a single corpus string."""
    cleaned = [clean_text(text) for text in texts if text]
    return "\n\n".join(chunk for chunk in cleaned if chunk)
