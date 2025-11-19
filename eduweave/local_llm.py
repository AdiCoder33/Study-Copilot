"""Utility for loading and reusing local Hugging Face text generation models."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LocalTextGenerator:
    """Wrap a text2text-generation pipeline with sane defaults."""

    def __init__(self, model_name: str, device: Optional[int | str] = None):
        self.model_name = model_name
        self.device = device
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if isinstance(self.device, int) else -1,
            )
        return self._pipeline

    def generate(self, prompt: str, temperature: float, max_new_tokens: int) -> str:
        generator = self._load()
        temperature = max(0.01, min(temperature, 2.0))
        generation = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.05,
            temperature=temperature,
            num_return_sequences=1,
        )
        result = generation[0]
        text = result.get("generated_text")
        if text is None:
            text = result.get("summary_text", "")
        return (text or "").strip()


@lru_cache(maxsize=3)
def get_text_generator(model_name: str, device: Optional[int | str] = None) -> LocalTextGenerator:
    return LocalTextGenerator(model_name=model_name, device=device)
