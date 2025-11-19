"""Lightweight logging for experimentation notes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict


def log_experiment(
    title: str,
    parameters: Dict[str, str | int | float],
    observation: str,
    log_path: str = "docs/experiments.md",
) -> None:
    """Append a markdown entry describing an experiment and its outcome."""
    path = Path(log_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    param_lines = ", ".join(f"{key}={value}" for key, value in parameters.items())
    entry = (
        f"### {timestamp} – {title}\n\n"
        f"**Parameters:** {param_lines or 'n/a'}\n\n"
        f"**Observation:** {observation.strip() or 'n/a'}\n\n"
        "---\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)
