"""Structured requests to a locally installed vision model through Ollama."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def json_from_text(raw: str) -> Any:
    """Parse JSON while tolerating explanatory text around the object."""

    raw = raw.strip()
    if not raw:
        raise ValueError("The model returned no text.")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The model response did not contain a JSON object.")
        return json.loads(raw[start : end + 1])


@dataclass(frozen=True)
class OllamaVisionModel:
    """An offline vision-model client shared by Qwen pipeline adapters."""

    model: str = "qwen3-vl"
    context_length: int = 16384

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
    ) -> Any:
        try:
            import ollama
        except ImportError as error:
            raise RuntimeError(
                "The 'ollama' Python package is missing; run 'uv sync'."
            ) from error

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(image_path)],
                    }
                ],
                format=schema,
                think=False,
                options={"temperature": 0, "num_ctx": self.context_length},
                stream=False,
            )
        except Exception as error:  # Ollama exposes transport-specific errors.
            raise RuntimeError(
                f"Local Ollama request failed for model '{self.model}': {error}"
            ) from error

        message = getattr(response, "message", None)
        if isinstance(message, dict):
            content = message.get("content", "")
            thinking = message.get("thinking", "")
        else:
            content = getattr(message, "content", "")
            thinking = getattr(message, "thinking", "")

        errors: list[str] = []
        for source_name, raw in (("content", content), ("thinking", thinking)):
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                return json_from_text(raw)
            except (TypeError, ValueError) as error:
                errors.append(f"{source_name}: {error}")
        if errors:
            raise ValueError("; ".join(errors))
        raise ValueError("The model returned neither content nor thinking text.")
