"""Send sanitized PNGs to selected HTTPS vision APIs through LiteLLM.

This adapter is not selected by the CLI yet. It accepts only the three direct
providers supported by this first hosted implementation: OpenAI, Anthropic,
and Gemini. The existing detector and cropper produce metadata-free PNGs.
"""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..model_config import HostedModelConfig
from ..ports import ModelResponse
from .ollama_client import DEFAULT_REQUEST_TIMEOUT_SECONDS, parse_model_response, response_field

PROVIDER_API_BASES = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _load_litellm() -> Any:
    try:
        import litellm
    except ImportError as error:
        raise RuntimeError("Hosted models require the optional 'hosted' package extra.") from error
    return litellm


def _supported_parameters(litellm: Any, model: str) -> set[str]:
    """Unknown models can still work through the JSON-only prompt path."""

    try:
        return set(litellm.get_supported_openai_params(model=model) or ())
    except Exception:
        return set()


def _token_logprobs(choice: Any, content: str) -> tuple[tuple[str, float], ...]:
    """Use probabilities only when they exactly cover the returned text."""

    logprobs = response_field(choice, "logprobs")
    entries = response_field(logprobs, "content")
    if not entries:
        return ()
    pairs: list[tuple[str, float]] = []
    for entry in entries:
        token = response_field(entry, "token")
        probability = response_field(entry, "logprob")
        if (
            not isinstance(token, str)
            or not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or not math.isfinite(probability)
            or probability > 0
        ):
            return ()
        pairs.append((token, float(probability)))
    return tuple(pairs) if "".join(token for token, _ in pairs) == content else ()


@dataclass(frozen=True)
class HostedVisionModel:
    config: HostedModelConfig = field(repr=False)
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.config.provider not in PROVIDER_API_BASES:
            raise ValueError("Hosted models currently support openai/, anthropic/, and gemini/.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Hosted request timeout must be a positive number.")

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
        *,
        validator: Callable[[Any], object] | None = None,
        logprobs: bool = False,
    ) -> ModelResponse:
        if image_path.suffix.casefold() != ".png":
            raise ValueError("Hosted model requests require a sanitized PNG image.")
        image_bytes = image_path.read_bytes()
        if not image_bytes.startswith(PNG_SIGNATURE):
            raise ValueError("Hosted model image is not a PNG.")
        image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        request_prompt = f"{prompt}\n\nReturn only one JSON object matching this schema: {schema_text}"
        litellm = _load_litellm()
        supported = _supported_parameters(litellm, self.config.model)
        arguments: dict[str, Any] = {
            "model": self.config.model,
            "api_key": self.config.api_key,
            "api_base": PROVIDER_API_BASES[self.config.provider],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "timeout": self.timeout_seconds,
            "stream": False,
            "num_retries": 0,
        }
        if "response_format" in supported:
            arguments["response_format"] = {"type": "json_object"}
        if logprobs and "logprobs" in supported:
            arguments["logprobs"] = True
        try:
            response = litellm.completion(**arguments)
        except Exception as error:
            # Provider exceptions may contain credentials, URLs, or reply text.
            raise RuntimeError(f"Hosted model request failed ({type(error).__name__}).") from None
        choices = response_field(response, "choices")
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ValueError("The hosted model returned an invalid response envelope.")
        choice = choices[0]
        if response_field(choice, "finish_reason") != "stop":
            raise ValueError("The hosted model did not finish its response.")
        content = response_field(response_field(choice, "message"), "content")
        if not isinstance(content, str):
            raise ValueError("The hosted model returned no text.")
        probabilities = _token_logprobs(choice, content) if logprobs else ()
        value, probabilities = parse_model_response(
            content, "", content_logprobs=probabilities, validator=validator
        )
        return ModelResponse(value=value, token_logprobs=probabilities)
