"""Send sanitized PNGs to selected HTTPS vision APIs through LiteLLM.

It accepts only the three direct providers supported by this first hosted
implementation: OpenAI, Anthropic, and Gemini. The existing detector and
cropper produce metadata-free PNGs.
"""

from __future__ import annotations

import base64
import json
import math
import random
import re
import time
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
TRANSIENT_HTTP_STATUSES = {500, 502, 503, 504}
MAX_SERVER_RETRIES = 2
HTTP_ERROR_HINTS = {
    400: "The provider rejected the request parameters.",
    401: "Check the API key.",
    403: "Check the API key restrictions and project access.",
    404: "Check the model name and its availability for this project.",
    429: "Check the project's request, token, and daily quotas in the provider console.",
    503: "The provider is temporarily unavailable or experiencing high demand; try again later.",
}


def _http_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if type(status) is not int:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if type(status) is int and 400 <= status <= 599 else None


def _logprobs_rejected(error: Exception) -> bool:
    """Inspect only to recognize an optional capability rejection; never log it."""

    return _http_status(error) == 400 and re.search(
        r"\blogprobs\b.{0,80}\bnot\s+(?:supported|enabled)\b"
        r"|\bunsupported\s+parameter\b.{0,80}\blogprobs\b",
        str(error), re.IGNORECASE,
    ) is not None


def _request_failure(error: Exception) -> RuntimeError:
    status = _http_status(error)
    detail = type(error).__name__
    if status is not None:
        detail += f"; HTTP {status}"
    hint = HTTP_ERROR_HINTS.get(status, "")
    message = f"Hosted model request failed ({detail})."
    return RuntimeError(f"{message} {hint}" if hint else message)


def _load_litellm() -> Any:
    try:
        import litellm
    except ImportError as error:
        raise RuntimeError("Hosted models require the optional 'hosted' package extra.") from error
    # LiteLLM otherwise prints its feedback banner to stdout on failures,
    # corrupting the CLI's machine-readable JSON output.
    litellm.suppress_debug_info = True
    return litellm


def _supported_parameters(litellm: Any, model: str) -> set[str]:
    """Unknown models can still work through the JSON-only prompt path."""

    try:
        return set(litellm.get_supported_openai_params(model=model) or ())
    except Exception:
        return set()


def _supports_schema(litellm: Any, model: str) -> bool:
    try:
        return litellm.supports_response_schema(model=model) is True
    except Exception:
        return False


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Use portable structure hints; local validators retain all limits."""

    def convert(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        converted = {key: convert(item) for key, item in value.items() if key not in {"minItems", "maxItems", "maxLength"}}
        if converted.get("type") == "object":
            converted["additionalProperties"] = False
        return converted

    return convert(schema)


def _can_use_strict_schema(value: Any) -> bool:
    """Strict provider schemas require every object property to be present."""

    if isinstance(value, dict):
        if value.get("type") == "object" and set(value.get("required", ())) != set(value.get("properties", {})):
            return False
        return all(_can_use_strict_schema(item) for item in value.values())
    if isinstance(value, list):
        return all(_can_use_strict_schema(item) for item in value)
    return True


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

    def preflight(self) -> dict[str, str]:
        """Check the optional client without sending a paid request."""

        _load_litellm()
        return {
            "model": self.config.model,
            "backend": "hosted",
            "provider": self.config.provider,
            "api_base": PROVIDER_API_BASES[self.config.provider],
        }

    def query(
        self,
        image_path: Path | None,
        prompt: str,
        schema: dict[str, Any],
        *,
        validator: Callable[[Any], object] | None = None,
        logprobs: bool = False,
    ) -> ModelResponse:
        image_url = None
        if image_path is not None:
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
        api_base = PROVIDER_API_BASES[self.config.provider]
        if self.config.provider == "gemini":
            # LiteLLM treats a supplied Gemini base as including its API version.
            # Keep the public metadata as the origin, but use the versioned base
            # so the SDK constructs /v1beta/models/<model>:generateContent.
            api_base += "/v1beta"
        content: list[dict[str, Any]] = [{"type": "text", "text": request_prompt}]
        if image_url is not None:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        arguments: dict[str, Any] = {
            "model": self.config.model,
            "api_key": self.config.api_key,
            "api_base": api_base,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "timeout": self.timeout_seconds,
            "stream": False,
            "num_retries": 0,
        }
        if "response_format" in supported:
            if _supports_schema(litellm, self.config.model) and _can_use_strict_schema(schema):
                arguments["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "archive_boxes", "strict": True, "schema": _strict_schema(schema)},
                }
            else:
                arguments["response_format"] = {"type": "json_object"}
        if logprobs and "logprobs" in supported:
            arguments["logprobs"] = True
        server_retries = 0
        while True:
            try:
                response = litellm.completion(**arguments)
                break
            except Exception as error:
                # SDK capability lists describe parameter translation, not
                # necessarily support by the selected model or account.
                if "logprobs" in arguments and _logprobs_rejected(error):
                    arguments.pop("logprobs")
                    continue
                if _http_status(error) in TRANSIENT_HTTP_STATUSES and server_retries < MAX_SERVER_RETRIES:
                    time.sleep(2 ** (server_retries + 1) + random.uniform(0, 1))
                    server_retries += 1
                    continue
                # Provider exceptions may contain credentials, URLs, or reply
                # text. Expose only a numeric status and our own fixed hint.
                raise _request_failure(error) from None
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
