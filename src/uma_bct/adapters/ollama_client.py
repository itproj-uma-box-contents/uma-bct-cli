"""Send safe, structured requests to Ollama.

Local addresses are used by default so photos stay on the computer. Responses
are streamed, capped at 4 MiB, and checked in both content and thinking.
Passing ``logprobs=True`` asks the server for each token's log-probability;
those pairs pass through unchanged as ``ModelResponse.token_logprobs`` for a
caller to score, matched to whichever of content or thinking actually won.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit, urlunsplit

from ..ports import ModelResponse

DEFAULT_MODEL = "qwen3-vl:8b-instruct-q4_K_M"
DEFAULT_CONTEXT_LENGTH = 16384
DEFAULT_TEMPERATURE = 0
DEFAULT_THINK = False
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
TIMEOUT_ENVIRONMENT_VARIABLE = "UMA_BCT_OLLAMA_TIMEOUT_SECONDS"
REMOTE_OLLAMA_ENVIRONMENT_VARIABLE = "UMA_BCT_ALLOW_REMOTE_OLLAMA"
CONTAINER_OLLAMA_ENVIRONMENT_VARIABLE = "UMA_BCT_ALLOW_CONTAINER_OLLAMA"
MAX_MODEL_RESPONSE_CHARACTERS = 4 * 1024 * 1024
MAX_MODEL_RESPONSE_ERROR_CHARACTERS = 512
ANSI_CONTROL_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
JSON_CONTAINER_CHARACTERS = frozenset("{}[]")
JSON_LITERAL_NAMES = ("false", "null", "true")


def environment_flag(name: str) -> bool:
    return os.getenv(name, "").casefold() in {"1", "true", "yes"}


def response_field(value: Any, field: str) -> Any:
    return value.get(field) if isinstance(value, dict) else getattr(value, field, None)


def message_text(chunk: Any, field: str) -> str:
    message = response_field(chunk, "message")
    value = response_field(message, field)
    return value if isinstance(value, str) else ""


def safe_progress_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = ANSI_CONTROL_SEQUENCE.sub("", value)
    printable = "".join(character if character.isprintable() else " " for character in value)
    return " ".join(printable.split())[:240]


def prefix_is_safe(value: str) -> bool:
    """Check that text before JSON has no broken brackets."""

    depth = 0
    for character in value:
        if character in "{}":
            return False
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def starts_with_simple_json(value: str) -> bool:
    """Check whether text starts with a simple JSON value."""

    candidate = value.lstrip()
    if not candidate:
        return False
    try:
        decoded, _ = json.JSONDecoder().raw_decode(candidate)
    except (json.JSONDecodeError, RecursionError):
        if candidate.startswith('"'):
            return True
        if candidate == "-":
            return True
        match = re.match(r"[A-Za-z]+", candidate)
        return bool(match and any(name.startswith(match.group(0)) for name in JSON_LITERAL_NAMES))
    return not isinstance(decoded, (dict, list))


def invalid_json_message(error: json.JSONDecodeError | RecursionError) -> str:
    location = f" at line {error.lineno}, column {error.colno}" if isinstance(error, json.JSONDecodeError) else ""
    return f"The model response contained invalid or incomplete JSON{location}."


def json_from_text(raw: str) -> Any:
    """Read one JSON result or a list of complete records."""

    if len(raw) > MAX_MODEL_RESPONSE_CHARACTERS:
        raise ValueError("The model response exceeded the 4 MiB safety limit.")
    text = raw.strip()
    if not text:
        raise ValueError("The model returned no text.")

    # Start at the first JSON shape so broken fragments cannot be skipped.
    starts = tuple(position for character in ("{", "[") if (position := text.find(character)) >= 0)
    if not starts:
        raise ValueError("The model response did not contain a JSON object or array.")
    start = min(starts)
    prefix = text[:start]
    if not prefix_is_safe(prefix):
        raise ValueError("The model response contained ambiguous JSON fragments.")
    if starts_with_simple_json(prefix):
        raise ValueError("The model response contained an unrelated JSON value.")

    decoder = json.JSONDecoder()
    try:
        first, end = decoder.raw_decode(text, start)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError(invalid_json_message(error)) from None
    if not isinstance(first, (dict, list)):
        raise ValueError("The model response must contain an object or array.")

    values = [first]
    cursor = end
    # Extra values are allowed only when they are complete record objects.
    while True:
        separator_start = cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break

        if text[cursor] == ",":
            cursor += 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor >= len(text) or text[cursor] != "{":
                raise ValueError("The model response used an illegal JSON record separator.")
        elif text[cursor] == "{":
            if cursor == separator_start:
                raise ValueError("The model response used an illegal JSON record separator.")
        else:
            suffix = text[cursor:]
            if any(character in JSON_CONTAINER_CHARACTERS for character in suffix):
                raise ValueError("The model response contained ambiguous JSON fragments.")
            if starts_with_simple_json(suffix):
                raise ValueError("The model response contained an unrelated or incomplete JSON value.")
            break

        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError(invalid_json_message(error)) from None
        if not isinstance(value, dict):
            raise ValueError("Adjacent JSON values must be complete record objects.")
        values.append(value)

    return first if len(values) == 1 else values


def chunk_token_logprobs(chunk: Any) -> list[tuple[str, float]]:
    """Read the (token, logprob) pairs behind one streamed chunk, if any."""

    pairs: list[tuple[str, float]] = []
    for entry in response_field(chunk, "logprobs") or ():
        token = response_field(entry, "token")
        logprob = response_field(entry, "logprob")
        if isinstance(token, str) and isinstance(logprob, (int, float)):
            pairs.append((token, float(logprob)))
    return pairs


def collect_stream_text(stream: Any) -> tuple[str, str, tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]]:
    content: list[str] = []
    thinking: list[str] = []
    content_logprobs: list[tuple[str, float]] = []
    thinking_logprobs: list[tuple[str, float]] = []
    characters = 0
    for number, chunk in enumerate(stream, start=1):
        chunk_content = message_text(chunk, "content")
        chunk_thinking = message_text(chunk, "thinking")
        characters += len(chunk_content) + len(chunk_thinking)
        if characters > MAX_MODEL_RESPONSE_CHARACTERS:
            raise ValueError("The model response exceeded the 4 MiB safety limit.")
        content.append(chunk_content)
        thinking.append(chunk_thinking)
        # A chunk belongs to whichever channel it carried text for.
        if chunk_content:
            content_logprobs.extend(chunk_token_logprobs(chunk))
        elif chunk_thinking:
            thinking_logprobs.extend(chunk_token_logprobs(chunk))
        if number % 20 == 0:
            print(".", end="", file=sys.stderr, flush=True)
    return ("".join(content).strip(), "".join(thinking).strip(), tuple(content_logprobs), tuple(thinking_logprobs))


def bounded_response_error(value: str) -> str:
    """Keep response errors short and hide the model's text."""

    printable = "".join(character if character.isprintable() else " " for character in value)
    message = " ".join(printable.split()) or "response interpretation failed"
    if len(message) <= MAX_MODEL_RESPONSE_ERROR_CHARACTERS:
        return message
    return f"{message[: MAX_MODEL_RESPONSE_ERROR_CHARACTERS - 3]}..."


def parse_model_response(
    content: str,
    thinking: str,
    *,
    content_logprobs: tuple[tuple[str, float], ...] = (),
    thinking_logprobs: tuple[tuple[str, float], ...] = (),
    validator: Callable[[Any], object] | None = None,
) -> tuple[Any, tuple[tuple[str, float], ...]]:
    errors: list[str] = []
    # Use content first, then thinking if parsing or validation fails. The
    # logprobs returned always match whichever channel actually won.
    sources = (("content", content, content_logprobs), ("thinking", thinking, thinking_logprobs))
    for source, raw, logprobs in sources:
        if not raw.strip():
            continue
        try:
            value = json_from_text(raw)
        except ValueError as error:
            errors.append(f"{source}: {bounded_response_error(str(error))}")
            continue
        if validator is not None:
            try:
                validator(value)
            except (OverflowError, TypeError, ValueError) as error:
                errors.append(f"{source}: parsed JSON failed semantic validation ({error.__class__.__name__})")
                continue
        return value, logprobs
    if errors:
        raise ValueError(bounded_response_error("; ".join(errors)))
    raise ValueError("The model returned neither content nor thinking text.")


def timeout_from_environment() -> float:
    raw = os.getenv(TIMEOUT_ENVIRONMENT_VARIABLE)
    if raw is None:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{TIMEOUT_ENVIRONMENT_VARIABLE} must be a positive number.") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{TIMEOUT_ENVIRONMENT_VARIABLE} must be a positive number.")
    return value


def normalize_ollama_host(value: str) -> str:
    """Allow local Ollama by default and require opt-in for remote hosts."""

    configured = value if "://" in value else f"http://{value}"
    parsed = urlsplit(configured)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OLLAMA_HOST must be an HTTP(S) URL with a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OLLAMA_HOST must not contain credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("OLLAMA_HOST must not contain a path, query, or fragment.")
    port = parsed.port
    if port == 0:
        raise ValueError("OLLAMA_HOST must not use port 0.")
    if port is None:
        port = 11434
    hostname = parsed.hostname.casefold()
    normalized_hostname = "127.0.0.1" if hostname == "localhost" else hostname
    try:
        loopback = ip_address(normalized_hostname).is_loopback
    except ValueError:
        loopback = False

    container = hostname == "ollama" and environment_flag(CONTAINER_OLLAMA_ENVIRONMENT_VARIABLE)
    remote = environment_flag(REMOTE_OLLAMA_ENVIRONMENT_VARIABLE)
    # Remote hosts need an explicit opt-in because image pixels leave this computer.
    if not loopback and not container and not remote:
        raise ValueError(
            "OLLAMA_HOST must use numeric loopback by default. Remote processing "
            f"requires {REMOTE_OLLAMA_ENVIRONMENT_VARIABLE}=1."
        )
    if remote and not loopback and not container and parsed.scheme != "https":
        raise ValueError("Remote Ollama endpoints must use HTTPS.")
    host_text = f"[{normalized_hostname}]" if ":" in normalized_hostname else normalized_hostname
    return urlunsplit((parsed.scheme, f"{host_text}:{port}", "", "", ""))


@dataclass(frozen=True)
class OllamaVisionModel:
    model: str = DEFAULT_MODEL
    context_length: int = DEFAULT_CONTEXT_LENGTH
    timeout_seconds: float | None = None
    host: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError("Ollama request timeout must be a positive number.")

    @property
    def effective_timeout(self) -> float:
        return self.timeout_seconds or timeout_from_environment()

    @property
    def effective_host(self) -> str:
        configured = self.host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        return normalize_ollama_host(configured)

    def client(self) -> Any:
        try:
            import ollama
        except ImportError as error:
            raise RuntimeError("The 'ollama' package is missing; reinstall UMA BCT.") from error
        return ollama.Client(
            host=self.effective_host, timeout=self.effective_timeout, follow_redirects=False, trust_env=False
        )

    def inventory(self) -> tuple[str, dict[str, str]]:
        try:
            host = self.effective_host
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        try:
            with closing(self.client()) as client:
                response = client.list()
        except Exception as error:
            raise RuntimeError(f"Cannot connect to Ollama at {host}: {error}") from error
        models = response_field(response, "models") or ()
        available = {str(response_field(item, "model") or ""): str(response_field(item, "digest") or "") for item in models}
        return host, available

    def metadata(self, host: str, available: dict[str, str]) -> dict[str, str]:
        if self.model not in available:
            raise RuntimeError(f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}")
        return {"model": self.model, "model_digest": available[self.model], "ollama_host": host}

    def preflight(self) -> dict[str, str]:
        metadata = self.metadata(*self.inventory())
        self.probe()
        return metadata

    def probe(self) -> None:
        """Load the model with the production context before image work starts."""

        try:
            with closing(self.client()) as client:
                client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": "Reply briefly."}],
                    think=DEFAULT_THINK,
                    options={"temperature": DEFAULT_TEMPERATURE, "num_ctx": self.context_length, "num_predict": 1},
                    stream=False,
                )
        except Exception as error:
            raise RuntimeError(f"Ollama inference preflight failed for model '{self.model}': {error}") from error

    def pull_model(self, output: TextIO) -> None:
        try:
            with closing(self.client()) as client:
                updates = client.pull(model=self.model, stream=True)
                previous = ""
                for update in updates:
                    status = safe_progress_text(response_field(update, "status"))
                    if status and status != previous:
                        print(status, file=output)
                        previous = status
        except Exception as error:
            raise RuntimeError(f"Could not download Ollama model '{self.model}': {error}") from error

    def ensure_model(self, *, progress_stream: TextIO | None = None) -> dict[str, str]:
        output = progress_stream or sys.stderr
        host, available = self.inventory()
        if self.model not in available:
            print(f"Downloading Ollama model '{self.model}'. This is a large download.", file=output)
            self.pull_model(output)
            host, available = self.inventory()
        metadata = self.metadata(host, available)
        self.probe()
        return metadata

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
        *,
        validator: Callable[[Any], object] | None = None,
        logprobs: bool = False,
    ) -> ModelResponse:
        try:
            with closing(self.client()) as client:
                stream = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt, "images": [str(image_path)]}],
                    format=schema,
                    think=DEFAULT_THINK,
                    options={"temperature": DEFAULT_TEMPERATURE, "num_ctx": self.context_length},
                    stream=True,
                    logprobs=logprobs,
                )
                content, thinking, content_logprobs, thinking_logprobs = collect_stream_text(stream)
        except Exception as error:
            raise RuntimeError(f"Local Ollama request failed for model '{self.model}': {error}") from error
        value, token_logprobs = parse_model_response(
            content, thinking, content_logprobs=content_logprobs, thinking_logprobs=thinking_logprobs, validator=validator
        )
        return ModelResponse(value=value, token_logprobs=token_logprobs)
