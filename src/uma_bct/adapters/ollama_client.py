"""Structured requests to a privacy-safe Ollama endpoint.

``normalize_ollama_host`` is the privacy boundary for the whole application:
by default it only accepts a numeric loopback address, and a remote or
Docker-hostname endpoint requires an explicit environment-variable opt-in
(``UMA_BCT_ALLOW_REMOTE_OLLAMA`` / ``UMA_BCT_ALLOW_CONTAINER_OLLAMA``) plus
HTTPS for anything not loopback -- this is what keeps archive photos on the
local workstation by default. ``query`` streams the model's response instead
of waiting for one payload, both to print progress dots and to enforce a
4 MiB response cap mid-stream (``collect_stream_text``) before an
unbounded or misbehaving model response is ever fully buffered.
``parse_model_response`` tries both the ``content`` and ``thinking`` stream
channels for JSON, since some model configurations put the structured
response in one or the other.
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
MAX_JSON_CONTAINER_ATTEMPTS = 32
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
    printable = "".join(
        character if character.isprintable() else " " for character in value
    )
    return " ".join(printable.split())[:240]


def _balanced_surrounding_prefix(value: str) -> bool:
    """Allow balanced prose brackets while rejecting structural fragments."""

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


def _starts_with_json_primitive(value: str) -> bool:
    """Recognize complete or truncated primitive JSON at a wrapper boundary."""

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
        return bool(
            match
            and any(name.startswith(match.group(0)) for name in JSON_LITERAL_NAMES)
        )
    return not isinstance(decoded, (dict, list))


def _invalid_json_message(error: json.JSONDecodeError | RecursionError) -> str:
    location = (
        f" at line {error.lineno}, column {error.colno}"
        if isinstance(error, json.JSONDecodeError)
        else ""
    )
    return f"The model response contained invalid or incomplete JSON{location}."


def _decode_first_json_container(
    text: str,
) -> tuple[json.JSONDecoder, int, Any, int]:
    """Find one bounded, unambiguous object or array in surrounding prose."""

    decoder = json.JSONDecoder()
    first_error: json.JSONDecodeError | RecursionError | None = None
    attempts = 0
    for match in re.finditer(r"[\[{]", text):
        start = match.start()
        if not _balanced_surrounding_prefix(text[:start]):
            continue
        attempts += 1
        if attempts > MAX_JSON_CONTAINER_ATTEMPTS:
            raise ValueError("The model response contained ambiguous JSON fragments.")
        try:
            value, end = decoder.raw_decode(text, start)
        except (json.JSONDecodeError, RecursionError) as error:
            if first_error is None:
                first_error = error
            continue
        return decoder, start, value, end
    if first_error is not None:
        raise ValueError(_invalid_json_message(first_error)) from None
    raise ValueError("The model response did not contain a JSON object or array.")


def json_from_text(raw: str) -> Any:
    """Decode one structured value or a sequence of complete record objects."""

    if len(raw) > MAX_MODEL_RESPONSE_CHARACTERS:
        raise ValueError("The model response exceeded the 4 MiB safety limit.")
    text = raw.strip()
    if not text:
        raise ValueError("The model returned no text.")

    decoder, start, first, end = _decode_first_json_container(text)

    prefix = text[:start]
    if not _balanced_surrounding_prefix(prefix):
        raise ValueError("The model response contained ambiguous JSON fragments.")
    if _starts_with_json_primitive(prefix):
        raise ValueError("The model response contained an unrelated JSON value.")
    if not isinstance(first, (dict, list)):
        raise ValueError("The model response must contain an object or array.")

    values = [first]
    cursor = end
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
                raise ValueError(
                    "The model response used an illegal JSON record separator."
                )
        elif text[cursor] == "{":
            if cursor == separator_start:
                raise ValueError(
                    "The model response used an illegal JSON record separator."
                )
        else:
            suffix = text[cursor:]
            if any(character in JSON_CONTAINER_CHARACTERS for character in suffix):
                raise ValueError(
                    "The model response contained ambiguous JSON fragments."
                )
            if _starts_with_json_primitive(suffix):
                raise ValueError(
                    "The model response contained an unrelated or incomplete "
                    "JSON value."
                )
            break

        try:
            value, cursor = decoder.raw_decode(text, cursor)
        except (json.JSONDecodeError, RecursionError) as error:
            raise ValueError(_invalid_json_message(error)) from None
        if not isinstance(value, dict):
            raise ValueError("Adjacent JSON values must be complete record objects.")
        values.append(value)

    return first if len(values) == 1 else values


def collect_stream_text(stream: Any) -> tuple[str, str]:
    content: list[str] = []
    thinking: list[str] = []
    characters = 0
    for number, chunk in enumerate(stream, start=1):
        chunk_content = message_text(chunk, "content")
        chunk_thinking = message_text(chunk, "thinking")
        characters += len(chunk_content) + len(chunk_thinking)
        if characters > MAX_MODEL_RESPONSE_CHARACTERS:
            raise ValueError("The model response exceeded the 4 MiB safety limit.")
        content.append(chunk_content)
        thinking.append(chunk_thinking)
        if number % 20 == 0:
            print(".", end="", file=sys.stderr, flush=True)
    return "".join(content).strip(), "".join(thinking).strip()


def bounded_response_error(value: str) -> str:
    """Keep channel diagnostics short and free of response payload text."""

    printable = "".join(
        character if character.isprintable() else " " for character in value
    )
    message = " ".join(printable.split()) or "response interpretation failed"
    if len(message) <= MAX_MODEL_RESPONSE_ERROR_CHARACTERS:
        return message
    return f"{message[: MAX_MODEL_RESPONSE_ERROR_CHARACTERS - 3]}..."


def parse_model_response(
    content: str,
    thinking: str,
    *,
    validator: Callable[[Any], object] | None = None,
) -> Any:
    errors: list[str] = []
    for source, raw in (("content", content), ("thinking", thinking)):
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
                errors.append(
                    f"{source}: parsed JSON failed semantic validation "
                    f"({error.__class__.__name__})"
                )
                continue
        return value
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
        raise ValueError(
            f"{TIMEOUT_ENVIRONMENT_VARIABLE} must be a positive number."
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{TIMEOUT_ENVIRONMENT_VARIABLE} must be a positive number.")
    return value


def normalize_ollama_host(value: str) -> str:
    """Allow loopback by default and require explicit, HTTPS remote opt-in."""

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

    container = hostname == "ollama" and environment_flag(
        CONTAINER_OLLAMA_ENVIRONMENT_VARIABLE
    )
    remote = environment_flag(REMOTE_OLLAMA_ENVIRONMENT_VARIABLE)
    if not loopback and not container and not remote:
        raise ValueError(
            "OLLAMA_HOST must use numeric loopback by default. Remote processing "
            f"requires {REMOTE_OLLAMA_ENVIRONMENT_VARIABLE}=1."
        )
    if remote and not loopback and not container and parsed.scheme != "https":
        raise ValueError("Remote Ollama endpoints must use HTTPS.")
    host_text = (
        f"[{normalized_hostname}]"
        if ":" in normalized_hostname
        else normalized_hostname
    )
    return urlunsplit((parsed.scheme, f"{host_text}:{port}", "", "", ""))


@dataclass(frozen=True)
class OllamaVisionModel:
    model: str = DEFAULT_MODEL
    context_length: int = DEFAULT_CONTEXT_LENGTH
    timeout_seconds: float | None = None
    host: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("Ollama request timeout must be a positive number.")

    @property
    def effective_timeout(self) -> float:
        return self.timeout_seconds or timeout_from_environment()

    @property
    def effective_host(self) -> str:
        configured = self.host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        return normalize_ollama_host(configured)

    def _client(self) -> Any:
        try:
            import ollama
        except ImportError as error:
            raise RuntimeError(
                "The 'ollama' package is missing; reinstall UMA BCT."
            ) from error
        return ollama.Client(
            host=self.effective_host,
            timeout=self.effective_timeout,
            follow_redirects=False,
            trust_env=False,
        )

    def _inventory(self) -> tuple[str, dict[str, str]]:
        try:
            host = self.effective_host
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        try:
            with closing(self._client()) as client:
                response = client.list()
        except Exception as error:
            raise RuntimeError(
                f"Cannot connect to Ollama at {host}: {error}"
            ) from error
        models = response_field(response, "models") or ()
        available = {
            str(response_field(item, "model") or ""): str(
                response_field(item, "digest") or ""
            )
            for item in models
        }
        return host, available

    def _metadata(self, host: str, available: dict[str, str]) -> dict[str, str]:
        if self.model not in available:
            raise RuntimeError(
                f"Ollama model '{self.model}' is not installed. "
                f"Run: ollama pull {self.model}"
            )
        return {
            "model": self.model,
            "model_digest": available[self.model],
            "ollama_host": host,
        }

    def preflight(self) -> dict[str, str]:
        return self._metadata(*self._inventory())

    def _pull_model(self, output: TextIO) -> None:
        try:
            with closing(self._client()) as client:
                updates = client.pull(model=self.model, stream=True)
                previous = ""
                for update in updates:
                    status = safe_progress_text(response_field(update, "status"))
                    if status and status != previous:
                        print(status, file=output)
                        previous = status
        except Exception as error:
            raise RuntimeError(
                f"Could not download Ollama model '{self.model}': {error}"
            ) from error

    def ensure_model(self, *, progress_stream: TextIO | None = None) -> dict[str, str]:
        output = progress_stream or sys.stderr
        host, available = self._inventory()
        if self.model not in available:
            print(
                f"Downloading Ollama model '{self.model}'. This is a large download.",
                file=output,
            )
            self._pull_model(output)
            host, available = self._inventory()
        return self._metadata(host, available)

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
        *,
        validator: Callable[[Any], object] | None = None,
    ) -> Any:
        try:
            with closing(self._client()) as client:
                stream = client.chat(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [str(image_path)],
                        }
                    ],
                    format=schema,
                    think=DEFAULT_THINK,
                    options={
                        "temperature": DEFAULT_TEMPERATURE,
                        "num_ctx": self.context_length,
                    },
                    stream=True,
                )
                content, thinking = collect_stream_text(stream)
        except Exception as error:
            raise RuntimeError(
                f"Local Ollama request failed for model '{self.model}': {error}"
            ) from error
        return parse_model_response(content, thinking, validator=validator)
