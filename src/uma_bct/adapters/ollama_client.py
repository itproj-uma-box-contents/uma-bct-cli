"""Structured requests to a locally installed vision model through Ollama."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from contextlib import closing
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
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
MINIMUM_OLLAMA_VERSION = (0, 12, 7)
TIMEOUT_ENVIRONMENT_VARIABLE = "UMA_BCT_OLLAMA_TIMEOUT_SECONDS"
REMOTE_OLLAMA_ENVIRONMENT_VARIABLE = "UMA_BCT_ALLOW_REMOTE_OLLAMA"
CONTAINER_OLLAMA_ENVIRONMENT_VARIABLE = "UMA_BCT_ALLOW_CONTAINER_OLLAMA"
MAX_MODEL_RESPONSE_CHARACTERS = 4 * 1024 * 1024
MAX_MODEL_STREAM_CHUNKS = 100_000
MAX_MODEL_PULL_PROGRESS_LINES = 1_000
MAX_MODEL_PULL_STATUS_CHARACTERS = 240
ANSI_CONTROL_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def environment_flag(name: str) -> bool:
    """Return whether an explicit boolean privacy opt-in is enabled."""

    return os.getenv(name, "").casefold() in {"1", "true", "yes"}


def message_text(chunk: Any, field: str) -> str:
    """Read a streamed Ollama message field from object or dictionary responses."""

    message = (
        chunk.get("message")
        if isinstance(chunk, dict)
        else getattr(chunk, "message", None)
    )
    if isinstance(message, dict):
        value = message.get(field, "")
    else:
        value = getattr(message, field, "")
    return value if isinstance(value, str) else ""


def response_field(response: Any, field: str) -> Any:
    """Read a top-level Ollama response field from objects or dictionaries."""

    if isinstance(response, dict):
        return response.get(field)
    return getattr(response, field, None)


def safe_progress_text(value: Any) -> str:
    """Return bounded single-line progress text safe for a terminal."""

    if not isinstance(value, str):
        return ""
    without_ansi = ANSI_CONTROL_SEQUENCE.sub("", value)
    printable = "".join(
        character if character.isprintable() else " " for character in without_ansi
    )
    return " ".join(printable.split())[:MAX_MODEL_PULL_STATUS_CHARACTERS]


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
            raise ValueError(
                "The model response did not contain a JSON object."
            ) from None
        return json.loads(raw[start : end + 1])


def collect_stream_text(stream: Any) -> tuple[str, str]:
    """Collect bounded content and thinking text from an Ollama stream."""

    content_parts: list[str] = []
    thinking_parts: list[str] = []
    response_characters = 0
    for chunk_number, chunk in enumerate(stream, start=1):
        if chunk_number > MAX_MODEL_STREAM_CHUNKS:
            raise ValueError("The model response exceeded the stream-chunk limit.")
        chunk_content = message_text(chunk, "content")
        chunk_thinking = message_text(chunk, "thinking")
        response_characters += len(chunk_content) + len(chunk_thinking)
        if response_characters > MAX_MODEL_RESPONSE_CHARACTERS:
            raise ValueError("The model response exceeded the 4 MiB safety limit.")
        if chunk_content:
            content_parts.append(chunk_content)
        if chunk_thinking:
            thinking_parts.append(chunk_thinking)
        if chunk_number % 20 == 0:
            print(".", end="", file=sys.stderr)
            sys.stderr.flush()
    return "".join(content_parts).strip(), "".join(thinking_parts).strip()


def parse_model_response(content: str, thinking: str) -> Any:
    """Parse content first, then thinking, preserving both validation errors."""

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


def timeout_from_environment() -> float:
    """Return the positive per-request timeout configured for local inference."""

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
    """Validate a privacy-safe Ollama endpoint and return its normalized URL."""

    configured = value if "://" in value else f"http://{value}"
    parsed = urlsplit(configured)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("OLLAMA_HOST must be an HTTP(S) URL with a hostname.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("OLLAMA_HOST must not contain credentials.")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("OLLAMA_HOST must not contain a path, query, or fragment.")
    # Accessing .port also validates malformed/non-numeric port values.
    port = parsed.port or 11434

    hostname = parsed.hostname.casefold()
    normalized_hostname = hostname
    if hostname == "localhost":
        # Avoid hosts-file or DNS rebinding of this special local spelling.
        normalized_hostname = "127.0.0.1"
        is_loopback = True
    else:
        try:
            is_loopback = ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False

    container_allowed = hostname == "ollama" and environment_flag(
        CONTAINER_OLLAMA_ENVIRONMENT_VARIABLE
    )
    remote_allowed = environment_flag(REMOTE_OLLAMA_ENVIRONMENT_VARIABLE)
    if not is_loopback and not container_allowed and not remote_allowed:
        raise ValueError(
            "OLLAMA_HOST must resolve to this device by default. Set "
            f"{CONTAINER_OLLAMA_ENVIRONMENT_VARIABLE}=1 only for the bundled "
            "Compose service, or set "
            f"{REMOTE_OLLAMA_ENVIRONMENT_VARIABLE}=1 only for an approved managed "
            "remote endpoint; doing so sends archive image data off-device."
        )
    if remote_allowed and not is_loopback and not container_allowed:
        if parsed.scheme != "https":
            raise ValueError(
                "Approved remote Ollama endpoints must use HTTPS because archive "
                "image data is sent to that endpoint."
            )
    normalized_netloc = (
        f"[{normalized_hostname}]:{port}"
        if ":" in normalized_hostname
        else f"{normalized_hostname}:{port}"
    )
    return urlunsplit((parsed.scheme, normalized_netloc, "", "", ""))


def fetch_ollama_version(host: str, timeout: float) -> str:
    """Fetch the server version without proxies or redirect following."""

    parsed = urlsplit(host)
    connection_type = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    connection = connection_type(
        parsed.hostname,
        parsed.port,
        timeout=min(timeout, 10.0),
    )
    try:
        connection.request(
            "GET", "/api/version", headers={"Accept": "application/json"}
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(
                f"Ollama version endpoint returned HTTP {response.status}."
            )
        payload = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    if not isinstance(payload, dict):
        raise ValueError("Ollama version endpoint returned a non-object response.")
    return str(payload.get("version", ""))


def version_tuple(value: str) -> tuple[int, int, int]:
    """Extract a comparable semantic-version triple from Ollama's response."""

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"Could not understand Ollama server version: {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


@dataclass(frozen=True)
class OllamaVisionModel:
    """An offline vision-model client shared by Qwen pipeline adapters."""

    model: str = DEFAULT_MODEL
    context_length: int = DEFAULT_CONTEXT_LENGTH
    timeout_seconds: float | None = None
    host: str | None = None

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0
        ):
            raise ValueError("Ollama request timeout must be a positive finite number.")

    @property
    def effective_timeout(self) -> float:
        return (
            self.timeout_seconds
            if self.timeout_seconds is not None
            else timeout_from_environment()
        )

    @property
    def effective_host(self) -> str:
        configured = self.host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
        return normalize_ollama_host(configured)

    def _client(self) -> Any:
        try:
            import ollama
        except ImportError as error:
            raise RuntimeError(
                "The 'ollama' Python package is missing; run 'uv sync'."
            ) from error
        return ollama.Client(
            host=self.effective_host,
            timeout=self.effective_timeout,
            follow_redirects=False,
            trust_env=False,
        )

    def _server_inventory(self) -> tuple[str, str, dict[str, str]]:
        """Validate the endpoint and server, then return its installed models."""

        try:
            host = self.effective_host
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        try:
            with closing(self._client()) as client:
                response = client.list()
            available = {
                str(getattr(item, "model", "")): str(getattr(item, "digest", ""))
                for item in getattr(response, "models", ())
            }
            version = fetch_ollama_version(host, self.effective_timeout)
        except Exception as error:
            raise RuntimeError(
                "Cannot connect to the local Ollama server. Start Ollama and verify "
                f"OLLAMA_HOST ({host}): {error}"
            ) from error

        if version_tuple(version) < MINIMUM_OLLAMA_VERSION:
            minimum = ".".join(str(part) for part in MINIMUM_OLLAMA_VERSION)
            raise RuntimeError(
                f"Ollama {minimum} or newer is required; server reported {version}."
            )
        return host, version, available

    def _model_metadata(
        self,
        host: str,
        version: str,
        available: dict[str, str],
    ) -> dict[str, str]:
        if self.model not in available:
            raise RuntimeError(
                f"Ollama model '{self.model}' is not installed. Run: "
                f"ollama pull {self.model}"
            )
        return {
            "model": self.model,
            "model_digest": available[self.model],
            "ollama_version": version,
            "ollama_host": host,
        }

    def preflight(self) -> dict[str, str]:
        """Validate the local server and requested model before a batch starts."""

        return self._model_metadata(*self._server_inventory())

    def _pull_model(self, progress_stream: TextIO) -> None:
        """Download the configured model while keeping progress off stdout."""

        try:
            with closing(self._client()) as client:
                updates = client.pull(model=self.model, stream=True)
                last_progress = ""
                progress_lines = 0
                for update in updates:
                    status = safe_progress_text(response_field(update, "status"))
                    completed = response_field(update, "completed")
                    total = response_field(update, "total")
                    detail = status
                    if (
                        isinstance(completed, (int, float))
                        and not isinstance(completed, bool)
                        and isinstance(total, (int, float))
                        and not isinstance(total, bool)
                        and total > 0
                    ):
                        percentage = max(0, min(100, int(completed * 100 / total)))
                        detail = (
                            f"{detail}: {percentage}%" if detail else f"{percentage}%"
                        )
                    if detail and detail != last_progress:
                        if progress_lines < MAX_MODEL_PULL_PROGRESS_LINES:
                            print(detail, file=progress_stream)
                        elif progress_lines == MAX_MODEL_PULL_PROGRESS_LINES:
                            print(
                                "Further model-download progress is suppressed.",
                                file=progress_stream,
                            )
                        progress_lines += 1
                        last_progress = detail
        except Exception as error:
            raise RuntimeError(
                f"Could not download Ollama model '{self.model}': {error}. "
                f"Try: ollama pull {self.model}"
            ) from error

    def ensure_model(self, *, progress_stream: TextIO | None = None) -> dict[str, str]:
        """Validate Ollama and download the configured model only when absent."""

        output = progress_stream if progress_stream is not None else sys.stderr
        host, server_version, available = self._server_inventory()
        if self.model not in available:
            print(
                f"Downloading Ollama model '{self.model}'. This is a large, "
                "one-time download.",
                file=output,
            )
            self._pull_model(output)
            host, server_version, available = self._server_inventory()
            if self.model not in available:
                raise RuntimeError(
                    f"Ollama model '{self.model}' is still unavailable after the "
                    f"download. Try: ollama pull {self.model}"
                )
            print(f"Ollama model '{self.model}' is ready.", file=output)
        return self._model_metadata(host, server_version, available)

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
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
        except Exception as error:  # Ollama exposes transport-specific errors.
            raise RuntimeError(
                f"Local Ollama request failed for model '{self.model}': {error}"
            ) from error
        return parse_model_response(content, thinking)
