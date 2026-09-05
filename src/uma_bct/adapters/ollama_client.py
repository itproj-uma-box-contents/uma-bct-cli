"""Structured requests to a privacy-safe Ollama endpoint."""

from __future__ import annotations

import json
import math
import os
import re
import sys
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
ANSI_CONTROL_SEQUENCE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


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


def json_from_text(raw: str) -> Any:
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


def parse_model_response(content: str, thinking: str) -> Any:
    errors: list[str] = []
    for source, raw in (("content", content), ("thinking", thinking)):
        if not raw.strip():
            continue
        try:
            return json_from_text(raw)
        except ValueError as error:
            errors.append(f"{source}: {error}")
    if errors:
        raise ValueError("; ".join(errors))
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
    if parsed.username or parsed.password:
        raise ValueError("OLLAMA_HOST must not contain credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("OLLAMA_HOST must not contain a path, query, or fragment.")
    port = parsed.port or 11434
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

    def query(self, image_path: Path, prompt: str, schema: dict[str, Any]) -> Any:
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
        return parse_model_response(content, thinking)
