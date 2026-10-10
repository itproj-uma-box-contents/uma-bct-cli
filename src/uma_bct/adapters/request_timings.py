"""Opt-in private request timings for the temporary demo; no text or pixels logged."""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from collections.abc import Iterator
from typing import Any

TIMINGS_ENV = "UMA_BCT_DEMO_TIMINGS_FILE"
FIELDS = ("total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration")


class RequestTimings:
    """Observe final Ollama timing fields without consuming or changing responses."""

    def __init__(self, kind: str) -> None:
        self.path = os.getenv(TIMINGS_ENV)
        if self.path and os.name != "posix":
            print("Demo request timings support macOS/Linux only; inference outcome is unchanged.", file=sys.stderr)
            self.path = None
        self.kind = kind
        self.started = time.monotonic() if self.path else None
        self.fields: dict[str, int] = {}

    def observe(self, response: Any) -> None:
        if self.path:
            for field in FIELDS:
                value = response.get(field) if isinstance(response, dict) else getattr(response, field, None)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    self.fields[field] = value

    def stream(self, stream: Any) -> Iterator[Any]:
        for chunk in stream:
            self.observe(chunk)
            yield chunk

    def finish(self, success: bool) -> None:
        if not self.path or self.started is None:
            return
        finished = time.monotonic()
        value = {"request": self.kind, "success": success, "wall_seconds": finished - self.started,
                 "started_monotonic_seconds": self.started, "finished_monotonic_seconds": finished,
                 "ollama_nanoseconds_and_counts": self.fields}
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NONBLOCK
                                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
            with os.fdopen(descriptor, "a") as output:
                if not stat.S_ISREG(os.fstat(output.fileno()).st_mode):
                    raise OSError("Timing destination must be a regular file.")
                output.write(json.dumps(value) + "\n")
        except OSError:
            # Diagnostics must not alter a transcription's success/partial state.
            print("Demo request timings could not be written; inference outcome is unchanged.", file=sys.stderr)
