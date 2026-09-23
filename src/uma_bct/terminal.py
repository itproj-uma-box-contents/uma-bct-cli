"""Show calm, readable progress without changing the CLI's JSON default."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO


def terminal_text(value: object) -> str:
    """Keep filenames and paths from moving the cursor or adding fake lines."""

    return "".join(character if character.isprintable() else "?" for character in str(value))


def elapsed_text(seconds: float) -> str:
    """Use a short duration that remains readable during a long run."""

    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s" if hours else f"{minutes}m {seconds}s"


class TerminalProgress:
    """Print milestones and a 30-second heartbeat while model calls are busy."""

    def __init__(
        self,
        total: int | None,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        heartbeat_seconds: float = 30,
    ) -> None:
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.clock = clock
        self.heartbeat_seconds = heartbeat_seconds
        self.started = clock()
        self.completed = 0
        self.current = "Preparing the run"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # Start one background timer so a slow model request never looks stuck.
    def __enter__(self) -> TerminalProgress:
        if self.total is None:
            self._write("Preparing UMA BCT...")
        else:
            self._write(f"UMA BCT: {self.total} photo{'s' if self.total != 1 else ''} queued.")
        self._thread = threading.Thread(target=self._heartbeat_loop, name="uma-bct-progress", daemon=True)
        self._thread.start()
        return self

    # Stop the timer before the final report or any exception leaves this run.
    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def _write(self, message: str) -> None:
        with self._lock:
            self._write_locked(message)

    def _write_locked(self, message: str) -> None:
        self.stream.write(f"{message}\n")
        self.stream.flush()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            self.heartbeat()

    def heartbeat(self) -> None:
        """Give a time and completed count even when the model is still busy."""

        with self._lock:
            finished = f" | {self.completed} of {self.total} finished" if self.total is not None else ""
            self._write_locked(
                f"Still working: {self.current}{finished} | {elapsed_text(self.elapsed_seconds())} elapsed"
            )

    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started)

    def phase(self, message: str) -> None:
        with self._lock:
            self.current = terminal_text(message)
            self._write_locked(self.current)

    def reused(self, count: int) -> None:
        with self._lock:
            self.completed = count
            if count:
                self._write_locked(f"Resuming: {count} photo{'s' if count != 1 else ''} already finished.")

    def photo_started(self, index: int, path: Path) -> None:
        with self._lock:
            self.current = f"photo {index} of {self.total}: {terminal_text(path.name)}"
            self._write_locked(f"Processing photo {index} of {self.total}: {terminal_text(path.name)}")

    def photo_finished(self, index: int, status: str, boxes: int) -> None:
        with self._lock:
            self.completed += 1
            self.current = "Preparing the next photo"
            self._write_locked(
                f"Photo {index} finished: {status} | {boxes} box{'es' if boxes != 1 else ''} found | "
                f"{self.completed} of {self.total} finished"
            )

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def human_summary(summary: dict[str, Any], *, elapsed_seconds: float, attempted: int | None = None) -> str:
    """Describe the published outcome using the same counts as the run summary."""

    status = str(summary.get("status", "failed"))
    label = {
        "complete": "Finished",
        "partial": "Finished with issues",
        "failed": "Run failed",
        "interrupted": "Run interrupted",
    }.get(status, terminal_text(status).capitalize())
    if status == "partial" and not summary.get("catalogue_output"):
        label = "Run incomplete"
    if summary.get("already_complete"):
        condition = " with issues" if status != "complete" else ""
        label = f"Already complete{condition} - no photos reprocessed"

    total = int(summary.get("images_total", 0))
    complete = int(summary.get("images_complete", 0))
    partial = int(summary.get("images_partial", 0))
    failed = int(summary.get("images_failed", 0))
    if attempted is not None:
        failed = min(failed, max(0, attempted - complete - partial))
    processed = complete + partial + failed

    lines = ["", label, f"Photos finished: {processed} of {total}"]
    if processed:
        lines.append(f"Photos: {complete} complete | {partial} partial | {failed} failed")
    catalogue = summary.get("catalogue_output")
    rows = int(summary.get("catalogue_records", 0))
    lines.append(f"Catalogue: {rows} rows" if catalogue else "Catalogue: not created")
    lines.append(f"Time this launch: {elapsed_text(elapsed_seconds)}")
    if batch := summary.get("batch_directory"):
        lines.append(f"Results: {terminal_text(batch)}")
    elif catalogue:
        lines.append(f"Results: {terminal_text(catalogue)}")
    if details := summary.get("run_manifest"):
        kind = "Progress" if Path(details).name == "Run Progress.json" else "Run details"
        lines.append(f"{kind}: {terminal_text(details)}")
    return "\n".join(lines)


def print_run_summary(
    summary: dict[str, Any], *, human: bool, elapsed_seconds: float, attempted: int | None = None
) -> None:
    """Use readable stdout only when requested; preserve the JSON contract."""

    if human:
        print(human_summary(summary, elapsed_seconds=elapsed_seconds, attempted=attempted))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
