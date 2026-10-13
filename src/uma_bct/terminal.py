"""Keep a small live terminal status and a readable final report."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

try:
    from rich.console import Console
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
except ImportError:
    # An older offline client runtime may have the new wheel but not Rich yet.
    Console = None
    Progress = None


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
    """Refresh one terminal display, or print milestones when output is captured."""

    def __init__(
        self,
        total: int | None,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        force_terminal: bool | None = None,
    ) -> None:
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.clock = clock
        self.started = clock()
        self.completed = 0
        self.current = "Preparing the run"
        self.photo_label = ""
        self.console = Console(file=self.stream, force_terminal=force_terminal) if Console is not None else None
        self._progress: Progress | None = None
        self._task_id: int | None = None

    def __enter__(self) -> TerminalProgress:
        # Rich refreshes one line on an interactive terminal. Captured output
        # gets milestones so logs do not fill with repeated progress frames.
        if self.console is not None and self.console.is_terminal and not self.console.is_dumb_terminal:
            try:
                columns = [SpinnerColumn(), TextColumn("{task.description}")]
                if self.total is not None:
                    columns.extend((BarColumn(bar_width=18), MofNCompleteColumn()))
                columns.append(TimeElapsedColumn())
                self._progress = Progress(
                    *columns, console=self.console, refresh_per_second=2, transient=True,
                    redirect_stdout=False, redirect_stderr=True,
                )
                self._task_id = self._progress.add_task(self.current, total=self.total)
                self._progress.start()
            except Exception:
                self.close()
                self._write("Live display unavailable; showing milestone updates.")
                self._write(
                    "Preparing UMA BCT" if self.total is None
                    else f"UMA BCT: {self.total} photo{'s' if self.total != 1 else ''} queued."
                )
        elif self.total is None:
            self._write("Preparing UMA BCT")
        else:
            self._write(f"UMA BCT: {self.total} photo{'s' if self.total != 1 else ''} queued.")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def _write(self, message: str) -> None:
        self.stream.write(f"{message}\n")
        self.stream.flush()

    def _show(self) -> None:
        description = f"{self.photo_label} | {self.current}" if self.photo_label else self.current
        if not self._update_live(description=description):
            self._write(description)

    def _update_live(self, **changes: object) -> bool:
        if self._progress is None or self._task_id is None:
            return False
        try:
            self._progress.update(self._task_id, refresh=True, **changes)
        except Exception:
            # A display failure must never turn a successful model result into
            # a failed photo; subsequent updates use plain stderr lines.
            self.close()
            self._write("Live display unavailable; showing milestone updates.")
            return False
        return True

    def elapsed_seconds(self) -> float:
        return max(0.0, self.clock() - self.started)

    def phase(self, message: str) -> None:
        self.current = terminal_text(message)
        self._show()

    def set_total(self, count: int) -> None:
        if self.total != count:
            self.total = count
            if not self._update_live(total=count):
                self._write(f"This batch contains {count} photo{'s' if count != 1 else ''}.")

    def reused(self, count: int) -> None:
        self.completed = count
        if not self._update_live(completed=count) and count:
            self._write(f"Resuming: {count} photo{'s' if count != 1 else ''} already finished.")

    def photo_started(self, index: int, path: Path) -> None:
        self.photo_label = f"Photo {index} of {self.total}: {terminal_text(path.name)}"
        self.current = "Detecting boxes"
        if not self._update_live(description=f"{self.photo_label} | {self.current}"):
            self._write(f"Processing {self.photo_label}")

    def photo_finished(self, index: int, status: str, boxes: int) -> None:
        self.completed += 1
        self.photo_label = ""
        self.current = f"Finished {self.completed} of {self.total} photos"
        if not self._update_live(advance=1, description=self.current):
            self._write(f"Photo {index} finished: {status} | {boxes} box{'es' if boxes != 1 else ''} found")

    def close(self) -> None:
        progress = self._progress
        self._progress = None
        self._task_id = None
        if progress is not None:
            try:
                progress.stop()
            except Exception:
                self._write("Warning: could not cleanly stop the live display.")


def human_summary(summary: dict[str, Any], *, elapsed_seconds: float, attempted: int | None = None) -> str:
    """Describe the published outcome using the same counts as the run summary."""

    status = str(summary.get("status", "failed"))
    label = {
        "complete": "Finished",
        "partial": "Finished with issues",
        "failed": "Run failed",
        "interrupted": "Run interrupted",
    }.get(status, terminal_text(status).capitalize())
    if status == "partial" and not (summary.get("catalogue_output") or summary.get("catalogue_outputs")):
        label = "Run incomplete"
    if summary.get("already_complete"):
        condition = " with issues" if status != "complete" else ""
        label = f"Already complete{condition} - no photos reprocessed"

    total = int(summary.get("images_total", 0))
    complete = int(summary.get("images_complete", 0))
    partial = int(summary.get("images_partial", 0))
    failed = int(summary.get("images_failed", 0))
    # During interruption the saved summary can include photos not attempted
    # in this launch. Keep the displayed failed count tied to actual attempts.
    if attempted is not None:
        failed = min(failed, max(0, attempted - complete - partial))
    processed = complete + partial + failed

    lines = ["", label, f"Photos finished: {processed} of {total}"]
    if processed:
        lines.append(f"Photos: {complete} complete | {partial} partial | {failed} failed")
    catalogue = summary.get("catalogue_output")
    catalogues = summary.get("catalogue_outputs") or ([catalogue] if catalogue else [])
    rows = int(summary.get("catalogue_records", 0))
    if catalogues:
        label = "Catalogues" if len(catalogues) > 1 else "Catalogue"
        count_label = "recorded rows" if summary.get("already_complete") else "rows"
        lines.append(f"{label}: {rows} {count_label}")
    else:
        lines.append("Catalogue: not created")
    if summary.get("already_complete") and "current_input_records" in summary:
        lines.append(f"Transcribed boxes in current photos: {summary['current_input_records']}")
    lines.append(f"Time this launch: {elapsed_text(elapsed_seconds)}")
    batches = summary.get("batch_directories") or ([summary["batch_directory"]] if summary.get("batch_directory") else [])
    if batches:
        lines.extend(f"Results: {terminal_text(batch)}" for batch in batches)
    elif catalogue:
        lines.append(f"Results: {terminal_text(catalogue)}")
    if not catalogue:
        lines.extend(f"Catalogue file: {terminal_text(path)}" for path in catalogues)
    manifests = summary.get("run_manifests") or ([summary["run_manifest"]] if summary.get("run_manifest") else [])
    for details in manifests:
        kind = "Progress" if Path(details).name == "Run Progress.json" else "Run details"
        lines.append(f"{kind}: {terminal_text(details)}")
    recovery_paths = summary.get("recovery_paths") or ([summary["recovery_path"]] if summary.get("recovery_path") else [])
    lines.extend(f"Recovery file: {terminal_text(path)}" for path in recovery_paths)
    return "\n".join(lines)


def print_run_summary(
    summary: dict[str, Any], *, human: bool, elapsed_seconds: float, attempted: int | None = None
) -> None:
    """Use readable stdout only when requested; preserve the JSON contract."""

    if human:
        print(human_summary(summary, elapsed_seconds=elapsed_seconds, attempted=attempted))
    else:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
