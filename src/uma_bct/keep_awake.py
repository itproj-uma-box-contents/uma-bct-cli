"""Ask the local operating system to stay awake during active CLI work."""

from __future__ import annotations

import ctypes
import os
import selectors
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def stop_helper(process: subprocess.Popen[bytes]) -> None:
    """Release an inhibitor by stopping only the helper this run started."""

    if process.stdin is not None:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
    finally:
        if process.stdout is not None:
            process.stdout.close()


def start_mac_awake() -> Callable[[], None]:
    """Ask macOS for confirmed idle-system and display assertions."""

    core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    iokit = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
    core.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    core.CFStringCreateWithCString.restype = ctypes.c_void_p
    core.CFRelease.argtypes = [ctypes.c_void_p]
    iokit.IOPMAssertionCreateWithName.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)
    ]
    iokit.IOPMAssertionCreateWithName.restype = ctypes.c_int32
    iokit.IOPMAssertionRelease.argtypes = [ctypes.c_uint32]
    iokit.IOPMAssertionRelease.restype = ctypes.c_int32

    def cf_string(value: bytes) -> int:
        reference = core.CFStringCreateWithCString(None, value, 0x08000100)
        if not reference:
            raise OSError("macOS could not create a power-assertion label")
        return reference

    reason = cf_string(b"UMA BCT is processing archive photos")
    assertions: list[int] = []
    # Both macOS requests belong to this process. If the second fails, release
    # the first immediately instead of claiming sleep protection was granted.
    try:
        for name in (b"PreventUserIdleSystemSleep", b"PreventUserIdleDisplaySleep"):
            kind = cf_string(name)
            try:
                assertion = ctypes.c_uint32()
                status = iokit.IOPMAssertionCreateWithName(kind, 255, reason, ctypes.byref(assertion))
                if status != 0:
                    raise OSError(f"macOS rejected the idle-sleep request (code {status})")
                assertions.append(assertion.value)
            finally:
                core.CFRelease(kind)
    except BaseException:
        for assertion_id in reversed(assertions):
            iokit.IOPMAssertionRelease(assertion_id)
        raise
    finally:
        core.CFRelease(reason)

    def release() -> None:
        failures = [
            assertion_id for assertion_id in reversed(assertions)
            if iokit.IOPMAssertionRelease(assertion_id) != 0
        ]
        if failures:
            raise OSError("macOS could not release an idle-sleep request")

    return release


LINUX_READY_TIMEOUT = 5.0
DIAGNOSTIC_LIMIT = 8192


def diagnostic_text(value: object) -> str:
    """Keep OS diagnostics readable without terminal escape sequences."""

    return "".join(character if character.isprintable() or character == "\n" else " " for character in str(value))


def start_linux_awake() -> Callable[[], None]:
    """Hold a confirmed systemd inhibitor and report its real failure output."""

    executable = next(
        (str(path) for path in (Path("/usr/bin/systemd-inhibit"), Path("/bin/systemd-inhibit")) if path.is_file()),
        None,
    ) or shutil.which("systemd-inhibit")
    if executable is None:
        raise OSError("Linux acquisition: systemd-inhibit is unavailable in the system locations or PATH")
    helper = "import sys; print('ready', flush=True); sys.stdin.buffer.read()"
    try:
        process = subprocess.Popen(
            [executable, "--what=idle:sleep", "--mode=block", "--no-ask-password", "--who=UMA BCT",
             "--why=Processing archive box photos", sys.executable, "-u", "-c", helper],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except OSError as error:
        raise OSError(f"Linux acquisition: could not launch {executable}: {error}") from error

    assert process.stdout is not None and process.stderr is not None
    diagnostics = bytearray()
    diagnostics_lock = threading.Lock()
    stopping = threading.Event()

    def drain_errors() -> None:
        try:
            while chunk := os.read(process.stderr.fileno(), 4096):
                with diagnostics_lock:
                    diagnostics.extend(chunk)
                    del diagnostics[:-DIAGNOSTIC_LIMIT]
        except (OSError, ValueError):
            # Cleanup can close an already-ended helper's pipe.
            pass

    def failure(stage: str, reason: str) -> str:
        with diagnostics_lock:
            output = diagnostics.decode("utf-8", errors="replace").strip()
        code = process.poll()
        status = f"; exit code {code}" if code is not None else ""
        return f"Linux {stage}: {executable}: {reason}{status}" + (f"\nHelper stderr:\n{output}" if output else "")

    reader = threading.Thread(target=drain_errors, daemon=True, name="uma-bct-inhibitor-stderr")
    watcher: threading.Thread | None = None

    def release() -> None:
        stopping.set()
        try:
            stop_helper(process)
        finally:
            if reader.ident is not None:
                reader.join(timeout=2)
            process.stderr.close()
            if watcher is not None and watcher.ident is not None:
                watcher.join(timeout=2)

    try:
        reader.start()
        deadline = time.monotonic() + LINUX_READY_TIMEOUT
        response = bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while b"\n" not in response:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(timeout=max(0, remaining)):
                    raise OSError(f"no readiness confirmation within {LINUX_READY_TIMEOUT:g} seconds")
                chunk = os.read(process.stdout.fileno(), 1024)
                if not chunk:
                    reader.join(timeout=0.2)
                    raise OSError("helper exited before confirming the lock")
                response.extend(chunk)
                if len(response) > 1024:
                    raise OSError("invalid readiness response")
            if bytes(response) != b"ready\n" or process.poll() is not None:
                reader.join(timeout=0.2)
                raise OSError("helper did not confirm a live idle-sleep lock")
        process.stdout.close()

        def watch_exit() -> None:
            process.wait()
            reader.join(timeout=0.2)
            if not stopping.is_set():
                print("Warning: idle-sleep protection was lost: " + diagnostic_text(
                    failure("runtime", "helper exited unexpectedly; processing continues")
                ), file=sys.stderr)

        watcher = threading.Thread(target=watch_exit, daemon=True, name="uma-bct-inhibitor-watch")
        watcher.start()
        return release
    except BaseException as error:
        # Stop the owned process before collecting any final error output.
        cleanup_failure = None
        try:
            release()
        except Exception as cleanup_error:
            cleanup_failure = f"Linux cleanup also failed: {cleanup_error}"
        if isinstance(error, Exception):
            detail = failure("acquisition", str(error))
            if cleanup_failure:
                detail += f"\n{cleanup_failure}"
            raise OSError(detail) from error
        if cleanup_failure:
            print(f"Warning: {diagnostic_text(cleanup_failure)}", file=sys.stderr)
        raise


def start_windows_awake() -> Callable[[], None]:
    """Set and clear Windows idle requests on this same CLI thread."""

    execution_state = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
    execution_state.argtypes = [ctypes.c_uint]
    execution_state.restype = ctypes.c_uint
    continuous = 0x80000000
    system_required = 0x00000001
    display_required = 0x00000002

    def request(flags: int, stage: str) -> None:
        ctypes.set_last_error(0)
        if not execution_state(flags):
            code = ctypes.get_last_error()
            detail = f"Windows error {code}: {ctypes.FormatError(code).strip()}" if code else "Windows supplied no extended error"
            raise OSError(f"Windows {stage}: SetThreadExecutionState(0x{flags:08X}) returned 0; {detail}")

    request(continuous | system_required | display_required, "acquisition")

    def release() -> None:
        request(continuous, "release")

    return release


@contextmanager
def keep_awake() -> Iterator[None]:
    """Continue safely if the host cannot provide an idle-sleep lock."""

    release: Callable[[], None] | None = None
    # Power management is best effort. The transcription result and JSON
    # stdout must not change just because this local OS request failed.
    try:
        if sys.platform == "darwin":
            release = start_mac_awake()
        elif sys.platform == "linux":
            release = start_linux_awake()
        elif sys.platform == "win32":
            release = start_windows_awake()
        else:
            raise OSError("this operating system has no supported idle-sleep control")
    except Exception as error:
        print(f"Warning: could not keep this computer awake: {diagnostic_text(error)}", file=sys.stderr)
    else:
        print("Idle-sleep protection request accepted for this run.", file=sys.stderr)
    try:
        yield
    finally:
        if release is not None:
            try:
                release()
            except Exception as error:
                print(f"Warning: could not release the idle-sleep request: {diagnostic_text(error)}", file=sys.stderr)
