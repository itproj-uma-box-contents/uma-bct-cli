"""Ask the local operating system to stay awake during active CLI work."""

from __future__ import annotations

import ctypes
import selectors
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path


def stop_helper(process: subprocess.Popen[str]) -> None:
    """Release an inhibitor by stopping only the helper this run started."""

    if process.stdin is not None:
        process.stdin.close()
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

    # Keep both requests owned by this process, so macOS releases them if it exits.
    def cf_string(value: bytes) -> int:
        reference = core.CFStringCreateWithCString(None, value, 0x08000100)
        if not reference:
            raise OSError("macOS could not create a power-assertion label")
        return reference

    reason = cf_string(b"UMA BCT is processing archive photos")
    assertions: list[int] = []
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


def start_linux_awake() -> Callable[[], None]:
    """Keep a systemd idle inhibitor until our pipe closes."""

    executable = next(
        (str(path) for path in (Path("/usr/bin/systemd-inhibit"), Path("/bin/systemd-inhibit")) if path.is_file()),
        None,
    )
    if executable is None:
        raise OSError("systemd-inhibit is unavailable")
    # The small child announces readiness only after systemd grants the lock.
    helper = "import sys; print('ready', flush=True); sys.stdin.buffer.read()"
    process = subprocess.Popen(
        [
            executable,
            "--what=idle:sleep",
            "--mode=block",
            "--who=UMA BCT",
            "--why=Processing archive box photos",
            sys.executable,
            "-u",
            "-c",
            helper,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        assert process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=5) or process.stdout.readline().strip() != "ready":
                raise OSError("systemd-inhibit could not acquire an idle lock")
        process.stdout.close()
        return lambda: stop_helper(process)
    except BaseException:
        stop_helper(process)
        raise


def start_windows_awake() -> Callable[[], None]:
    """Set and clear Windows idle requests on this same CLI thread."""

    execution_state = ctypes.windll.kernel32.SetThreadExecutionState
    execution_state.argtypes = [ctypes.c_uint]
    execution_state.restype = ctypes.c_uint
    continuous = 0x80000000
    system_required = 0x00000001
    display_required = 0x00000002
    if not execution_state(continuous | system_required | display_required):
        raise OSError("Windows did not accept the idle-sleep request")

    def release() -> None:
        if not execution_state(continuous):
            raise OSError("Windows did not release the idle-sleep request")

    return release


@contextmanager
def keep_awake() -> Iterator[None]:
    """Continue safely if the host cannot provide an idle-sleep lock."""

    release: Callable[[], None] | None = None
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
        print(f"Warning: could not keep this computer awake: {error}", file=sys.stderr)
    else:
        print("Idle-sleep protection request accepted for this run.", file=sys.stderr)
    try:
        yield
    finally:
        if release is not None:
            try:
                release()
            except Exception as error:
                print(f"Warning: could not release the idle-sleep request: {error}", file=sys.stderr)
