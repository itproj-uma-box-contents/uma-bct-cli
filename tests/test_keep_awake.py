import contextlib
import ctypes
import io
import os
import subprocess
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from uma_bct.keep_awake import keep_awake, start_linux_awake, start_mac_awake, start_windows_awake, stop_helper


class KeepAwakeTest(unittest.TestCase):
    def test_mac_assertions_are_confirmed_and_released(self):
        core = SimpleNamespace(CFStringCreateWithCString=Mock(side_effect=[11, 12, 13]), CFRelease=Mock())
        assertion_ids = iter((101, 102))

        def create(_kind, _level, _reason, target):
            ctypes.cast(target, ctypes.POINTER(ctypes.c_uint32)).contents.value = next(assertion_ids)
            return 0

        iokit = SimpleNamespace(IOPMAssertionCreateWithName=Mock(side_effect=create), IOPMAssertionRelease=Mock(return_value=0))
        with patch("uma_bct.keep_awake.ctypes.CDLL", side_effect=[core, iokit]):
            release = start_mac_awake()
            release()
        self.assertEqual(iokit.IOPMAssertionRelease.call_args_list, [call(102), call(101)])
        self.assertEqual(core.CFRelease.call_args_list, [call(12), call(13), call(11)])

        core.CFStringCreateWithCString.side_effect = [11, 12, 13]
        attempts = [0]

        def fail_second(_kind, _level, _reason, target):
            attempts[0] += 1
            if attempts[0] == 1:
                ctypes.cast(target, ctypes.POINTER(ctypes.c_uint32)).contents.value = 101
                return 0
            return 5

        iokit.IOPMAssertionCreateWithName = Mock(side_effect=fail_second)
        iokit.IOPMAssertionRelease.reset_mock()
        with patch("uma_bct.keep_awake.ctypes.CDLL", side_effect=[core, iokit]):
            with self.assertRaisesRegex(OSError, "code 5"):
                start_mac_awake()
        iokit.IOPMAssertionRelease.assert_called_once_with(101)

    def test_idle_request_lifecycle_and_unavailable_control(self):
        for platform, starter in (
            ("darwin", "start_mac_awake"),
            ("linux", "start_linux_awake"),
            ("win32", "start_windows_awake"),
        ):
            with self.subTest(platform=platform):
                release = Mock()
                with patch("uma_bct.keep_awake.sys.platform", platform), patch(
                    f"uma_bct.keep_awake.{starter}", return_value=release
                ) as start:
                    with keep_awake():
                        start.assert_called_once_with()
                        release.assert_not_called()
                release.assert_called_once_with()

        execution_state = Mock(return_value=1)
        kernel = SimpleNamespace(SetThreadExecutionState=execution_state)
        with windows_api(kernel):
            release = start_windows_awake()
            release()
        self.assertEqual(execution_state.call_args_list, [call(0x80000003), call(0x80000000)])

        warning = io.StringIO()
        with (
            patch("uma_bct.keep_awake.sys.platform", "linux"),
            patch("uma_bct.keep_awake.start_linux_awake", side_effect=OSError("no system bus")),
            contextlib.redirect_stderr(warning),
        ):
            with keep_awake():
                pass
        self.assertIn("could not keep this computer awake: no system bus", warning.getvalue())


@contextlib.contextmanager
def windows_api(kernel, *, error=0):
    with (
        patch("uma_bct.keep_awake.ctypes.WinDLL", return_value=kernel, create=True),
        patch("uma_bct.keep_awake.ctypes.set_last_error", create=True) as clear,
        patch("uma_bct.keep_awake.ctypes.get_last_error", return_value=error, create=True),
        patch("uma_bct.keep_awake.ctypes.FormatError", return_value="Access is denied.", create=True),
    ):
        yield clear


class WindowsDiagnosticsTest(unittest.TestCase):
    def test_failed_request_reports_only_fresh_native_error(self):
        for code, detail in ((5, "Windows error 5: Access is denied."), (0, "Windows supplied no extended error")):
            with self.subTest(code=code):
                kernel = SimpleNamespace(SetThreadExecutionState=Mock(return_value=0))
                with windows_api(kernel, error=code) as clear:
                    with self.assertRaisesRegex(OSError, detail) as raised:
                        start_windows_awake()
                clear.assert_called_once_with(0)
                self.assertIn("acquisition", str(raised.exception))
                self.assertIn("0x80000003", str(raised.exception))

    def test_release_failure_keeps_processing_result_and_json_stdout(self):
        kernel = SimpleNamespace(SetThreadExecutionState=Mock(side_effect=[1, 0]))
        stdout, stderr = io.StringIO(), io.StringIO()
        with (windows_api(kernel, error=5) as clear, patch("uma_bct.keep_awake.sys.platform", "win32"),
              contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
            with keep_awake():
                print('{"status": "complete"}')
        self.assertEqual(stdout.getvalue(), '{"status": "complete"}\n')
        self.assertIn("Windows release", stderr.getvalue())
        self.assertIn("Access is denied", stderr.getvalue())
        self.assertEqual(clear.call_args_list, [call(0), call(0)])

    def test_acquisition_failure_warns_and_runs_body(self):
        kernel = SimpleNamespace(SetThreadExecutionState=Mock(return_value=0))
        stderr = io.StringIO()
        with windows_api(kernel, error=5), patch("uma_bct.keep_awake.sys.platform", "win32"), contextlib.redirect_stderr(stderr):
            with keep_awake():
                completed = True
        self.assertTrue(completed)
        self.assertIn("Access is denied", stderr.getvalue())
        self.assertNotIn("request accepted", stderr.getvalue())


@unittest.skipIf(os.name == "nt", "systemd helper uses POSIX pipes")
class LinuxInhibitorTest(unittest.TestCase):
    @contextlib.contextmanager
    def helper(self, script):
        real_popen = subprocess.Popen
        processes = []

        def launch(command, **kwargs):
            self.assertEqual(command[0], "/fake/systemd-inhibit")
            self.assertIn("--what=idle:sleep", command)
            self.assertIn("--no-ask-password", command)
            process = real_popen([sys.executable, "-u", "-c", script], **kwargs)
            processes.append(process)
            return process

        with (patch("uma_bct.keep_awake.Path.is_file", return_value=False),
              patch("uma_bct.keep_awake.shutil.which", return_value="/fake/systemd-inhibit"),
              patch("uma_bct.keep_awake.subprocess.Popen", side_effect=launch)):
            try:
                yield processes
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=2)

    def test_confirmed_lock_uses_path_fallback_and_releases_cleanly(self):
        stderr = io.StringIO()
        with self.helper("import sys; print('ready', flush=True); sys.stdin.buffer.read()") as processes:
            with contextlib.redirect_stderr(stderr):
                release = start_linux_awake()
                process = processes[0]
                self.assertIsNone(process.poll())
                release()
            self.assertEqual(process.returncode, 0)
            for stream in (process.stdin, process.stdout, process.stderr):
                self.assertTrue(stream.closed)
        self.assertEqual(stderr.getvalue(), "")

    def test_denied_access_and_missing_bus_report_helper_output(self):
        for detail in ("Access denied by polkit", "Failed to connect to system bus: No such file or directory"):
            with self.subTest(detail=detail), self.helper(f"import sys; sys.stderr.write({detail!r}); sys.exit(7)") as processes:
                with self.assertRaises(OSError) as raised:
                    start_linux_awake()
                self.assertIn(detail, str(raised.exception))
                self.assertIn("exit code 7", str(raised.exception))
                self.assertIn("/fake/systemd-inhibit", str(raised.exception))
                self.assertIsNotNone(processes[0].poll())

    def test_partial_readiness_cannot_block_past_deadline(self):
        script = "import sys, time; sys.stdout.write('rea'); sys.stdout.flush(); time.sleep(20)"
        with self.helper(script) as processes, patch("uma_bct.keep_awake.LINUX_READY_TIMEOUT", 0.1):
            started = time.monotonic()
            with self.assertRaisesRegex(OSError, "no readiness confirmation within 0.1 seconds"):
                start_linux_awake()
            self.assertLess(time.monotonic() - started, 4)
            self.assertIsNotNone(processes[0].poll())

    def test_cleanup_failure_does_not_hide_acquisition_error_or_interruption(self):
        def failing_cleanup(process):
            stop_helper(process)
            raise PermissionError(13, "cleanup denied")

        script = "import sys; sys.stderr.write('Failed to connect to system bus'); sys.exit(7)"
        with self.helper(script), patch("uma_bct.keep_awake.stop_helper", side_effect=failing_cleanup):
            with self.assertRaises(OSError) as raised:
                start_linux_awake()
        self.assertIn("helper exited before confirming the lock", str(raised.exception))
        self.assertIn("Failed to connect to system bus", str(raised.exception))
        self.assertIn("exit code 7", str(raised.exception))
        self.assertIn("cleanup denied", str(raised.exception))

        stderr = io.StringIO()
        with (self.helper("import sys; sys.stdin.buffer.read()"),
              patch("uma_bct.keep_awake.selectors.DefaultSelector", side_effect=KeyboardInterrupt),
              patch("uma_bct.keep_awake.stop_helper", side_effect=failing_cleanup),
              contextlib.redirect_stderr(stderr)):
            with self.assertRaises(KeyboardInterrupt):
                start_linux_awake()
        self.assertIn("cleanup denied", stderr.getvalue())

    def test_stderr_is_drained_and_bounded_without_hiding_reason(self):
        script = "import sys; sys.stderr.write('x' * 100000 + 'Access denied'); sys.exit(1)"
        with self.helper(script):
            with self.assertRaises(OSError) as raised:
                start_linux_awake()
        self.assertIn("Access denied", str(raised.exception))
        self.assertLess(len(str(raised.exception)), 8500)

    def test_unexpected_exit_reports_loss_and_actual_reason(self):
        script = "import sys, time; print('ready', flush=True); time.sleep(0.1); sys.stderr.write('system bus closed'); sys.exit(9)"
        stderr = io.StringIO()
        with self.helper(script), contextlib.redirect_stderr(stderr):
            release = start_linux_awake()
            try:
                deadline = time.monotonic() + 3
                while "protection was lost" not in stderr.getvalue() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIn("protection was lost", stderr.getvalue())
                self.assertIn("system bus closed", stderr.getvalue())
                self.assertIn("exit code 9", stderr.getvalue())
            finally:
                release()

    def test_missing_tool_and_launch_denial_are_actionable(self):
        with patch("uma_bct.keep_awake.Path.is_file", return_value=False), patch("uma_bct.keep_awake.shutil.which", return_value=None):
            with self.assertRaisesRegex(OSError, "unavailable.*PATH"):
                start_linux_awake()
        with self.helper("pass"), patch("uma_bct.keep_awake.subprocess.Popen", side_effect=PermissionError(13, "Permission denied")):
            with self.assertRaisesRegex(OSError, "could not launch.*Permission denied"):
                start_linux_awake()

    def test_warning_sanitizes_terminal_control_characters(self):
        stderr = io.StringIO()
        with (patch("uma_bct.keep_awake.sys.platform", "linux"),
              patch("uma_bct.keep_awake.start_linux_awake", side_effect=OSError("denied\x1b[31m\r\nHelper error")),
              contextlib.redirect_stderr(stderr)):
            with keep_awake():
                pass
        self.assertIn("Helper error", stderr.getvalue())
        self.assertNotIn("\x1b", stderr.getvalue())
        self.assertNotIn("\r", stderr.getvalue())
