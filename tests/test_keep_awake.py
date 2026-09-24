import contextlib
import ctypes
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from uma_bct.keep_awake import keep_awake, start_mac_awake, start_windows_awake


class KeepAwakeTest(unittest.TestCase):
    # Confirm both macOS requests before reporting success, and release partial work on error.
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

    # Check that every supported host releases its request and errors stay nonfatal.
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
        with patch("uma_bct.keep_awake.ctypes.windll", SimpleNamespace(kernel32=kernel), create=True):
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
