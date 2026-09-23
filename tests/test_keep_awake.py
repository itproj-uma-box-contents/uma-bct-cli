import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from uma_bct.keep_awake import keep_awake, start_windows_awake


class KeepAwakeTest(unittest.TestCase):
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
