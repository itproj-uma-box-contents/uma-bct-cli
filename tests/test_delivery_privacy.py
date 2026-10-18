import subprocess
import unittest

from scripts.check_docker_context import PRIVATE_CONTEXT_PATHS, ROOT


class DeliveryPrivacyTest(unittest.TestCase):
    def test_private_outputs_and_credentials_are_ignored_on_case_sensitive_git(self):
        for path in PRIVATE_CONTEXT_PATHS:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "-c", "core.ignorecase=false", "check-ignore", "--no-index", "--quiet", path],
                    cwd=ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, path)
