"""Exercise the changed-path classifier against real Git history."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "classify_ci_changes.sh"


@unittest.skipIf(os.name == "nt" or not shutil.which("bash") or not shutil.which("git"), "Requires Unix, Bash, and Git")
class CIChangesTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.repo = Path(self.temp_dir.name)
        self.git("init", "-q")
        self.git("config", "user.name", "CI Test")
        self.git("config", "user.email", "ci-test@example.invalid")
        self.write("README.md", "Initial documentation\n")
        self.write("src/app.py", "print('hello')\n")
        self.commit("Initial commit")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, check=True)

    def write(self, path, contents):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-qm", message)
        return self.git("rev-parse", "HEAD").stdout.strip()

    def classify(self, event, before, after):
        result = subprocess.run(
            ["bash", str(SCRIPT), event, before, after],
            cwd=self.repo,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    def test_markdown_and_docs_assets_are_documentation_only(self):
        self.write("README.md", "Updated documentation\n")
        self.write("docs/diagram.png", "Placeholder")
        head = self.commit("Update docs")

        self.assertEqual(self.classify("pull_request", self.base, head), "true")
        self.assertEqual(self.classify("push", self.base, head), "true")

    def test_mixed_documentation_and_code_runs_full_ci(self):
        self.write("README.md", "Updated documentation\n")
        self.write("src/app.py", "print('changed')\n")
        head = self.commit("Update docs and code")

        self.assertEqual(self.classify("pull_request", self.base, head), "false")

    def test_workflow_configuration_runs_full_ci(self):
        self.write(".github/workflows/ci.yml", "name: CI\n")
        head = self.commit("Change CI configuration")

        self.assertEqual(self.classify("pull_request", self.base, head), "false")
        self.assertEqual(self.classify("push", self.base, head), "false")

    def test_latest_docs_commit_does_not_hide_earlier_code_change(self):
        self.write("src/app.py", "print('changed')\n")
        code_commit = self.commit("Change code")
        self.write("README.md", "Updated documentation\n")
        head = self.commit("Update docs")

        self.assertEqual(self.classify("pull_request", code_commit, head), "true")
        self.assertEqual(self.classify("pull_request", self.base, head), "false")
        self.assertEqual(self.classify("push", self.base, head), "false")

    def test_renaming_code_to_docs_runs_full_ci(self):
        (self.repo / "docs").mkdir(exist_ok=True)
        (self.repo / "src/app.py").rename(self.repo / "docs/app.md")
        head = self.commit("Move code into docs")

        self.assertEqual(self.classify("pull_request", self.base, head), "false")

    def test_missing_or_empty_comparison_runs_full_ci(self):
        self.assertEqual(self.classify("pull_request", self.base, self.base), "false")
        self.assertEqual(self.classify("push", "not-a-commit", self.base), "false")


if __name__ == "__main__":
    unittest.main()
