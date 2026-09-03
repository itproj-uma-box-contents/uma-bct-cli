from __future__ import annotations

import contextlib
import io
import stat
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.check_release_artifacts import inspect_artifact, main


class ReleaseArtifactPrivacyTests(unittest.TestCase):
    def test_archive_traversal_and_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wheel = root / "unsafe.whl"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                archive.writestr("../outside.txt", b"unsafe")

            linked_wheel = root / "linked.whl"
            with zipfile.ZipFile(linked_wheel, mode="w") as archive:
                link = zipfile.ZipInfo("package/link")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "target")

            sdist = root / "unsafe.tar.gz"
            with tarfile.open(sdist, mode="w:gz") as archive:
                link = tarfile.TarInfo("package/link")
                link.type = tarfile.SYMTYPE
                link.linkname = "target"
                archive.addfile(link)

            self.assertIn("unsafe archive path", inspect_artifact(wheel)[0])
            self.assertIn("archive link", inspect_artifact(linked_wheel)[0])
            self.assertIn("archive link", inspect_artifact(sdist)[0])

    def test_private_suffixes_are_rejected_before_backup_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheel = Path(temporary_directory) / "unsafe.whl"
            names = (
                "package/photo.jpg.gz",
                "package/catalogue.csv.bak",
                "package/model.safetensors.index.json",
                "package/catalogue.run.json.bak",
            )
            with zipfile.ZipFile(wheel, mode="w") as archive:
                for name in names:
                    archive.writestr(name, b"private")

            findings = inspect_artifact(wheel)

            self.assertEqual(len(findings), len(names))
            for name in names:
                self.assertTrue(
                    any(name in finding for finding in findings),
                    msg=f"expected a privacy finding for {name}",
                )

    def test_safe_wheel_and_sdist_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            wheel = root / "uma_bct_cli-0.1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                archive.writestr("uma_bct/__init__.py", "")
            sdist = root / "uma_bct_cli-0.1.0.tar.gz"
            with tarfile.open(sdist, mode="w:gz") as archive:
                content = b"safe"
                member = tarfile.TarInfo("uma_bct_cli-0.1.0/README.md")
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

            self.assertEqual([], inspect_artifact(wheel))
            self.assertEqual([], inspect_artifact(sdist))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, main([str(root)]))

    def test_private_directories_and_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheel = Path(temporary_directory) / "unsafe.whl"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                archive.writestr("Archive box photos/private.JPG", b"photo")
                archive.writestr("package/outputs/catalogue.json", b"{}")
                archive.writestr("package/weights/model.gguf", b"weights")

            findings = inspect_artifact(wheel)

            self.assertEqual(3, len(findings))
            self.assertTrue(
                any("private/runtime directory" in item for item in findings)
            )
            self.assertTrue(any(".gguf" in item for item in findings))

    def test_generated_catalogues_and_images_are_rejected_anywhere(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            sdist = Path(temporary_directory) / "unsafe.tar.gz"
            with tarfile.open(sdist, mode="w:gz") as archive:
                for name in (
                    "package/fixture.csv",
                    "package/example.webp",
                    "package/private/catalogue.run.json",
                ):
                    member = tarfile.TarInfo(name)
                    member.size = 1
                    archive.addfile(member, io.BytesIO(b"x"))

            findings = inspect_artifact(sdist)

            self.assertEqual(3, len(findings))
            self.assertTrue(any(".csv" in item for item in findings))
            self.assertTrue(any(".webp" in item for item in findings))
            self.assertTrue(any("run manifest" in item for item in findings))

    def test_private_runtime_temporary_files_are_rejected_regardless_of_suffix(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            wheel = Path(temporary_directory) / "unsafe.whl"
            with zipfile.ZipFile(wheel, mode="w") as archive:
                archive.writestr("package/.uma-bct-json-secret.tmp", b"{}")
                archive.writestr(
                    "package/nested/.uma-bct-catalogue-backup-secret.tmp",
                    b"private catalogue",
                )
                archive.writestr(
                    "package/.uma-bct-catalogue-export-secret.data",
                    b"private catalogue",
                )

            findings = inspect_artifact(wheel)

            self.assertEqual(3, len(findings))
            self.assertTrue(
                all("generated private runtime file" in item for item in findings)
            )

    def test_missing_or_unsupported_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            unsupported = root / "release.zip"
            unsupported.write_bytes(b"not an artifact")

            self.assertNotEqual([], inspect_artifact(unsupported))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(1, main([str(root)]))


if __name__ == "__main__":
    unittest.main()
