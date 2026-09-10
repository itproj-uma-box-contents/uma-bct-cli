import io
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_release_artifacts import inspect_artifact, unsafe_reason
from scripts.extract_ollama_archive import extract_archive, validate_link_target


class IdentityDecompressor:
    def stream_reader(self, source):
        return source


class ArtifactSafetyTest(unittest.TestCase):
    def test_ollama_extractor_accepts_archive_internal_link_targets(self):
        cases = (
            (tarfile.SYMTYPE, "bin/ollama-link", "ollama"),
            (tarfile.SYMTYPE, "bin/ollama-link", "../lib/ollama"),
            (tarfile.SYMTYPE, "bin/root-link", ".."),
            (tarfile.LNKTYPE, "bin/ollama-hardlink", "bin/ollama"),
        )
        for entry_type, name, linkname in cases:
            member = tarfile.TarInfo(name)
            member.type = entry_type
            member.linkname = linkname
            with self.subTest(entry_type=entry_type, linkname=linkname):
                validate_link_target(member)

    @unittest.skipIf(os.name == "nt", "safe-link extraction needs POSIX links")
    def test_extracts_safe_archive_internal_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "ollama.tar.zst"
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                content = b"ollama"
                executable = tarfile.TarInfo("bin/ollama")
                executable.size = len(content)
                archive.addfile(executable, io.BytesIO(content))

                symlink = tarfile.TarInfo("bin/ollama-link")
                symlink.type = tarfile.SYMTYPE
                symlink.linkname = "ollama"
                archive.addfile(symlink)

                hardlink = tarfile.TarInfo("bin/ollama-hardlink")
                hardlink.type = tarfile.LNKTYPE
                hardlink.linkname = "bin/ollama"
                archive.addfile(hardlink)
            archive_path.write_bytes(buffer.getvalue())

            with patch.dict("sys.modules", {"zstandard": SimpleNamespace(ZstdDecompressor=IdentityDecompressor)}):
                extract_archive(archive_path, root / "extracted")
            self.assertEqual((root / "extracted/bin/ollama-link").read_bytes(), b"ollama")
            self.assertEqual((root / "extracted/bin/ollama-hardlink").read_bytes(), b"ollama")

    def test_release_scan_accepts_code_and_rejects_private_content(self):
        for name in ("Archive box photos/shelf.jpg", "package/models/model.gguf", "outputs/catalogue.csv", "../escape.py"):
            with self.subTest(name=name):
                self.assertIsNotNone(unsafe_reason(name))
        self.assertIsNone(unsafe_reason("uma_bct/cli.py"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = root / "safe.whl"
            private = root / "private.zip"
            with zipfile.ZipFile(safe, "w") as archive:
                archive.writestr("uma_bct/cli.py", "code")
            with zipfile.ZipFile(private, "w") as archive:
                archive.writestr("inputs/shelf.jpg", "private")
            self.assertEqual(inspect_artifact(safe), [])
            self.assertIn("private", inspect_artifact(private)[0])

    def test_linux_ollama_extractor_accepts_files_and_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "ollama.tar.zst"

            def write_archive(name, *, entry_type=tarfile.REGTYPE, linkname=""):
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w") as archive:
                    member = tarfile.TarInfo(name)
                    member.type = entry_type
                    member.linkname = linkname
                    if entry_type == tarfile.REGTYPE:
                        content = b"ollama"
                        member.size = len(content)
                        archive.addfile(member, io.BytesIO(content))
                    else:
                        archive.addfile(member)
                archive_path.write_bytes(buffer.getvalue())

            write_archive("bin/ollama")
            destination = root / "safe"
            fake_zstandard = SimpleNamespace(ZstdDecompressor=IdentityDecompressor)
            with patch.dict("sys.modules", {"zstandard": fake_zstandard}):
                extract_archive(archive_path, destination)
            self.assertEqual((destination / "bin/ollama").read_bytes(), b"ollama")

            write_archive("../outside")
            with (
                patch.dict("sys.modules", {"zstandard": fake_zstandard}),
                self.assertRaisesRegex(ValueError, "Unsafe archive path"),
            ):
                extract_archive(archive_path, root / "unsafe")

            unsafe_links = (
                (tarfile.SYMTYPE, "../../outside"),
                (tarfile.SYMTYPE, "..\\..\\outside"),
                (tarfile.SYMTYPE, "/outside"),
                (tarfile.LNKTYPE, "../outside"),
                (tarfile.LNKTYPE, "/outside"),
            )
            for index, (entry_type, linkname) in enumerate(unsafe_links):
                write_archive("bin/ollama-link", entry_type=entry_type, linkname=linkname)
                with (
                    self.subTest(entry_type=entry_type, linkname=linkname),
                    patch.dict("sys.modules", {"zstandard": fake_zstandard}),
                    self.assertRaisesRegex(ValueError, "Unsafe archive link target"),
                ):
                    extract_archive(archive_path, root / f"unsafe-link-{index}")
            self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
