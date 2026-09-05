import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.check_release_artifacts import inspect_artifact, unsafe_reason
from scripts.extract_ollama_archive import extract_archive


class IdentityDecompressor:
    def stream_reader(self, source):
        return source


class ArtifactSafetyTest(unittest.TestCase):
    def test_release_scan_accepts_code_and_rejects_private_content(self):
        for name in (
            "Archive box photos/shelf.jpg",
            "package/models/model.gguf",
            "outputs/catalogue.csv",
            "../escape.py",
        ):
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

            def write_archive(name):
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w") as archive:
                    content = b"ollama"
                    member = tarfile.TarInfo(name)
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
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


if __name__ == "__main__":
    unittest.main()
