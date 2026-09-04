from __future__ import annotations

import contextlib
import io
import os
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.extract_ollama_archive import extract_archive


class _IdentityDecompressor:
    def stream_reader(self, source):
        return contextlib.nullcontext(source)


class OllamaArchiveTest(unittest.TestCase):
    def test_extracts_regular_verified_archive_content(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "ollama.tar.zst"
            content = b"executable"
            with tarfile.open(archive_path, mode="w") as archive:
                member = tarfile.TarInfo("bin/ollama")
                member.mode = 0o755
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))

            with patch.dict(
                sys.modules,
                {"zstandard": SimpleNamespace(ZstdDecompressor=_IdentityDecompressor)},
            ):
                extract_archive(archive_path, root / "installed")

            executable = root / "installed" / "bin" / "ollama"
            self.assertEqual(executable.read_bytes(), content)
            if os.name != "nt":
                self.assertTrue(executable.stat().st_mode & 0o100)

    @unittest.skipIf(os.name == "nt", "Linux archive link behavior")
    def test_preserves_an_internal_relative_library_link(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "ollama.tar.zst"
            content = b"shared library"
            with tarfile.open(archive_path, mode="w") as archive:
                target = tarfile.TarInfo("lib/ollama/cuda_v12/libexample.so")
                target.size = len(content)
                archive.addfile(target, io.BytesIO(content))
                link = tarfile.TarInfo("lib/ollama/mlx_cuda_v12/libexample.so")
                link.type = tarfile.SYMTYPE
                link.linkname = "../cuda_v12/libexample.so"
                archive.addfile(link)

            with patch.dict(
                sys.modules,
                {"zstandard": SimpleNamespace(ZstdDecompressor=_IdentityDecompressor)},
            ):
                extract_archive(archive_path, root / "installed")

            extracted_link = root / "installed/lib/ollama/mlx_cuda_v12/libexample.so"
            self.assertTrue(extracted_link.is_symlink())
            self.assertEqual(extracted_link.read_bytes(), content)

    def test_rejects_escaping_links_and_parent_traversal(self):
        for name, member_type, linkname, expected_error in (
            ("bin/link", tarfile.SYMTYPE, "../../outside", tarfile.FilterError),
            ("../outside", tarfile.REGTYPE, "", ValueError),
        ):
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    archive_path = root / "unsafe.tar.zst"
                    with tarfile.open(archive_path, mode="w") as archive:
                        member = tarfile.TarInfo(name)
                        member.type = member_type
                        if member_type == tarfile.SYMTYPE:
                            member.linkname = linkname
                        archive.addfile(member)

                    with (
                        patch.dict(
                            sys.modules,
                            {
                                "zstandard": SimpleNamespace(
                                    ZstdDecompressor=_IdentityDecompressor
                                )
                            },
                        ),
                        self.assertRaises(expected_error),
                    ):
                        extract_archive(archive_path, root / "installed")

                    self.assertFalse((root / "outside").exists())


if __name__ == "__main__":
    unittest.main()
