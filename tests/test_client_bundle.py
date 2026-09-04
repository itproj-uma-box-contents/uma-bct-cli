from __future__ import annotations

import contextlib
import hashlib
import io
import os
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_client_bundle import (
    CLIENT_FILES,
    build_client_bundle,
    main,
    verify_client_bundle,
)

_WINDOWS_HELPER_SOURCE = r"""
using System;
using System.IO;
public class Program {
    public static int Main(string[] args) {
        string log = Environment.GetEnvironmentVariable("UMA_BCT_TEST_LOG");
        string name = Path.GetFileName(Environment.GetCommandLineArgs()[0]);
        if (name.StartsWith("uma-bct")) name = "uma-bct";
        string line = name + " " + String.Join(" ", args)
            + " host=" + Environment.GetEnvironmentVariable("OLLAMA_HOST")
            + " no_cloud=" + Environment.GetEnvironmentVariable("OLLAMA_NO_CLOUD")
            + " remote=" + Environment.GetEnvironmentVariable("UMA_BCT_ALLOW_REMOTE_OLLAMA")
            + " container=" + Environment.GetEnvironmentVariable("UMA_BCT_ALLOW_CONTAINER_OLLAMA") + "\n";
        File.AppendAllText(log, line);
        if (name == "OllamaSetup.exe") {
            if (File.Exists(Environment.GetEnvironmentVariable("UMA_BCT_TEST_INPUT"))) {
                return 80;
            }
            if (!String.IsNullOrEmpty(Environment.GetEnvironmentVariable("OLLAMA_UNINSTALL"))
                    || !String.IsNullOrEmpty(Environment.GetEnvironmentVariable("OLLAMA_INSTALL_DIR"))
                    || !String.IsNullOrEmpty(Environment.GetEnvironmentVariable("OLLAMA_NO_START"))) {
                return 81;
            }
            string destination = Path.Combine(
                Environment.GetEnvironmentVariable("LOCALAPPDATA"), "Programs", "Ollama");
            Directory.CreateDirectory(destination);
            File.Copy(Environment.GetCommandLineArgs()[0],
                Path.Combine(destination, "ollama.exe"), true);
            File.AppendAllText(log, "ollama installer\n");
            return 0;
        }
        if (args.Length == 1 && args[0] == "--version") {
            Console.WriteLine(name.StartsWith("ollama")
                ? "ollama version is 0.30.8" : "uv 0.12.9");
        }
        if (Array.IndexOf(args, "venv") >= 0) {
            string app = Environment.GetEnvironmentVariable("UMA_BCT_TEST_APP");
            string scripts = Path.Combine(app, "Scripts");
            Directory.CreateDirectory(scripts);
            string executable = Environment.GetCommandLineArgs()[0];
            File.Copy(executable, Path.Combine(scripts, "python.exe"), true);
            File.Copy(executable, Path.Combine(scripts, "uma-bct.exe"), true);
        }
        if (Array.IndexOf(args, "setup") >= 0) {
            string input = Environment.GetEnvironmentVariable("UMA_BCT_TEST_INPUT");
            if (!String.IsNullOrEmpty(input)) {
                Directory.CreateDirectory(Path.GetDirectoryName(input));
                File.WriteAllBytes(input, new byte[0]);
            }
        }
        string requestedStatus = Environment.GetEnvironmentVariable("UMA_BCT_TEST_RUN_STATUS");
        if (Array.IndexOf(args, "--force") >= 0 && !String.IsNullOrEmpty(requestedStatus)) {
            return Int32.Parse(requestedStatus);
        }
        return 0;
    }
}
"""


def replace_fixture_once(source: str, original: str, replacement: str) -> str:
    """Replace one launcher fixture token, failing if the fixture drifted."""

    occurrences = source.count(original)
    if occurrences != 1:
        raise AssertionError(
            f"Expected one launcher fixture token, found {occurrences}: {original!r}"
        )
    return source.replace(original, replacement, 1)


class ClientBundleTest(unittest.TestCase):
    def test_bundle_command_builds_and_then_verifies_the_release(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            distributions = Path(temporary_directory)
            (distributions / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(distributions)]), 0)
                self.assertEqual(main(["--verify", str(distributions)]), 0)

    def test_readme_leads_with_the_single_client_launcher(self):
        readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")

        self.assertIn("double-click `Run UMA BCT.cmd`", readme)
        self.assertIn("sh run-uma-bct.sh", readme)
        self.assertNotIn("uv tool install", readme)
        self.assertNotIn("ollama serve", readme)
        self.assertNotIn("docker compose", readme)

    def test_windows_installer_waits_only_for_the_installer_process(self):
        launcher = (Path(__file__).parents[1] / "run-uma-bct.ps1").read_text(
            encoding="utf-8"
        )

        self.assertIn("$Installer.WaitForExit()", launcher)
        self.assertNotIn("-PassThru -Wait", launcher)

    def test_posix_launcher_pins_each_supported_download(self):
        launcher = (Path(__file__).parents[1] / "run-uma-bct.sh").read_text(
            encoding="utf-8"
        )

        expected_mappings = (
            (
                "Darwin-*",
                "ollama_",
                "ollama-darwin.tgz",
                "52acbca4e89c53db9abc586a22b5633fd101db293177264b9a0fe5d64a42a064",
            ),
            (
                "Linux-x86_64|Linux-amd64",
                "ollama_",
                "ollama-linux-amd64.tar.zst",
                "ffe2b2c2f2f5f5b30c081ec353c2e0bb2d9ead516064a8e22663b24b8fd8dca0",
            ),
            (
                "Linux-arm64|Linux-aarch64",
                "ollama_",
                "ollama-linux-arm64.tar.zst",
                "668a6f934b0b0455128bb4a76c9e50b9e5f274f9dc7710a066b7073e5bd36588",
            ),
            (
                "Darwin-arm64",
                "",
                "uv-aarch64-apple-darwin.tar.gz",
                "301f72afaf54060f92da7016cb0115bd077f43a9c8e39c1d8170a0bac80fd398",
            ),
            (
                "Darwin-x86_64",
                "",
                "uv-x86_64-apple-darwin.tar.gz",
                "e1ca175824f1056589ce9908f7631879ebc3c36535b5e63dc06510beb370b4c1",
            ),
            (
                "Linux-arm64|Linux-aarch64",
                "",
                "uv-aarch64-unknown-linux-gnu.tar.gz",
                "c36fe17937ff6bd16dc42fc13854b5465999fcab2efe0af559381e945e3c6001",
            ),
            (
                "Linux-x86_64|Linux-amd64",
                "",
                "uv-x86_64-unknown-linux-gnu.tar.gz",
                "ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460",
            ),
        )
        for platform, variable_prefix, archive, sha256 in expected_mappings:
            with self.subTest(platform=platform, archive=archive):
                self.assertIn(
                    f"{platform})\n"
                    f'            {variable_prefix}archive_name="{archive}"\n'
                    f'            {variable_prefix}archive_sha256="{sha256}"',
                    launcher,
                )

    def test_builds_one_client_folder_around_the_wheel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            distributions = root / "dist"
            source.mkdir()
            distributions.mkdir()
            for name in CLIENT_FILES:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
                if name.endswith(".sh"):
                    path.chmod(0o755)
            wheel = distributions / "uma_bct_cli-0.1.0-py3-none-any.whl"
            wheel.write_bytes(b"wheel")

            bundle = build_client_bundle(distributions, source_root=source)

            self.assertEqual(bundle, distributions / "uma-bct-client.zip")
            with zipfile.ZipFile(bundle) as archive:
                names = set(archive.namelist())
                self.assertEqual(
                    names,
                    {
                        *(f"UMA-BCT/{Path(name).name}" for name in CLIENT_FILES),
                        f"UMA-BCT/{wheel.name}",
                    },
                )
                shell_mode = archive.getinfo("UMA-BCT/run-uma-bct.sh").external_attr
                self.assertTrue((shell_mode >> 16) & 0o100)
                self.assertEqual(
                    archive.read(f"UMA-BCT/{wheel.name}"),
                    wheel.read_bytes(),
                )
                self.assertFalse(any("inputs/" in name for name in names))
                self.assertFalse(any("outputs/" in name for name in names))

    def test_requires_exactly_one_wheel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in CLIENT_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
            distributions = root / "dist"
            distributions.mkdir()

            with self.assertRaisesRegex(ValueError, "exactly one wheel"):
                build_client_bundle(distributions, source_root=root)

            (distributions / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            (distributions / "uma_bct_cli-0.2.0-py3-none-any.whl").touch()
            with self.assertRaisesRegex(ValueError, "exactly one wheel"):
                build_client_bundle(distributions, source_root=root)

    def test_verification_rejects_a_bundle_that_does_not_match_the_wheel(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in CLIENT_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
            distributions = root / "dist"
            distributions.mkdir()
            wheel = distributions / "uma_bct_cli-0.1.0-py3-none-any.whl"
            wheel.write_bytes(b"original wheel")
            build_client_bundle(distributions, source_root=root)
            wheel.write_bytes(b"different wheel")

            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_client_bundle(distributions, source_root=root)

    def test_verification_rejects_unexpected_and_non_regular_entries(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name in CLIENT_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
            distributions = root / "dist"
            distributions.mkdir()
            (distributions / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            bundle = build_client_bundle(distributions, source_root=root)

            with zipfile.ZipFile(bundle, mode="a") as archive:
                archive.writestr("UMA-BCT/unexpected.txt", "unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected or missing"):
                verify_client_bundle(distributions, source_root=root)

            bundle = build_client_bundle(distributions, source_root=root)
            with zipfile.ZipFile(bundle) as archive:
                contents = [
                    (member.filename, archive.read(member))
                    for member in archive.infolist()
                ]
            with zipfile.ZipFile(bundle, mode="w") as archive:
                for name, content in contents:
                    member = zipfile.ZipInfo(name)
                    member.create_system = 3
                    member.external_attr = (
                        (stat.S_IFLNK if name.endswith("README.md") else stat.S_IFREG)
                        | 0o644
                    ) << 16
                    archive.writestr(member, content)
            with self.assertRaisesRegex(ValueError, "not a regular file"):
                verify_client_bundle(distributions, source_root=root)

    @unittest.skipIf(os.name == "nt", "POSIX launcher test")
    def test_posix_launcher_uses_the_local_private_workflow(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            distributions = temporary_root / "dist"
            distributions.mkdir()
            (distributions / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            bundle = build_client_bundle(
                distributions,
                source_root=Path(__file__).parents[1],
            )
            with zipfile.ZipFile(bundle) as archive:
                archive.extractall(temporary_root)
            root = temporary_root / "UMA-BCT"
            launcher = root / "run-uma-bct.sh"
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").touch()
            outputs = root / "outputs"
            outputs.mkdir()
            (outputs / "catalogue.csv").write_bytes(b"reviewed catalogue")
            (outputs / "catalogue.run.json").write_bytes(b"reviewed manifest")
            runtime_bin = root / ".uma-bct-runtime" / "bin"
            runtime_bin.mkdir(parents=True)
            app_bin = root / ".uma-bct-runtime" / "app" / "bin"
            app_bin.mkdir(parents=True)
            fake_home = root / "home"
            ollama = root / ".uma-bct-runtime" / "ollama" / "ollama"
            ollama.parent.mkdir(parents=True)
            call_log = root / "calls.txt"
            for executable in (ollama, runtime_bin / "uv"):
                executable.write_text(
                    '#!/bin/sh\nif [ "$1" = "--version" ]; then '
                    'case "$(basename "$0")" in ollama) '
                    'printf "ollama version is 0.30.8\\n";; *) '
                    'printf "uv 0.12.9\\n";; esac; exit 0; fi\n'
                    'printf "%s %s host=%s no_cloud=%s remote=%s '
                    'container=%s\\n" '
                    '"$(basename "$0")" "$*" "${OLLAMA_HOST:-}" '
                    '"${OLLAMA_NO_CLOUD:-}" '
                    '"${UMA_BCT_ALLOW_REMOTE_OLLAMA:-}" '
                    '"${UMA_BCT_ALLOW_CONTAINER_OLLAMA:-}" '
                    '>> "$UMA_BCT_TEST_LOG"\nexit 0\n',
                    encoding="utf-8",
                )
                executable.chmod(0o755)
            app_cli = app_bin / "uma-bct"
            app_cli.write_text(
                '#!/bin/sh\nprintf "uma-bct %s\\n" "$*" '
                '>> "$UMA_BCT_TEST_LOG"\n'
                'exit "${UMA_BCT_TEST_RUN_STATUS:-0}"\n',
                encoding="utf-8",
            )
            app_cli.chmod(0o755)
            app_python = app_bin / "python"
            app_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            app_python.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(fake_home),
                    "UV_VERSION": "hostile-inherited-version",
                    "UV_RELEASE_URL": "https://remote.invalid/uv",
                    "OLLAMA_HOST": "https://remote.invalid",
                    "OLLAMA_NO_CLOUD": "0",
                    "UMA_BCT_ALLOW_CONTAINER_OLLAMA": "1",
                    "UMA_BCT_ALLOW_REMOTE_OLLAMA": "1",
                    "UMA_BCT_NO_OPEN": "1",
                    "UMA_BCT_TEST_LOG": str(call_log),
                }
            )

            completed = subprocess.run(
                ["sh", str(launcher)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = call_log.read_text(encoding="utf-8")
            self.assertIn("ollama list", calls)
            self.assertNotIn("uma-bct setup", calls)
            self.assertIn("uma-bct --force", calls)
            self.assertIn("pip install --no-config --offline", calls)
            self.assertIn("host=http://127.0.0.1:11434", calls)
            self.assertIn("no_cloud=1", calls)
            self.assertIn("remote= container=", calls)
            saved_catalogues = list(outputs.glob("catalogue.previous-*.csv"))
            saved_manifests = list(outputs.glob("catalogue.previous-*.run.json"))
            self.assertEqual(len(saved_catalogues), 1)
            self.assertEqual(len(saved_manifests), 1)
            self.assertEqual(saved_catalogues[0].read_bytes(), b"reviewed catalogue")
            self.assertEqual(saved_manifests[0].read_bytes(), b"reviewed manifest")

            partial_environment = {**environment, "UMA_BCT_TEST_RUN_STATUS": "1"}
            partial = subprocess.run(
                ["sh", str(launcher)],
                cwd=root,
                env=partial_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(partial.returncode, 1)
            self.assertIn("Processing was incomplete", partial.stderr)
            self.assertNotIn("Finished. Review", partial.stdout)

    @unittest.skipIf(os.name == "nt", "POSIX launcher test")
    def test_posix_launcher_rejects_a_redirected_runtime_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            launcher = root / "run-uma-bct.sh"
            launcher.write_text(
                (Path(__file__).parents[1] / "run-uma-bct.sh").read_text(
                    encoding="utf-8"
                ),
                encoding="utf-8",
            )
            target = root / "target"
            target.mkdir()
            runtime = root / ".uma-bct-runtime"
            try:
                runtime.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            completed = subprocess.run(
                ["sh", str(launcher)],
                cwd=root,
                env={**os.environ, "UMA_BCT_NO_OPEN": "1"},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("must not be a symbolic link", completed.stderr)
            self.assertEqual(list(target.iterdir()), [])

    @unittest.skipIf(os.name == "nt", "POSIX launcher test")
    def test_posix_launcher_does_not_install_code_beside_existing_photos(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            launcher = root / "run-uma-bct.sh"
            launcher.write_text(
                (Path(__file__).parents[1] / launcher.name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            (root / "client-constraints.txt").touch()
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "private.heic").touch()

            completed = subprocess.run(
                ["sh", str(launcher)],
                cwd=root,
                env={**os.environ, "HOME": str(root / "home")},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn(
                "Move private files out of inputs and outputs", completed.stderr
            )
            self.assertFalse((root / ".uma-bct-runtime" / "uv-install.sh").exists())

    @unittest.skipIf(os.name == "nt", "POSIX launcher test")
    def test_posix_launcher_does_not_download_if_private_folders_cannot_be_inspected(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            launcher = root / "run-uma-bct.sh"
            launcher_text = (Path(__file__).parents[1] / launcher.name).read_text(
                encoding="utf-8"
            )
            launcher_text = replace_fixture_once(
                launcher_text,
                'find "$INPUTS" "$OUTPUTS" -mindepth 1 -maxdepth 1 -print -quit',
                'sh "$ROOT/failing-find" "$INPUTS" "$OUTPUTS" '
                "-mindepth 1 -maxdepth 1 -print -quit",
            )
            launcher_text = replace_fixture_once(
                launcher_text,
                "curl -q --proto '=https' --proto-redir '=https' --tlsv1.2 "
                '\\\n        -LsSf "$url" -o "$destination"',
                ': > "$ROOT/download-attempted"',
            )
            launcher.write_text(launcher_text, encoding="utf-8")
            (root / "failing-find").write_text("#!/bin/sh\nexit 73\n", encoding="utf-8")
            (root / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            (root / "client-constraints.txt").touch()

            completed = subprocess.run(
                ["sh", str(launcher)],
                cwd=root,
                env={**os.environ, "HOME": str(root / "home")},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("could not be inspected safely", completed.stderr)
            self.assertFalse((root / "download-attempted").exists())

    @unittest.skipIf(os.name == "nt", "POSIX launcher test")
    def test_posix_launcher_rejects_an_unverified_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            launcher = root / "run-uma-bct.sh"
            launcher_text = (Path(__file__).parents[1] / launcher.name).read_text(
                encoding="utf-8"
            )
            download = root / "unverified.tar.gz"
            download.write_bytes(b"not the verified uv archive")
            launcher_text = replace_fixture_once(
                launcher_text,
                '"$BUNDLED_UV_RELEASE_URL/$archive_name"',
                f'"{download.as_uri()}"',
            )
            launcher_text = replace_fixture_once(
                launcher_text,
                "curl -q --proto '=https' --proto-redir '=https' --tlsv1.2 "
                '\\\n        -LsSf "$url" -o "$destination"',
                'cp "${url#file://}" "$destination"',
            )
            launcher.write_text(launcher_text, encoding="utf-8")
            (root / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            (root / "client-constraints.txt").touch()

            completed = subprocess.run(
                ["sh", str(launcher)],
                cwd=root,
                env={**os.environ, "HOME": str(root / "home")},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("checksum verification failed", completed.stderr)
            self.assertFalse((root / ".uma-bct-runtime" / "bin" / "uv").exists())

    @unittest.skipIf(os.name == "nt", "POSIX launcher test")
    def test_posix_launcher_bootstraps_before_adding_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            launcher = root / "run-uma-bct.sh"
            launcher_text = (Path(__file__).parents[1] / launcher.name).read_text(
                encoding="utf-8"
            )
            (root / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            (root / "client-constraints.txt").touch()
            fake_home = root / "home"
            fake_home.mkdir()
            input_image = root / "inputs" / "shelf.jpg"
            call_log = root / "calls.txt"
            fixture_directory = root / "fixtures"
            fixture_directory.mkdir()

            app_program = fixture_directory / "uma-bct"
            app_program.write_text(
                '#!/bin/sh\nprintf "uma-bct %s\\n" "$*" '
                '>> "$UMA_BCT_TEST_LOG"\n'
                'if [ "$1" = "setup" ]; then '
                ': > "$UMA_BCT_TEST_INPUT"; fi\nexit 0\n',
                encoding="utf-8",
            )
            app_program.chmod(0o755)
            uv_program = fixture_directory / "uv"
            uv_program.write_text(
                '#!/bin/sh\nif [ "$1" = "--version" ]; then '
                'printf "uv 0.12.9\\n"; exit 0; fi\n'
                'printf "uv %s\\n" "$*" >> "$UMA_BCT_TEST_LOG"\n'
                'case " $* " in *" venv "*) '
                'mkdir -p "$UMA_BCT_TEST_APP/bin"; '
                'cp "$UMA_BCT_TEST_APP_PROGRAM" "$UMA_BCT_TEST_APP/bin/python"; '
                'cp "$UMA_BCT_TEST_APP_PROGRAM" "$UMA_BCT_TEST_APP/bin/uma-bct"; '
                'chmod 755 "$UMA_BCT_TEST_APP/bin/python" '
                '"$UMA_BCT_TEST_APP/bin/uma-bct";; esac\nexit 0\n',
                encoding="utf-8",
            )
            uv_program.chmod(0o755)
            ollama_program = fixture_directory / "ollama"
            ollama_program.write_text(
                '#!/bin/sh\nif [ "$1" = "--version" ]; then '
                'printf "ollama version is 0.30.8\\n"; exit 0; fi\n'
                'printf "ollama %s\\n" "$*" '
                '>> "$UMA_BCT_TEST_LOG"\nexit 0\n',
                encoding="utf-8",
            )
            ollama_program.chmod(0o755)
            ollama_archive = fixture_directory / "ollama.tgz"
            with tarfile.open(ollama_archive, mode="w:gz") as archive:
                archive.add(ollama_program, arcname="ollama")
                archive.add(ollama_program, arcname="bin/ollama")

            uv_archive = fixture_directory / "uv.tar.gz"
            with tarfile.open(uv_archive, mode="w:gz") as archive:
                archive.add(uv_program, arcname="uv-test/uv")
                archive.add(uv_program, arcname="uv-test/uvx")

            replacements = {
                '"$BUNDLED_UV_RELEASE_URL/$archive_name"': f'"{uv_archive.as_uri()}"',
                '"$OLLAMA_RELEASE_URL/$ollama_archive_name"': f'"{ollama_archive.as_uri()}"',
                "curl -q --proto '=https' --proto-redir '=https' --tlsv1.2 "
                '\\\n        -LsSf "$url" -o "$destination"': (
                    'cp "${url#file://}" "$destination"'
                ),
            }
            for official_hash in (
                "ffe2b2c2f2f5f5b30c081ec353c2e0bb2d9ead516064a8e22663b24b8fd8dca0",
                "668a6f934b0b0455128bb4a76c9e50b9e5f274f9dc7710a066b7073e5bd36588",
            ):
                launcher_text = replace_fixture_once(
                    launcher_text,
                    f'ollama_archive_sha256="{official_hash}"\n'
                    '            ollama_archive_type="zstd"',
                    f'ollama_archive_sha256="{official_hash}"\n'
                    '            ollama_archive_type="gzip"',
                )
            uv_archive_hash = hashlib.sha256(uv_archive.read_bytes()).hexdigest()
            for official_hash in (
                "301f72afaf54060f92da7016cb0115bd077f43a9c8e39c1d8170a0bac80fd398",
                "e1ca175824f1056589ce9908f7631879ebc3c36535b5e63dc06510beb370b4c1",
                "c36fe17937ff6bd16dc42fc13854b5465999fcab2efe0af559381e945e3c6001",
                "ec7a99cd05e0cd7f80243f135ce1361c76835cb0ee60055d14d20eba8eba1460",
            ):
                replacements[official_hash] = uv_archive_hash
            ollama_archive_hash = hashlib.sha256(
                ollama_archive.read_bytes()
            ).hexdigest()
            for official_hash in (
                "52acbca4e89c53db9abc586a22b5633fd101db293177264b9a0fe5d64a42a064",
                "ffe2b2c2f2f5f5b30c081ec353c2e0bb2d9ead516064a8e22663b24b8fd8dca0",
                "668a6f934b0b0455128bb4a76c9e50b9e5f274f9dc7710a066b7073e5bd36588",
            ):
                replacements[official_hash] = ollama_archive_hash
            for original, replacement in replacements.items():
                launcher_text = replace_fixture_once(
                    launcher_text, original, replacement
                )
            launcher.write_text(launcher_text, encoding="utf-8")

            environment = os.environ.copy()
            environment.update(
                {
                    "HOME": str(fake_home),
                    "UMA_BCT_NO_OPEN": "1",
                    "UMA_BCT_TEST_APP": str(root / ".uma-bct-runtime" / "app"),
                    "UMA_BCT_TEST_APP_PROGRAM": str(app_program),
                    "UMA_BCT_TEST_INPUT": str(input_image),
                    "UMA_BCT_TEST_LOG": str(call_log),
                }
            )

            completed = subprocess.run(
                ["sh", str(launcher)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(input_image.is_file())
            calls = call_log.read_text(encoding="utf-8")
            self.assertLess(calls.index("uv venv"), calls.index("uma-bct setup"))
            self.assertLess(calls.index("ollama list"), calls.index("uma-bct setup"))
            self.assertIn("pip sync --no-config --require-hashes", calls)
            self.assertIn("--offline", calls)

    @unittest.skipUnless(os.name == "nt", "Windows launcher test")
    def test_windows_launcher_bootstraps_before_adding_images(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = Path(__file__).parents[1]
            launcher = root / "run-uma-bct.ps1"
            launcher_text = (repository / launcher.name).read_text(encoding="utf-8")
            (root / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            (root / "client-constraints.txt").touch()
            local_app_data = root / "local-app-data"
            input_image = root / "inputs" / "shelf.jpg"
            call_log = root / "calls.txt"

            powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
            self.assertIsNotNone(powershell)
            uv_executable = root / "uv.exe"
            compile_environment = {
                **os.environ,
                "UMA_BCT_HELPER_SOURCE": _WINDOWS_HELPER_SOURCE,
                "UMA_BCT_HELPER_OUTPUT": str(uv_executable),
            }
            compiled = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-Command",
                    "Add-Type -TypeDefinition $env:UMA_BCT_HELPER_SOURCE "
                    "-OutputAssembly $env:UMA_BCT_HELPER_OUTPUT "
                    "-OutputType ConsoleApplication",
                ],
                env=compile_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            uv_archive = root / "uv.zip"
            with zipfile.ZipFile(uv_archive, mode="w") as archive:
                archive.write(uv_executable, "uv.exe")
                archive.write(uv_executable, "uvx.exe")
            ollama_installer = root / "OllamaSetup.exe"
            shutil.copy2(uv_executable, ollama_installer)
            replacements = {
                "https://releases.astral.sh/github/uv/releases/download/"
                "$UvVersion/uv-x86_64-pc-windows-msvc.zip": str(uv_archive),
                "ddbfcee1ac615a0499f6aa97b5ec8ebdf3ee4a7714a48055ec2ba0030e3cf810": hashlib.sha256(
                    uv_archive.read_bytes()
                ).hexdigest(),
                "https://github.com/ollama/ollama/releases/download/"
                "v$OllamaVersion/OllamaSetup.exe": str(ollama_installer),
                "d7cfbb2509d9e9464972d162899a5f085c1802619b6bbe0c4e210b47bcf2d446": hashlib.sha256(
                    ollama_installer.read_bytes()
                ).hexdigest(),
                "Invoke-WebRequest -Uri $UvArchiveUrl -OutFile $UvArchive "
                "-UseBasicParsing": "Copy-Item -LiteralPath $UvArchiveUrl "
                "-Destination $UvArchive",
                "Invoke-WebRequest -Uri $OllamaInstallerUrl "
                "-OutFile $OllamaInstaller -UseBasicParsing": "Copy-Item "
                "-LiteralPath $OllamaInstallerUrl -Destination $OllamaInstaller",
                "Assert-OllamaSignature $OllamaInstaller": "# Test fixture hash verified above",
                '(Join-Path $env:ProgramFiles "Ollama\\ollama.exe")': (
                    '(Join-Path $env:UMA_BCT_TEST_PROGRAM_FILES "Ollama\\ollama.exe")'
                ),
            }
            for original, replacement in replacements.items():
                launcher_text = replace_fixture_once(
                    launcher_text, original, replacement
                )
            launcher.write_text(launcher_text, encoding="utf-8")

            environment = os.environ.copy()
            environment.update(
                {
                    "LOCALAPPDATA": str(local_app_data),
                    "UMA_BCT_TEST_PROGRAM_FILES": str(root / "program-files"),
                    "OLLAMA_INSTALL_DIR": "C:\\untrusted",
                    "OLLAMA_NO_START": "1",
                    "OLLAMA_UNINSTALL": "1",
                    "UMA_BCT_NO_OPEN": "1",
                    "UMA_BCT_TEST_APP": str(root / ".uma-bct-runtime" / "app"),
                    "UMA_BCT_TEST_INPUT": str(input_image),
                    "UMA_BCT_TEST_LOG": str(call_log),
                }
            )

            launcher_command = [
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(launcher),
            ]
            completed = subprocess.run(
                launcher_command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(input_image.is_file())
            calls = call_log.read_text(encoding="utf-8")
            self.assertLess(calls.index("uv venv"), calls.index("uma-bct setup"))
            self.assertLess(
                calls.index("ollama installer"),
                calls.index("uma-bct setup"),
            )
            self.assertIn("pip sync --no-config --require-hashes", calls)
            self.assertIn("--offline", calls)

            input_image.unlink()
            zero_images = subprocess.run(
                launcher_command,
                cwd=root,
                env={**environment, "UMA_BCT_TEST_INPUT": ""},
                input="\n",
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(zero_images.returncode, 2)
            self.assertIn("No supported photos were found", zero_images.stderr)

            input_image.touch()
            (input_image.parent / "second-shelf.jpg").touch()
            many_images = subprocess.run(
                launcher_command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(many_images.returncode, 0, many_images.stderr)

    @unittest.skipUnless(os.name == "nt", "Windows launcher test")
    def test_windows_launcher_uses_the_local_private_workflow(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            repository = Path(__file__).parents[1]
            launcher = root / "run-uma-bct.ps1"
            launcher.write_text(
                (repository / launcher.name).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (root / "uma_bct_cli-0.1.0-py3-none-any.whl").touch()
            (root / "client-constraints.txt").touch()
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").touch()
            outputs = root / "outputs"
            outputs.mkdir()
            (outputs / "catalogue.csv").write_bytes(b"reviewed catalogue")
            (outputs / "catalogue.run.json").write_bytes(b"reviewed manifest")
            runtime_bin = root / ".uma-bct-runtime" / "bin"
            runtime_bin.mkdir(parents=True)
            local_app_data = root / "local-app-data"
            ollama = local_app_data / "Programs" / "Ollama" / "ollama.exe"
            ollama.parent.mkdir(parents=True)
            call_log = root / "calls.txt"
            powershell = shutil.which("powershell.exe") or shutil.which("pwsh")
            self.assertIsNotNone(powershell)
            compile_environment = os.environ.copy()
            compile_environment.update(
                {
                    "UMA_BCT_HELPER_SOURCE": _WINDOWS_HELPER_SOURCE,
                    "UMA_BCT_HELPER_OUTPUT": str(runtime_bin / "uv.exe"),
                }
            )
            compiled = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-Command",
                    "Add-Type -TypeDefinition $env:UMA_BCT_HELPER_SOURCE "
                    "-OutputAssembly $env:UMA_BCT_HELPER_OUTPUT "
                    "-OutputType ConsoleApplication",
                ],
                env=compile_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            shutil.copy2(runtime_bin / "uv.exe", ollama)
            app_scripts = root / ".uma-bct-runtime" / "app" / "Scripts"
            app_scripts.mkdir(parents=True)
            shutil.copy2(runtime_bin / "uv.exe", app_scripts / "python.exe")
            shutil.copy2(runtime_bin / "uv.exe", app_scripts / "uma-bct.exe")
            environment = os.environ.copy()
            environment.update(
                {
                    "LOCALAPPDATA": str(local_app_data),
                    "OLLAMA_HOST": "https://remote.invalid",
                    "OLLAMA_NO_CLOUD": "0",
                    "UMA_BCT_ALLOW_CONTAINER_OLLAMA": "1",
                    "UMA_BCT_ALLOW_REMOTE_OLLAMA": "1",
                    "UMA_BCT_NO_OPEN": "1",
                    "UMA_BCT_TEST_LOG": str(call_log),
                }
            )

            completed = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(launcher),
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = call_log.read_text(encoding="utf-8")
            self.assertIn("ollama list", calls)
            self.assertNotIn("uma-bct setup", calls)
            self.assertIn("uma-bct --force", calls)
            self.assertIn("pip install --no-config --offline", calls)
            self.assertIn("host=http://127.0.0.1:11434", calls)
            self.assertIn("no_cloud=1", calls)
            self.assertIn("remote= container=", calls)
            saved_catalogues = list(outputs.glob("catalogue.previous-*.csv"))
            saved_manifests = list(outputs.glob("catalogue.previous-*.run.json"))
            self.assertEqual(len(saved_catalogues), 1)
            self.assertEqual(len(saved_manifests), 1)
            self.assertEqual(saved_catalogues[0].read_bytes(), b"reviewed catalogue")
            self.assertEqual(saved_manifests[0].read_bytes(), b"reviewed manifest")

            partial_environment = {**environment, "UMA_BCT_TEST_RUN_STATUS": "1"}
            partial = subprocess.run(
                [
                    powershell,
                    "-NoLogo",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(launcher),
                ],
                cwd=root,
                env=partial_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(partial.returncode, 1)
            self.assertIn("Processing was incomplete", partial.stderr)
            self.assertNotIn("Finished. Review", partial.stdout)


if __name__ == "__main__":
    unittest.main()
