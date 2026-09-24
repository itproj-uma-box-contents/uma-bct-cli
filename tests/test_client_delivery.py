import hashlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import zipfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from scripts.build_client_bundle import (
    BUNDLE_ROOT,
    CLIENT_FILES,
    FRIENDLY_INPUT_DIRECTORY,
    FRIENDLY_RESULTS_DIRECTORY,
    build_client_bundle,
    verify_client_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OLLAMA_VERSION = "0.32.9"
EXPECTED_OLLAMA_DARWIN_SHA256 = "17a5b096d4515d00a6415012db847a2b353b389ed7ab33d025e3b98c2f05b49c"
EXPECTED_OLLAMA_LINUX_AMD64_SHA256 = "5d747a43369f61e38f20b5a39fcc5c90647e562cdc61e2e56034f1c5b113d540"
EXPECTED_OLLAMA_LINUX_ARM64_SHA256 = "79617139521db251c716d6229505d7530171ff4d68e476d0c22dabc15726a237"
EXPECTED_OLLAMA_WINDOWS_AMD64_SHA256 = "7b4f6ce09c1f2c3b21561b323779beaf3ca3c7012f8e4522605a13cbbb19f0b8"
EXPECTED_OLLAMA_CONTAINER_DIGEST = "1685741456770df6e3cceb2a945a5f75e020f658d1701509668d6f4688f1dd3f"

FAKE_OLLAMA = f"""#!{sys.executable}
import os
import signal
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit


def write_event(name):
    path = os.environ.get("UMA_BCT_FAKE_OLLAMA_EVENTS")
    if path:
        with Path(path).open("a", encoding="utf-8") as output:
            output.write(f"{{name}} {{os.environ.get('OLLAMA_HOST', '')}} {{os.getpid()}}\\n")


command = sys.argv[1] if len(sys.argv) > 1 else ""
if command == "--version":
    if os.environ.get("UMA_BCT_FAKE_OLLAMA_STALL_VERSION") == "1":
        time.sleep(10)
    print("ollama version is {EXPECTED_OLLAMA_VERSION}")
    raise SystemExit(0)

endpoint = urlsplit(os.environ["OLLAMA_HOST"])
address = (endpoint.hostname, endpoint.port)
if command == "list":
    write_event("unexpected-list")
    raise SystemExit(1)

if command != "serve":
    raise SystemExit(1)

counter_path = os.environ.get("UMA_BCT_FAKE_OLLAMA_COUNTER")
count = 1
if counter_path:
    counter = Path(counter_path)
    count = int(counter.read_text(encoding="utf-8")) if counter.exists() else 0
    count += 1
    counter.write_text(str(count), encoding="utf-8")
    if count <= int(os.environ.get("UMA_BCT_FAKE_OLLAMA_FAIL_STARTS", "0")):
        write_event("bind-failed")
        raise SystemExit(98)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    server.bind(address)
    server.listen()
except OSError:
    write_event("bind-failed")
    raise

if count <= int(os.environ.get("UMA_BCT_FAKE_OLLAMA_STALL_STARTS", "0")):
    write_event("stalled")
    connection, _ = server.accept()
    with connection:
        connection.recv(4096)
        time.sleep(2)
    server.close()
    write_event("stopped")
    raise SystemExit(98)

write_event("started")


def stop(_signal, _frame):
    raise SystemExit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
try:
    while True:
        connection, _ = server.accept()
        with connection:
            connection.recv(4096)
            connection.sendall(b'HTTP/1.0 200 OK\\r\\nContent-Type: application/json\\r\\n\\r\\n{{"models":[]}}')
finally:
    server.close()
    write_event("stopped")
"""


# Write the executable.
def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)


# Prepare the launcher root.
def prepare_launcher_root(root: Path) -> tuple[Path, dict[str, str], Path]:
    (root / "client-constraints.txt").write_text("locked", encoding="utf-8")
    (root / "uma_bct_cli-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "shelf.jpg").write_bytes(b"private photo fixture")
    (root / "outputs").mkdir()
    runtime = root / ".uma-bct-runtime"
    events = root / "ollama-events.txt"
    environment = os.environ.copy()
    environment.update(
        UMA_BCT_NO_OPEN="1",
        UMA_BCT_FAKE_APP_HOST=str(root / "app-host.txt"),
        UMA_BCT_FAKE_APP_PID=str(root / "app-pid.txt"),
        UMA_BCT_FAKE_APP_SIGNALS=str(root / "app-signals.txt"),
        UMA_BCT_FAKE_OLLAMA_EVENTS=str(events),
        UMA_BCT_FAKE_OLLAMA_COUNTER=str(root / "ollama-counter.txt"),
    )
    return runtime, environment, events


# Prepare the POSIX launcher.
def prepare_posix_launcher(root: Path, *, companion: bool = True) -> tuple[Path, dict[str, str], Path]:
    launcher = root / "run-uma-bct.sh"
    shutil.copy2(ROOT / "run-uma-bct.sh", launcher)
    runtime, environment, events = prepare_launcher_root(root)
    write_executable(
        runtime / "bin/uv", '#!/bin/sh\nif [ "${1:-}" = "--version" ]; then printf \'uv 0.12.9\\n\'; fi\nexit 0\n'
    )
    write_executable(runtime / "app/bin/python", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    write_executable(
        runtime / "app/bin/uma-bct",
        '#!/bin/sh\nprintf \'%s\\n\' "$OLLAMA_HOST" >"$UMA_BCT_FAKE_APP_HOST"\n'
        'if [ "${UMA_BCT_FAKE_INTERRUPT:-}" = "1" ]; then kill -TERM "$PPID"; fi\n'
        'if [ "${UMA_BCT_FAKE_APP_WAIT:-}" = "1" ]; then\n'
        '  exec "$UMA_BCT_FAKE_SYSTEM_PYTHON" -c \'import os, signal, time\n'
        'path = os.environ["UMA_BCT_FAKE_APP_SIGNALS"]\n'
        'def stop(name, code):\n'
        '    def handler(_signum, _frame):\n'
        '        with open(path, "a", encoding="utf-8") as output: output.write(name + "\\n")\n'
        '        raise SystemExit(code)\n'
        '    return handler\n'
        'signal.signal(signal.SIGINT, stop("INT", 130))\n'
        'signal.signal(signal.SIGTERM, stop("TERM", 143))\n'
        'signal.signal(signal.SIGHUP, stop("HUP", 129))\n'
        'open(os.environ["UMA_BCT_FAKE_APP_READY"], "w", encoding="utf-8").write("ready\\n")\n'
        'while True: time.sleep(1)\'\n'
        'fi\n'
        'exit "${UMA_BCT_FAKE_APP_EXIT:-0}"\n',
    )
    ollama = runtime / "ollama/ollama"
    write_executable(ollama, FAKE_OLLAMA)
    if companion:
        if sys.platform == "darwin":
            helper = ollama.with_name("llama-server")
        else:
            helper = runtime / "ollama/lib/ollama/llama-server"
        write_executable(helper, "#!/bin/sh\nexit 0\n")

    environment.update(
        UMA_BCT_FAKE_APP_READY=str(root / "app-ready.txt"),
        UMA_BCT_FAKE_SYSTEM_PYTHON=sys.executable,
    )
    return launcher, environment, events


# Prepare the windows launcher.
def prepare_windows_launcher(
    root: Path, fake_executable: Path, *, companion: bool = True
) -> tuple[Path, dict[str, str], Path]:
    launcher = root / "run-uma-bct.ps1"
    shutil.copy2(ROOT / "run-uma-bct.ps1", launcher)
    runtime, environment, events = prepare_launcher_root(root)
    executable_paths = (runtime / "bin/uv.exe", runtime / "app/Scripts/python.exe", runtime / "app/Scripts/uma-bct.exe")
    for path in executable_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fake_executable, path)

    local_app_data = root / "local-app-data"
    ollama = runtime / "ollama/ollama.exe"
    ollama.parent.mkdir(parents=True)
    shutil.copy2(fake_executable, ollama)
    if companion:
        helper = ollama.parent / "lib/ollama/llama-server.exe"
        helper.parent.mkdir(parents=True)
        shutil.copy2(fake_executable, helper)

    environment.update(
        LOCALAPPDATA=str(local_app_data),
        ProgramFiles=str(root / "program-files"),
    )
    return launcher, environment, events


# Handle the launcher command step.
def launcher_command(launcher: Path) -> list[str]:
    if launcher.suffix == ".ps1":
        return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)]
    return ["sh", str(launcher)]


# Run the launcher.
def run_launcher(launcher: Path, environment: dict[str, str], timeout: int = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        launcher_command(launcher), env=environment, capture_output=True, text=True, timeout=timeout
    )


# Handle the start launcher step.
def start_launcher(launcher: Path, environment: dict[str, str], **options) -> subprocess.Popen[str]:
    return subprocess.Popen(
        launcher_command(launcher),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **options,
    )


# Handle the event names step.
def event_names(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines()]


# Handle the wait until step.
def wait_until(process: subprocess.Popen[str], ready: Callable[[], bool], message: str, timeout: int = 10) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready():
            return
        if process.poll() is not None:
            raise AssertionError(message)
        time.sleep(0.1)
    raise AssertionError(message)


# Handle the launcher fixture step.
@contextmanager
def launcher_fixture(prepare, *args, root_name: str | None = None, **kwargs):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        if root_name is not None:
            root = root / root_name
            root.mkdir()
        yield (root, *prepare(root, *args, **kwargs))


# Group the client delivery checks.
class ClientDeliveryTest(unittest.TestCase):
    # Prepare the shared fixture for this test group.
    @classmethod
    def setUpClass(cls):
        cls.compiled_fixture = None
        cls.fixture_directory = None
        if os.name != "nt":
            return
        cls.fixture_directory = tempfile.TemporaryDirectory()
        fixture_root = Path(cls.fixture_directory.name)
        cls.compiled_fixture = fixture_root / "fake-launcher-tool.exe"
        source = ROOT / "tests/fixtures/fake_launcher_tool.cs"
        quoted_source = str(source).replace("'", "''")
        quoted_output = str(cls.compiled_fixture).replace("'", "''")
        command = f"Add-Type -Path '{quoted_source}' -OutputAssembly '{quoted_output}' -OutputType ConsoleApplication"
        subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True, capture_output=True, text=True)

    # Clean up the shared fixture after this test group.
    @classmethod
    def tearDownClass(cls):
        if cls.fixture_directory is not None:
            cls.fixture_directory.cleanup()

    # Prepare the bundle sources.
    def prepare_bundle_sources(self, root: Path) -> tuple[Path, Path]:
        dist = root / "dist"
        dist.mkdir()
        wheel = dist / "uma_bct_cli-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"wheel")
        for name in CLIENT_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        return dist, wheel

    # Check that bundle contains only the client files and one wheel.
    def test_bundle_contains_only_the_client_files_and_one_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist, wheel = self.prepare_bundle_sources(root)

            bundle = build_client_bundle(dist, source_root=root)
            self.assertEqual(bundle.name, "archive-box-transcription.zip")
            self.assertEqual(BUNDLE_ROOT, "Archive Box Transcription")
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {
                        *(f"{BUNDLE_ROOT}/{Path(name).name}" for name in CLIENT_FILES),
                        f"{BUNDLE_ROOT}/{wheel.name}",
                        f"{BUNDLE_ROOT}/{FRIENDLY_INPUT_DIRECTORY}/",
                        f"{BUNDLE_ROOT}/{FRIENDLY_RESULTS_DIRECTORY}/",
                    },
                )
            bundle.write_bytes(b"not a zip")
            with self.assertRaisesRegex(ValueError, "verify"):
                verify_client_bundle(dist, source_root=root)

    # Check that bundle rejects a linked source file.
    def test_bundle_rejects_a_linked_source_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist, _wheel = self.prepare_bundle_sources(root)
            private = root / "private-transcript.txt"
            private.write_bytes(b"private archive transcription")
            readme = root / "README.md"
            readme.unlink()
            try:
                readme.symlink_to(private)
            except OSError as error:
                self.skipTest(str(error))

            with self.assertRaisesRegex(ValueError, "unlinked regular file"):
                build_client_bundle(dist, source_root=root)

    # Check that bundle verifier rejects duplicate member names.
    def test_bundle_verifier_rejects_duplicate_member_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist, _wheel = self.prepare_bundle_sources(root)
            bundle = build_client_bundle(dist, source_root=root)
            with zipfile.ZipFile(bundle) as source:
                entries = [(item, source.read(item)) for item in source.infolist()]
            duplicate_name = f"{BUNDLE_ROOT}/README.md"
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(duplicate_name, b"private earlier duplicate")
                for item, content in entries:
                    archive.writestr(item, content)

            with self.assertRaisesRegex(ValueError, "duplicate"):
                verify_client_bundle(dist, source_root=root)

    # Check that bundle verifier rejects link metadata even when bytes match.
    def test_bundle_verifier_rejects_link_metadata_even_when_bytes_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist, _wheel = self.prepare_bundle_sources(root)
            bundle = build_client_bundle(dist, source_root=root)
            with zipfile.ZipFile(bundle) as source:
                entries = [(item, source.read(item)) for item in source.infolist()]
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
                for item, content in entries:
                    if item.filename == f"{BUNDLE_ROOT}/README.md":
                        item.create_system = 3
                        item.external_attr = (0o120777 << 16)
                    archive.writestr(item, content)

            with self.assertRaisesRegex(ValueError, "link or special"):
                verify_client_bundle(dist, source_root=root)

    # Check that dependency lock is generated and excludes deferred exporters.
    def test_dependency_lock_is_generated_and_excludes_deferred_exporters(self):
        constraints = (ROOT / "client-constraints.txt").read_text(encoding="utf-8")
        self.assertIn("autogenerated by uv", constraints)
        self.assertIn("pillow-heif==", constraints)
        self.assertIn("rawpy==", constraints)
        self.assertIn("zstandard==", constraints)
        self.assertNotIn("pandas==", constraints)
        self.assertNotIn("openpyxl==", constraints)
        self.assertNotIn("tifffile==", constraints)

    # Check that compose disables Ollama cloud for every service.
    def test_compose_disables_ollama_cloud_for_every_service(self):
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        self.assertEqual(compose.count('OLLAMA_NO_CLOUD: "1"'), 3)
        self.assertIn("- --json", compose)

    # Check that delivery paths and fixtures agree on the Ollama version.
    def test_delivery_paths_and_fixtures_agree_on_the_ollama_version(self):
        posix = (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8")
        windows = (ROOT / "run-uma-bct.ps1").read_text(encoding="utf-8")
        compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
        windows_fixture = (ROOT / "tests/fixtures/fake_launcher_tool.cs").read_text(encoding="utf-8")

        with self.subTest(delivery_path="POSIX launcher"):
            self.assertIn(f'OLLAMA_VERSION="{EXPECTED_OLLAMA_VERSION}"', posix)
        with self.subTest(delivery_path="Windows launcher"):
            self.assertIn(f'$OllamaVersion = "{EXPECTED_OLLAMA_VERSION}"', windows)
        with self.subTest(delivery_path="Compose services"):
            image = (
                f"image: ollama/ollama:{EXPECTED_OLLAMA_VERSION}"
                f"@sha256:{EXPECTED_OLLAMA_CONTAINER_DIGEST}"
            )
            self.assertEqual(compose.count(image), 2)
        for checksum in (
            EXPECTED_OLLAMA_DARWIN_SHA256,
            EXPECTED_OLLAMA_LINUX_AMD64_SHA256,
            EXPECTED_OLLAMA_LINUX_ARM64_SHA256,
        ):
            with self.subTest(delivery_path="POSIX archive", checksum=checksum):
                self.assertIn(checksum, posix)
        with self.subTest(delivery_path="Windows archive"):
            self.assertIn(EXPECTED_OLLAMA_WINDOWS_AMD64_SHA256, windows)
        with self.subTest(delivery_path="POSIX launcher fixture"):
            self.assertIn(f"ollama version is {EXPECTED_OLLAMA_VERSION}", FAKE_OLLAMA)
        with self.subTest(delivery_path="Windows launcher fixture"):
            self.assertIn(f"client version is {EXPECTED_OLLAMA_VERSION}", windows_fixture)

    # Check that launchers pin downloads and prepare before accepting photos.
    def test_launchers_pin_downloads_and_prepare_before_accepting_photos(self):
        posix = (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8")
        windows = (ROOT / "run-uma-bct.ps1").read_text(encoding="utf-8")
        for launcher in (posix, windows):
            self.assertIn("0.12.9", launcher)
            self.assertIn(EXPECTED_OLLAMA_VERSION, launcher)
            self.assertIn("127.0.0.1", launcher)
            self.assertIn("client-constraints.txt", launcher)
            self.assertIn("Add Photos Here", launcher)
            self.assertIn("Results", launcher)
            self.assertIn("--results-directory", launcher)
            self.assertIn(".uma-bct-launcher.lock", launcher)
        for extension in ("avif", "bmp", "dng", "gif", "heic", "heif", "jpeg", "jpg", "png", "tif", "tiff", "webp"):
            self.assertIn(f'".{extension}"', windows.casefold())
        for pattern in (
            ".[Aa][Vv][Ii][Ff]", ".[Bb][Mm][Pp]", ".[Dd][Nn][Gg]", ".[Gg][Ii][Ff]",
            ".[Hh][Ee][Ii][Cc]", ".[Hh][Ee][Ii][Ff]", ".[Jj][Pp][Ee][Gg]", ".[Jj][Pp][Gg]",
            ".[Pp][Nn][Gg]", ".[Tt][Ii][Ff]", ".[Tt][Ii][Ff][Ff]", ".[Ww][Ee][Bb][Pp]",
        ):
            self.assertIn(pattern, posix)
        self.assertLess(posix.index("setup --path-to-inputs"), posix.index("Press Enter after"))
        self.assertLess(windows.index('"setup --path-to-inputs "'), windows.index("Read-Host"))
        self.assertIn("Get-AuthenticodeSignature", windows)
        self.assertIn("ollama-windows-amd64.zip", windows)
        self.assertNotIn("OllamaSetup.exe", windows)
        self.assertNotIn(r"Programs\Ollama", windows)

    # Check that launchers own a dynamic service and have bounded startup.
    def test_launchers_own_a_dynamic_service_and_have_bounded_startup(self):
        posix = (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8")
        windows = (ROOT / "run-uma-bct.ps1").read_text(encoding="utf-8")
        self.assertNotIn('OLLAMA_HOST="http://127.0.0.1:11434"', posix)
        self.assertNotIn('$env:OLLAMA_HOST = "http://127.0.0.1:11434"', windows)
        self.assertIn("socket.AF_INET", posix)
        self.assertIn("1 2 3", posix)
        self.assertIn('environment["OLLAMA_HOST"] = "http://127.0.0.1:1"', posix)
        self.assertIn("timeout=0.5", posix)
        self.assertIn("timeout=5", posix)
        self.assertIn("TcpListener", windows)
        self.assertIn("1..3", windows)
        self.assertIn('$StartInfo.EnvironmentVariables["OLLAMA_HOST"] = "http://127.0.0.1:1"', windows)
        self.assertIn("WaitForExit(5000)", windows)
        self.assertIn(".Wait(500)", windows)
        self.assertIn("ReceiveTimeout = 500", windows)
        self.assertIn("finally", windows)
        for launcher in (posix, windows):
            self.assertIn("llama-server", launcher)
            self.assertIn("ollama.log", launcher)
            self.assertIn("/api/version", launcher)
            self.assertNotIn(" list ", launcher)
        self.assertIn("taskkill.exe", windows)
        self.assertIn(" /T /F", windows)

    # Check that POSIX launcher preflights glibc before runtime preparation.
    def test_posix_launcher_preflights_glibc_before_runtime_preparation(self):
        posix = (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8")
        self.assertIn('MINIMUM_GLIBC_VERSION="2.28"', posix)
        self.assertIn("getconf GNU_LIBC_VERSION", posix)
        self.assertIn("musl Linux is not supported", posix)
        self.assertLess(posix.index("require_supported_glibc\n"), posix.index('mkdir -p "$INPUTS"'))

    # Check that ci pins and exercises the supported platform matrix.
    def test_ci_pins_and_exercises_the_supported_platform_matrix(self):
        workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
        for runner in ("windows-2025", "macos-15", "macos-15-intel", "ubuntu-24.04", "ubuntu-24.04-arm"):
            with self.subTest(runner=runner):
                self.assertIn(f"os: {runner}", workflow)
        self.assertNotIn("runs-on: ubuntu-latest", workflow)
        self.assertNotIn("os: macos-latest", workflow)
        self.assertNotIn("os: windows-latest", workflow)
        for command in ("uv run uma-bct --help", "uv run python -m uma_bct --help", "uv run python main.py --help"):
            self.assertEqual(workflow.count(command), 1)
        self.assertEqual(workflow.count("name: Docker smoke (${{ matrix.name }})"), 1)

    # Check that POSIX launcher has valid shell syntax.
    @unittest.skipIf(os.name == "nt", "POSIX shell test")
    def test_posix_launcher_has_valid_shell_syntax(self):
        subprocess.run(["sh", "-n", str(ROOT / "run-uma-bct.sh")], check=True, capture_output=True, text=True)
        self.assertIn('"$APP_CLI" --path-to-inputs', (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8"))
        self.assertIn('"--path-to-inputs "', (ROOT / "run-uma-bct.ps1").read_text(encoding="utf-8"))

    # Check that macOS asks Finder for the photo folder and gives its path if opening fails.
    @unittest.skipIf(os.name == "nt", "POSIX shell fixture")
    def test_posix_folder_open_reports_a_finder_failure(self):
        source = (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8")
        start = source.index("open_folder() {")
        function = source[start:source.index("\n}\n", start) + 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "Add Photos Here"
            folder.mkdir()
            calls = root / "open-calls.txt"
            write_executable(root / "uname", "#!/bin/sh\nprintf 'Darwin\\n'\n")
            write_executable(
                root / "open",
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >>\"$TEST_OPEN_CALLS\"\n"
                "[ \"${TEST_OPEN_FAIL:-0}\" != 1 ]\n",
            )
            environment = os.environ.copy()
            environment.pop("UMA_BCT_NO_OPEN", None)
            environment.update(PATH=f"{root}:/usr/bin:/bin", TEST_FOLDER=str(folder), TEST_OPEN_CALLS=str(calls))
            command = f"set -eu\n{function}\nopen_folder \"$TEST_FOLDER\""

            opened = subprocess.run(["sh", "-c", command], env=environment, capture_output=True, text=True)
            self.assertEqual(opened.returncode, 0, opened.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["-a", "Finder", str(folder)])
            self.assertEqual(opened.stderr, "")

            calls.unlink()
            environment["TEST_OPEN_FAIL"] = "1"
            failed = subprocess.run(["sh", "-c", command], env=environment, capture_output=True, text=True)
            self.assertEqual(failed.returncode, 0, failed.stderr)
            self.assertIn(str(folder), failed.stderr)
            self.assertEqual(calls.read_text(encoding="utf-8").splitlines(), ["-a", "Finder", str(folder), str(folder)])

    # Check that POSIX launcher ignores a foreign default service and cleans up its child.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_ignores_a_foreign_default_service_and_cleans_up_its_child(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        owns_listener = False
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                listener.bind(("127.0.0.1", 11434))
                listener.listen()
                owns_listener = True
                thread = threading.Thread(target=lambda: listener.accept()[0].close(), daemon=True)
                thread.start()
            except OSError:
                listener.close()

            with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
                environment["OLLAMA_HOST"] = "http://127.0.0.1:1"
                result = run_launcher(launcher, environment)

                self.assertEqual(result.returncode, 0, result.stderr)
                selected_host = (root / "app-host.txt").read_text(encoding="utf-8").strip()
                self.assertRegex(selected_host, r"^http://127\.0\.0\.1:\d+$")
                self.assertNotEqual(selected_host, "http://127.0.0.1:11434")
                self.assertNotEqual(selected_host, "http://127.0.0.1:1")
                event_lines = events.read_text(encoding="utf-8").splitlines()
                self.assertEqual(event_names(events), ["started", "stopped"])
                child_pid = int(event_lines[0].split()[-1])
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)

            with socket.create_connection(("127.0.0.1", 11434), timeout=1):
                pass
        finally:
            if owns_listener:
                listener.close()

    # Check that POSIX launcher retries a failed bind and propagates all app exit codes.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_retries_a_failed_bind_and_propagates_all_app_exit_codes(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "1"
            result = run_launcher(launcher, environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(event_names(events), ["bind-failed", "started", "stopped"])

        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_OLLAMA_STALL_STARTS"] = "1"
            result = run_launcher(launcher, environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(event_names(events), ["stalled", "stopped", "started", "stopped"])

        for exit_code in (0, 1, 2):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_posix_launcher(root)
                environment["UMA_BCT_FAKE_APP_EXIT"] = str(exit_code)
                result = run_launcher(launcher, environment)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(event_names(events), ["started", "stopped"])

    # Check that POSIX launcher rejects missing companion and bounded startup failure.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_rejects_missing_companion_and_bounded_startup_failure(self):
        with launcher_fixture(prepare_posix_launcher, companion=False) as (root, launcher, environment, events):
            result = run_launcher(launcher, environment, 10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("llama-server", result.stderr)
            self.assertIn("ollama.log", result.stderr)
            self.assertFalse(events.exists())

        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "3"
            result = run_launcher(launcher, environment, 10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("three attempts", result.stderr)
            self.assertIn("ollama.log", result.stderr)
            self.assertEqual(len(events.read_text(encoding="utf-8").splitlines()), 3)

        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_OLLAMA_STALL_VERSION"] = "1"
            result = run_launcher(launcher, environment, 8)
            self.assertEqual(result.returncode, 2)
            self.assertIn("incomplete or unsupported", result.stderr)
            self.assertFalse(events.exists())

    # Check that POSIX interrupted Ollama extraction repairs cleanly on restart.
    @unittest.skipUnless(sys.platform == "darwin", "macOS staged-extraction fixture")
    def test_posix_interrupted_ollama_extraction_repairs_cleanly_on_restart(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            (root / "inputs/shelf.jpg").unlink()
            shutil.rmtree(root / ".uma-bct-runtime/ollama")

            archive_content = root / "archive-content"
            ollama = archive_content / "ollama"
            helper = archive_content / "llama-server"
            write_executable(ollama, FAKE_OLLAMA)
            write_executable(helper, "#!/bin/sh\nexit 0\n")
            archive = root / "fake-ollama.tgz"
            with tarfile.open(archive, "w:gz") as output:
                output.add(ollama, arcname="ollama")
                output.add(helper, arcname="llama-server")
            archive_digest = hashlib.sha256(archive.read_bytes()).hexdigest()

            fake_bin = root / "fake-bin"
            write_executable(
                fake_bin / "curl",
                "#!/bin/sh\ndestination=\"\"\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  if [ \"$1\" = \"-o\" ]; then shift; destination=$1; fi\n"
                "  shift\n"
                "done\n"
                "/bin/cp \"$UMA_BCT_FAKE_OLLAMA_ARCHIVE\" \"$destination\"\n",
            )
            write_executable(
                fake_bin / "tar",
                "#!/bin/sh\n"
                "if [ \"${UMA_BCT_FAKE_EXTRACT_STALL:-}\" = \"1\" ]; then\n"
                "  : >\"$UMA_BCT_FAKE_EXTRACT_MARKER\"\n"
                "  while :; do sleep 1; done\n"
                "fi\n"
                "exec /usr/bin/tar \"$@\"\n",
            )
            launcher_text = launcher.read_text(encoding="utf-8")
            launcher_text = launcher_text.replace(
                'PATH="/usr/bin:/bin:/usr/sbin:/sbin"',
                f'PATH="{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin"',
            )
            launcher_text = launcher_text.replace(
                EXPECTED_OLLAMA_DARWIN_SHA256,
                archive_digest,
            )
            launcher.write_text(launcher_text, encoding="utf-8")
            marker = root / "extract-started.txt"
            environment.update(
                UMA_BCT_FAKE_OLLAMA_ARCHIVE=str(archive),
                UMA_BCT_FAKE_EXTRACT_MARKER=str(marker),
                UMA_BCT_FAKE_EXTRACT_STALL="1",
            )
            first = start_launcher(launcher, environment, start_new_session=True)
            try:
                wait_until(first, marker.exists, "Launcher exited before staged extraction started")
                os.killpg(first.pid, signal.SIGKILL)
                first.communicate(timeout=10)
            finally:
                if first.poll() is None:
                    os.killpg(first.pid, signal.SIGKILL)
                    first.communicate()
            self.assertTrue((root / ".uma-bct-runtime/ollama-install").is_dir())

            environment.pop("UMA_BCT_FAKE_EXTRACT_STALL")
            result = run_launcher(launcher, environment, 25)

            self.assertEqual(result.returncode, 2)
            self.assertIn("No supported photos", result.stderr)
            self.assertIn(str((root / "Add Photos Here").resolve()), result.stdout)
            self.assertTrue((root / ".uma-bct-runtime/ollama/ollama").is_file())
            self.assertTrue((root / ".uma-bct-runtime/ollama/llama-server").is_file())
            self.assertFalse((root / ".uma-bct-runtime/ollama-install").exists())
            self.assertEqual(event_names(events), ["started", "stopped"])

    # Check that POSIX launcher cleans up when interrupted.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_cleans_up_when_interrupted(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_INTERRUPT"] = "1"
            result = run_launcher(launcher, environment)

            self.assertEqual(result.returncode, 143)
            self.assertEqual(event_names(events), ["started", "stopped"])

    # Check that POSIX launcher uses friendly layout and rejects mixed private data.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_uses_friendly_layout_and_rejects_mixed_private_data(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            (root / "inputs").rename(root / "Add Photos Here")
            (root / "outputs").rename(root / "Results")

            result = run_launcher(launcher, environment)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(events.is_file())

        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            friendly = root / "Add Photos Here"
            friendly.mkdir()
            (friendly / "other.jpg").write_bytes(b"other private photo")
            (root / "Results").mkdir()

            result = run_launcher(launcher, environment, 10)

            self.assertEqual(result.returncode, 2)
            self.assertIn("both", result.stderr)
            self.assertIn("Add Photos Here", result.stderr)
            self.assertIn("inputs", result.stderr)
            self.assertFalse(events.exists())

    # Check that POSIX layout inspection rejects unreadable and noncanonical entries.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_layout_inspection_rejects_unreadable_and_noncanonical_entries(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            inputs = root / "inputs"
            inputs.chmod(0)
            try:
                result = run_launcher(launcher, environment, 10)
            finally:
                inputs.chmod(0o700)
            self.assertEqual(result.returncode, 2)
            self.assertIn("inspected safely", result.stderr)
            self.assertFalse(events.exists())

        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            try:
                (root / "Add Photos Here").symlink_to(root / "missing-friendly-directory")
            except OSError as error:
                self.skipTest(str(error))
            result = run_launcher(launcher, environment, 10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("symbolic link", result.stderr)
            self.assertFalse(events.exists())

        for malformed in ("multiline", "case", "hardlink"):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_posix_launcher(root)
                results = root / "Results"
                results.mkdir()
                lock = results / ".uma-bct.lock"
                if malformed == "hardlink":
                    source = root / "lock-source"
                    source.write_text('{"schema_version": 1, "pid": 1}\n', encoding="utf-8")
                    try:
                        os.link(source, lock)
                    except OSError as error:
                        self.skipTest(str(error))
                elif malformed == "case":
                    lock.write_text('{"Schema_version": 1, "pid": 1}\n', encoding="utf-8")
                else:
                    lock.write_text('{"schema_version": 1, "pid": 1}\nextra\n', encoding="utf-8")

                result = run_launcher(launcher, environment, 10)

                self.assertEqual(result.returncode, 2)
                self.assertIn("both", result.stderr)
                self.assertFalse(events.exists())

    # Check that POSIX layout ignores only a canonical unlinked results lock.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_layout_ignores_only_a_canonical_unlinked_results_lock(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            results = root / "Results"
            results.mkdir()
            (results / ".uma-bct.lock").write_text(
                '{"schema_version": 1, "pid": 1}\n', encoding="utf-8"
            )

            result = run_launcher(launcher, environment)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(events.is_file())

    # Check that POSIX launcher supports bundle paths with spaces and unicode.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_supports_bundle_paths_with_spaces_and_unicode(self):
        with launcher_fixture(
            prepare_posix_launcher, root_name="Archive Client ü"
        ) as (root, launcher, environment, events):

            result = run_launcher(launcher, environment)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(event_names(events), ["started", "stopped"])

    # Check that two starts of one POSIX bundle are exclusive.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_two_starts_of_one_posix_bundle_are_exclusive(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            first = start_launcher(launcher, environment)
            try:
                wait_until(
                    first,
                    lambda: events.exists() and (root / "app-host.txt").exists(),
                    "First launcher exited before its app became ready",
                )
                second = subprocess.run(
                    ["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=10
                )
                self.assertEqual(second.returncode, 2)
                self.assertIn("already running", second.stderr)
                self.assertEqual(event_names(events), ["started"])
            finally:
                if first.poll() is None:
                    first.terminate()
                first.communicate(timeout=10)

    # Check that POSIX launcher recovers a well formed dead owner reservation.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_recovers_a_well_formed_dead_owner_reservation(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            reservation = root / ".uma-bct-launcher.lock"
            reservation.mkdir(mode=0o700)
            (reservation / "pid").write_text("999999999\n", encoding="utf-8")

            result = run_launcher(launcher, environment)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(reservation.exists())
            self.assertEqual(event_names(events), ["started", "stopped"])

        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            (root / ".uma-bct-launcher.lock").mkdir(mode=0o700)

            result = run_launcher(launcher, environment)

            self.assertEqual(result.returncode, 2)
            self.assertIn("malformed", result.stderr)
            self.assertFalse(events.exists())

        for malformed in ("multiline", "nul", "hardlink"):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_posix_launcher(root)
                reservation = root / ".uma-bct-launcher.lock"
                reservation.mkdir(mode=0o700)
                owner = reservation / "pid"
                if malformed == "multiline":
                    owner.write_text("999999999\ndo-not-delete\n", encoding="utf-8")
                elif malformed == "nul":
                    owner.write_bytes(b"999999999\x00\n")
                else:
                    sentinel = root / "owner-sentinel.txt"
                    sentinel.write_text("999999999\n", encoding="utf-8")
                    try:
                        os.link(sentinel, owner)
                    except OSError as error:
                        self.skipTest(str(error))

                result = run_launcher(launcher, environment, 10)

                self.assertEqual(result.returncode, 2)
                self.assertTrue(reservation.is_dir())
                self.assertTrue(owner.is_file())
                self.assertFalse(events.exists())

    # Check that two POSIX recoverers cannot steal the winners new reservation.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_two_posix_recoverers_cannot_steal_the_winners_new_reservation(self):
        with launcher_fixture(prepare_posix_launcher) as (root, launcher, environment, events):
            reservation = root / ".uma-bct-launcher.lock"
            reservation.mkdir(mode=0o700)
            (reservation / "pid").write_text("999999999\n", encoding="utf-8")
            environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            processes = [
                start_launcher(launcher, environment)
                for _index in range(2)
            ]
            try:
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    exited = [process for process in processes if process.poll() is not None]
                    running = [process for process in processes if process.poll() is None]
                    if len(exited) == 1 and len(running) == 1 and (root / "app-ready.txt").exists():
                        break
                    time.sleep(0.1)
                else:
                    self.fail("Concurrent reservation recovery did not select one winner")
                self.assertEqual(exited[0].returncode, 2)
                self.assertTrue(reservation.is_dir())
                self.assertEqual(len(list(reservation.iterdir())), 1)
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                    process.communicate(timeout=10)

            self.assertEqual(event_names(events), ["started", "stopped"])

    # Check that two POSIX bundles use independent services.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_two_posix_bundles_use_independent_services(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first_root = parent / "Client One"
            second_root = parent / "Client Two ü"
            first_root.mkdir()
            second_root.mkdir()
            first_launcher, first_environment, first_events = prepare_posix_launcher(first_root)
            second_launcher, second_environment, second_events = prepare_posix_launcher(second_root)
            first_environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            second_environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            first = start_launcher(first_launcher, first_environment)
            second = start_launcher(second_launcher, second_environment)
            try:
                deadline = time.monotonic() + 12
                while time.monotonic() < deadline:
                    if (first_root / "app-host.txt").exists() and (second_root / "app-host.txt").exists():
                        break
                    if first.poll() is not None or second.poll() is not None:
                        self.fail("A concurrent launcher exited before both apps became ready")
                    time.sleep(0.1)
                first_host = (first_root / "app-host.txt").read_text(encoding="utf-8").strip()
                second_host = (second_root / "app-host.txt").read_text(encoding="utf-8").strip()
                self.assertNotEqual(first_host, second_host)

                first.terminate()
                first.communicate(timeout=10)
                self.assertIsNone(second.poll())
                second_port = int(second_host.rsplit(":", 1)[1])
                with socket.create_connection(("127.0.0.1", second_port), timeout=1) as connection:
                    connection.sendall(b"GET /api/version HTTP/1.0\r\n\r\n")
                    self.assertIn(b"200 OK", connection.recv(128))
            finally:
                for process in (first, second):
                    if process.poll() is None:
                        process.terminate()
                    process.communicate(timeout=10)

            self.assertEqual(event_names(first_events), ["started", "stopped"])
            self.assertEqual(event_names(second_events), ["started", "stopped"])

    # Check that POSIX launcher forwards real interrupts to a blocked CLI.
    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_forwards_real_interrupts_to_a_blocked_cli(self):
        for sent_signal, expected_status in ((signal.SIGINT, 130), (signal.SIGHUP, 129)):
            with self.subTest(sent_signal=sent_signal), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_posix_launcher(root)
                environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
                process = start_launcher(launcher, environment)
                try:
                    wait_until(
                        process,
                        lambda events=events, root=root: events.exists() and (root / "app-ready.txt").exists(),
                        "Launcher exited before the blocked CLI was ready",
                    )
                    process.send_signal(sent_signal)
                    process.communicate(timeout=10)
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.communicate()

                self.assertEqual(process.returncode, expected_status)
                expected_name = "INT" if sent_signal == signal.SIGINT else "HUP"
                self.assertEqual((root / "app-signals.txt").read_text(encoding="utf-8").splitlines(), [expected_name])
                self.assertEqual(event_names(events), ["started", "stopped"])

    # Check that windows launcher has valid powershell syntax.
    @unittest.skipUnless(os.name == "nt", "Windows launcher test")
    def test_windows_launcher_has_valid_powershell_syntax(self):
        script = ROOT / "run-uma-bct.ps1"
        command = (
            "$errors=$null; [void][System.Management.Automation.Language.Parser]::"
            f"ParseFile('{script}', [ref]$null, [ref]$errors); "
            "if ($errors.Count) { $errors | Out-String | Write-Error }"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True, capture_output=True, text=True)

    # Check that windows launcher uses dynamic ports retries and cleans up.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_launcher_uses_dynamic_ports_retries_and_cleans_up(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        owns_listener = False
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 11434))
            listener.listen()
            owns_listener = True
        except OSError:
            listener.close()
        try:
            with launcher_fixture(
                prepare_windows_launcher, self.compiled_fixture
            ) as (root, launcher, environment, events):
                environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "1"
                environment["OLLAMA_HOST"] = "http://127.0.0.1:1"
                result = run_launcher(launcher, environment, 20)
                self.assertEqual(result.returncode, 0, result.stderr)
                selected_host = (root / "app-host.txt").read_text(encoding="utf-8").strip()
                self.assertRegex(selected_host, r"^http://127\.0\.0\.1:\d+$")
                self.assertNotEqual(selected_host, "http://127.0.0.1:11434")
                self.assertNotEqual(selected_host, "http://127.0.0.1:1")
                event_lines = events.read_text(encoding="utf-8").splitlines()
                self.assertEqual(event_names(events), ["bind-failed", "runner-started", "started"])
                for line in event_lines[-2:]:
                    with self.assertRaises(OSError):
                        os.kill(int(line.split()[-1]), 0)

            with socket.create_connection(("127.0.0.1", 11434), timeout=1):
                pass
        finally:
            if owns_listener:
                listener.close()

    # Check that windows cmd entrypoint runs the powershell launcher.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_cmd_entrypoint_runs_the_powershell_launcher(self):
        with launcher_fixture(
            prepare_windows_launcher, self.compiled_fixture
        ) as (root, _launcher, environment, events):
            command = root / "Run UMA BCT.cmd"
            shutil.copy2(ROOT / "Run UMA BCT.cmd", command)

            result = subprocess.run(
                ["cmd.exe", "/d", "/c", str(command)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(events.is_file())

    # Check that windows cmd entrypoint propagates ctrl break as 130.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_cmd_entrypoint_propagates_ctrl_break_as_130(self):
        with launcher_fixture(
            prepare_windows_launcher, self.compiled_fixture
        ) as (root, _launcher, environment, _events):
            command = root / "Run UMA BCT.cmd"
            shutil.copy2(ROOT / "Run UMA BCT.cmd", command)
            environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            process = subprocess.Popen(
                ["cmd.exe", "/d", "/c", str(command)],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and not (root / "app-host.txt").exists():
                    if process.poll() is not None:
                        self.fail("CMD launcher exited before the interruption fixture was ready")
                    time.sleep(0.1)
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.communicate(timeout=20)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

            self.assertEqual(process.returncode, 130)
            self.assertEqual(
                (root / "app-signals.txt").read_text(encoding="utf-8").splitlines(), ["ControlBreak"]
            )

    # Check that windows launcher uses friendly unicode layout and rejects mixed data.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_launcher_uses_friendly_unicode_layout_and_rejects_mixed_data(self):
        with launcher_fixture(
            prepare_windows_launcher, self.compiled_fixture, root_name="Archive Client ü"
        ) as (root, launcher, environment, events):
            (root / "inputs").rename(root / "Add Photos Here")
            (root / "outputs").rename(root / "Results")

            result = run_launcher(launcher, environment, 20)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(events.is_file())

        with launcher_fixture(prepare_windows_launcher, self.compiled_fixture) as (root, launcher, environment, events):
            friendly = root / "Add Photos Here"
            friendly.mkdir()
            (friendly / "other.jpg").write_bytes(b"other private photo")
            (root / "Results").mkdir()
            result = run_launcher(launcher, environment, 10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("both", result.stderr)
            self.assertFalse(events.exists())

    # Check that two starts of one windows bundle are exclusive.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_two_starts_of_one_windows_bundle_are_exclusive(self):
        with launcher_fixture(
            prepare_windows_launcher, self.compiled_fixture
        ) as (root, launcher, environment, _events):
            environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            first = start_launcher(
                launcher, environment, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            try:
                wait_until(
                    first,
                    (root / "app-pid.txt").exists,
                    "First Windows launcher exited before its app was ready",
                )
                second = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(second.returncode, 2)
                self.assertIn("already running", second.stderr)
            finally:
                if first.poll() is None:
                    first.send_signal(signal.CTRL_BREAK_EVENT)
                first.communicate(timeout=20)

    # Check that windows launcher rejects a hardlinked bundle reservation without modifying target.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_launcher_rejects_a_hardlinked_bundle_reservation_without_modifying_target(self):
        with launcher_fixture(
            prepare_windows_launcher, self.compiled_fixture
        ) as (root, launcher, environment, _events):
            sentinel = root / "reservation-sentinel.txt"
            sentinel.write_bytes(b"do not overwrite")
            try:
                os.link(sentinel, root / ".uma-bct-launcher.lock")
            except OSError as error:
                self.skipTest(str(error))

            result = run_launcher(launcher, environment, 10)

            self.assertEqual(result.returncode, 2)
            self.assertEqual(sentinel.read_bytes(), b"do not overwrite")

    # Check that windows launcher rejects missing companion and propagates exit codes.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_launcher_rejects_missing_companion_and_propagates_exit_codes(self):
        with launcher_fixture(
            prepare_windows_launcher, self.compiled_fixture, companion=False
        ) as (root, launcher, environment, events):
            result = run_launcher(launcher, environment, 10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("llama-server.exe", result.stderr)
            self.assertIn("ollama.log", result.stderr)
            self.assertFalse(events.exists())

        with launcher_fixture(prepare_windows_launcher, self.compiled_fixture) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_OLLAMA_STALL_STARTS"] = "1"
            result = run_launcher(launcher, environment)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(event_names(events), ["stalled", "stopped", "runner-started", "started"])
            for line in events.read_text(encoding="utf-8").splitlines()[-2:]:
                with self.assertRaises(OSError):
                    os.kill(int(line.split()[-1]), 0)

        with launcher_fixture(prepare_windows_launcher, self.compiled_fixture) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "3"
            result = run_launcher(launcher, environment)
            self.assertEqual(result.returncode, 2)
            self.assertIn("three attempts", result.stderr)
            self.assertEqual(len(events.read_text(encoding="utf-8").splitlines()), 3)

        with launcher_fixture(prepare_windows_launcher, self.compiled_fixture) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_OLLAMA_STALL_VERSION"] = "1"
            result = run_launcher(launcher, environment, 10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("incomplete or unsupported", result.stderr)
            self.assertFalse(events.exists())

        for exit_code in (0, 1, 2):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture)
                environment["UMA_BCT_FAKE_APP_EXIT"] = str(exit_code)
                result = run_launcher(launcher, environment, 20)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                event_lines = events.read_text(encoding="utf-8").splitlines()
                self.assertEqual(event_names(events), ["runner-started", "started"])
                for line in event_lines:
                    with self.assertRaises(OSError):
                        os.kill(int(line.split()[-1]), 0)

    # Check that windows launcher cleans up its process tree on ctrl break.
    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_launcher_cleans_up_its_process_tree_on_ctrl_break(self):
        with launcher_fixture(prepare_windows_launcher, self.compiled_fixture) as (root, launcher, environment, events):
            environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            process = start_launcher(
                launcher, environment, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            try:
                wait_until(
                    process,
                    lambda: events.exists() and (root / "app-host.txt").exists(),
                    "Windows interruption fixture did not become ready",
                )

                event_lines = events.read_text(encoding="utf-8").splitlines()
                app_pid = int((root / "app-pid.txt").read_text(encoding="utf-8"))
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.communicate(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

            self.assertEqual(event_names(events), ["runner-started", "started"])
            self.assertEqual(process.returncode, 130)
            self.assertEqual(
                (root / "app-signals.txt").read_text(encoding="utf-8").splitlines(), ["ControlBreak"]
            )
            with self.assertRaises(OSError):
                os.kill(app_pid, 0)
            for line in event_lines:
                with self.assertRaises(OSError):
                    os.kill(int(line.split()[-1]), 0)


if __name__ == "__main__":
    unittest.main()
