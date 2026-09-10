import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path

from scripts.build_client_bundle import BUNDLE_ROOT, CLIENT_FILES, build_client_bundle, verify_client_bundle

ROOT = Path(__file__).resolve().parents[1]

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
    print("ollama version is 0.30.8")
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


def write_executable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o700)


def prepare_posix_launcher(root: Path, *, companion: bool = True) -> tuple[Path, dict[str, str], Path]:
    launcher = root / "run-uma-bct.sh"
    shutil.copy2(ROOT / "run-uma-bct.sh", launcher)
    (root / "client-constraints.txt").write_text("locked", encoding="utf-8")
    (root / "uma_bct_cli-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "shelf.jpg").write_bytes(b"private photo fixture")
    (root / "outputs").mkdir()

    runtime = root / ".uma-bct-runtime"
    write_executable(
        runtime / "bin/uv", '#!/bin/sh\nif [ "${1:-}" = "--version" ]; then printf \'uv 0.12.9\\n\'; fi\nexit 0\n'
    )
    write_executable(runtime / "app/bin/python", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    write_executable(
        runtime / "app/bin/uma-bct",
        '#!/bin/sh\nprintf \'%s\\n\' "$OLLAMA_HOST" >"$UMA_BCT_FAKE_APP_HOST"\n'
        'if [ "${UMA_BCT_FAKE_INTERRUPT:-}" = "1" ]; then kill -TERM "$PPID"; fi\n'
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

    events = root / "ollama-events.txt"
    environment = os.environ.copy()
    environment.update(
        UMA_BCT_NO_OPEN="1",
        UMA_BCT_FAKE_APP_HOST=str(root / "app-host.txt"),
        UMA_BCT_FAKE_OLLAMA_EVENTS=str(events),
        UMA_BCT_FAKE_OLLAMA_COUNTER=str(root / "ollama-counter.txt"),
    )
    return launcher, environment, events


def prepare_windows_launcher(
    root: Path, fake_executable: Path, *, companion: bool = True
) -> tuple[Path, dict[str, str], Path]:
    launcher = root / "run-uma-bct.ps1"
    shutil.copy2(ROOT / "run-uma-bct.ps1", launcher)
    (root / "client-constraints.txt").write_text("locked", encoding="utf-8")
    (root / "uma_bct_cli-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    inputs = root / "inputs"
    inputs.mkdir()
    (inputs / "shelf.jpg").write_bytes(b"private photo fixture")
    (root / "outputs").mkdir()

    runtime = root / ".uma-bct-runtime"
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

    events = root / "ollama-events.txt"
    environment = os.environ.copy()
    environment.update(
        LOCALAPPDATA=str(local_app_data),
        ProgramFiles=str(root / "program-files"),
        UMA_BCT_NO_OPEN="1",
        UMA_BCT_FAKE_APP_HOST=str(root / "app-host.txt"),
        UMA_BCT_FAKE_OLLAMA_EVENTS=str(events),
        UMA_BCT_FAKE_OLLAMA_COUNTER=str(root / "ollama-counter.txt"),
    )
    return launcher, environment, events


class ClientDeliveryTest(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls):
        if cls.fixture_directory is not None:
            cls.fixture_directory.cleanup()

    def test_bundle_contains_only_the_client_files_and_one_wheel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / "dist"
            dist.mkdir()
            wheel = dist / "uma_bct_cli-0.1.0-py3-none-any.whl"
            wheel.write_bytes(b"wheel")
            for name in CLIENT_FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")

            bundle = build_client_bundle(dist, source_root=root)
            with zipfile.ZipFile(bundle) as archive:
                self.assertEqual(
                    set(archive.namelist()),
                    {*(f"{BUNDLE_ROOT}/{Path(name).name}" for name in CLIENT_FILES), f"{BUNDLE_ROOT}/{wheel.name}"},
                )
            bundle.write_bytes(b"not a zip")
            with self.assertRaisesRegex(ValueError, "verify"):
                verify_client_bundle(dist, source_root=root)

    def test_dependency_lock_is_generated_and_excludes_deferred_exporters(self):
        constraints = (ROOT / "client-constraints.txt").read_text(encoding="utf-8")
        self.assertIn("autogenerated by uv", constraints)
        self.assertIn("zstandard==", constraints)
        self.assertNotIn("pandas==", constraints)
        self.assertNotIn("openpyxl==", constraints)

    def test_launchers_pin_downloads_and_prepare_before_accepting_photos(self):
        posix = (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8")
        windows = (ROOT / "run-uma-bct.ps1").read_text(encoding="utf-8")
        for launcher in (posix, windows):
            self.assertIn("0.12.9", launcher)
            self.assertIn("0.30.8", launcher)
            self.assertIn("127.0.0.1", launcher)
            self.assertIn("client-constraints.txt", launcher)
        self.assertLess(posix.index('setup >"$RUNTIME/setup.json"'), posix.index("Press Enter after"))
        self.assertLess(windows.index("& $AppCli setup"), windows.index("Read-Host"))
        self.assertIn("Get-AuthenticodeSignature", windows)
        self.assertIn("ollama-windows-amd64.zip", windows)
        self.assertNotIn("OllamaSetup.exe", windows)
        self.assertNotIn(r"Programs\Ollama", windows)

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

    def test_posix_launcher_preflights_glibc_before_runtime_preparation(self):
        posix = (ROOT / "run-uma-bct.sh").read_text(encoding="utf-8")
        self.assertIn('MINIMUM_GLIBC_VERSION="2.28"', posix)
        self.assertIn("getconf GNU_LIBC_VERSION", posix)
        self.assertIn("musl Linux is not supported", posix)
        self.assertLess(posix.index("require_supported_glibc\n"), posix.index('mkdir -p "$INPUTS"'))

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

    @unittest.skipIf(os.name == "nt", "POSIX shell test")
    def test_posix_launcher_has_valid_shell_syntax(self):
        subprocess.run(["sh", "-n", str(ROOT / "run-uma-bct.sh")], check=True, capture_output=True, text=True)

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

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_posix_launcher(root)
                result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=15)

                self.assertEqual(result.returncode, 0, result.stderr)
                selected_host = (root / "app-host.txt").read_text(encoding="utf-8").strip()
                self.assertRegex(selected_host, r"^http://127\.0\.0\.1:\d+$")
                self.assertNotEqual(selected_host, "http://127.0.0.1:11434")
                event_lines = events.read_text(encoding="utf-8").splitlines()
                self.assertEqual([line.split()[0] for line in event_lines], ["started", "stopped"])
                child_pid = int(event_lines[0].split()[-1])
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)

            with socket.create_connection(("127.0.0.1", 11434), timeout=1):
                pass
        finally:
            if owns_listener:
                listener.close()

    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_retries_a_failed_bind_and_propagates_all_app_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_posix_launcher(root)
            environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "1"
            result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [line.split()[0] for line in events.read_text(encoding="utf-8").splitlines()],
                ["bind-failed", "started", "stopped"],
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_posix_launcher(root)
            environment["UMA_BCT_FAKE_OLLAMA_STALL_STARTS"] = "1"
            result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [line.split()[0] for line in events.read_text(encoding="utf-8").splitlines()],
                ["stalled", "stopped", "started", "stopped"],
            )

        for exit_code in (0, 1, 2):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_posix_launcher(root)
                environment["UMA_BCT_FAKE_APP_EXIT"] = str(exit_code)
                result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, exit_code, result.stderr)
                self.assertEqual(
                    [line.split()[0] for line in events.read_text(encoding="utf-8").splitlines()], ["started", "stopped"]
                )

    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_rejects_missing_companion_and_bounded_startup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_posix_launcher(root, companion=False)
            result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("llama-server", result.stderr)
            self.assertIn("ollama.log", result.stderr)
            self.assertFalse(events.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_posix_launcher(root)
            environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "3"
            result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("three attempts", result.stderr)
            self.assertIn("ollama.log", result.stderr)
            self.assertEqual(len(events.read_text(encoding="utf-8").splitlines()), 3)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_posix_launcher(root)
            environment["UMA_BCT_FAKE_OLLAMA_STALL_VERSION"] = "1"
            result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=8)
            self.assertEqual(result.returncode, 2)
            self.assertIn("incomplete or unsupported", result.stderr)
            self.assertFalse(events.exists())

    @unittest.skipIf(os.name == "nt", "POSIX launcher fixture")
    def test_posix_launcher_cleans_up_when_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_posix_launcher(root)
            environment["UMA_BCT_FAKE_INTERRUPT"] = "1"
            result = subprocess.run(["sh", str(launcher)], env=environment, capture_output=True, text=True, timeout=15)

            self.assertEqual(result.returncode, 143)
            self.assertEqual(
                [line.split()[0] for line in events.read_text(encoding="utf-8").splitlines()], ["started", "stopped"]
            )

    @unittest.skipUnless(os.name == "nt", "Windows launcher test")
    def test_windows_launcher_has_valid_powershell_syntax(self):
        script = ROOT / "run-uma-bct.ps1"
        command = (
            "$errors=$null; [void][System.Management.Automation.Language.Parser]::"
            f"ParseFile('{script}', [ref]$null, [ref]$errors); "
            "if ($errors.Count) { $errors | Out-String | Write-Error }"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True, capture_output=True, text=True)

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
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture)
                environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "1"
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                selected_host = (root / "app-host.txt").read_text(encoding="utf-8").strip()
                self.assertRegex(selected_host, r"^http://127\.0\.0\.1:\d+$")
                self.assertNotEqual(selected_host, "http://127.0.0.1:11434")
                event_lines = events.read_text(encoding="utf-8").splitlines()
                self.assertEqual([line.split()[0] for line in event_lines], ["bind-failed", "runner-started", "started"])
                for line in event_lines[-2:]:
                    with self.assertRaises(OSError):
                        os.kill(int(line.split()[-1]), 0)

            with socket.create_connection(("127.0.0.1", 11434), timeout=1):
                pass
        finally:
            if owns_listener:
                listener.close()

    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_launcher_rejects_missing_companion_and_propagates_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture, companion=False)
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("llama-server.exe", result.stderr)
            self.assertIn("ollama.log", result.stderr)
            self.assertFalse(events.exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture)
            environment["UMA_BCT_FAKE_OLLAMA_STALL_STARTS"] = "1"
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [line.split()[0] for line in events.read_text(encoding="utf-8").splitlines()],
                ["stalled", "stopped", "runner-started", "started"],
            )
            for line in events.read_text(encoding="utf-8").splitlines()[-2:]:
                with self.assertRaises(OSError):
                    os.kill(int(line.split()[-1]), 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture)
            environment["UMA_BCT_FAKE_OLLAMA_FAIL_STARTS"] = "3"
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("three attempts", result.stderr)
            self.assertEqual(len(events.read_text(encoding="utf-8").splitlines()), 3)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture)
            environment["UMA_BCT_FAKE_OLLAMA_STALL_VERSION"] = "1"
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("incomplete or unsupported", result.stderr)
            self.assertFalse(events.exists())

        for exit_code in (0, 1, 2):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture)
                environment["UMA_BCT_FAKE_APP_EXIT"] = str(exit_code)
                result = subprocess.run(
                    ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, exit_code, result.stderr)
                event_lines = events.read_text(encoding="utf-8").splitlines()
                self.assertEqual([line.split()[0] for line in event_lines], ["runner-started", "started"])
                for line in event_lines:
                    with self.assertRaises(OSError):
                        os.kill(int(line.split()[-1]), 0)

    @unittest.skipUnless(os.name == "nt", "Windows launcher fixture")
    def test_windows_launcher_cleans_up_its_process_tree_on_ctrl_break(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher, environment, events = prepare_windows_launcher(root, self.compiled_fixture)
            environment["UMA_BCT_FAKE_APP_WAIT"] = "1"
            process = subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(launcher)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if events.exists() and (root / "app-host.txt").exists():
                        break
                    if process.poll() is not None:
                        self.fail("Windows launcher exited before the interruption fixture was ready")
                    time.sleep(0.1)
                else:
                    self.fail("Windows interruption fixture did not become ready")

                event_lines = events.read_text(encoding="utf-8").splitlines()
                process.send_signal(signal.CTRL_BREAK_EVENT)
                process.communicate(timeout=10)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()

            self.assertEqual([line.split()[0] for line in event_lines], ["runner-started", "started"])
            for line in event_lines:
                with self.assertRaises(OSError):
                    os.kill(int(line.split()[-1]), 0)


if __name__ == "__main__":
    unittest.main()
