"""Exercise the real CLI and adapters against a deterministic local Ollama API.

Only the API response is scripted: discovery, pixels, HTTP, parsing, crops,
verification, checkpoints, exports and exit codes use production code.
"""
from __future__ import annotations

import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from helpers import read_excel
from PIL import Image

from uma_bct.adapters.ollama_client import DEFAULT_MODEL

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def scripted_ollama(*, added_photo: Path | None = None, block_verification: bool = False):
    counts = Counter()
    ready, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, value, *, streaming=False):
            body = (json.dumps(value) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson" if streaming else "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # The interruption test deliberately disconnects a waiting client.

        def do_GET(self):
            if self.path != "/api/tags":
                self.send_error(404)
                return
            self.respond({"models": [{"model": DEFAULT_MODEL, "digest": "sha256:synthetic"}]})

        def do_POST(self):
            nonlocal block_verification, added_photo
            if self.path != "/api/chat":
                self.send_error(404)
                return
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            message = request["messages"][-1]
            prompt = message["content"]
            if not message.get("images"):
                counts["preflight"] += 1
                content = "ok"
            else:
                assert base64.b64decode(message["images"][0]).startswith(b"\x89PNG")
                properties = request["format"]["properties"]["boxes"]["items"]["properties"]
                if "bbox_2d" in properties:
                    counts["detection"] += 1
                    if added_photo is not None:
                        Image.new("RGB", (320, 200), "tan").save(added_photo)
                        added_photo = None
                    value = {"boxes": [{"bbox_2d": [100, 100, 900, 900]}]}
                else:
                    if "Two readings disagree" in prompt:
                        counts["third"] += 1
                        unit = "5"
                    elif "Independently check every visible character" in prompt:
                        counts["second"] += 1
                        if block_verification:
                            block_verification = False
                            ready.set()
                            release.wait(timeout=30)
                        unit = "5"
                    else:
                        counts["first"] += 1
                        unit = "4"
                    value = {"boxes": [{"position": "middle-left", "record_identifiers": ["1960.0009"],
                                         "unit": unit, "typed": ["1960.0009", f"Unit {unit}"],
                                         "handwritten": ["Review annotation"], "auxiliary_annotations": []}]}
                content = json.dumps(value)
            self.respond({"model": DEFAULT_MODEL, "message": {"role": "assistant", "content": content},
                          "done": True}, streaming=request.get("stream", False))

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", counts, ready, release
        finally:
            release.set()
            server.shutdown()
            thread.join(timeout=5)


class CliIntegrationTest(unittest.TestCase):
    def command(self, root, *, optional_features=True):
        options = ["--file-type", "excel", "--verification", "always"] if optional_features else []
        return [sys.executable, "-m", "uma_bct", "-i", str(root / "inputs"),
                "--results-directory", str(root / "results"), *options, "--json"]

    def environment(self, endpoint):
        return dict(os.environ, OLLAMA_HOST=endpoint, OLLAMA_NO_CLOUD="1", PYTHONPATH=str(ROOT / "src"))

    def prepare(self, root):
        (root / "inputs").mkdir()
        Image.new("RGB", (320, 200), "tan").save(root / "inputs/01 é #.png")

    def run_cli(self, root, endpoint):
        result = subprocess.run(self.command(root), cwd=root, env=self.environment(endpoint),
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout)

    def test_default_cli_has_one_reading_csv_and_no_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            with scripted_ollama() as (endpoint, counts, _ready, _release):
                environment = self.environment(endpoint)
                environment["UMA_BCT_VERIFICATION"] = "always"
                result = subprocess.run(self.command(root, optional_features=False), cwd=root, env=environment,
                                        capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(Path(summary["catalogue_output"]).suffix, ".csv")
            self.assertNotIn("review_output", summary)
            self.assertEqual(counts["detection"], 1)
            self.assertEqual(counts["first"], 1)
            self.assertEqual(counts["second"], 0)
            self.assertEqual(counts["third"], 0)

    def test_real_cli_defers_additions_and_checks_only_disputed_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            with scripted_ollama(added_photo=root / "inputs/02.png") as (endpoint, counts, _ready, _release):
                first = self.run_cli(root, endpoint)
                self.assertEqual(first["images_total"], 1)
                self.assertEqual(counts["detection"], 1)
                self.assertEqual(counts["first"], 1)
                self.assertEqual(counts["second"], 1)
                self.assertEqual(counts["third"], 1)
                self.assertEqual(Path(first["catalogue_output"]).suffix, ".xlsx")
                rows, links = read_excel(Path(first["catalogue_output"]))
                self.assertEqual(rows[1][1], "0005")
                self.assertIn("Verification discrepancy", rows[1][5])
                self.assertIn("G2", links)
                self.assertFalse((Path(first["batch_directory"]) / "Review.html").exists())
                second = self.run_cli(root, endpoint)
                self.assertEqual(second["images_total"], 1)
                self.assertNotEqual(first["batch_directory"], second["batch_directory"])
                self.assertEqual(counts["detection"], 2)
                before = counts.copy()
                again = self.run_cli(root, endpoint)
                self.assertTrue(again["already_complete"])
                self.assertEqual(again["images_total"], 2)
                self.assertEqual(len(again["catalogue_outputs"]), 2)
                self.assertEqual(counts, before)

    @unittest.skipIf(os.name == "nt", "POSIX interruption; Windows launcher signals have dedicated fixtures")
    def test_real_cli_interrupt_and_resume_preserve_first_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            with scripted_ollama(block_verification=True) as (endpoint, counts, ready, release):
                process = subprocess.Popen(self.command(root), cwd=root, env=self.environment(endpoint),
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    self.assertTrue(ready.wait(timeout=20), "CLI never reached its second reading")
                    process.send_signal(signal.SIGINT)
                    stdout, stderr = process.communicate(timeout=15)
                    self.assertEqual(process.returncode, 130, stderr)
                    self.assertEqual(json.loads(stdout)["status"], "interrupted")
                finally:
                    release.set()
                    if process.poll() is None:
                        process.kill()
                        process.communicate()
                self.assertEqual(counts["detection"], 1)
                resumed = self.run_cli(root, endpoint)
                self.assertTrue(resumed["resumed"])
                self.assertEqual(counts["detection"], 1)
                self.assertEqual(counts["first"], 1)
                self.assertEqual(counts["second"], 2)  # Only the interrupted request repeats.
                self.assertEqual(counts["third"], 1)
                self.assertTrue(Path(resumed["catalogue_output"]).is_file())
