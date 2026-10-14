"""Synthetic private worker tests; no AWS, model weights, or photographs."""

import contextlib
import http.client
import io
import json
import os
import shutil
import socket
import ssl
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts import demo_ec2_pair as pair
from scripts import demo_ec2_peer as peer
from uma_bct.adapters.ollama_client import OllamaVisionModel
from uma_bct.bootstrap import build_pipeline


@unittest.skipUnless(os.name == "posix", "Temporary EC2 demo scripts target macOS/Linux")
class WorkerTest(unittest.TestCase):
    def test_private_client_loads_mutual_tls_without_proxies_or_redirects(self):
        factory, context = Mock(), Mock()
        environment = {"UMA_BCT_ALLOW_REMOTE_OLLAMA": "1", "UMA_BCT_DEMO_TLS_DIRECTORY": "/private/tls",
                       "UMA_BCT_DEMO_READER_HOSTS": '["https://172.31.7.196:9443"]'}
        with (patch.dict(os.environ, environment, clear=True), patch("ssl.create_default_context", return_value=context),
              patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=factory)})):
            OllamaVisionModel(host="https://172.31.7.196:9443").client()
        context.load_cert_chain.assert_called_once_with("/private/tls/client.crt", "/private/tls/client.key")
        factory.assert_called_once_with(host="https://172.31.7.196:9443", timeout=600.0,
                                       follow_redirects=False, trust_env=False, verify=context)

    def test_explicit_reader_list_retains_order_and_declared_count(self):
        hosts = ["http://ollama-parallel:11434", "http://ollama-parallel-3:11434", "http://ollama-parallel-4:11434",
                 *[f"https://172.31.7.196:{port}" for port in range(9443, 9447)]]
        environment = {"UMA_BCT_ALLOW_REMOTE_OLLAMA": "1", "UMA_BCT_ALLOW_CONTAINER_OLLAMA": "1",
                       "UMA_BCT_DEMO_TRANSCRIPTION_WORKERS": "8", "OLLAMA_HOST": "http://ollama:11434",
                       "UMA_BCT_DEMO_READER_HOSTS": json.dumps(hosts)}
        with (patch.dict(os.environ, environment, clear=True),
              patch.object(OllamaVisionModel, "preflight", return_value={"model": "same", "model_digest": "same"})):
            pipeline = build_pipeline(detector_name="qwen", transcriber_name="qwen", model_name="same",
                                      work_directory=Path("synthetic"), minimum_detector_confidence=0.5)
            self.assertEqual(pipeline.model_metadata["demo_parallel_ollama_host"], ",".join(hosts))
            os.environ["UMA_BCT_DEMO_READER_HOSTS"] = json.dumps(hosts[:-1])
            with self.assertRaisesRegex(ValueError, "declared worker count"):
                build_pipeline(detector_name="qwen", transcriber_name="qwen", model_name="same",
                               work_directory=Path("synthetic"), minimum_detector_confidence=0.5)

    def test_bind_address_requires_rfc1918(self):
        for value in ("0.0.0.0", "127.0.0.1", "169.254.1.2", "8.8.8.8", "224.0.0.1", "240.0.0.1", "::1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                peer.private_address(value)
        self.assertEqual(peer.private_address("172.31.7.196"), "172.31.7.196")

    def test_loader_registers_module_for_dataclasses(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "dependency.py"
            source.write_text("from dataclasses import dataclass\n@dataclass\nclass Example:\n    value: int = 1\n")
            self.assertEqual(peer.module(str(source)).Example().value, 1)

    def test_requests_retain_production_settings_and_payload(self):
        request = dict(model=peer.MODEL, think=False, options={"num_ctx": 16384, "temperature": 0},
                       messages=[{"images": ["synthetic-pixels"]}], logprobs=True, stream=True)
        body = json.dumps(request).encode()
        peer.validate_chat(body)
        self.assertEqual(json.loads(body), request)
        for change in ({"model": "other"}, {"think": True}, {"options": {"num_ctx": 8192, "temperature": 0}}):
            with self.assertRaises(ValueError):
                peer.validate_chat(json.dumps(request | change).encode())

    def test_incomplete_response_never_gets_terminal_http_chunk(self):
        for remainder, line in ((1, b'{"done":true}\n'), (None, b'{"message":{"content":"valid partial"}}\n')):
            response = SimpleNamespace(read1=Mock(side_effect=[line, b""]), length=remainder)
            output = io.BytesIO()
            with self.assertRaises((http.client.IncompleteRead, ValueError)):
                peer.relay(response, output, require_done=True)
            self.assertFalse(output.getvalue().endswith(b"0\r\n\r\n"))
            wire = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + output.getvalue()
            parsed = http.client.HTTPResponse(SimpleNamespace(makefile=lambda *args, wire=wire: io.BytesIO(wire)))
            parsed.begin()
            with self.assertRaises(http.client.IncompleteRead):
                parsed.read()

    def test_complete_stream_preserves_content_and_framing(self):
        body = b'{"message":{"content":"synthetic"},"done":false}\n{"done":true}\n'
        response = SimpleNamespace(read1=Mock(side_effect=[body[:25], body[25:], b""]), length=0)
        output = io.BytesIO()
        peer.relay(response, output, require_done=True)
        wire = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n" + output.getvalue()
        parsed = http.client.HTTPResponse(SimpleNamespace(makefile=lambda *args: io.BytesIO(wire)))
        parsed.begin()
        self.assertEqual(parsed.read(), body)

    def test_stream_accepts_separate_lines_and_split_terminal_line(self):
        for blocks in ([b'{"done":false}\n', b'{"done":true}\n'],
                       [b'{"done":false}\n{"do', b'ne":true}', b'\n']):
            output = io.BytesIO()
            response = SimpleNamespace(read1=Mock(side_effect=[*blocks, b""]), length=0)
            peer.relay(response, output, require_done=True)
            self.assertTrue(output.getvalue().endswith(b"0\r\n\r\n"))

    @unittest.skipUnless(os.name == "posix", "Linux worker uses flock")
    def test_inference_excludes_configuration_until_response_finishes(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "lock"
            lock.touch(mode=0o600)
            with patch.object(peer, "LOCK", lock), peer.inference_lock(), peer.inference_lock():
                with lock.open() as owner, self.assertRaises(BlockingIOError):
                    fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with lock.open() as owner:
                fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_tls_publication_occurs_inside_exclusive_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            incoming, base = root / "incoming", root / "peer"
            incoming.mkdir()
            staging = incoming / ("a" * 32 + "-tls")
            staging.mkdir(mode=0o700)
            for name in ("ca.crt", "client.crt", "client.key"):
                path = staging / name
                path.write_bytes(b"synthetic")
                path.chmod(0o600)
            locked = []

            @contextlib.contextmanager
            def lock():
                locked.append(True)
                yield
                locked.pop()

            def write(path, content, *args):
                self.assertTrue(locked)
                path.write_bytes(content)

            helper = SimpleNamespace(configuration_lock=lock, ubuntu_identity=lambda: (os.getuid(), os.getgid()), write=write)
            with (patch.object(peer, "module", return_value=helper), patch.object(peer.os, "geteuid", return_value=0),
                  patch.object(peer, "INCOMING", incoming), patch.object(peer, "BASE", base),
                  patch.object(peer, "INSTALLED", root / "installed"), patch.object(peer, "COORDINATOR", root / "coordinator"),
                  patch.object(peer, "client_context") as validate):
                peer.configure("coordinator", {"workers": 4, "private_ip": "172.31.7.196", "tls_staging": str(staging)})
                validate.assert_called_once_with(staging)
            self.assertFalse(staging.exists())
            value = json.loads((root / "coordinator").read_text())
            self.assertEqual(Path(value["tls_directory"]), base / staging.name)
            self.assertTrue((base / staging.name / "client.key").is_file())

    def test_pair_rejects_same_instance_before_preparation(self):
        identity = {"instance_id": "i-example", "private_ip": "172.31.1.1", "region": "test", "vpc": "vpc-test"}
        args = SimpleNamespace(peer_host="ubuntu@peer", peer_key=None, image=Path("synthetic.jpg"))
        with patch.object(pair, "identity", return_value=identity), self.assertRaisesRegex(RuntimeError, "distinct"):
            pair.prepare(args)

    @unittest.skipUnless(shutil.which("openssl"), "Synthetic certificates require OpenSSL")
    def test_mutual_tls_rejects_missing_client_certificate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair.certificates(root, "127.0.0.1")
            for authenticated in (True, False):
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    listener.listen()
                    result = []

                    def server(result=result):
                        raw, _ = listener.accept()
                        try:
                            with peer.server_context(root).wrap_socket(raw, server_side=True) as secured:
                                secured.sendall(b"ready")
                                result.append(True)
                        except ssl.SSLError:
                            raw.close()
                            result.append(False)

                    thread = threading.Thread(target=server, daemon=True)
                    thread.start()
                    context = peer.client_context(root) if authenticated else ssl.create_default_context(cafile=str(root / "ca.crt"))
                    with socket.create_connection(listener.getsockname(), timeout=5) as raw:
                        try:
                            with context.wrap_socket(raw, server_hostname="127.0.0.1") as secured:
                                received = secured.recv(5)
                        except ssl.SSLError:
                            received = b""
                    thread.join(timeout=5)
                    self.assertEqual(received, b"ready" if authenticated else b"")
                    self.assertEqual(result, [authenticated])


if __name__ == "__main__":
    unittest.main()
