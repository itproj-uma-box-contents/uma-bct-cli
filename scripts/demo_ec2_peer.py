#!/usr/bin/env python3
"""Private mutually authenticated worker gateway; no public Ollama listener."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.client
import importlib.machinery
import importlib.util
import json
import os
import ssl
import stat
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import ip_address, ip_network
from pathlib import Path
from urllib.parse import urlsplit

BASE = Path("/home/ubuntu/demo_peer")
TLS = BASE / "tls"
GATEWAY = BASE / "gateway.json"
COORDINATOR = Path("/opt/uma-bct-demo/peer-coordinator.json")
INSTALLED = Path("/usr/local/bin/uma-bct-demo-peer")
UNIT = Path("/etc/systemd/system/uma-bct-demo-peer.service")
MODEL = "qwen3-vl:8b-instruct-q4_K_M"
DIGEST = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
MAXIMUM = 16 * 1024 * 1024
LOCK = Path("/home/ubuntu/.uma-bct-demo.lock")
INCOMING = Path("/home/ubuntu/demo_incoming")


def module(path: str):
    name = "uma_demo_peer_" + hashlib.sha256(path.encode()).hexdigest()[:16]
    loader = importlib.machinery.SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    loader.exec_module(value)
    return value


def private_address(value: str) -> str:
    address = ip_address(value)
    if address.version != 4 or not any(address in ip_network(network) for network in
                                     ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")):
        raise ValueError("Worker transport requires one private VPC IPv4 address.")
    return str(address)


@contextlib.contextmanager
def inference_lock():
    """Concurrent requests share the worker; configuration requires exclusivity."""
    import fcntl

    descriptor = os.open(LOCK, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "r") as owner:
        info = os.fstat(owner.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            raise RuntimeError("Worker lock must be private and owned by the service user.")
        fcntl.flock(owner.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def relay(response, destination, *, require_done: bool) -> None:
    """Only a complete upstream response receives the final downstream chunk."""
    tail, last_line = b"", b""
    while block := response.read1(64 * 1024):
        if require_done:
            lines = (tail + block).split(b"\n")
            tail = lines[-1]
            completed = [line for line in lines[:-1] if line.strip()]
            if completed:
                last_line = completed[-1]
            if len(tail) > MAXIMUM:
                raise ValueError("Worker response line exceeds its bound.")
        destination.write(f"{len(block):x}\r\n".encode() + block + b"\r\n")
        destination.flush()
    if response.length not in (None, 0):
        raise http.client.IncompleteRead(b"")
    if require_done and json.loads(tail.strip() or last_line).get("done") is not True:
        raise ValueError("Worker stream ended without a completed inference response.")
    destination.write(b"0\r\n\r\n")
    destination.flush()


def client_context(directory: Path = TLS) -> ssl.SSLContext:
    context = ssl.create_default_context(cafile=str(directory / "ca.crt"))
    context.load_cert_chain(str(directory / "client.crt"), str(directory / "client.key"))
    return context


def server_context(directory: Path = TLS) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=str(directory / "ca.crt"))
    context.load_cert_chain(str(directory / "server.crt"), str(directory / "server.key"))
    return context


def validate_chat(body: bytes) -> None:
    value = json.loads(body)
    if (not isinstance(value, dict) or value.get("model") != MODEL or value.get("think") is not False
            or value.get("options", {}).get("num_ctx") != 16384
            or value.get("options", {}).get("temperature") != 0):
        raise ValueError("Worker requests must retain the exact production model/context/settings.")


def endpoint(index: int) -> str:
    health = module("/usr/local/bin/uma-bct-demo-health")
    service = "ollama" if index == 1 else f"ollama-parallel{'' if index == 2 else f'-{index}'}"
    compose = health.COMPOSE if index == 1 else "/opt/uma-bct-demo/parallel.yaml"
    _container, info = health.container_info(service=service, compose=compose)
    return info["NetworkSettings"]["Networks"][health.NETWORK]["IPAddress"]


def gateway_readiness(workers: int) -> dict:
    controller = module("/home/ubuntu/demo_control.py")
    ready = controller.health(transcription_workers=workers)
    deployment = controller.deployment()
    return {"status": "ready", "workers": workers, "readiness": ready,
            "application_source_sha256": deployment["source_sha256"],
            "gateway_sha256": hashlib.sha256(INSTALLED.read_bytes()).hexdigest()}


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass  # Do not put photo/prompt payloads in service logs.

    def do_GET(self):
        self.forward("GET")

    def do_POST(self):
        self.forward("POST")

    def forward(self, method: str) -> None:
        try:
            with inference_lock():
                self.forward_locked(method)
        except Exception:
            self.close_connection = True
            self.send_error(503, "Private worker is being configured")

    def forward_locked(self, method: str) -> None:
        started = False
        try:
            if method == "GET" and self.path == "/demo/ready":
                payload = json.dumps(gateway_readiness(self.server.workers)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                started = True
                self.wfile.write(payload)
                return
            allowed = {"GET": {"/api/version", "/api/tags", "/api/ps"}, "POST": {"/api/chat"}}
            if self.path not in allowed[method] or "Transfer-Encoding" in self.headers:
                raise ValueError("Unsupported worker operation.")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 <= size <= MAXIMUM:
                raise ValueError("Worker request exceeds its bound.")
            self.connection.settimeout(60)
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Incomplete worker request.")
            if method == "POST":
                validate_chat(body)
            connection = http.client.HTTPConnection(self.server.address, 11434, timeout=900)
            try:
                connection.request(method, self.path, body=body or None, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                self.connection.settimeout(900)
                relay(response, self.wfile, require_done=method == "POST" and response.status == 200
                      and json.loads(body).get("stream", True) is True)
                self.close_connection = True
            finally:
                connection.close()
        except Exception:
            self.close_connection = True
            # Plain operational failure only; never include request text or pixels.
            if not started:
                self.send_error(503, "Private worker is not ready or request is invalid")


class GatewayServer(ThreadingHTTPServer):
    def get_request(self):
        connection, address = super().get_request()
        try:
            connection.settimeout(8)
            connection.do_handshake()
            connection.settimeout(60)
            return connection, address
        except Exception:
            connection.close()
            raise


def serve() -> None:
    config = json.loads(GATEWAY.read_text())
    address = private_address(config["private_ip"])
    workers = config["workers"]
    context = server_context(Path(config["tls_directory"]))
    servers = []
    for index in range(1, workers + 1):
        server = GatewayServer((address, 9442 + index), GatewayHandler)
        server.daemon_threads = True
        server.workers, server.address = workers, endpoint(index)
        server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Event().wait()


def configure(role: str, request: dict) -> dict:
    if os.geteuid() != 0:
        raise RuntimeError("Private worker configuration requires sudo.")
    parallel = module("/usr/local/bin/uma-bct-demo-parallel")
    with parallel.configuration_lock():
        workers = request["workers"]
        if workers != 4:
            raise ValueError("The prepared pair uses four readers on each GPU.")
        address = private_address(request["private_ip"])
        staged = Path(request["tls_staging"])
        import re

        if (staged.parent != INCOMING
                or not re.fullmatch(r"[a-f0-9]{32}-tls", staged.name)
                or staged.is_symlink() or not staged.is_dir()):
            raise ValueError("Invalid private TLS staging directory.")
        account = parallel.ubuntu_identity()
        names = ("ca.crt", "server.crt", "server.key") if role == "gateway" else ("ca.crt", "client.crt", "client.key")
        for name in names:
            info = (staged / name).lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != account[0]
                    or info.st_nlink != 1 or info.st_mode & 0o077):
                raise ValueError("TLS files must be private regular files owned by ubuntu.")
        (server_context if role == "gateway" else client_context)(staged)
        BASE.mkdir(mode=0o700, exist_ok=True)
        os.chown(BASE, *account)
        published = BASE / staged.name
        # Immutable credentials are published while configuration owns the lock.
        # Existing live credentials are never overwritten or deleted.
        staged.rename(published)
        parallel.write(INSTALLED, Path(__file__).read_bytes(), 0o755)
        if role == "gateway":
            parallel.write(GATEWAY, json.dumps({"private_ip": address, "workers": workers,
                                               "tls_directory": str(published)}).encode(), 0o600)
            # The uploaded TLS and gateway files are private and Ubuntu-owned.
            os.chown(GATEWAY, *account)
            unit = f'''[Unit]
Description=UMA BCT private authenticated GPU worker
Requires=uma-bct-demo-parallel.service
After=uma-bct-demo-parallel.service network-online.target
PartOf=uma-bct-demo-parallel.service
[Service]
Type=simple
User=ubuntu
SupplementaryGroups=docker
ExecStart={INSTALLED} serve
Restart=on-failure
UMask=0077
[Install]
WantedBy=multi-user.target
'''
            parallel.write(UNIT, unit.encode())
            subprocess.run(["systemctl", "daemon-reload"], check=True)
            subprocess.run(["systemctl", "enable", "--now", "uma-bct-demo-peer.service"], check=True)
            subprocess.run(["systemctl", "restart", "uma-bct-demo-peer.service"], check=True)
        else:
            urls = [f"https://{address}:{9443 + index}" for index in range(workers)]
            parallel.write(COORDINATOR, json.dumps({"local_workers": workers, "total_workers": workers * 2,
                           "worker_urls": urls, "tls_directory": str(published)}).encode())
        return {"status": "configured", "role": role, "private_ip": address, "workers": workers}


def combined_readiness(controller, config: dict) -> dict:
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise RuntimeError("Private worker redirects are not allowed.")
    def remote() -> dict:
        url = config["worker_urls"][0]
        parsed = urlsplit(url)
        private_address(parsed.hostname)
        if parsed.scheme != "https" or parsed.port != 9443:
            raise RuntimeError("Invalid prepared private worker endpoint.")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect(),
                    urllib.request.HTTPSHandler(context=client_context(Path(config["tls_directory"]))))
        with opener.open(url + "/demo/ready", timeout=30) as response:
            value = json.load(response)
        if value.get("status") != "ready" or value.get("workers") != config["local_workers"]:
            raise RuntimeError("Peer GPU is not ready with its declared reader count.")
        if value.get("application_source_sha256") != controller.deployment()["source_sha256"]:
            raise RuntimeError("GPU nodes run different application revisions; prepare both first.")
        for runner in value["readiness"]["runners"]:
            model = runner["running_model"]
            if (model.get("digest") != DIGEST or model.get("context_length") != 16384
                    or not model.get("size_vram") or model.get("size_vram") != model.get("size")):
                raise RuntimeError("Peer model is cold, different, or spills to CPU.")
        return value["readiness"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        local_future = pool.submit(controller.health, transcription_workers=config["local_workers"])
        remote_future = pool.submit(remote)
        local, peer = local_future.result(), remote_future.result()
    return {"status": "ready", "runners": [*local["runners"], *peer["runners"]],
            "nodes": [local, peer], "settings": {**local["settings"], "transcription_workers": config["total_workers"],
                                               "independent_runners": config["total_workers"]},
            "backend": local["backend"], "transport": "mutually authenticated private HTTPS"}


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gateway", "coordinator", "serve"))
    args = parser.parse_args()
    try:
        if args.action == "serve":
            serve()
        else:
            print(json.dumps(configure(args.action, json.load(sys.stdin))))
        return 0
    except Exception as error:
        print(json.dumps({"status": "not_ready", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
