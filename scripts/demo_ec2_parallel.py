#!/usr/bin/env python3
"""Prepare/check two private independent runners; never launch AWS or process photos."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

BASE = Path("/opt/uma-bct-demo")
HEALTH = Path("/usr/local/bin/uma-bct-demo-health")
INSTALLED = Path("/usr/local/bin/uma-bct-demo-parallel")
COMPOSE = BASE / "parallel.yaml"
CONFIG = BASE / "parallel.json"
UNIT = Path("/etc/systemd/system/uma-bct-demo-parallel.service")
SERVICE = "uma-bct-demo-parallel.service"
LOCK = Path("/home/ubuntu/.uma-bct-demo.lock")
UNIT_TEXT = """[Unit]
Description=UMA BCT optional second private Qwen runner
Requires=uma-bct-demo.service
After=uma-bct-demo.service docker.service
PartOf=uma-bct-demo.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/bin/uma-bct-demo-parallel start
ExecStop=/usr/bin/docker compose -p uma-bct-demo -f /opt/uma-bct-demo/parallel.yaml stop
TimeoutStartSec=1200
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
"""


def health_module(path: Path | None = None) -> ModuleType:
    loader = importlib.machinery.SourceFileLoader("uma_demo_health", str(path or HEALTH))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("Prepared health helper could not be loaded.")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def command(*arguments: str) -> None:
    subprocess.run(arguments, check=True, stdout=sys.stderr, stderr=sys.stderr)


def write(path: Path, content: bytes, mode: int = 0o644) -> None:
    temporary = path.with_name(path.name + ".new-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as output:
            output.write(content)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def ubuntu_identity() -> tuple[int, int]:
    import pwd

    account = pwd.getpwnam("ubuntu")
    return account.pw_uid, account.pw_gid


@contextmanager
def configuration_lock() -> Iterator[None]:
    """Exclude live runs/prepares without blocking the subordinate startup unit."""
    import fcntl

    uid, gid = ubuntu_identity()
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(LOCK, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        descriptor = os.open(LOCK, flags)
    else:
        if os.fstat(descriptor).st_uid != uid:
            os.fchown(descriptor, uid, gid)
    with os.fdopen(descriptor, "a") as owner:
        info = os.fstat(owner.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_nlink != 1 or info.st_mode & 0o077:
            raise RuntimeError("Demo lock must be a private regular file owned by ubuntu.")
        try:
            fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another demo run/preparation owns the server; configuration was not changed.") from error
        yield


def wait_api(health: ModuleType, *, service: str = "ollama", compose: str | None = None) -> None:
    """Only preparation waits; live status never retries or warms a cold model."""
    deadline = time.monotonic() + 60
    last_error = "service is starting"
    while time.monotonic() < deadline:
        try:
            _container, info = health.container_info(service=service, compose=compose or health.COMPOSE)
            version = health.api(info, "version")
            if version.get("version") != "0.32.9":
                raise ValueError("Prepared Ollama version differs from 0.32.9.")
            return
        except ValueError:
            raise
        except Exception as error:
            last_error = str(error)
            time.sleep(0.5)
    raise RuntimeError(f"Owned model service API did not become ready in 60 seconds: {last_error}")


def require_two_runners() -> None:
    if json.loads(CONFIG.read_text()).get("workers") != 2:
        raise RuntimeError("Two runners are not prepared; run prepare --transcription-workers 2.")


def status(*, warm_second: bool = False) -> dict:
    require_two_runners()
    health = health_module()
    if health.backend_config()["mode"] != "gpu":
        raise RuntimeError("Parallel demo mode requires GPU; CPU fallback is refused.")
    first = health.readiness()
    second = health.readiness(warm=warm_second, service="ollama-parallel", compose=str(COMPOSE))
    # Loading the second runner must not cause the first to leave the GPU.
    first = health.readiness()
    return {"status": "ready", "backend": first["backend"], "gpu": first["gpu"],
            "runners": [first, second], "settings": {**first["settings"], "transcription_workers": 2,
                                                     "independent_runners": 2},
            "note": "Each Qwen3-VL runner has one slot; both hold the same production model/context."}


def stop_second() -> None:
    if UNIT.exists():
        command("systemctl", "disable", "--now", SERVICE)
    if COMPOSE.exists():
        command("docker", "compose", "-p", "uma-bct-demo", "-f", str(COMPOSE), "stop")
    write(CONFIG, b'{"workers": 1}\n')


def configure(args: argparse.Namespace) -> dict:
    with configuration_lock():
        return configure_locked(args)


def configure_locked(args: argparse.Namespace) -> dict:
    for source in (args.health_file, args.compose_file):
        if source.is_symlink() or not source.is_file():
            raise ValueError("Preparation requires regular helper/configuration source files.")
    health = health_module(args.health_file)
    if args.workers == 2 and health.backend_config()["mode"] != "gpu":
        raise ValueError("Parallel demo mode is supported only on its prepared GPU server.")
    write(HEALTH, args.health_file.read_bytes(), 0o755)
    write(INSTALLED, Path(__file__).read_bytes(), 0o755)
    write(COMPOSE, args.compose_file.read_bytes())
    if args.workers == 1:
        stop_second()
        return {"status": "configured", "transcription_workers": 1}
    if not UNIT.exists() or UNIT.read_text() != UNIT_TEXT:
        write(UNIT, UNIT_TEXT.encode())
        command("systemctl", "daemon-reload")
        # Some NVIDIA/runtime combinations lose device access after daemon-reload.
        # Recover the owned service here, outside all live-run timing.
        container, _info = health.container_info()
        try:
            health.command("docker", "exec", container, "nvidia-smi", "-L")
        except subprocess.CalledProcessError:
            print("Recreating owned model service GPU access after daemon-reload.", file=sys.stderr)
            command("docker", "compose", "-p", "uma-bct-demo", "-f", health.COMPOSE, "restart", "ollama")
            wait_api(health)
            health.readiness(warm=True)
    write(CONFIG, b'{"workers": 2}\n')
    try:
        command("systemctl", "enable", SERVICE)
        command("systemctl", "restart", SERVICE)
        return status()
    except Exception as error:
        stop_second()
        raise RuntimeError(f"Two-runner GPU preparation failed; restored one-runner configuration: {error}") from error


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare = commands.add_parser("configure")
    prepare.add_argument("--workers", type=int, choices=(1, 2), required=True)
    prepare.add_argument("--health-file", type=Path, required=True)
    prepare.add_argument("--compose-file", type=Path, required=True)
    commands.add_parser("status")
    commands.add_parser("start")
    args = parser.parse_args()
    try:
        if args.action in {"configure", "start"} and os.geteuid() != 0:
            raise RuntimeError("Runner preparation/startup requires sudo; status is read-only.")
        if args.action == "configure":
            value = configure(args)
        elif args.action == "start":
            require_two_runners()
            try:
                command("docker", "compose", "-p", "uma-bct-demo", "-f", str(COMPOSE), "up", "--detach")
                wait_api(health_module(), service="ollama-parallel", compose=str(COMPOSE))
                value = status(warm_second=True)
            except Exception:
                command("docker", "compose", "-p", "uma-bct-demo", "-f", str(COMPOSE), "stop")
                raise
        else:
            value = status()
        print(json.dumps(value))
        return 0
    except Exception as error:
        print(json.dumps({"status": "not_ready", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
