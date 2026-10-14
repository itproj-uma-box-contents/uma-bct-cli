#!/usr/bin/env python3
"""Opt-in H100 CUDA multiprocess scheduling; only owned, idle demo services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

BASE = Path("/opt/uma-bct-demo")
CONFIG = BASE / "mps.json"
COMPOSE = BASE / "compose.yaml"
BACKUP = BASE / "compose.without-mps.yaml"
PIPE = Path("/var/lib/uma-bct-demo/mps/pipe")
LOG = PIPE.parent / "logs"
INSTALLED = Path("/usr/local/bin/uma-bct-demo-mps")
UNIT = Path("/etc/systemd/system/uma-bct-demo-mps.service")
DROPIN = Path("/etc/systemd/system/uma-bct-demo.service.d/mps.conf")
LOCK = Path("/home/ubuntu/.uma-bct-demo.lock")
SERVICE = "uma-bct-demo-mps.service"
PRIMARY = "uma-bct-demo.service"
PARALLEL = "uma-bct-demo-parallel.service"


def command(*args: str, text: str | None = None, mps: bool = False) -> str:
    environment = dict(os.environ)
    if mps:
        environment.update(CUDA_MPS_PIPE_DIRECTORY=str(PIPE), CUDA_MPS_LOG_DIRECTORY=str(LOG))
    return subprocess.check_output(args, input=text, text=True, env=environment, stderr=subprocess.PIPE, timeout=180).strip()


def write(path: Path, content: bytes, mode: int = 0o644) -> None:
    temporary = path.with_name(path.name + ".new-" + uuid.uuid4().hex)
    try:
        temporary.write_bytes(content)
        temporary.chmod(mode)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def configuration_lock() -> Iterator[None]:
    import fcntl
    import pwd

    uid = pwd.getpwnam("ubuntu").pw_uid
    try:
        descriptor = os.open(LOCK, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    except FileExistsError:
        descriptor = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW)
    else:
        os.fchown(descriptor, uid, pwd.getpwnam("ubuntu").pw_gid)
    with os.fdopen(descriptor, "a") as lock:
        info = os.fstat(lock.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != uid or info.st_nlink != 1 or info.st_mode & 0o077:
            raise RuntimeError("Demo lock must be a private regular file owned by ubuntu.")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def enabled() -> bool:
    return CONFIG.exists() and json.loads(CONFIG.read_text()).get("enabled") is True


def decorate(source: bytes) -> bytes:
    """Add scheduler IPC only; the pinned inference image/settings remain unchanged."""
    text = source.decode()
    if "CUDA_MPS_PIPE_DIRECTORY" in text:
        raise ValueError("Compose source already contains MPS settings.")
    if re.search(r"^\s*ports\s*:", text, re.MULTILINE):
        raise ValueError("Scheduler preparation cannot publish model API ports.")
    if "    environment:\n" not in text or "    volumes:\n" not in text:
        raise ValueError("Expected the existing private model Compose services.")
    text = text.replace("    environment:\n", '    ipc: host\n    environment:\n      CUDA_MPS_PIPE_DIRECTORY: "/tmp/nvidia-mps"\n')
    text = text.replace("    volumes:\n", f"    volumes:\n      - {PIPE}:/tmp/nvidia-mps\n")
    return text.encode()


def foreign_controller() -> bool:
    for executable in Path("/proc").glob("[0-9]*/exe"):
        try:
            if executable.resolve(strict=True).name == "nvidia-cuda-mps-control":
                return True
        except (FileNotFoundError, PermissionError):
            continue
    return False


def stop_models() -> None:
    if Path("/etc/systemd/system/uma-bct-demo-parallel.service").exists():
        command("systemctl", "disable", "--now", PARALLEL)
        write(BASE / "parallel.json", b'{"workers": 1}\n')
    command("systemctl", "stop", PRIMARY)
    command("docker", "compose", "-p", "uma-bct-demo", "-f", str(COMPOSE), "stop", "ollama")


def configure(use_mps: bool) -> dict:
    with configuration_lock():
        if use_mps and json.loads((BASE / "backend.json").read_text()) != {"mode": "gpu", "gpu_name": "H100"}:
            raise RuntimeError("This measured scheduler experiment is H100-only.")
        if enabled() == use_mps:
            decorated = b"CUDA_MPS_PIPE_DIRECTORY" in COMPOSE.read_bytes()
            if use_mps:
                state = json.loads(CONFIG.read_text())
                if (not decorated or not DROPIN.exists() or not UNIT.exists()
                        or hashlib.sha256(COMPOSE.read_bytes()).hexdigest() != state.get("compose_sha256")):
                    raise RuntimeError("Incomplete or changed scheduler configuration; inspect/restore it before preparation.")
                status(1)  # A JSON flag alone does not establish a live scheduler/client.
            elif decorated or DROPIN.exists():
                raise RuntimeError("Scheduler configuration is incomplete; do not claim it is disabled. Inspect the saved backup.")
            elif UNIT.exists():
                unit_state = command("systemctl", "show", SERVICE, "-p", "ActiveState", "-p", "UnitFileState")
                if "ActiveState=active" in unit_state or "UnitFileState=enabled" in unit_state:
                    raise RuntimeError("Scheduler unit is still active/enabled; inspect it before claiming normal mode.")
            if INSTALLED.exists():
                write(INSTALLED, Path(__file__).read_bytes(), 0o755)
            return {"status": "configured", "gpu_mps": use_mps}
        if use_mps:
            command("nvidia-cuda-mps-control", "-h")
            if foreign_controller():
                raise RuntimeError("An existing CUDA scheduler is not owned by this demo; inspect it before enabling.")
            original = COMPOSE.read_bytes()
            transformed = decorate(original)
            write(BACKUP, original)
        else:
            original = BACKUP.read_bytes()
            expected = json.loads(CONFIG.read_text())["compose_sha256"]
            if hashlib.sha256(COMPOSE.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Owned Compose changed after MPS preparation; inspect before reverting it.")
        stop_models()  # Lock excludes live CLI invocations: never stop an active MPS client.
        try:
            if use_mps:
                for path in (PIPE, LOG):
                    path.mkdir(mode=0o700, parents=True, exist_ok=True)
                write(INSTALLED, Path(__file__).read_bytes(), 0o755)
                unit = f'''[Unit]
Description=UMA BCT opt-in H100 CUDA scheduler
Before=docker.service {PRIMARY}
[Service]
Type=forking
Environment=CUDA_MPS_PIPE_DIRECTORY={PIPE}
Environment=CUDA_MPS_LOG_DIRECTORY={LOG}
ExecStart=/usr/bin/nvidia-cuda-mps-control -d
ExecStop={INSTALLED} stop
TimeoutStopSec=30
[Install]
WantedBy=multi-user.target
'''
                write(UNIT, unit.encode())
                DROPIN.parent.mkdir(parents=True, exist_ok=True)
                write(DROPIN, f"[Unit]\nRequires={SERVICE}\nAfter={SERVICE}\n".encode())
                write(COMPOSE, transformed)
                write(CONFIG, json.dumps({"enabled": True, "compose_sha256": hashlib.sha256(transformed).hexdigest()}).encode())
                command("systemctl", "daemon-reload")
                command("systemctl", "enable", "--now", SERVICE)
            else:
                command("systemctl", "disable", "--now", SERVICE)
                DROPIN.unlink(missing_ok=True)
                write(COMPOSE, original)
                write(CONFIG, b'{"enabled": false}\n')
                command("systemctl", "daemon-reload")
            # Production-context loading stays in explicit preparation, never in live timing.
            command("systemctl", "restart", PRIMARY)
            return {"status": "configured", "gpu_mps": use_mps}
        except Exception as error:
            # Failed opt-in returns the owned model service to its original configuration.
            stop_models()
            if UNIT.exists():
                command("systemctl", "disable", "--now", SERVICE)
            DROPIN.unlink(missing_ok=True)
            write(COMPOSE, original)
            write(CONFIG, b'{"enabled": false}\n')
            command("systemctl", "daemon-reload")
            command("systemctl", "restart", PRIMARY)
            raise RuntimeError(f"MPS preparation failed; original GPU service restored: {error}") from error


def status(workers: int) -> dict:
    if not enabled():
        return {"gpu_mps": False}
    if command("systemctl", "is-active", SERVICE) != "active":
        raise RuntimeError("Selected CUDA scheduler is stopped; prepare before running.")
    servers = command("nvidia-cuda-mps-control", text="get_server_list\n", mps=True).splitlines()
    clients = set()
    for server in servers:
        if not server.isdecimal():
            raise RuntimeError("Invalid CUDA scheduler server identity.")
        clients.update(command("nvidia-cuda-mps-control", text=f"get_client_list {server}\n", mps=True).splitlines())
    names = ["uma-bct-demo-ollama-1", *[f"uma-bct-demo-ollama-parallel{'' if i == 2 else f'-{i}'}-1"
                                      for i in range(2, workers + 1)]]
    containers = json.loads(command("docker", "inspect", *names))
    covered = []
    for info in containers:
        if ("CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps" not in info["Config"]["Env"]
                or info["HostConfig"].get("IpcMode") != "host"
                or not any(m.get("Source") == str(PIPE) and m.get("Destination") == "/tmp/nvidia-mps"
                           for m in info["Mounts"])):
            raise RuntimeError("A model runner bypasses the selected CUDA scheduler.")
        cgroups = []
        for pid in clients:
            if pid.isdecimal():
                try:
                    cgroups.append(Path(f"/proc/{pid}/cgroup").read_text())
                except FileNotFoundError:
                    continue
        if not any(info["Id"] in cgroup for cgroup in cgroups):
            raise RuntimeError("A model runner has no verified CUDA scheduler client.")
        covered.append(info["Name"])
    return {"gpu_mps": True, "server_pids": servers, "verified_model_clients": covered,
            "helper_sha256": hashlib.sha256(INSTALLED.read_bytes()).hexdigest()}


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("enable", "disable", "status", "stop"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8), default=1)
    args = parser.parse_args()
    try:
        if os.geteuid() != 0:
            raise RuntimeError("Scheduler management and client verification require sudo.")
        if args.action == "stop":
            command("nvidia-cuda-mps-control", text="quit\n", mps=True)
            return 0
        result = status(args.workers) if args.action == "status" else configure(args.action == "enable")
        print(json.dumps(result))
        return 0
    except Exception as error:
        print(json.dumps({"status": "not_ready", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
