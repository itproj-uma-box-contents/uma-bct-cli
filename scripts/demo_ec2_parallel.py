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
from concurrent.futures import ThreadPoolExecutor
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


def require_two_runners() -> int:
    workers = json.loads(CONFIG.read_text()).get("workers")
    if workers not in {2, 4, 6, 8}:
        raise RuntimeError("Parallel runners are not prepared; run prepare --transcription-workers 2, 4, 6 or 8.")
    return workers


def service_name(index: int) -> str:
    return "ollama" if index == 1 else f"ollama-parallel{'' if index == 2 else f'-{index}'}"


def runner_compose(source: bytes, workers: int) -> bytes:
    """Duplicate only the pinned private service; no new ports or model settings."""
    text = source.decode()
    header, remainder = text.split("services:\n", 1)
    service, network = remainder.split("\nnetworks:\n", 1)
    if "  ollama-parallel:\n" not in service:
        raise ValueError("Expected the pinned private parallel Compose template.")
    copies = [service.replace("  ollama-parallel:\n", f"  {service_name(index)}:\n", 1)
              for index in range(2, max(workers, 2) + 1)]
    return (header + "services:\n" + "\n".join(copies) + "\nnetworks:\n" + network).encode()


def scheduled_readiness(health: ModuleType, workers: int) -> list[dict]:
    """Batch fresh checks for MPS clients; strict API residency plus real NVIDIA tooling."""
    config = health.backend_config()
    if config.get("mode") != "gpu" or config.get("gpu_name") not in {"H100", "RTX PRO 6000 Blackwell Server Edition"}:
        raise RuntimeError("The selected scheduler requires its H100 or Blackwell backend.")
    gpu = health.command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
    name = gpu.split(",", 1)[0].strip().removeprefix("NVIDIA ")
    if len(gpu.splitlines()) != 1 or not (name == config["gpu_name"] or name.startswith(config["gpu_name"] + " ")):
        raise RuntimeError("The scheduler cannot fall back to another GPU or CPU.")
    names = [f"uma-bct-demo-{service_name(index)}-1" for index in range(1, workers + 1)]
    containers = json.loads(health.command("docker", "inspect", *names))
    required = {"OLLAMA_NO_CLOUD=1", "OLLAMA_KEEP_ALIVE=-1", "OLLAMA_MAX_LOADED_MODELS=1",
                "OLLAMA_NUM_PARALLEL=1", "OLLAMA_CONTEXT_LENGTH=16384", "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"}
    for info in containers:
        if (not info["State"]["Running"] or info["Config"]["Image"] != health.OLLAMA_IMAGE
                or info["HostConfig"].get("PortBindings")
                or not required.issubset(info["Config"].get("Env", []))
                or not any(d.get("Driver") == "nvidia" and d.get("Count") == 1
                           for d in info["HostConfig"].get("DeviceRequests", []))):
            raise RuntimeError("A scheduler runner differs from the pinned private GPU configuration.")
    # MPS health additionally verifies every container's actual CUDA client PID.
    # This checks the driver inside a container, as well as the host tooling above.
    device = health.command("docker", "exec", containers[0]["Id"], "nvidia-smi", "-L")
    def check(info: dict) -> dict:
        model = health.inventory(info)
        running = next((item for item in health.api(info, "ps").get("models", [])
                        if item.get("name") == health.MODEL), None)
        # The pinned Ollama CLI uses this same equality for its '100% GPU' display.
        if (not running or running.get("digest") != health.MODEL_DIGEST
                or running.get("context_length") != health.CONTEXT
                or not running.get("size_vram") or running.get("size_vram") != running.get("size")):
            raise RuntimeError("A scheduler model is cold, has wrong context, or spills onto CPU; prepare first.")
        return {"status": "ready", "backend": config, "gpu": gpu, "model": model, "running_model": running,
                "settings": health.SETTINGS, "cold_or_reheat_probe": None,
                "gpu_evidence": {"host_nvidia_smi": gpu, "container_nvidia_smi": device,
                                 "ollama_running_model_api": "all model bytes reported resident on GPU"}}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(check, containers))


def status(*, warm_second: bool = False) -> dict:
    workers = require_two_runners()
    health = health_module()
    if health.backend_config()["mode"] != "gpu":
        raise RuntimeError("Parallel demo mode requires GPU; CPU fallback is refused.")
    mps_config = BASE / "mps.json"
    if not warm_second and mps_config.exists() and json.loads(mps_config.read_text()).get("enabled") is True:
        checked = scheduled_readiness(health, workers)
        scheduling = json.loads(health.command("sudo", "/usr/local/bin/uma-bct-demo-mps", "status", "--workers", str(workers)))
        if scheduling.get("gpu_mps") is not True:
            raise RuntimeError("The selected CUDA scheduler has no verified model clients.")
        first = checked[0]
        return {"status": "ready", "backend": first["backend"], "gpu": first["gpu"], "runners": checked,
                "gpu_scheduling": scheduling,
                "settings": {**first["settings"], "transcription_workers": workers, "independent_runners": workers},
                "note": "Fresh batch checks; every CUDA scheduler client is additionally verified by the controller."}
    def check(index: int) -> dict:
        if index == 1:
            return health.readiness()
        return health.readiness(warm=warm_second, service=service_name(index), compose=str(COMPOSE))
    if not warm_second:
        # Status does not load/unload anything: every independent runner can be
        # verified concurrently, including the primary, without dropping checks.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            checked = list(pool.map(check, range(1, workers + 1)))
        first, additional = checked[0], checked[1:]
        return {"status": "ready", "backend": first["backend"], "gpu": first["gpu"],
                "runners": checked, "settings": {**first["settings"], "transcription_workers": workers,
                                                "independent_runners": workers},
                "note": "Each runner retains the exact production model/context; status never warms."}
    first = health.readiness()
    with ThreadPoolExecutor(max_workers=workers - 1) as pool:
        additional = list(pool.map(check, range(2, workers + 1)))
    if warm_second:
        # Loading later services must not displace any earlier runner onto CPU.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            additional = list(pool.map(lambda index: health.readiness(service=service_name(index), compose=str(COMPOSE)),
                                       range(2, workers + 1)))
        first = health.readiness()
    return {"status": "ready", "backend": first["backend"], "gpu": first["gpu"],
            "runners": [first, *additional], "settings": {**first["settings"], "transcription_workers": workers,
                                                        "independent_runners": workers},
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
    if args.workers not in {1, 2, 4, 6, 8}:
        raise ValueError("Select 1, 2, 4, 6 or 8 independent runners.")
    for source in (args.health_file, args.compose_file):
        if source.is_symlink() or not source.is_file():
            raise ValueError("Preparation requires regular helper/configuration source files.")
    health = health_module(args.health_file)
    if args.workers != 1 and health.backend_config()["mode"] != "gpu":
        raise ValueError("Parallel demo mode is supported only on its prepared GPU server.")
    compose_bytes = runner_compose(args.compose_file.read_bytes(), args.workers)
    mps_config = BASE / "mps.json"
    if mps_config.exists() and json.loads(mps_config.read_text()).get("enabled") is True:
        compose_bytes = health_module(Path("/usr/local/bin/uma-bct-demo-mps")).decorate(compose_bytes)
    # Stop the OLD service set before replacing its Compose file, including when shrinking.
    stop_second()
    write(HEALTH, args.health_file.read_bytes(), 0o755)
    write(INSTALLED, Path(__file__).read_bytes(), 0o755)
    write(COMPOSE, compose_bytes)
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
    if args.workers > 2:
        primary = health.readiness()
        free_mib = int(health.command("nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"))
        # Allow for vision buffers beyond Ollama's model VRAM report, plus headroom.
        required = (args.workers - 1) * (primary["running_model"]["size_vram"] + 2 * 1024**3) + 2 * 1024**3
        if free_mib * 1024**2 < required:
            raise RuntimeError("Insufficient free GPU memory for the selected runners; keep fewer workers. No CPU fallback.")
    write(CONFIG, json.dumps({"workers": args.workers}).encode() + b"\n")
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
    prepare.add_argument("--workers", type=int, choices=(1, 2, 4, 6, 8), required=True)
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
                for index in range(2, require_two_runners() + 1):
                    wait_api(health_module(), service=service_name(index), compose=str(COMPOSE))
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
