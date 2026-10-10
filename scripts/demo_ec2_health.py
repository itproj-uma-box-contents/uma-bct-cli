#!/usr/bin/env python3
"""Private EC2 readiness/warming. No photos; status never loads the model."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

MODEL = "qwen3-vl:8b-instruct-q4_K_M"
MODEL_DIGEST = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
OLLAMA_IMAGE = "ollama/ollama:0.32.9@sha256:1685741456770df6e3cceb2a945a5f75e020f658d1701509668d6f4688f1dd3f"
CONTEXT = 16384
COMPOSE = "/opt/uma-bct-demo/compose.yaml"
NETWORK = "uma-bct-demo_default"
BACKEND_FILE = Path("/opt/uma-bct-demo/backend.json")
SETTINGS = {"model": MODEL, "num_ctx": CONTEXT, "temperature": 0, "think": False,
            "transcription_logprobs": True, "parallel_requests": 1}


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.PIPE, timeout=15).strip()


def backend_config() -> dict:
    try:
        value = json.loads(BACKEND_FILE.read_text())
    except FileNotFoundError:
        value = {"mode": "gpu", "gpu_name": "L4"}  # Original G6 installations.
    if not isinstance(value, dict) or value.get("mode") not in {"gpu", "cpu"}:
        raise RuntimeError("Invalid prepared backend configuration.")
    if value["mode"] == "gpu" and value.get("gpu_name") not in {"L4", "A10G", "H100"}:
        raise RuntimeError("GPU setup must specify L4, A10G or H100.")
    return value


def container_info(config: dict | None = None, *, service: str = "ollama", compose: str = COMPOSE) -> tuple[str, dict]:
    config = config or backend_config()
    container = command("docker", "compose", "-p", "uma-bct-demo", "-f", compose, "ps", "-q", service)
    if not container:
        raise RuntimeError("Ollama service is stopped. Inspect sudo journalctl -u uma-bct-demo.service.")
    info = json.loads(command("docker", "inspect", container))[0]
    if not info["State"]["Running"] or info["Config"]["Image"] != OLLAMA_IMAGE:
        raise RuntimeError("The pinned Ollama container is not running.")
    if info["HostConfig"].get("PortBindings"):
        raise RuntimeError("Ollama has published ports; remove them before using the demo.")
    required = {"OLLAMA_NO_CLOUD=1", "OLLAMA_KEEP_ALIVE=-1", "OLLAMA_MAX_LOADED_MODELS=1",
                "OLLAMA_NUM_PARALLEL=1", "OLLAMA_CONTEXT_LENGTH=16384"}
    if not required.issubset(info["Config"].get("Env", [])):
        raise RuntimeError("Ollama service settings differ from the prepared demo configuration.")
    devices = info["HostConfig"].get("DeviceRequests") or []
    if config["mode"] == "cpu":
        if (devices or info["HostConfig"].get("Devices") or info["HostConfig"].get("Runtime") != "runc"
                or not {"CUDA_VISIBLE_DEVICES=-1", "OLLAMA_VULKAN=0"}.issubset(info["Config"].get("Env", []))):
            raise RuntimeError("CPU rehearsal must disable GPU access explicitly.")
    elif not any(item.get("Driver") == "nvidia" and item.get("Count") == 1 for item in devices):
        raise RuntimeError("The Ollama container does not request its NVIDIA GPU.")
    return container, info


def api(info: dict, endpoint: str, payload: dict | None = None) -> dict:
    address = info["NetworkSettings"]["Networks"][NETWORK]["IPAddress"]
    if not address:
        raise RuntimeError("The private Ollama bridge has no address.")
    request = urllib.request.Request(
        f"http://{address}:11434/api/{endpoint}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    # Do not forward this private request through inherited HTTP proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=900 if payload is not None else 3) as response:
        return json.load(response)


def inventory(info: dict) -> dict:
    if api(info, "version").get("version") != "0.32.9":
        raise RuntimeError("Ollama must be version 0.32.9.")
    models = api(info, "tags").get("models", [])
    model = next((item for item in models if item.get("name") == MODEL), None)
    if model is None:
        raise RuntimeError("The exact model is missing. Restart uma-bct-demo.service during preparation.")
    if model.get("digest") != MODEL_DIGEST or model.get("details", {}).get("quantization_level") != "Q4_K_M":
        raise RuntimeError("Model identity differs from the established local model; do not benchmark changed weights.")
    return model


def readiness(*, warm: bool = False, expected_backend: str = "gpu", service: str = "ollama", compose: str = COMPOSE) -> dict:
    config = backend_config()
    if config["mode"] != expected_backend:
        raise RuntimeError(f"Server is {config['mode']}; select --backend {config['mode']} explicitly. No automatic fallback.")
    gpu = None
    if config["mode"] == "gpu":
        gpu = command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
        if len(gpu.splitlines()) != 1 or config["gpu_name"] not in gpu:
            raise RuntimeError(f"GPU setup requires one NVIDIA {config['gpu_name']}; CPU fallback is not permitted.")
    container, info = container_info(config, service=service, compose=compose)
    if config["mode"] == "gpu":
        command("docker", "exec", container, "nvidia-smi", "-L")
    else:
        command("docker", "exec", container, "sh", "-c",
                "test ! -e /dev/nvidiactl && test ! -e /dev/dri && test ! -e /dev/kfd")
    model = inventory(info)
    timing = None
    if warm:
        started = time.monotonic()
        response = api(info, "chat", {
            "model": MODEL, "messages": [{"role": "user", "content": "Reply briefly."}],
            "think": False, "stream": False, "keep_alive": -1,
            "options": {"num_ctx": CONTEXT, "temperature": 0, "num_predict": 1},
        })
        timing = {"wall_seconds": time.monotonic() - started,
                  **{key: response.get(key) for key in ("load_duration", "total_duration", "prompt_eval_duration",
                                                       "eval_duration", "prompt_eval_count", "eval_count")}}
    running = next((item for item in api(info, "ps").get("models", []) if item.get("name") == MODEL), None)
    display = command("docker", "exec", container, "ollama", "ps")
    processor = "GPU" if config["mode"] == "gpu" else "CPU"
    # On CPU, native AMX buffers can appear as VRAM in Ollama 0.32.9. Device
    # isolation above is authoritative; preserve the raw report below.
    correct_memory = running is not None and (processor == "CPU" or bool(running.get("size_vram")))
    correct_display = any(MODEL in line and (processor == "CPU" or "100% GPU" in line) for line in display.splitlines())
    if (running is None or running.get("digest") != MODEL_DIGEST or running.get("context_length") != CONTEXT
            or not correct_memory or not correct_display):
        raise RuntimeError(f"Model is cold, has wrong context, or is not fully on {processor}. Run prepare before a live run.")
    return {"status": "ready", "backend": config, "gpu": gpu, "model": model, "running_model": running,
            "ollama_ps": display,
            "settings": SETTINGS, "cold_or_reheat_probe": timing}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "inventory", "warm"), nargs="?", default="status")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    try:
        if backend_config()["mode"] != args.backend:
            raise RuntimeError("Backend mismatch; CPU rehearsal requires explicit --backend cpu.")
        if args.action == "inventory":
            _container, info = container_info()
            value = inventory(info)
        else:
            value = readiness(warm=args.action == "warm", expected_backend=args.backend)
        print(json.dumps(value))
        return 0
    except Exception as error:
        print(json.dumps({"status": "not_ready", "error": str(error),
                          "preparation": "Inspect /var/log/uma-bct-demo-setup.log and uma-bct-demo.service."}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
