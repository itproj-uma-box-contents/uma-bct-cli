#!/bin/bash
# Paste the WHOLE file into EC2 Advanced details > User data.
# Ubuntu 24.04 LTS amd64, one NVIDIA A10G, encrypted >=100 GiB gp3 root EBS.
# No credentials, repository tokens, private keys or image download URLs belong here.
# Application deployment is deliberately a later authenticated SSH prepare step.
set -Eeuo pipefail
umask 077
export DEBIAN_FRONTEND=noninteractive
install -m 0700 -d /var/lib/uma-bct-demo /var/lib/uma-bct-demo/models
install -m 0755 -d /opt/uma-bct-demo
exec > >(tee -a /var/log/uma-bct-demo-setup.log) 2>&1
chmod 0600 /var/log/uma-bct-demo-setup.log
trap 'rc=$?; echo "SETUP FAILED: line $LINENO, exit $rc; inspect this log and journalctl -u uma-bct-demo.service"; exit "$rc"' ERR
printf 'Bootstrap began: %s\n' "$(date -Is)"
. /etc/os-release
if [ "$ID" != ubuntu ] || [ "$VERSION_ID" != 24.04 ] || [ "$(dpkg --print-architecture)" != amd64 ]; then
    echo 'This setup requires Ubuntu 24.04 LTS amd64.'; exit 1
fi

# Check the provisioned block device, not filesystem capacity (which has overhead).
# A filesystem grow command cannot enlarge the underlying EBS volume.
root_source=$(findmnt -n -o SOURCE /)
root_device=$(lsblk -snb -o TYPE,PATH "$root_source" | awk '$1 == "disk" {print $2; exit}')
[ -n "$root_device" ] || { echo 'Cannot identify root EBS disk.'; exit 1; }
root_bytes=$(lsblk -bdn -o SIZE "$root_device")
free_bytes=$(df -B1 --output=avail / | tail -n 1 | tr -d ' ')
[ "$root_bytes" -ge 107374182400 ] && [ "$free_bytes" -ge 32212254720 ] || {
    echo 'Insufficient storage: provision encrypted >=100 GiB gp3 in the Console, with >=30 GiB free.'
    exit 1
}

# Official Docker apt repository; Ubuntu's docker.io is not installed on this fresh AMI.
apt-get update
apt-get install -y ca-certificates curl gnupg python3 ubuntu-drivers-common pciutils linux-headers-aws linux-aws
install -m 0755 -d /etc/apt/keyrings
curl -fsSL --retry 3 https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat >/etc/apt/sources.list.d/docker.sources <<'DOCKER_APT'
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: noble
Components: stable
Architectures: amd64
Signed-By: /etc/apt/keyrings/docker.asc
DOCKER_APT
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Server driver branch >=580 covers Ollama's supported A10G CUDA backends.
# Ubuntu's utility selects modules for the AWS kernel; reboot is mandatory below.
lspci | grep -qi NVIDIA || { echo 'No NVIDIA device. This User Data is GPU-only; no CPU fallback.'; exit 1; }
ubuntu-drivers install --gpgpu nvidia:580-server
apt-get install -y nvidia-utils-580-server

# NVIDIA's signed stable toolkit repository. Pin the documented compatible release.
curl -fsSL --retry 3 https://nvidia.github.io/libnvidia-container/gpgkey | \
    gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
chmod 0644 /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL --retry 3 https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    >/etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
demo_toolkit_version=1.20.1-1
apt-get install -y "nvidia-container-toolkit=$demo_toolkit_version" \
    "nvidia-container-toolkit-base=$demo_toolkit_version" "libnvidia-container-tools=$demo_toolkit_version" \
    "libnvidia-container1=$demo_toolkit_version"
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
usermod -aG docker ubuntu
# This is a single-user demo: docker group access is root-equivalent.
install -o ubuntu -g ubuntu -m 0700 -d /home/ubuntu/demo_images /home/ubuntu/demo_results \
    /home/ubuntu/demo_app /home/ubuntu/demo_incoming

# Embedded service/health helper; no repository credentials needed.
cat >/opt/uma-bct-demo/backend.json <<'DEMO_BACKEND'
{"mode": "gpu", "gpu_name": "A10G"}
DEMO_BACKEND
chmod 0644 /opt/uma-bct-demo/backend.json
cat >/opt/uma-bct-demo/compose.yaml <<'DEMO_COMPOSE'
# Standalone temporary EC2 service: no photographs or application build context.
# User Data embeds this exact configuration so Ollama can start before deployment.
name: uma-bct-demo
services:
  ollama:
    image: ollama/ollama:0.32.9@sha256:1685741456770df6e3cceb2a945a5f75e020f658d1701509668d6f4688f1dd3f
    restart: unless-stopped
    environment:
      OLLAMA_NO_CLOUD: "1"
      OLLAMA_KEEP_ALIVE: "-1"
      OLLAMA_MAX_LOADED_MODELS: "1"
      OLLAMA_NUM_PARALLEL: "1"
      OLLAMA_CONTEXT_LENGTH: "16384"
    volumes:
      - /var/lib/uma-bct-demo/models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 5s
      timeout: 5s
      retries: 24
    # No ports: the API is reachable only on the local Docker bridge.
DEMO_COMPOSE
chmod 0644 /opt/uma-bct-demo/compose.yaml
cat >/usr/local/bin/uma-bct-demo-health <<'DEMO_HEALTH'
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
DEMO_HEALTH
chmod 0755 /usr/local/bin/uma-bct-demo-health
cat >/usr/local/sbin/uma-bct-demo-start <<'DEMO_START'
#!/bin/bash
set -Eeuo pipefail
umask 077
exec >>/var/log/uma-bct-demo-setup.log 2>&1
trap 'rc=$?; echo "SERVICE SETUP FAILED: line $LINENO, exit $rc"; exit "$rc"' ERR
printf 'Post-boot setup began: %s\n' "$(date -Is)"
nvidia-smi
# No app/photograph context: this only pulls the pinned public Ollama image.
docker compose -p uma-bct-demo -f /opt/uma-bct-demo/compose.yaml up -d ollama
container=$(docker compose -p uma-bct-demo -f /opt/uma-bct-demo/compose.yaml ps -q ollama)
# Bounded wait for API startup. The image itself contains the ollama CLI.
api_ready=0
for attempt in $(seq 1 60); do
    if docker exec "$container" ollama list; then api_ready=1; break; fi
    sleep 2
done
[ "$api_ready" -eq 1 ] || { echo 'Ollama API startup timed out.'; exit 1; }
# Only pull if absent. A changed digest is a hard failure, not a silent repull.
if ! docker exec "$container" ollama list | awk '{print $1}' | grep -Fxq qwen3-vl:8b-instruct-q4_K_M; then
    echo 'Downloading the exact required model; this is cold preparation.'
    docker exec "$container" ollama pull qwen3-vl:8b-instruct-q4_K_M
fi
/usr/local/bin/uma-bct-demo-health inventory
# A production-context, one-token text probe loads the model, without image loops.
/usr/local/bin/uma-bct-demo-health warm >/var/lib/uma-bct-demo/latest-warm.json
chmod 0644 /var/lib/uma-bct-demo/latest-warm.json
printf 'INFRASTRUCTURE READY: %s; deploy application and selected input with prepare.\n' "$(date -Is)"
dpkg-query -W docker-ce docker-compose-plugin nvidia-driver-580-server nvidia-container-toolkit || true
DEMO_START
chmod 0755 /usr/local/sbin/uma-bct-demo-start

# User Data normally runs only on first launch. This enabled unit handles the
# driver reboot and EVERY subsequent stop/start. Enable without starting on the
# first boot; start after reboot. Do not order after cloud-final: it orders after
# multi-user.target itself, which would create a boot ordering cycle.
cat >/etc/systemd/system/uma-bct-demo.service <<'DEMO_UNIT'
[Unit]
Description=UMA BCT private GPU demo model preparation
Wants=network-online.target
After=network-online.target docker.service
Requires=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/uma-bct-demo-start
TimeoutStartSec=90min
UMask=0077
[Install]
WantedBy=multi-user.target
DEMO_UNIT
systemctl daemon-reload
systemctl enable uma-bct-demo.service
printf 'Bootstrap installed: %s. Reboot scheduled; wait for post-boot INFRASTRUCTURE READY.\n' "$(date -Is)"
# Delay lets cloud-final finish and mark User Data complete. Never repeatedly
# reboot from the persistent service; driver failures stay visible for repair.
shutdown -r +1
