#!/usr/bin/env python3
"""Keep self-contained EC2 User Data scripts in sync with their shared helpers.

The G6 script is the common shell template. Other variants replace only the
backend configuration, driver/startup steps, and embedded service configuration.
This tool writes repository scripts only; it never contacts or launches AWS.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"g6": ("gpu", "L4"), "g5": ("gpu", "A10G"), "p5": ("gpu", "H100"), "m7i": ("cpu", None)}
FILENAMES = {"g6": "demo-ec2-user-data.sh", "g5": "demo-ec2-g5-user-data.sh",
             "p5": "demo-ec2-p5-user-data.sh", "m7i": "demo-ec2-m7i-user-data.sh"}


def embedded(script: str, marker: str, content: str) -> str:
    before, remaining = script.split(f"<<'{marker}'\n", 1)
    _old, after = remaining.split(f"\n{marker}\n", 1)
    return before + f"<<'{marker}'\n" + content.rstrip("\n") + f"\n{marker}\n" + after


def render(variant: str) -> str:
    mode, gpu_name = VARIANTS[variant]
    script = (ROOT / "scripts/demo-ec2-user-data.sh").read_text()
    configuration = json.dumps({"mode": mode, "gpu_name": gpu_name})
    if "<<'DEMO_BACKEND'\n" not in script:
        anchor = "cat >/opt/uma-bct-demo/compose.yaml <<'DEMO_COMPOSE'"
        script = script.replace(anchor, "cat >/opt/uma-bct-demo/backend.json <<'DEMO_BACKEND'\n"
                                + configuration + "\nDEMO_BACKEND\nchmod 0644 /opt/uma-bct-demo/backend.json\n" + anchor)
    else:
        script = embedded(script, "DEMO_BACKEND", configuration)
    compose = (ROOT / ("compose.ec2-demo.cpu.yaml" if mode == "cpu" else "compose.ec2-demo.yaml")).read_text()
    script = embedded(script, "DEMO_COMPOSE", compose)
    script = embedded(script, "DEMO_HEALTH", (ROOT / "scripts/demo_ec2_health.py").read_text())
    if variant in {"g5", "p5"}:
        script = script.replace("one NVIDIA L4", f"one NVIDIA {gpu_name}")
        script = script.replace("supported L4 CUDA backends", f"supported {gpu_name} CUDA backends")
    elif variant == "m7i":
        script = script.replace("one NVIDIA L4", "M7i CPU rehearsal (>=32 GiB RAM)")
        script = script.replace("python3 ubuntu-drivers-common pciutils linux-headers-aws linux-aws", "python3")
        start = script.index("# Server driver branch >=580")
        end = script.index("systemctl enable docker", start)
        script = script[:start] + "# CPU rehearsal: no NVIDIA driver or Container Toolkit installation.\n" + script[end:]
        memory_check = ("# M7i CPU rehearsal needs 32 GiB provisioned RAM (allow kernel reservation).\n"
                        "demo_memory_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)\n"
                        "[ \"$demo_memory_kib\" -ge 31457280 ] || { echo 'CPU rehearsal requires >=32 GiB RAM.'; exit 1; }\n\n")
        script = script.replace("# Official Docker apt repository;", memory_check + "# Official Docker apt repository;", 1)
        script = script.replace("\nnvidia-smi\n# No app/photograph context", "\n# No app/photograph context", 1)
        script = script.replace("uma-bct-demo-health inventory\n", "uma-bct-demo-health inventory --backend cpu\n")
        script = script.replace("uma-bct-demo-health warm >", "uma-bct-demo-health warm --backend cpu >")
        script = script.replace(" nvidia-driver-580-server nvidia-container-toolkit || true", " || true")
        script = script.replace("Description=UMA BCT private GPU demo model preparation",
                                "Description=UMA BCT private CPU rehearsal model preparation")
        comment_start = script.index("# User Data normally runs only")
        unit_start = script.index("cat >/etc/systemd/system/uma-bct-demo.service", comment_start)
        script = script[:comment_start] + ("# Enable for every stop/start. Initial CPU startup is queued after installation;\n"
                                          "# no driver reboot is needed. Never order this unit after cloud-final.\n") + script[unit_start:]
        reboot_start = script.index("printf 'Bootstrap installed:")
        script = script[:reboot_start] + ("printf 'CPU rehearsal bootstrap installed: %s. Model preparation queued.\\n' \"$(date -Is)\"\n"
                                        "systemctl start --no-block uma-bct-demo.service\n")
    if len(script.encode()) > 16 * 1024:
        raise ValueError(f"{variant} User Data exceeds EC2's 16 KiB raw limit.")
    return script


def main() -> None:
    # Render all before replacing any output; a size/template failure leaves files intact.
    rendered = {variant: render(variant) for variant in VARIANTS}
    for variant, content in rendered.items():
        destination = ROOT / "scripts" / FILENAMES[variant]
        destination.write_text(content)
        destination.chmod(0o755)


if __name__ == "__main__":
    main()
