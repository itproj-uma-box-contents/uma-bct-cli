"""Model-free demo control tests: no AWS account, GPU, model, or private photos."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from scripts import build_demo_userdata as userdata
from scripts import demo_benchmark as benchmark
from scripts import demo_ec2 as client
from scripts import demo_ec2_health as health
from scripts import demo_ec2_mps as mps
from scripts import demo_ec2_parallel as parallel
from scripts import demo_ec2_remote as remote
from uma_bct.adapters.ollama_client import OllamaVisionModel
from uma_bct.adapters.request_timings import RequestTimings

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == "posix", "Temporary demo scripts target macOS/Linux")
class DemoConfigTest(unittest.TestCase):
    def test_g6e_and_g7e_scripts_match_gpu_and_driver_requirements(self):
        for variant, gpu in (("g6e", "L40S"), ("g7e", "RTX PRO 6000 Blackwell Server Edition")):
            with self.subTest(variant=variant):
                path = ROOT / "scripts" / f"demo-ec2-{variant}-user-data.sh"
                script = path.read_text()
                self.assertEqual(script, userdata.render(variant))
                self.assertLessEqual(len(script.encode()), 16 * 1024)
                backend = json.loads(script.split("<<'DEMO_BACKEND'\n", 1)[1].split("\nDEMO_BACKEND\n", 1)[0])
                self.assertEqual(backend, {"mode": "gpu", "gpu_name": gpu})
                self.assertIn("shutdown -r +1", script)
                self.assertIn("107374182400", script)
                self.assertIn(health.OLLAMA_IMAGE, script)
                self.assertIn(health.MODEL, script)
                helper = script.split("<<'DEMO_HEALTH'\n", 1)[1].split("\nDEMO_HEALTH\n", 1)[0] + "\n"
                self.assertEqual(helper, (ROOT / "scripts/demo_ec2_health.py").read_text())
                if variant == "g7e":
                    self.assertIn("apt-get install -y nvidia-driver-580-server-open", script)
                    self.assertNotIn("ubuntu-drivers install --gpgpu nvidia:580-server\n", script)
                else:
                    self.assertIn("ubuntu-drivers install --gpgpu nvidia:580-server\n", script)
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_p5_userdata_is_self_contained_and_requires_one_h100(self):
        path = ROOT / "scripts/demo-ec2-p5-user-data.sh"
        script = path.read_text()
        self.assertEqual(script, userdata.render("p5"))
        self.assertLessEqual(len(script.encode()), 16 * 1024)
        backend = json.loads(script.split("<<'DEMO_BACKEND'\n", 1)[1].split("\nDEMO_BACKEND\n", 1)[0])
        self.assertEqual(backend, {"mode": "gpu", "gpu_name": "H100"})
        self.assertIn("one NVIDIA H100", script)
        self.assertIn("supported H100 CUDA backends", script)
        self.assertIn("shutdown -r +1", script)
        self.assertIn("107374182400", script)
        self.assertIn(health.OLLAMA_IMAGE, script)
        self.assertIn("OLLAMA_NUM_PARALLEL: \"1\"", script)
        helper = script.split("<<'DEMO_HEALTH'\n", 1)[1].split("\nDEMO_HEALTH\n", 1)[0] + "\n"
        self.assertEqual(helper, (ROOT / "scripts/demo_ec2_health.py").read_text())
        subprocess.run(["bash", "-n", str(path)], check=True)

    def test_self_contained_userdata_is_valid_and_matches_source(self):
        script = (ROOT / "scripts/demo-ec2-user-data.sh").read_text()
        self.assertLessEqual(len(script.encode()), 16 * 1024)
        for marker, path in (("DEMO_COMPOSE", ROOT / "compose.ec2-demo.yaml"),
                             ("DEMO_HEALTH", ROOT / "scripts/demo_ec2_health.py")):
            embedded = script.split(f"<<'{marker}'\n", 1)[1].split(f"\n{marker}\n", 1)[0] + "\n"
            self.assertEqual(embedded, path.read_text())
        unit = script.split("<<'DEMO_UNIT'\n", 1)[1].split("\nDEMO_UNIT\n", 1)[0]
        self.assertNotIn("cloud-final.service", unit)  # Prevent multi-user.target boot ordering cycle.
        self.assertIn("systemctl enable uma-bct-demo.service", script)
        self.assertNotIn("systemctl start uma-bct-demo.service", script)
        self.assertIn("shutdown -r +1", script)
        self.assertIn("107374182400", script)
        self.assertIn("chmod 0644 /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg", script)
        self.assertIn("apt-get install -y nvidia-utils-580-server", script)
        subprocess.run(["bash", "-n", str(ROOT / "scripts/demo-ec2-user-data.sh")], check=True)
        subprocess.run(["bash", "-n", str(ROOT / "scripts/demo-ec2.sh")], check=True)

    def test_g5_and_m7i_userdata_are_self_contained_and_match_their_backend(self):
        for variant in ("g5", "m7i"):
            with self.subTest(variant=variant):
                path = ROOT / "scripts" / userdata.FILENAMES[variant]
                script = path.read_text()
                self.assertEqual(script, userdata.render(variant))
                self.assertLessEqual(len(script.encode()), 16 * 1024)
                config = json.loads(script.split("<<'DEMO_BACKEND'\n", 1)[1].split("\nDEMO_BACKEND\n", 1)[0])
                self.assertEqual(config["mode"], "cpu" if variant == "m7i" else "gpu")
                self.assertEqual(config["gpu_name"], None if variant == "m7i" else "A10G")
                helper = script.split("<<'DEMO_HEALTH'\n", 1)[1].split("\nDEMO_HEALTH\n", 1)[0] + "\n"
                self.assertEqual(helper, (ROOT / "scripts/demo_ec2_health.py").read_text())
                startup = script.split("<<'DEMO_START'\n", 1)[1].split("\nDEMO_START\n", 1)[0]
                if variant == "m7i":
                    self.assertNotIn("ubuntu-drivers install", script)
                    self.assertNotIn("apt-get install -y \"nvidia-container-toolkit", script)
                    self.assertNotIn("\nnvidia-smi\n", startup)
                    self.assertNotIn("shutdown -r", script)
                    self.assertIn("start --no-block uma-bct-demo.service", script)
                    self.assertIn("31457280", script)
                    self.assertIn("uma-bct-demo-health warm --backend cpu", startup)
                    cpu_compose = script.split("<<'DEMO_COMPOSE'\n", 1)[1].split("\nDEMO_COMPOSE\n", 1)[0] + "\n"
                    self.assertEqual(cpu_compose, (ROOT / "compose.ec2-demo.cpu.yaml").read_text())
                    self.assertNotIn("capabilities: [gpu]", cpu_compose)
                else:
                    self.assertIn("shutdown -r +1", script)
                    self.assertIn("ubuntu-drivers install --gpgpu", script)
                    self.assertIn("apt-get install -y nvidia-utils-580-server", script)
                    self.assertIn("chmod 0644 /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg", script)
                subprocess.run(["bash", "-n", str(path)], check=True)

    def test_exact_model_pins_agree(self):
        self.assertEqual(health.MODEL, remote.MODEL)
        self.assertEqual(health.MODEL_DIGEST, remote.MODEL_DIGEST)
        self.assertEqual(health.MODEL_DIGEST, benchmark.MODEL_DIGEST)
        self.assertIn(health.OLLAMA_IMAGE, (ROOT / "compose.yaml").read_text())

    def test_deployment_is_only_source_even_with_private_files_present(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "source.tar"
            metadata = client.source_bundle(destination)
            with tarfile.open(destination) as archive:
                names = archive.getnames()
                self.assertEqual(set(names), set(metadata["files"]) | {"Deployment.json"})
                self.assertTrue(all(remote.source_allowed(name) for name in metadata["files"]))
                self.assertFalse(any(name.endswith((".jpg", ".png", ".gguf")) or "/.git/" in name for name in names))


class DemoHealthTest(unittest.TestCase):
    def test_l4_readiness_does_not_accept_l40s_substring(self):
        with patch.object(health, "backend_config", return_value={"mode": "gpu", "gpu_name": "L4"}), \
                patch.object(health, "command", return_value="NVIDIA L40S, 580, 46068 MiB"):
            with self.assertRaisesRegex(RuntimeError, "L4"):
                health.readiness()

    def test_l40s_and_blackwell_readiness_require_the_selected_gpu_and_full_residency(self):
        running = {"name": health.MODEL, "digest": health.MODEL_DIGEST, "context_length": 16384, "size_vram": 1}
        for name in ("L40S", "RTX PRO 6000 Blackwell Server Edition"):
            with tempfile.TemporaryDirectory() as temporary:
                backend = Path(temporary) / "backend.json"
                backend.write_text(json.dumps({"mode": "gpu", "gpu_name": name}))
                with patch.object(health, "BACKEND_FILE", backend):
                    self.assertEqual(health.backend_config()["gpu_name"], name)
            for gpu, processor, accepted in ((f"NVIDIA {name}", "100% GPU", True),
                                             ("NVIDIA L4", "100% GPU", False),
                                             (f"NVIDIA {name}", "100% CPU", False)):
                with self.subTest(name=name, gpu=gpu, processor=processor), \
                        patch.object(health, "backend_config", return_value={"mode": "gpu", "gpu_name": name}), \
                        patch.object(health, "command", side_effect=[gpu, "GPU device", f"{health.MODEL} {processor}"]), \
                        patch.object(health, "container_info", return_value=("container", {})), \
                        patch.object(health, "inventory", return_value={}), \
                        patch.object(health, "api", return_value={"models": [running]}):
                    if accepted:
                        self.assertEqual(health.readiness()["status"], "ready")
                    else:
                        with self.assertRaises(RuntimeError):
                            health.readiness()

    def test_h100_configuration_and_readiness_reject_other_devices_and_cpu_spill(self):
        with tempfile.TemporaryDirectory() as temporary:
            backend = Path(temporary) / "backend.json"
            backend.write_text('{"mode":"gpu","gpu_name":"H100"}')
            with patch.object(health, "BACKEND_FILE", backend):
                self.assertEqual(health.backend_config()["gpu_name"], "H100")
        running = {"name": health.MODEL, "digest": health.MODEL_DIGEST, "context_length": 16384, "size_vram": 1}
        for gpu, processor, accepted in (("NVIDIA H100 80GB HBM3", "100% GPU", True),
                                         ("NVIDIA L4", "100% GPU", False),
                                         ("H100\nH100", "100% GPU", False),
                                         ("H100", "60%/40% CPU/GPU", False)):
            with self.subTest(gpu=gpu, processor=processor), \
                    patch.object(health, "backend_config", return_value={"mode": "gpu", "gpu_name": "H100"}), \
                    patch.object(health, "command", side_effect=[gpu, "GPU device", f"{health.MODEL} {processor}"]), \
                    patch.object(health, "container_info", return_value=("container", {})), \
                    patch.object(health, "inventory", return_value={}), \
                    patch.object(health, "api", return_value={"models": [running]}):
                if accepted:
                    self.assertEqual(health.readiness()["status"], "ready")
                else:
                    with self.assertRaises(RuntimeError):
                        health.readiness()

    def info(self):
        return {"State": {"Running": True}, "Config": {"Image": health.OLLAMA_IMAGE, "Env": [
            "OLLAMA_NO_CLOUD=1", "OLLAMA_KEEP_ALIVE=-1", "OLLAMA_MAX_LOADED_MODELS=1", "OLLAMA_NUM_PARALLEL=1",
                "OLLAMA_CONTEXT_LENGTH=16384"]}, "HostConfig": {"Runtime": "runc", "PortBindings": {}, "DeviceRequests": [
                {"Driver": "nvidia", "Count": 1}]}}

    def test_published_api_or_wrong_settings_are_rejected(self):
        for edit in ("port", "settings", "gpu", "image"):
            with self.subTest(edit=edit):
                info = self.info()
                if edit == "port":
                    info["HostConfig"]["PortBindings"] = {"11434/tcp": [{"HostPort": "11434"}]}
                elif edit == "settings":
                    info["Config"]["Env"] = []
                elif edit == "gpu":
                    info["HostConfig"]["DeviceRequests"] = []
                else:
                    info["Config"]["Image"] = "latest"
                with patch.object(health, "command", side_effect=["container", json.dumps([info])]):
                    with self.assertRaises(RuntimeError):
                        health.container_info()

    def test_status_never_warms_and_requires_gpu_context_and_identity(self):
        running = {"name": health.MODEL, "digest": health.MODEL_DIGEST, "context_length": 16384, "size_vram": 123}
        for state in ("ready", "cold", "context", "cpu"):
            with self.subTest(state=state):
                value = dict(running)
                if state == "context":
                    value["context_length"] = 4096
                models = [] if state == "cold" else [value]
                display = f"{health.MODEL} 100% {'CPU' if state == 'cpu' else 'GPU'}"
                with patch.object(health, "command", side_effect=["NVIDIA L4, 580, 23034 MiB", "GPU device", display]), \
                        patch.object(health, "container_info", return_value=("container", {})), \
                        patch.object(health, "inventory", return_value={"digest": health.MODEL_DIGEST}), \
                        patch.object(health, "api", return_value={"models": models}) as api:
                    if state == "ready":
                        self.assertEqual(health.readiness()["status"], "ready")
                    else:
                        with self.assertRaises(RuntimeError):
                            health.readiness()
                    self.assertEqual(api.call_args.args[1], "ps")
                    self.assertEqual(len(api.call_args.args), 2)

    def test_warming_uses_production_settings_and_no_image(self):
        running = {"name": health.MODEL, "digest": health.MODEL_DIGEST, "context_length": 16384, "size_vram": 1}
        with patch.object(health, "command", side_effect=["NVIDIA L4", "GPU device", f"{health.MODEL} 100% GPU"]), \
                patch.object(health, "container_info", return_value=("container", {})), \
                patch.object(health, "inventory", return_value={}), \
                patch.object(health, "api", side_effect=[{"load_duration": 10}, {"models": [running]}]) as api:
            value = health.readiness(warm=True)
        payload = api.call_args_list[0].args[2]
        self.assertEqual(payload["options"], {"num_ctx": 16384, "temperature": 0, "num_predict": 1})
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], -1)
        self.assertNotIn("images", payload["messages"][0])
        self.assertEqual(value["cold_or_reheat_probe"]["load_duration"], 10)

    def test_changed_weights_are_rejected(self):
        with patch.object(health, "api", side_effect=[{"version": "0.32.9"}, {"models": [
            {"name": health.MODEL, "digest": "changed", "details": {"quantization_level": "Q4_K_M"}}]}]):
            with self.assertRaisesRegex(RuntimeError, "identity"):
                health.inventory({})

    def test_g5_requires_a10g_and_full_gpu_residency(self):
        running = {"name": health.MODEL, "digest": health.MODEL_DIGEST, "context_length": 16384, "size_vram": 1}
        with patch.object(health, "backend_config", return_value={"mode": "gpu", "gpu_name": "A10G"}), \
                patch.object(health, "command", side_effect=["NVIDIA A10G", "GPU device", f"{health.MODEL} 100% GPU"]), \
                patch.object(health, "container_info", return_value=("container", {})), \
                patch.object(health, "inventory", return_value={}), \
                patch.object(health, "api", return_value={"models": [running]}):
            self.assertEqual(health.readiness()["backend"]["gpu_name"], "A10G")
        with patch.object(health, "backend_config", return_value={"mode": "gpu", "gpu_name": "A10G"}), \
                patch.object(health, "command", return_value="NVIDIA L4"):
            with self.assertRaisesRegex(RuntimeError, "A10G"):
                health.readiness()

    def test_cpu_readiness_is_explicit_and_never_executes_nvidia_tooling(self):
        running = {"name": health.MODEL, "digest": health.MODEL_DIGEST, "context_length": 16384, "size_vram": 0}
        config = {"mode": "cpu", "gpu_name": None}
        with patch.object(health, "backend_config", return_value=config), \
                patch.object(health, "command", return_value=f"{health.MODEL} 100% CPU") as command, \
                patch.object(health, "container_info", return_value=("container", {})), \
                patch.object(health, "inventory", return_value={}), \
                patch.object(health, "api", return_value={"models": [running]}) as api:
            with self.assertRaisesRegex(RuntimeError, "explicitly"):
                health.readiness(warm=True)
            api.assert_not_called()
            value = health.readiness(expected_backend="cpu")
            self.assertEqual(value["backend"]["mode"], "cpu")
            self.assertIsNone(value["gpu"])
            self.assertEqual(command.call_args.args, ("docker", "exec", "container", "ollama", "ps"))
            self.assertIn("test ! -e /dev/nvidiactl", command.call_args_list[0].args[-1])
            running["size_vram"] = 4761698959
            command.return_value = f"{health.MODEL} 42%/58% CPU/GPU"
            value = health.readiness(expected_backend="cpu")
            self.assertIn("42%/58%", value["ollama_ps"])
            running["context_length"] = 4096
            with self.assertRaisesRegex(RuntimeError, "CPU"):
                health.readiness(expected_backend="cpu")

    def test_cpu_container_cannot_request_a_gpu_or_omit_explicit_cpu_settings(self):
        config = {"mode": "cpu", "gpu_name": None}
        info = self.info()
        with patch.object(health, "command", side_effect=["container", json.dumps([info])]):
            with self.assertRaisesRegex(RuntimeError, "disable GPU"):
                health.container_info(config)
        info["HostConfig"]["DeviceRequests"] = []
        info["Config"]["Env"] += ["CUDA_VISIBLE_DEVICES=-1", "OLLAMA_VULKAN=0"]
        with patch.object(health, "command", side_effect=["container", json.dumps([info])]):
            self.assertEqual(health.container_info(config)[0], "container")

    def test_malformed_backend_configuration_is_rejected(self):
        for configuration in ({"mode": "auto"}, {"mode": "gpu", "gpu_name": "unknown"}, []):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "backend.json"
                path.write_text(json.dumps(configuration))
                with patch.object(health, "BACKEND_FILE", path):
                    with self.assertRaises(RuntimeError):
                        health.backend_config()


@unittest.skipUnless(os.name == "posix", "EC2 controller targets Linux")
class DemoRemoteTest(unittest.TestCase):
    def test_unsafe_archives_are_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, kind in (("../escape.py", tarfile.REGTYPE), ("src/uma_bct/model.py", tarfile.SYMTYPE),
                               ("approved.jpg", tarfile.REGTYPE)):
                with self.subTest(name=name, kind=kind):
                    archive = root / "upload.tar"
                    with tarfile.open(archive, "w") as bundle:
                        member = tarfile.TarInfo(name)
                        member.type = kind
                        member.size = 0
                        bundle.addfile(member, io.BytesIO())
                    with patch.object(remote, "APPLICATION", root / "app"), patch.object(remote.subprocess, "run") as build:
                        with self.assertRaises(ValueError):
                            remote.install_source(archive)
                        build.assert_not_called()
            self.assertFalse((root / "escape.py").exists())

    def test_each_run_is_fresh_and_retains_status_and_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, results = root / "images", root / "results"
            inputs.mkdir()
            results.mkdir()
            image = inputs / "approved.jpg"
            image.write_bytes(b"synthetic image fixture")
            for cli_code, status in ((0, "complete"), (1, "partial"), (2, "failed")):
                run_id = f"{cli_code + 1:032x}"
                def execute(args, *, stdout, stderr, run_id=run_id, status=status, cli_code=cli_code):
                    self.assertIn("--reprocess", args)
                    self.assertEqual(args[args.index("--verification") + 1], "never")
                    self.assertEqual(args[args.index("--model") + 1], remote.MODEL)
                    self.assertEqual(args[args.index("--results-directory") + 1], "/data/outputs/Batches")
                    self.assertNotIn("--gpus", args)  # CLI itself does not own the persistent model service.
                    batch = results / run_id / "Batches" / "Batch fixture"
                    batch.mkdir(parents=True)
                    for name in ("Catalogue.csv", "Review.html"):
                        (batch / name).write_text("fixture")
                    (batch / "Run Details.json").write_text(json.dumps({"images": [
                        {"boxes": [{"box_id": "1"}, {"box_id": "2"}], "fallback_path": []}]}))
                    json.dump({"status": status, "images_total": 1, "images_reused": 0,
                               "images_processed_this_start": 1, "catalogue_records": 3,
                               "batch_directory": "/data/outputs/Batches/Batch fixture"}, stdout)
                    return SimpleNamespace(returncode=cli_code)
                with patch.object(remote, "INPUTS", inputs), patch.object(remote, "RESULTS", results), \
                        patch.object(remote, "health", return_value={"status": "ready"}), \
                        patch.object(remote, "deployment", return_value={"image": "prepared"}), \
                        patch.object(remote, "staged", return_value=(image, {"source_sha256": remote.digest(image)})), \
                        patch.object(remote.subprocess, "run", side_effect=execute):
                    response = remote.run({"run_id": run_id})
                    self.assertEqual(response["envelope"]["cli_exit_code"], cli_code)
                    self.assertEqual(response["envelope"]["status"], status)
                    self.assertEqual(response["envelope"]["physical_box_records"], 2)
                    self.assertEqual(response["envelope"]["catalogue_rows"], 3)
                    self.assertIn("Batches/Batch fixture/Review.html", response["transfer"]["files"])
                    with self.assertRaises(FileExistsError):
                        remote.run({"run_id": run_id})

    def test_cold_server_fails_before_inference_or_result_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(remote, "RESULTS", Path(temporary)), \
                    patch.object(remote, "health", side_effect=RuntimeError("cold")), \
                    patch.object(remote.subprocess, "run") as execute:
                with self.assertRaisesRegex(RuntimeError, "cold"):
                    remote.run({"run_id": "a" * 32})
                execute.assert_not_called()
                self.assertEqual(list(Path(temporary).iterdir()), [])


@unittest.skipUnless(os.name == "posix", "SSH wrapper targets macOS/Linux")
class DemoParallelServiceTest(unittest.TestCase):
    def test_scheduler_batch_readiness_checks_exact_gpu_residency_for_each_runner(self):
        info = {"Id": "fixture", "State": {"Running": True},
                "Config": {"Image": health.OLLAMA_IMAGE, "Env": ["OLLAMA_NO_CLOUD=1", "OLLAMA_KEEP_ALIVE=-1",
                    "OLLAMA_MAX_LOADED_MODELS=1", "OLLAMA_NUM_PARALLEL=1", "OLLAMA_CONTEXT_LENGTH=16384",
                    "CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"]},
                "HostConfig": {"PortBindings": {}, "DeviceRequests": [{"Driver": "nvidia", "Count": 1}]}}
        running = {"name": health.MODEL, "digest": health.MODEL_DIGEST, "context_length": 16384,
                   "size": 100, "size_vram": 100}
        def command(*args):
            if args[0] == "nvidia-smi":
                return "NVIDIA H100 80GB HBM3, 580.178.04, 81559 MiB"
            if args[:2] == ("docker", "inspect"):
                return json.dumps([info, info])
            return "GPU device"
        module = SimpleNamespace(backend_config=lambda: {"mode": "gpu", "gpu_name": "H100"}, command=command,
            OLLAMA_IMAGE=health.OLLAMA_IMAGE, MODEL=health.MODEL, MODEL_DIGEST=health.MODEL_DIGEST,
            CONTEXT=health.CONTEXT, SETTINGS=health.SETTINGS, inventory=lambda _: {},
            api=lambda *_: {"models": [running]})
        self.assertEqual(len(parallel.scheduled_readiness(module, 2)), 2)
        for key, value in (("size_vram", 99), ("context_length", 8192), ("digest", "changed")):
            with self.subTest(key=key):
                changed = {**running, key: value}
                module.api = lambda *_, changed=changed: {"models": [changed]}
                with self.assertRaisesRegex(RuntimeError, "cold|CPU"):
                    parallel.scheduled_readiness(module, 2)

    def test_read_only_readiness_overlaps_all_independent_runners(self):
        gate = threading.Barrier(4)
        def readiness(**kwargs):
            gate.wait(timeout=2)
            return {"backend": {"mode": "gpu"}, "gpu": "H100", "settings": health.SETTINGS}
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text('{"workers": 4}')
            module = SimpleNamespace(backend_config=lambda: {"mode": "gpu"}, readiness=readiness)
            with patch.object(parallel, "CONFIG", config), patch.object(parallel, "health_module", return_value=module):
                result = parallel.status()
            self.assertEqual(len(result["runners"]), 4)
            self.assertEqual(result["settings"]["transcription_workers"], 4)

    def test_many_runners_keep_pinned_private_services_and_recheck_every_loaded_model(self):
        template = (ROOT / "compose.ec2-demo.parallel.yaml").read_bytes()
        for count in (4, 6):
            compose = parallel.runner_compose(template, count).decode()
            self.assertEqual(compose.count("    image: " + health.OLLAMA_IMAGE), count - 1)
            self.assertNotIn("ports:", compose)
            self.assertEqual(compose.count("OLLAMA_CONTEXT_LENGTH: \"16384\""), count - 1)
            with tempfile.TemporaryDirectory() as temporary:
                config = Path(temporary) / "config.json"
                config.write_text(json.dumps({"workers": count}))
                def readiness(**kwargs):
                    if kwargs.get("service") == "ollama-parallel-3" and not kwargs.get("warm"):
                        raise RuntimeError("earlier reader spilled onto CPU")
                    return {"backend": {"mode": "gpu"}, "gpu": "H100", "settings": health.SETTINGS}
                module = SimpleNamespace(backend_config=lambda: {"mode": "gpu"}, readiness=readiness)
                with patch.object(parallel, "CONFIG", config), patch.object(parallel, "health_module", return_value=module):
                    with self.assertRaisesRegex(RuntimeError, "spilled"):
                        parallel.status(warm_second=True)

    def test_active_run_excludes_configuration_before_any_mutation(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            lock = Path(temporary) / "lock"
            lock.touch(mode=0o600)
            with lock.open("a") as active, patch.object(parallel, "LOCK", lock), \
                    patch.object(parallel, "ubuntu_identity", return_value=(os.getuid(), os.getgid())), \
                    patch.object(parallel, "configure_locked") as configure:
                fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "configuration was not changed"):
                    parallel.configure(SimpleNamespace(workers=1))
                configure.assert_not_called()

    def test_startup_wait_is_bounded_api_only_and_rejects_changed_version(self):
        module = SimpleNamespace(COMPOSE="fixture", container_info=Mock(return_value=("second", {})),
                                 api=Mock(side_effect=[ConnectionError("starting"), {"version": "0.32.9"}]),
                                 readiness=Mock())
        with patch.object(parallel.time, "sleep") as sleep:
            parallel.wait_api(module, service="ollama-parallel", compose="parallel fixture")
        self.assertEqual(module.api.call_count, 2)
        module.readiness.assert_not_called()
        sleep.assert_called_once()
        module.api = Mock(return_value={"version": "other"})
        with self.assertRaisesRegex(ValueError, "version"):
            parallel.wait_api(module)

    def test_status_requires_both_runners_on_gpu_and_rechecks_primary_after_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "parallel.json"
            config.write_text('{"workers": 2}')
            ready = {"backend": {"mode": "gpu"}, "gpu": "L4", "settings": health.SETTINGS}
            module = SimpleNamespace(backend_config=lambda: {"mode": "gpu"})
            with patch.object(parallel, "CONFIG", config), patch.object(parallel, "health_module", return_value=module):
                module.readiness = Mock(side_effect=[ready, ready, RuntimeError("primary left GPU")])
                with self.assertRaisesRegex(RuntimeError, "primary left GPU"):
                    parallel.status(warm_second=True)
                self.assertEqual(module.readiness.call_args_list[1].kwargs,
                                 {"warm": True, "service": "ollama-parallel", "compose": str(parallel.COMPOSE)})
                module.readiness = Mock(side_effect=[ready, ready, ready])
                value = parallel.status()
                self.assertEqual(len(value["runners"]), 2)
                self.assertEqual(value["settings"]["transcription_workers"], 2)

    def test_second_runner_failure_is_not_accepted_as_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "parallel.json"
            config.write_text('{"workers": 2}')
            module = SimpleNamespace(backend_config=lambda: {"mode": "gpu"},
                                     readiness=Mock(side_effect=[{}, RuntimeError("wrong model digest")]))
            with patch.object(parallel, "CONFIG", config), patch.object(parallel, "health_module", return_value=module):
                with self.assertRaisesRegex(RuntimeError, "wrong model digest"):
                    parallel.status()

    def test_failed_preparation_restores_one_runner_without_cpu_fallback(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            for name in ("HEALTH", "INSTALLED", "COMPOSE", "CONFIG", "UNIT", "LOCK"):
                stack.enter_context(patch.object(parallel, name, root / name))
            stack.enter_context(patch.object(parallel, "ubuntu_identity", return_value=(os.getuid(), os.getgid())))
            source_health, source_compose = root / "source-health", root / "source-compose"
            source_health.write_text("source helper")
            source_compose.write_bytes((ROOT / "compose.ec2-demo.parallel.yaml").read_bytes())
            module = SimpleNamespace(backend_config=lambda: {"mode": "gpu"},
                                     container_info=lambda: ("first", {}), command=Mock())
            stack.enter_context(patch.object(parallel, "health_module", return_value=module))

            def command(*args):
                if args[:2] == ("systemctl", "restart"):
                    raise subprocess.CalledProcessError(1, args)

            execute = stack.enter_context(patch.object(parallel, "command", side_effect=command))
            with self.assertRaisesRegex(RuntimeError, "restored one-runner"):
                parallel.configure(SimpleNamespace(workers=2, health_file=source_health, compose_file=source_compose))
            self.assertEqual(json.loads(parallel.CONFIG.read_text()), {"workers": 1})
            self.assertIn(call("systemctl", "disable", "--now", parallel.SERVICE), execute.call_args_list)


@unittest.skipUnless(os.name == "posix", "SSH wrapper targets macOS/Linux")
class DemoMPSTest(unittest.TestCase):
    def test_matching_json_flag_cannot_hide_incomplete_scheduler_configuration(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(mps, "BASE", root))
            (root / "backend.json").write_text('{"mode":"gpu","gpu_name":"H100"}')
            for name in ("COMPOSE", "DROPIN", "UNIT", "CONFIG"):
                stack.enter_context(patch.object(mps, name, root / name))
            stack.enter_context(patch.object(mps, "configuration_lock", return_value=contextlib.nullcontext()))
            mps.COMPOSE.write_text("CUDA_MPS_PIPE_DIRECTORY: stale")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                mps.configure(False)
            mps.COMPOSE.write_text("normal Compose")
            mps.DROPIN.write_text("stale dependency")
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                mps.configure(False)
            mps.CONFIG.write_text('{"enabled":true}')
            with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                mps.configure(True)

    def test_scheduler_preserves_settings_and_requires_private_api(self):
        source = (ROOT / "compose.ec2-demo.parallel.yaml").read_bytes()
        configured = mps.decorate(parallel.runner_compose(source, 4)).decode()
        self.assertEqual(configured.count(health.OLLAMA_IMAGE), 3)
        self.assertEqual(configured.count('OLLAMA_CONTEXT_LENGTH: "16384"'), 3)
        self.assertEqual(configured.count('CUDA_MPS_PIPE_DIRECTORY: "/tmp/nvidia-mps"'), 3)
        self.assertEqual(configured.count("    ipc: host"), 3)
        self.assertNotIn("ports:", configured)
        with self.assertRaisesRegex(ValueError, "publish"):
            mps.decorate(source + b'\n    ports: ["11434:11434"]\n')
        with self.assertRaisesRegex(ValueError, "already"):
            mps.decorate(configured.encode())

    def test_scheduler_requires_actual_client_mapping_and_shared_ipc(self):
        info = {"Id": "fixture-container-id", "Name": "/uma-bct-demo-ollama-1",
                "Config": {"Env": ["CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps"]},
                "HostConfig": {"IpcMode": "host"},
                "Mounts": [{"Source": str(mps.PIPE), "Destination": "/tmp/nvidia-mps"}]}
        def command(*args, **kwargs):
            if args[0] == "systemctl":
                return "active"
            if args[:2] == ("docker", "inspect"):
                return json.dumps([info])
            return "123" if kwargs.get("text") == "get_server_list\n" else "456"
        with patch.object(mps, "enabled", return_value=True), patch.object(mps, "command", side_effect=command), \
                patch.object(Path, "read_bytes", return_value=b"helper"), \
                patch.object(Path, "read_text", return_value="/system.slice/docker-fixture-container-id.scope"):
            self.assertTrue(mps.status(1)["gpu_mps"])
            with patch.object(Path, "read_text", return_value="other-container"):
                with self.assertRaisesRegex(RuntimeError, "verified"):
                    mps.status(1)
            info["HostConfig"]["IpcMode"] = "private"
            with self.assertRaisesRegex(RuntimeError, "bypasses"):
                mps.status(1)

    def test_scheduler_failed_preparation_restores_original_primary(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(mps, "BASE", root))
            for name in ("CONFIG", "COMPOSE", "BACKUP", "INSTALLED", "UNIT", "DROPIN", "PIPE", "LOG"):
                stack.enter_context(patch.object(mps, name, root / name))
            original = (ROOT / "compose.ec2-demo.yaml").read_bytes()
            mps.COMPOSE.write_bytes(original)
            (root / "backend.json").write_text('{"mode":"gpu","gpu_name":"H100"}')
            stack.enter_context(patch.object(mps, "configuration_lock", return_value=contextlib.nullcontext()))
            stack.enter_context(patch.object(mps, "foreign_controller", return_value=False))
            stopped = stack.enter_context(patch.object(mps, "stop_models"))
            attempts = []
            def command(*args, **kwargs):
                if args == ("systemctl", "restart", mps.PRIMARY):
                    attempts.append(args)
                    if len(attempts) == 1:
                        raise RuntimeError("scheduler startup failed")
                return ""
            stack.enter_context(patch.object(mps, "command", side_effect=command))
            with self.assertRaisesRegex(RuntimeError, "original GPU service restored"):
                mps.configure(True)
            self.assertEqual(mps.COMPOSE.read_bytes(), original)
            self.assertFalse(json.loads(mps.CONFIG.read_text())["enabled"])
            self.assertFalse(mps.DROPIN.exists())
            self.assertEqual(stopped.call_count, 2)
            self.assertEqual(len(attempts), 2)


class DemoClientTest(unittest.TestCase):
    def test_mismatched_stream_cannot_supply_another_runs_nonzero_cli_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.jpg"
            image.write_bytes(b"synthetic")
            response = {"protocol": "uma-bct-stream-v1", "envelope": {"run_id": "a" * 32, "cli_exit_code": 1,
                "status": "partial", "backend": "gpu", "source": {"source_sha256": "wrong"},
                "settings": {"transcription_workers": 1}}}
            process = SimpleNamespace(stdin=Mock(), stdout=io.BytesIO(json.dumps(response).encode() + b"\n"),
                                      wait=Mock(return_value=0), poll=Mock(return_value=0))
            with patch.object(client, "connection", return_value=("ubuntu@fixture", [])), \
                    patch.object(client, "result_root", return_value=root / "local"), \
                    patch.object(client.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                    patch.object(client.subprocess, "Popen", return_value=process):
                value, code = client.demo_run(image, "never", time.monotonic(), stream_results=True)
            self.assertEqual(code, 2)
            self.assertIsNone(value["cli_exit_code"])

    def test_single_ssh_stream_validates_and_publishes_all_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.jpg"
            image.write_bytes(b"synthetic approved image")
            source = root / "server"
            (source / "Batch fixture/Crops").mkdir(parents=True)
            (source / "Batch fixture/Crops/box.png").write_bytes(b"synthetic crop")
            (source / "Batch fixture/Review.html").write_text('<a href="Crops/box.png">View</a>')
            response = {"protocol": "uma-bct-stream-v1", "envelope": {
                "run_id": "a" * 32, "cli_exit_code": 0, "status": "complete", "backend": "gpu",
                "source": {"source_sha256": client.digest(image)}, "settings": {"transcription_workers": 4}},
                "transfer": remote.transfer_manifest(source)}
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w") as bundle:
                bundle.add(source, arcname=".")
            process = SimpleNamespace(stdin=Mock(), stdout=io.BytesIO(json.dumps(response).encode() + b"\n" + archive.getvalue()),
                                      wait=Mock(return_value=0), poll=Mock(return_value=0))
            with patch.object(client, "connection", return_value=("ubuntu@fixture", [])), \
                    patch.object(client, "result_root", return_value=root / "local"), \
                    patch.object(client.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                    patch.object(client.subprocess, "Popen", return_value=process) as ssh, \
                    patch.object(client, "rpc") as rpc:
                value, code = client.demo_run(image, "never", time.monotonic(), transcription_workers=4, stream_results=True)
            self.assertEqual(code, 0)
            self.assertTrue(value["retrieval_complete"])
            self.assertEqual(value["result_transport"], "ssh-stream")
            self.assertEqual((Path(value["local_results"]) / "Batch fixture/Crops/box.png").read_bytes(), b"synthetic crop")
            self.assertEqual(json.loads(process.stdin.write.call_args.args[0])["action"], "run-stream-v1")
            ssh.assert_called_once()
            rpc.assert_not_called()

    def test_stream_legacy_server_and_source_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for response in ({"status": "failed", "error": "Unknown demo action."},
                             {"protocol": "uma-bct-stream-v1", "envelope": {"run_id": "wrong"}}, []):
                process = SimpleNamespace(stdin=Mock(), stdout=io.BytesIO(json.dumps(response).encode() + b"\n"),
                                          wait=Mock(return_value=0), poll=Mock(return_value=0))
                with patch.object(client, "connection", return_value=("ubuntu@fixture", [])), \
                        patch.object(client.subprocess, "Popen", return_value=process):
                    with self.assertRaises(client.StreamTransferError):
                        client.stream_rpc({"run_id": "a" * 32, "source_sha256": "b" * 64,
                                           "backend": "gpu", "transcription_workers": 4}, root)
                self.assertFalse(list(root.rglob("Results")))

    def test_failed_stream_retains_partial_cli_exit_code_and_transfer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.jpg"
            image.write_bytes(b"synthetic")
            response = {"protocol": "uma-bct-stream-v1", "envelope": {"run_id": "a" * 32, "cli_exit_code": 1,
                "status": "partial", "backend": "gpu", "source": {"source_sha256": client.digest(image)},
                "settings": {"transcription_workers": 1}}, "transfer": {"files": {}}}
            process = SimpleNamespace(stdin=Mock(), stdout=io.BytesIO(json.dumps(response).encode() + b"\npartial"),
                                      wait=Mock(return_value=255), poll=Mock(return_value=255))
            with patch.object(client, "connection", return_value=("ubuntu@fixture", [])), \
                    patch.object(client, "result_root", return_value=root / "local"), \
                    patch.object(client.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                    patch.object(client.subprocess, "Popen", return_value=process):
                value, code = client.demo_run(image, "never", time.monotonic(), stream_results=True)
            self.assertEqual(code, 1)
            self.assertEqual(value["cli_status"], "partial")
            self.assertEqual(value["status"], "failed")
            self.assertTrue(list(root.rglob("Transfer.tar")))
            self.assertFalse(list(root.rglob("Results")))

    def test_archive_retrieval_preserves_exact_files_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, extracted = root / "source", root / "extracted"
            source.mkdir()
            extracted.mkdir()
            (source / "Batch fixture/Box Images").mkdir(parents=True)
            (source / "Batch fixture/Review.html").write_text('<a href="Box Images/crop.png">crop</a>')
            (source / "Batch fixture/Box Images/crop.png").write_bytes(b"synthetic crop bytes")
            transfer = remote.transfer_manifest(source)
            archive = root / "transfer.tar"
            with tarfile.open(archive, "w") as bundle:
                bundle.add(source, arcname=".")
            client.extract_result_archive(archive, extracted, transfer)
            client.validate_transfer(extracted, transfer)
            self.assertEqual((extracted / "Batch fixture/Box Images/crop.png").read_bytes(), b"synthetic crop bytes")

    def test_archive_rejects_traversal_links_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, attack in enumerate(("../escape", "symlink", "duplicate")):
                directory = root / str(index)
                directory.mkdir()
                archive = root / f"attack-{index}.tar"
                with tarfile.open(archive, "w") as bundle:
                    member = tarfile.TarInfo("../escape" if attack == "../escape" else "fixture")
                    if attack == "symlink":
                        member.type, member.linkname = tarfile.SYMTYPE, "../escape"
                    else:
                        member.size = 3
                    bundle.addfile(member, None if attack == "symlink" else io.BytesIO(b"abc"))
                    if attack == "duplicate":
                        bundle.addfile(member, io.BytesIO(b"abc"))
                with self.assertRaises((ValueError, RuntimeError)):
                    client.extract_result_archive(archive, directory, {"files": {"fixture": {"size": 3}}})
            self.assertFalse((root / "escape").exists())

    def test_parallel_mode_rejects_legacy_single_runner_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.jpg"
            image.write_bytes(b"synthetic")
            with patch.object(client, "result_root", return_value=root), \
                    patch.object(client, "rpc", return_value={"status": "ready", "readiness": {"settings": {}}}) as rpc:
                value, code = client.demo_run(image, "never", time.monotonic(), transcription_workers=2)
            self.assertEqual(code, 2)
            self.assertIn("two independent runners", value["error"])
            rpc.assert_called_once_with({"action": "status", "backend": "gpu", "transcription_workers": 2})

    def test_parallel_cpu_prepare_fails_before_transfer(self):
        with patch.object(client, "connection", return_value=("ubuntu@fixture", [])), \
                patch.object(client, "copy") as copy, patch.object(client, "ssh_json") as ssh:
            with self.assertRaises(ValueError):
                client.prepare(None, backend="cpu", transcription_workers=2)
            copy.assert_not_called()
            ssh.assert_not_called()

    def test_cpu_backend_is_an_explicit_wrapper_option(self):
        args = client.parser().parse_args(["run", "--image", "/fixture.jpg"])
        self.assertEqual(args.backend, "gpu")
        args = client.parser().parse_args(["run", "--image", "/fixture.jpg", "--backend", "cpu"])
        self.assertEqual(args.backend, "cpu")
        with patch.object(client, "ssh_json") as ssh:
            with self.assertRaises(ValueError):
                client.prepare(None, backend="cpu; shell-injection")
            ssh.assert_not_called()

    def test_prepare_retains_legacy_gpu_health_arguments(self):
        for backend, expected in (("gpu", "/usr/local/bin/uma-bct-demo-health warm"),
                                  ("cpu", "/usr/local/bin/uma-bct-demo-health warm --backend cpu")):
            with patch.object(client, "connection", return_value=("ubuntu@fixture", [])), \
                    patch.object(client, "ssh_json", side_effect=RuntimeError("stop before transfer")) as ssh:
                with self.assertRaises(RuntimeError):
                    client.prepare(None, backend=backend)
                self.assertEqual(ssh.call_args.args[0], expected)

    def test_cpu_run_rejects_a_legacy_gpu_controller_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "fixture.jpg"
            image.write_bytes(b"synthetic")
            with patch.object(client, "result_root", return_value=root), \
                    patch.object(client, "rpc", return_value={"status": "ready", "readiness": {"gpu": "NVIDIA L4"}}) as rpc:
                value, code = client.demo_run(image, "never", time.monotonic(), backend="cpu")
            self.assertEqual(code, 2)
            self.assertIn("explicit CPU", value["error"])
            rpc.assert_called_once_with({"action": "status", "backend": "cpu"})
    def test_output_root_cannot_enter_checkout_even_through_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            alias = Path(temporary) / "alias"
            alias.symlink_to(ROOT, target_is_directory=True)
            for destination in (ROOT / "unsafe-results", alias / "unsafe-results"):
                with patch.dict(os.environ, {"UMA_BCT_DEMO_RESULTS": str(destination)}):
                    with self.assertRaisesRegex(ValueError, "outside"):
                        client.result_root()

    def test_transfer_checks_full_file_set_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "Review.html").write_text('<a href="Box Images/box.png">View</a>')
            crops = directory / "Box Images"
            crops.mkdir()
            (crops / "box.png").write_bytes(b"synthetic crop")
            transfer = remote.transfer_manifest(directory)
            client.validate_transfer(directory, transfer)
            (crops / "box.png").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                client.validate_transfer(directory, transfer)
            (crops / "box.png").unlink()
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                client.validate_transfer(directory, transfer)

    def test_retrieve_reserves_new_folder_each_time_and_retains_failed_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "server"
            fixture.mkdir()
            (fixture / "Review.html").write_text("fixture")
            response = {"envelope": {"run_id": "a" * 32}, "transfer": remote.transfer_manifest(fixture)}
            def copy(_source, target, *, recursive):
                shutil.copytree(fixture, target, dirs_exist_ok=True)
            with patch.object(client, "connection", return_value=("ubuntu@fixture", [])), \
                    patch.object(client, "copy", side_effect=copy):
                first = client.retrieve(response, root / "local")
                second = client.retrieve(response, root / "local")
                self.assertNotEqual(first, second)
                self.assertTrue(first.is_dir())
                (fixture / "Review.html").unlink()
                with self.assertRaisesRegex(RuntimeError, "partial copy retained"):
                    client.retrieve(response, root / "local")
                self.assertTrue(first.is_dir())
                self.assertEqual(len(list((root / "local").iterdir())), 3)

    def test_retrieval_failure_preserves_nonzero_cli_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "fixture.jpg"
            image.write_bytes(b"synthetic")
            response = {"envelope": {"run_id": "a" * 32, "cli_exit_code": 1, "status": "partial",
                                     "source": {"source_sha256": client.digest(image)}}}
            with patch.object(client, "result_root", return_value=Path(temporary)), \
                    patch.object(client.uuid, "uuid4", return_value=SimpleNamespace(hex="a" * 32)), \
                    patch.object(client, "rpc", return_value=response), \
                    patch.object(client, "retrieve", side_effect=OSError("transfer unavailable")):
                value, code = client.demo_run(image, "never", time.monotonic())
            self.assertEqual(code, 1)
            self.assertEqual(value["cli_status"], "partial")
            self.assertEqual(value["status"], "failed")

    def test_timer_finalization_preserves_failure_code_and_clean_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary, timer = root / "summary.json", root / "timer.txt"
            summary.write_text(json.dumps({"status": "partial", "cli_exit_code": 1, "local_results": str(root)}))
            timer.write_text("real 12.34\nuser 0.10\nsys 0.05\n")
            with patch.object(client, "write_private_json", side_effect=OSError("disk unavailable")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                code = client.finish_timing([str(summary), str(timer), "1", "1"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "failed")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = client.finish_timing([str(summary), str(root / "missing"), "1", "1"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "failed")

    def test_actual_shell_failed_run_has_clean_json_and_external_wall_time(self):
        process = subprocess.run([str(ROOT / "scripts/demo-ec2.sh"), "run", "--image", "/missing.jpg", "--json"],
                                 capture_output=True, text=True)
        value = json.loads(process.stdout)
        self.assertEqual(process.returncode, 2)
        self.assertEqual(value["status"], "failed")
        self.assertIn("end_to_end_seconds", value)
        self.assertFalse(value["timing_target_met_provisionally"])


class RequestTimingsTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "Opt-in demo timings target macOS/Linux")
    def test_stream_and_logprob_response_are_unchanged_with_opt_in(self):
        chunks = [{"message": {"content": '{"unit":"4"}'}, "logprobs": [{"token": "4", "logprob": -0.1}]},
                  {"done": True, "eval_count": 4, "eval_duration": 123, "load_duration": 0}]
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "timings.jsonl"
            fake = SimpleNamespace(chat=lambda **kwargs: iter(chunks), close=lambda: None)
            with patch.dict(os.environ, {"UMA_BCT_DEMO_TIMINGS_FILE": str(destination)}), \
                    patch.object(OllamaVisionModel, "client", return_value=fake):
                response = OllamaVisionModel().query(Path("fixture.png"), "secret text", {}, logprobs=True)
            self.assertEqual(response.value, {"unit": "4"})
            self.assertEqual(response.token_logprobs, (("4", -0.1),))
            value = json.loads(destination.read_text())
            self.assertTrue(value["success"])
            self.assertEqual(value["ollama_nanoseconds_and_counts"]["eval_duration"], 123)
            self.assertNotIn("secret", destination.read_text())
            self.assertNotIn("unit", destination.read_text())

    def test_disabled_diagnostics_have_no_file_io(self):
        with patch.dict(os.environ, {}, clear=True), patch("os.open") as opened:
            timings = RequestTimings("detection")
            self.assertEqual(list(timings.stream([{"eval_count": 3}])), [{"eval_count": 3}])
            timings.finish(True)
            opened.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "Unix FIFOs and symlinks")
    def test_fifo_and_symlink_do_not_block_or_change_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            target = root / "target"
            target.write_text("retained")
            link = root / "link"
            link.symlink_to(target)
            for destination in (fifo, link):
                with patch.dict(os.environ, {"UMA_BCT_DEMO_TIMINGS_FILE": str(destination)}), \
                        contextlib.redirect_stderr(io.StringIO()) as output:
                    RequestTimings("detection").finish(True)
                self.assertIn("inference outcome is unchanged", output.getvalue())
            self.assertEqual(target.read_text(), "retained")


class BenchmarkReportTest(unittest.TestCase):
    def test_parallel_request_spans_are_distinct_from_cumulative_wall_cost(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "Request Timings.jsonl").write_text("\n".join(json.dumps(row) for row in (
                {"request": "transcription", "wall_seconds": 3, "started_monotonic_seconds": 10,
                 "finished_monotonic_seconds": 13},
                {"request": "transcription", "wall_seconds": 4, "started_monotonic_seconds": 10,
                 "finished_monotonic_seconds": 14},
            )))
            value = benchmark.request_totals(root)
            self.assertEqual(value["wall_seconds_by_request"]["transcription"], 7)
            self.assertEqual(value["request_span_seconds_by_kind"]["transcription"], 4)
            (root / "Request Timings.jsonl").write_text(json.dumps({"request": "transcription", "wall_seconds": 3}))
            self.assertIsNone(benchmark.request_totals(root)["request_span_seconds_by_kind"]["transcription"])

    def test_every_duration_failures_and_physical_count_are_preserved(self):
        runs = [{"status": "complete", "cli_exit_code": 0, "cli_elapsed_seconds": duration,
                 "physical_box_records": 2, "catalogue_rows": 3} for duration in (3, 4, 8)]
        value = benchmark.report(runs)
        self.assertEqual(value["cli_execution"]["every_duration_seconds"], [3, 4, 8])
        self.assertEqual(value["cli_execution"]["median_seconds"], 4)
        self.assertEqual(value["cli_execution"]["slowest_seconds"], 8)
        self.assertFalse(value["completion_target"]["verified_correct_configuration"])
        self.assertFalse(value["completion_target"]["met_for_all_measured_runs_provisionally"])
        runs[2].update(status="partial", cli_elapsed_seconds=2)
        self.assertFalse(benchmark.report(runs)["completion_target"]["met_for_all_measured_runs_provisionally"])

    def test_full_mac_command_is_the_ec2_acceptance_measure(self):
        value = benchmark.report([{"status": "complete", "cli_exit_code": 0, "cli_elapsed_seconds": 3,
                                   "full_mac_command_seconds": 6}])
        self.assertEqual(value["completion_target"]["measurement"], "full_mac_command_seconds")
        self.assertFalse(value["completion_target"]["met_for_all_measured_runs_provisionally"])


if __name__ == "__main__":
    unittest.main()
