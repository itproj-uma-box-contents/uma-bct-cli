#!/usr/bin/env python3
"""Private, fresh local/EC2 benchmarks and a feasibility report. No ground-truth claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from types import SimpleNamespace

MODEL = "qwen3-vl:8b-instruct-q4_K_M"
MODEL_DIGEST = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
ROOT = Path(__file__).resolve().parents[1]


def private_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(ROOT) and not any(resolved.is_relative_to(ROOT / name)
                                               for name in ("add_repository_photos", "Archive box photos", "outputs")):
        raise ValueError("Keep benchmarks outside Git/build contexts or inside an ignored private-photo/outputs directory.")
    resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    if resolved.stat().st_mode & 0o077:
        raise ValueError("Benchmark results directory must have private permissions (chmod 700).")
    return resolved


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def local_api(endpoint: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request("http://127.0.0.1:11434/api/" + endpoint,
                                     data=json.dumps(payload).encode() if payload is not None else None,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=900) as response:
        return json.load(response)


def local_gpu() -> dict:
    running = next((model for model in local_api("ps").get("models", []) if model.get("name") == MODEL), None)
    environment = {**os.environ, "OLLAMA_HOST": "http://127.0.0.1:11434"}
    display = subprocess.check_output(["ollama", "ps"], env=environment, text=True)
    if (not running or running.get("digest") != MODEL_DIGEST or running.get("context_length") != 16384
            or not any(MODEL in line and "100% GPU" in line for line in display.splitlines())):
        raise RuntimeError("Local model is not fully GPU-resident at production context; CPU results are not a GPU baseline.")
    return {"running_model": running, "ollama_ps": display}


def reheating(directory: Path) -> dict:
    before = local_api("ps")
    model = next((item for item in local_api("tags").get("models", []) if item.get("name") == MODEL), None)
    version = local_api("version")
    if version.get("version") != "0.32.9" or not model or model.get("digest") != MODEL_DIGEST:
        raise ValueError("Prepare the pinned Ollama version and exact model first; this benchmark never downloads them.")
    started = time.monotonic()
    response = local_api("chat", {"model": MODEL, "messages": [{"role": "user", "content": "Reply briefly."}],
                                  "think": False, "stream": False, "keep_alive": -1,
                                  "options": {"num_ctx": 16384, "temperature": 0, "num_predict": 1}})
    value = {"probe_wall_seconds": time.monotonic() - started, "before": before, "gpu_after": local_gpu(),
             "model": model, "ollama": version,
             "ollama_nanoseconds_and_counts": {key: response.get(key) for key in
                                               ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
                                                "prompt_eval_count", "eval_count")},
             "note": "Preparation probe, not image inference. Installed-model download time is not measured."}
    save(directory / "Cold Preparation.json", value)
    return value


def request_totals(root: Path) -> dict:
    files = list(root.rglob("Request Timings.jsonl"))
    requests = [json.loads(line) for file in files for line in file.read_text().splitlines() if line]
    return {"requests": requests,
            "wall_seconds_by_request": {kind: sum(row["wall_seconds"] for row in requests if row["request"] == kind)
                                        for kind in ("preflight", "detection", "transcription")}}


def statistics_report(runs: list[dict], field: str) -> dict:
    durations = [run[field] for run in runs if isinstance(run.get(field), (int, float))]
    return {"every_duration_seconds": durations, "median_seconds": statistics.median(durations) if durations else None,
            "slowest_seconds": max(durations) if durations else None, "fastest_seconds": min(durations) if durations else None}


def report(runs: list[dict], preparation: dict | None = None) -> dict:
    completed = all(run.get("status") == "complete" and run.get("cli_exit_code") == 0
                    and run.get("wrapper_exit_code", 0) == 0 for run in runs)
    field = "full_mac_command_seconds" if any("full_mac_command_seconds" in run for run in runs) else "cli_elapsed_seconds"
    durations = [run[field] for run in runs if isinstance(run.get(field), (int, float))]
    return {"schema_version": 1, "runs": runs, "preparation": preparation,
            "cli_execution": statistics_report(runs, "cli_elapsed_seconds"),
            "full_mac_command": statistics_report(runs, "full_mac_command_seconds"),
            "accuracy": "unreviewed; historical outputs are not ground truth",
            "completion_target": {"seconds": 5, "measurement": field,
                                  "minimum_trials": 5, "measured_trials": len(runs),
                                  "met_for_all_measured_runs_provisionally": len(runs) >= 5 and completed
                                  and len(durations) == len(runs) and max(durations) <= 5,
                                  "verified_correct_configuration": False}}


def application_identity() -> dict:
    files = [ROOT / "pyproject.toml", ROOT / "uv.lock", *sorted((ROOT / "src/uma_bct").rglob("*.py"))]
    fingerprints = {}
    for path in files:
        with path.open("rb") as source:
            fingerprints[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(source, "sha256").hexdigest()
    return {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
            "source_sha256": hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest(),
            "source_files": fingerprints}


def benchmark(args: argparse.Namespace) -> dict:
    image = args.image.expanduser()
    if not image.is_absolute() or not image.is_file() or image.is_symlink() or image.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError("Select exactly one approved absolute JPEG path.")
    with image.open("rb") as source:
        fingerprint = hashlib.file_digest(source, "sha256").hexdigest()
    destination = private_root(args.results_directory) / f"benchmark-{uuid.uuid4().hex}"
    destination.mkdir(mode=0o700)
    application = application_identity()
    preparation = reheating(destination) if args.command == "local" else None
    environment = dict(os.environ)
    if args.command == "local":
        environment.update(OLLAMA_HOST="http://127.0.0.1:11434", OLLAMA_NO_CLOUD="1")
        environment.pop("UMA_BCT_ALLOW_REMOTE_OLLAMA", None)
        environment.pop("UMA_BCT_ALLOW_CONTAINER_OLLAMA", None)
    environment.pop("UMA_BCT_DEMO_TIMINGS_FILE", None)
    runs = []
    for index in range(args.runs):
        trial = destination / f"trial-{index + 1:02}"
        trial.mkdir(mode=0o700)
        print(f"Fresh {args.command} benchmark {index + 1}/{args.runs}; verification={args.verification}", file=sys.stderr)
        if args.command == "local":
            gpu_before = local_gpu()
            command = [sys.executable, "-m", "uma_bct", "--json", "--path-to-inputs", str(image),
                       "--results-directory", str(trial), "--reprocess", "--verification", args.verification, "--model", MODEL]
            if args.timings:
                environment["UMA_BCT_DEMO_TIMINGS_FILE"] = str(trial / "Request Timings.jsonl")
        else:
            gpu_before = None
            command = [str(ROOT / "scripts/demo-ec2.sh"), "run", "--image", str(image), "--json",
                       "--verification", args.verification, "--backend", args.backend,
                       *(["--timings"] if args.timings else [])]
        started = time.monotonic()
        interrupted = False
        with (trial / "CLI Summary.json").open("w") as output:
            try:
                process = subprocess.run(command, env=environment, stdout=output, stderr=sys.stderr)
            except KeyboardInterrupt:
                interrupted = True
                process = SimpleNamespace(returncode=130)
        elapsed = time.monotonic() - started
        try:
            summary = json.loads((trial / "CLI Summary.json").read_text())
        except ValueError:
            summary = {"status": "failed", "error": "No valid JSON summary."}
        if interrupted:
            summary["status"] = "interrupted"
        value = {"status": summary.get("status", "failed"),
                 "cli_exit_code": process.returncode if args.command == "local" else summary.get("cli_exit_code"),
                 "wrapper_exit_code": process.returncode if args.command == "ec2" else None,
                 "source_sha256": fingerprint, "source_image": str(image), "application": application,
                 "settings": {"model": MODEL, "model_digest": MODEL_DIGEST, "num_ctx": 16384, "temperature": 0,
                              "think": False, "verification": args.verification, "diagnostics_enabled": args.timings},
                 "summary": summary, "accuracy": "unreviewed"}
        if args.command == "local":
            value.pop("wrapper_exit_code")
            value.update(cli_elapsed_seconds=elapsed, gpu_before=gpu_before)
            try:
                value["gpu_after"] = local_gpu()
            except Exception as error:
                value.update(status="failed", gpu_error=str(error))
            details_paths = list(trial.glob("Batch */Run Details.json"))
            if details_paths:
                details = json.loads(details_paths[0].read_text())
                value["physical_box_records"] = sum(len(item.get("boxes", [])) for item in details.get("images", []))
                value["catalogue_rows"] = details.get("catalogue_records")
                value["fallbacks"] = [item.get("fallback_path", []) for item in details.get("images", [])]
            value["request_timings"] = request_totals(trial)
        else:
            value.update(full_mac_command_seconds=elapsed, cli_elapsed_seconds=summary.get("cli_elapsed_seconds"),
                         physical_box_records=summary.get("physical_box_records"), catalogue_rows=summary.get("catalogue_rows"),
                         fallbacks=summary.get("fallbacks"), local_results=summary.get("local_results"))
            value["settings"]["backend"] = summary.get("backend", args.backend)
            value["application"] = summary.get("application", application)
            if summary.get("local_results"):
                value["request_timings"] = request_totals(Path(summary["local_results"]))
        runs.append(value)
        save(trial / "Benchmark Run.json", value)
        save(destination / "Feasibility Report.json", report(runs, preparation))
        print(f"Trial {index + 1}: {value['status']}, {elapsed:.3f}s; results {trial}", file=sys.stderr)
        if interrupted:
            break
    value = report(runs, preparation)
    value["report_location"] = str(destination / "Feasibility Report.json")
    save(destination / "Feasibility Report.json", value)
    return value


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("local", "ec2"):
        command = commands.add_parser(name)
        command.add_argument("--image", type=Path, required=True)
        command.add_argument("--results-directory", type=Path, required=True)
        command.add_argument("--runs", type=int, default=5)
        command.add_argument("--verification", choices=("never", "always"), default="never")
        command.add_argument("--timings", action="store_true")
        if name == "ec2":
            command.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    try:
        value = benchmark(args)
        print(json.dumps(value))
        if any(run["status"] == "interrupted" for run in value["runs"]):
            return 130
        return 0 if all(run["status"] == "complete" for run in value["runs"]) else 1
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "results_root": str(args.results_directory),
                          "error": "Benchmark stopped; retain completed reports and interrupted checkpoints. No completion claim."}))
        return 130
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
