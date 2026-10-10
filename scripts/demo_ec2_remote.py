#!/usr/bin/env python3
"""Private SSH control helper. AWS resources are managed only by the presenter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path, PurePosixPath

BASE = Path("/home/ubuntu")
INCOMING = BASE / "demo_incoming"
APPLICATION = BASE / "demo_app"
INPUTS = BASE / "demo_images"
RESULTS = BASE / "demo_results"
HEALTH = "/usr/local/bin/uma-bct-demo-health"
PARALLEL_HEALTH = "/usr/local/bin/uma-bct-demo-parallel"
MODEL = "qwen3-vl:8b-instruct-q4_K_M"
MODEL_DIGEST = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
SETTINGS = {"model": MODEL, "num_ctx": 16384, "temperature": 0, "think": False,
            "transcription_logprobs": True, "parallel_requests": 1}
SOURCE_FILES = {"Dockerfile", ".dockerignore", "README.md", "pyproject.toml", "uv.lock",
                "scripts/demo_ec2_remote.py"}


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Unsafe relative artifact path.")
    return Path(*path.parts)


def token(value: str, *, length: int = 32) -> str:
    if not isinstance(value, str) or not re.fullmatch(rf"[a-f0-9]{{{length}}}", value):
        raise ValueError("Invalid generated identifier.")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def health(action: str = "status", *, backend: str = "gpu", transcription_workers: int = 1) -> dict:
    if backend not in {"gpu", "cpu"}:
        raise ValueError("Backend must be gpu or cpu.")
    if transcription_workers not in {1, 2} or (backend == "cpu" and transcription_workers != 1):
        raise ValueError("Two independent runners require GPU mode.")
    args = ([PARALLEL_HEALTH, "status"] if transcription_workers == 2 else
            [HEALTH, action, *(["--backend", "cpu"] if backend == "cpu" else [])])
    process = subprocess.run(args, capture_output=True, text=True,
                             timeout=960 if action == "warm" else 90)
    value = json.loads(process.stdout)
    if process.returncode:
        raise RuntimeError(value.get("error", "GPU/model readiness failed."))
    return value


def deployment() -> dict:
    value = json.loads((APPLICATION / "current.json").read_text())
    source_hash = token(value["source_sha256"], length=64)
    if value["image"] != f"uma-bct-demo:{source_hash}":
        raise RuntimeError("Invalid application image identity; run prepare again.")
    inspected = subprocess.check_output(["docker", "image", "inspect", value["image"]], text=True)
    labels = json.loads(inspected)[0]["Config"].get("Labels") or {}
    if labels.get("com.uma-bct-demo.source-sha256") != source_hash:
        raise RuntimeError("Prepared application image is missing or changed; run prepare again.")
    return value


def staged(request: dict) -> tuple[Path, dict]:
    fingerprint = token(request["source_sha256"], length=64)
    metadata = json.loads((INPUTS / fingerprint / "Source.json").read_text())
    if metadata.get("source_sha256") != fingerprint or metadata.get("staged_sha256") != fingerprint:
        raise RuntimeError("Staged source association changed; run prepare --image again.")
    name = metadata["filename"]
    if Path(name).name != name or not name.lower().endswith((".jpg", ".jpeg")):
        raise ValueError("Staged demo input must be one JPEG filename.")
    path = INPUTS / fingerprint / name
    if path.is_symlink() or not path.is_file() or digest(path) != fingerprint:
        raise RuntimeError("The staged photograph is missing or changed. Run prepare --image for this approved input.")
    return path, metadata


def source_allowed(name: str) -> bool:
    return name in SOURCE_FILES or (name.startswith("src/uma_bct/") and name.endswith(".py"))


def install_source(archive: Path) -> dict:
    staging = APPLICATION / f".deploy-{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700, parents=True)
    try:
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            seen = set()
            if sum(member.size for member in members) > 32 * 1024 * 1024:
                raise ValueError("Application source bundle exceeds 32 MiB.")
            for member in members:
                relative = safe_relative(member.name)
                if (not member.isfile() or member.name in seen
                        or (member.name != "Deployment.json" and not source_allowed(member.name))):
                    raise ValueError("Deployment accepts only allowlisted regular source files.")
                seen.add(member.name)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                with bundle.extractfile(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
                destination.chmod(0o600)
        metadata = json.loads((staging / "Deployment.json").read_text())
        expected = metadata["files"]
        if set(expected) != seen - {"Deployment.json"} or not SOURCE_FILES.issubset(expected):
            raise ValueError("Deployment file manifest is incomplete.")
        for name, fingerprint in expected.items():
            if digest(staging / safe_relative(name)) != fingerprint:
                raise ValueError("Application source checksum mismatch.")
        source_hash = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
        if source_hash != metadata["source_sha256"]:
            raise ValueError("Application bundle checksum mismatch.")
        image = f"uma-bct-demo:{source_hash}"
        subprocess.run(["docker", "build", "--target", "production", "--label",
                        f"com.uma-bct-demo.source-sha256={source_hash}", "--label",
                        f"org.opencontainers.image.revision={metadata['revision']}", "-t", image, str(staging)],
                       check=True, stdout=sys.stderr, stderr=sys.stderr)
        installed = APPLICATION / source_hash
        if not installed.exists():
            staging.rename(installed)
        control = installed / "scripts/demo_ec2_remote.py"
        temporary_control = BASE / f".demo-control-{uuid.uuid4().hex}"
        shutil.copyfile(control, temporary_control)
        temporary_control.chmod(0o600)
        os.replace(temporary_control, BASE / "demo_control.py")
        result = {**metadata, "image": image}
        write_json(APPLICATION / "current.json", result)
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def prepare(request: dict) -> dict:
    # The Mac has explicitly reheated before transfer; live runs never do so.
    ready = health(backend=request.get("backend", "gpu"), transcription_workers=request.get("transcription_workers", 1))
    upload_id = token(request["upload_id"])
    image_name = request["filename"]
    if image_name is None:
        app = install_source(INCOMING / f"{upload_id}.tar")
        for suffix in (".tar", ".py", "-health.py", "-parallel.py", "-parallel.yaml"):
            (INCOMING / f"{upload_id}{suffix}").unlink(missing_ok=True)
        return {"status": "prepared", "application": app, "readiness": ready, "image_staged": False}
    if Path(image_name).name != image_name or image_name.startswith(".") or not image_name.lower().endswith((".jpg", ".jpeg")):
        raise ValueError("Select a JPEG with a plain, non-hidden filename.")
    fingerprint = token(request["source_sha256"], length=64)
    uploaded = INCOMING / f"{upload_id}.jpg"
    if uploaded.is_symlink() or digest(uploaded) != fingerprint:
        raise ValueError("Uploaded photograph checksum mismatch.")
    app = install_source(INCOMING / f"{upload_id}.tar")
    image_directory = INPUTS / fingerprint
    image_directory.mkdir(mode=0o700, exist_ok=True)
    target = image_directory / image_name
    if target.exists() and digest(target) != fingerprint:
        raise ValueError("Staged photograph conflicts with earlier preparation.")
    if not target.exists():
        os.replace(uploaded, target)
        target.chmod(0o600)
    write_json(image_directory / "Source.json", {"filename": image_name, "source_sha256": fingerprint,
               "staged_sha256": fingerprint, "metadata_policy": "unchanged original JPEG", "source_path": request["source_path"]})
    for suffix in (".jpg", ".tar", ".py"):
        (INCOMING / f"{upload_id}{suffix}").unlink(missing_ok=True)
    for suffix in ("-health.py", "-parallel.py", "-parallel.yaml"):
        (INCOMING / f"{upload_id}{suffix}").unlink(missing_ok=True)
    return {"status": "prepared", "source_sha256": fingerprint, "application": app, "readiness": ready}


def transfer_manifest(directory: Path) -> dict:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Result directory contains a symbolic link; retrieval refused.")
        if path.is_file() and path.name != "Transfer Manifest.json":
            files[path.relative_to(directory).as_posix()] = {"sha256": digest(path), "size": path.stat().st_size}
        elif not path.is_file() and not path.is_dir():
            raise RuntimeError("Result directory contains a special file; retrieval refused.")
    write_json(directory / "Transfer Manifest.json", {"files": files})
    return {"files": files, "manifest_sha256": digest(directory / "Transfer Manifest.json")}


def validate_summary(summary: dict, directory: Path, exit_code: int) -> Path | None:
    batch = summary.get("batch_directory")
    if batch is None:
        if exit_code == 0:
            raise RuntimeError("Successful CLI output has no published batch.")
        return None
    relative = safe_relative(str(PurePosixPath(batch).relative_to("/data/outputs")))
    batch_path = directory / relative
    if exit_code == 0:
        if (summary.get("status") != "complete" or summary.get("images_reused") != 0
                or summary.get("images_processed_this_start") != 1 or summary.get("images_total") != 1):
            raise RuntimeError("CLI did not complete one fresh photograph.")
        for name in ("Catalogue.csv", "Review.html", "Run Details.json"):
            if not (batch_path / name).is_file():
                raise RuntimeError(f"Completed CLI publication is missing {name}.")
    return batch_path


def run(request: dict) -> dict:
    run_id = token(request["run_id"])
    mode = request.get("verification", "never")
    if mode not in {"never", "always"}:
        raise ValueError("Demo verification must be never or always.")
    backend = request.get("backend", "gpu")
    workers = request.get("transcription_workers", 1)
    before = health(backend=backend, transcription_workers=workers)
    app = deployment()
    image, source = staged(request)
    directory = RESULTS / run_id
    directory.mkdir(mode=0o700)  # Never replace or resume an earlier demo.
    args = ["docker", "run", "--rm", "--name", f"uma-bct-run-{run_id}", "--network", "uma-bct-demo_default",
            "--user", f"{os.getuid()}:{os.getgid()}", "--read-only", "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true", "--init", "--tmpfs", "/tmp:size=256m",
            "-e", "HOME=/tmp", "-e", "PYTHONUNBUFFERED=1", "-e", "OLLAMA_HOST=http://ollama:11434",
            "-e", "OLLAMA_NO_CLOUD=1", "-e", "UMA_BCT_ALLOW_CONTAINER_OLLAMA=1",
            "--mount", f"type=bind,source={INPUTS},target=/data/inputs,readonly",
            "--mount", f"type=bind,source={directory},target=/data/outputs", app["image"],
            "--json", "--path-to-inputs", f"/data/inputs/{image.relative_to(INPUTS).as_posix()}",
            # Managed locking needs a writable parent on the same filesystem.
            "--results-directory", "/data/outputs/Batches", "--reprocess", "--verification", mode, "--model", MODEL]
    if request.get("timings") is True:
        args[2:2] = ["-e", "UMA_BCT_DEMO_TIMINGS_FILE=/data/outputs/Request Timings.jsonl"]
    if workers == 2:
        args[2:2] = ["-e", "UMA_BCT_DEMO_TRANSCRIPTION_WORKERS=2"]
    print(f"Running fresh EC2 inference on {backend.upper()}; CLI progress follows.", file=sys.stderr, flush=True)
    started = time.monotonic()
    with (directory / "CLI stdout.json").open("w") as output:
        process = subprocess.run(args, stdout=output, stderr=sys.stderr)
    elapsed = time.monotonic() - started  # Includes Docker and CLI/Python startup and publication.
    error = None
    summary = {}
    batch = None
    after = None
    try:
        summary = json.loads((directory / "CLI stdout.json").read_text())
        batch = validate_summary(summary, directory, process.returncode)
        after = health(backend=backend, transcription_workers=workers)
    except Exception as failure:
        error = str(failure)
    details = {}
    if batch and (batch / "Run Details.json").is_file():
        details = json.loads((batch / "Run Details.json").read_text())
    records = [box for image_result in details.get("images", []) for box in image_result.get("boxes", [])]
    envelope = {"schema_version": 1, "run_id": run_id, "status": summary.get("status", "failed"),
                "cli_exit_code": process.returncode, "cli": summary, "cli_elapsed_seconds": elapsed,
                "application": app, "source": source, "backend": backend,
                "settings": {**SETTINGS, "verification": mode, "backend": backend,
                             "transcription_workers": workers, "independent_runners": workers},
                "physical_box_records": len(records), "catalogue_rows": summary.get("catalogue_records"),
                "accuracy": "unreviewed", "readiness_before": before, "readiness_after": after,
                "orchestration_error": error,
                "batch_relative_path": batch.relative_to(directory).as_posix() if batch else None,
                "fallbacks": [item.get("fallback_path", []) for item in details.get("images", [])]}
    write_json(directory / "EC2 Run.json", envelope)
    return {"envelope": envelope, "transfer": transfer_manifest(directory)}


def main() -> int:
    os.umask(0o077)
    try:
        if sys.platform != "linux":
            raise RuntimeError("The EC2 control helper requires Linux; use the Mac SSH wrapper instead.")
        import fcntl

        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise ValueError("Expected a JSON request object.")
        for directory in (INCOMING, APPLICATION, INPUTS, RESULTS):
            directory.mkdir(mode=0o700, exist_ok=True)
        # Serialise demo clients; an explicit run can use two independently served crop readers.
        with (BASE / ".uma-bct-demo.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            action = request.get("action")
            if action == "prepare":
                result = prepare(request)
            elif action == "status":
                result = {"status": "ready", "readiness": health(backend=request.get("backend", "gpu"),
                           transcription_workers=request.get("transcription_workers", 1)),
                          "application": deployment()}
            elif action == "run":
                result = run(request)
            elif action == "retrieve":
                run_id = token(request["run_id"])
                directory = RESULTS / run_id
                result = {"envelope": json.loads((directory / "EC2 Run.json").read_text()),
                          "transfer": transfer_manifest(directory)}
            else:
                raise ValueError("Unknown demo action.")
        print(json.dumps(result))
        return 0  # CLI exit code is retained in the envelope; SSH transport remains distinguishable.
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
