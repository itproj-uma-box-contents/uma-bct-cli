#!/usr/bin/env python3
"""Mac SSH demo client; stdlib only, no AWS API or infrastructure mutations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REMOTE_BASE = "/home/ubuntu"
SOURCE_FILES = ("Dockerfile", ".dockerignore", "README.md", "pyproject.toml", "uv.lock", "scripts/demo_ec2_remote.py",
                "scripts/demo_ec2_peer.py")


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def selected_image(value: Path) -> tuple[Path, str]:
    path = value.expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("--image must be an absolute path to one regular approved JPEG.")
    if path.suffix.lower() not in {".jpg", ".jpeg"} or path.name.startswith(".") or not path.name.isprintable():
        raise ValueError("The demo accepts one JPEG with a visible, printable filename.")
    return path, digest(path)


def connection() -> tuple[str, list[str]]:
    host = os.getenv("UMA_BCT_EC2_HOST", "")
    if not re.fullmatch(r"ubuntu@[A-Za-z0-9][A-Za-z0-9.-]*", host):
        raise ValueError("Set UMA_BCT_EC2_HOST=ubuntu@<current EC2 public DNS or IPv4>.")
    key = Path(os.getenv("UMA_BCT_EC2_KEY", "")).expanduser()
    if not key.is_absolute() or not key.is_file() or key.stat().st_mode & 0o077:
        raise ValueError("Set UMA_BCT_EC2_KEY to an absolute private-key path with permissions 600 or 400.")
    options = ["-i", str(key), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2"]
    return host, options


def ssh_json(remote_command: str, request: dict | None = None) -> dict:
    host, options = connection()
    process = subprocess.run(["ssh", "-T", *options, host, remote_command],
                             input=json.dumps(request) if request is not None else "",
                             stdout=subprocess.PIPE, text=True, stderr=sys.stderr)
    try:
        response = json.loads(process.stdout)
    except ValueError:
        raise RuntimeError("SSH returned no valid demo response. Check server state, current address, key, and trusted host key.") from None
    if process.returncode:
        raise RuntimeError(response.get("error", f"SSH/control exit code {process.returncode}."))
    return response


def rpc(request: dict) -> dict:
    return ssh_json("python3 /home/ubuntu/demo_control.py", request)


def require_cpu_status(value: dict) -> None:
    if value.get("readiness", {}).get("backend", {}).get("mode") != "cpu":
        raise RuntimeError("Server did not confirm explicit CPU mode. Install M7i User Data and prepare --backend cpu first.")


def copy(source: str, destination: str, *, recursive: bool = False) -> None:
    _host, options = connection()
    # OpenSSH's default SFTP transport retains host verification and handles spaces.
    subprocess.run(["scp", "-q", *(["-r"] if recursive else []), *options, source, destination],
                   check=True, stdout=sys.stderr, stderr=sys.stderr)


def source_bundle(destination: Path, *, root: Path = ROOT) -> dict:
    files = [root / name for name in SOURCE_FILES]
    files.extend(sorted((root / "src/uma_bct").rglob("*.py")))
    fingerprints = {}
    for path in files:
        if path.is_symlink() or not path.is_file() or "__pycache__" in path.parts:
            raise ValueError(f"Deployment requires regular source files: {path.name}")
        fingerprints[path.relative_to(root).as_posix()] = digest(path)
    source_hash = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    metadata = {"revision": revision, "working_tree_dirty": dirty, "source_sha256": source_hash, "files": fingerprints}
    with tarfile.open(destination, "w") as bundle:
        for path in files:
            bundle.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
        encoded = (json.dumps(metadata) + "\n").encode()
        member = tarfile.TarInfo("Deployment.json")
        member.size, member.mode = len(encoded), 0o600
        bundle.addfile(member, io.BytesIO(encoded))
    # A source file changed while packaging: do not deploy a mixed application.
    for path in files:
        if digest(path) != fingerprints[path.relative_to(root).as_posix()]:
            raise RuntimeError("Source changed during deployment packaging. Retry prepare.")
    return metadata


def prepare(image: Path | None, *, backend: str = "gpu", transcription_workers: int = 1, gpu_mps: bool = False) -> dict:
    if backend not in {"gpu", "cpu"}:
        raise ValueError("Backend must be gpu or cpu.")
    host, options = connection()
    if transcription_workers not in {1, 2, 4, 6, 8} or (backend == "cpu" and (transcription_workers != 1 or gpu_mps)):
        raise ValueError("Use one worker normally; independent runners require GPU demo mode.")
    chosen, fingerprint = selected_image(image) if image is not None else (None, None)
    print("Preparing the server, application, and selected input; this is outside live-run timing.", file=sys.stderr)
    # Verify infrastructure/reheat before any private transfer. A stopped or unfinished
    # server fails here with setup instructions; the wrapper does not start EC2.
    ready = ssh_json("/usr/local/bin/uma-bct-demo-health warm" + (" --backend cpu" if backend == "cpu" else ""))
    subprocess.run(["ssh", "-T", *options, host, "install -d -m 0700 /home/ubuntu/demo_incoming"], check=True,
                   stdout=sys.stderr, stderr=sys.stderr)
    upload_id = uuid.uuid4().hex
    # Source-only setup before any private photograph transfer. Configure/reheat
    # is explicit preparation, never hidden inside a timed run.
    for source, suffix in (("scripts/demo_ec2_health.py", "health.py"),
                           ("scripts/demo_ec2_parallel.py", "parallel.py"),
                           ("scripts/demo_ec2_mps.py", "mps.py"),
                           ("compose.ec2-demo.parallel.yaml", "parallel.yaml")):
        copy(str(ROOT / source), f"{host}:{REMOTE_BASE}/demo_incoming/{upload_id}-{suffix}")
    ssh_json(f"sudo python3 {REMOTE_BASE}/demo_incoming/{upload_id}-mps.py {'enable' if gpu_mps else 'disable'}")
    ssh_json(f"sudo python3 {REMOTE_BASE}/demo_incoming/{upload_id}-parallel.py configure "
             f"--workers {transcription_workers} --health-file {REMOTE_BASE}/demo_incoming/{upload_id}-health.py "
             f"--compose-file {REMOTE_BASE}/demo_incoming/{upload_id}-parallel.yaml")
    if transcription_workers > 1:
        ready = {**ssh_json("/usr/local/bin/uma-bct-demo-parallel status"), "primary_reheat": ready}
    with tempfile.TemporaryDirectory(prefix="uma-bct-demo-deploy-") as temporary:
        directory = Path(temporary)
        bundle = directory / "application.tar"
        source_bundle(bundle)
        copy(str(bundle), f"{host}:{REMOTE_BASE}/demo_incoming/{upload_id}.tar")
        copy(str(ROOT / "scripts/demo_ec2_remote.py"), f"{host}:{REMOTE_BASE}/demo_incoming/{upload_id}.py")
        if chosen:
            staged_copy = directory / "approved.jpg"
            shutil.copyfile(chosen, staged_copy)
            staged_copy.chmod(0o600)
            if digest(staged_copy) != fingerprint or digest(chosen) != fingerprint:
                raise RuntimeError("Selected photograph changed during preparation; retry with a stable original.")
            copy(str(staged_copy), f"{host}:{REMOTE_BASE}/demo_incoming/{upload_id}.jpg")
        response = ssh_json(f"python3 /home/ubuntu/demo_incoming/{upload_id}.py", {
            "action": "prepare", "upload_id": upload_id, "filename": chosen.name if chosen else None,
            "source_sha256": fingerprint, "source_path": str(chosen) if chosen else None, "backend": backend,
            "transcription_workers": transcription_workers,
        })
    response["initial_reheat"] = ready
    return response


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Unsafe path in result manifest.")
    return Path(*path.parts)


def validate_transfer(directory: Path, transfer: dict) -> None:
    manifest = directory / "Transfer Manifest.json"
    if manifest.is_symlink() or digest(manifest) != transfer["manifest_sha256"]:
        raise RuntimeError("Transferred result manifest is missing or changed.")
    expected = transfer["files"]
    if json.loads(manifest.read_text()).get("files") != expected:
        raise RuntimeError("Transferred file list differs from the server's authenticated manifest.")
    found = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError("Transferred results contain a link or special file.")
        if path.is_file() and path != manifest:
            found.add(path.relative_to(directory).as_posix())
    if found != set(expected):
        raise RuntimeError("Result retrieval is incomplete or contains unexpected files.")
    for name, metadata in expected.items():
        path = directory / safe_relative(name)
        if path.stat().st_size != metadata["size"] or digest(path) != metadata["sha256"]:
            raise RuntimeError(f"Result checksum mismatch: {name}")


def extract_result_archive(archive: Path, directory: Path, transfer: dict) -> None:
    """Extract only authenticated regular result files; never trust tar paths/types."""
    expected = transfer["files"]
    allowed = {*expected, "Transfer Manifest.json"}
    parents = {parent.as_posix() for name in allowed for parent in PurePosixPath(name).parents if parent.as_posix() != "."}
    seen = set()
    with tarfile.open(archive, "r:") as bundle:
        for member in bundle:
            name = member.name.removeprefix("./")
            if member.isdir() and name == ".":
                continue
            relative = safe_relative(name)
            if member.isdir() and name in parents:
                (directory / relative).mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not member.isfile() or name not in allowed or name in seen or member.sparse is not None:
                raise RuntimeError("Result archive contains an unexpected file, duplicate, link or special entry.")
            maximum = expected[name]["size"] if name in expected else 1024 * 1024
            if member.size < 0 or (member.size != maximum if name in expected else member.size > maximum):
                raise RuntimeError("Result archive file size differs from its authenticated manifest.")
            seen.add(name)
            target = directory / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=64 * 1024)
            target.chmod(0o600)
    if seen != allowed:
        raise RuntimeError("Result archive is incomplete.")


def retrieve(response: dict, root: Path, *, archive_results: bool = False, local_archive: Path | None = None) -> Path:
    run_id = response["envelope"]["run_id"]
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("Invalid remote run identifier.")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o077:
        raise ValueError("Local demo result root must be a private directory (chmod 700).")
    # A new enclosing directory is exclusively reserved even when retrieving an old
    # run again. Earlier results are never replaced, including interrupted copies.
    reserved = local_archive.parent if local_archive else Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=root))
    staging = reserved / "Incoming"
    staging.mkdir(mode=0o700)
    host, options = connection()
    try:
        if archive_results or local_archive:
            archive = local_archive or reserved / "Transfer.tar"
            if local_archive is None:
                with archive.open("xb") as output:
                    archive.chmod(0o600)
                    # The only interpolated value is the validated hexadecimal run ID.
                    command = f"tar --directory={REMOTE_BASE}/demo_results/{run_id} --create --file=- ."
                    subprocess.run(["ssh", "-T", *options, host, command], check=True,
                                   stdout=output, stderr=sys.stderr)
            extract_result_archive(archive, staging, response["transfer"])
        else:
            copy(f"{host}:{REMOTE_BASE}/demo_results/{run_id}/.", str(staging), recursive=True)
        validate_transfer(staging, response["transfer"])
        if archive_results or local_archive:
            archive.unlink()
        final = reserved / "Results"
        staging.rename(final)
        return final
    except Exception as error:
        raise RuntimeError(f"Retrieval failed; partial copy retained in {staging}. Retry retrieve --run-id {run_id}. {error}") from error


class StreamTransferError(RuntimeError):
    """Retain an authenticated CLI outcome when its subsequent transfer fails."""

    def __init__(self, message: str, response: dict):
        super().__init__(message)
        self.response = response if isinstance(response, dict) else {}


def stream_rpc(request: dict, root: Path) -> tuple[dict, Path, float, float]:
    """Receive a bounded JSON header and complete archive through one private SSH session."""
    host, options = connection()
    reserved = Path(tempfile.mkdtemp(prefix=f"{request['run_id']}-", dir=root))
    archive = reserved / "Transfer.tar"
    response = {}
    process = subprocess.Popen(["ssh", "-T", *options, host, "python3 /home/ubuntu/demo_control.py"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)
    started = time.monotonic()
    try:
        process.stdin.write(json.dumps(request).encode())
        process.stdin.close()
        header = process.stdout.readline(1024 * 1024 + 1)
        header_seconds = time.monotonic() - started
        if len(header) > 1024 * 1024 or not header.endswith(b"\n"):
            raise RuntimeError("Server returned no bounded stream header. Update/prepare the server first.")
        parsed = json.loads(header)
        if not isinstance(parsed, dict) or parsed.get("protocol") != "uma-bct-stream-v1":
            raise RuntimeError(parsed.get("error", "Server does not support streaming; run prepare first.")
                               if isinstance(parsed, dict) else "Invalid stream response.")
        envelope = parsed["envelope"]
        if (envelope.get("run_id") != request["run_id"]
                or envelope.get("source", {}).get("source_sha256") != request["source_sha256"]
                or envelope.get("backend") != request["backend"]
                or envelope.get("settings", {}).get("transcription_workers") != request["transcription_workers"]):
            raise RuntimeError("Stream outcome does not match the selected source, backend or runner count.")
        code = envelope.get("cli_exit_code")
        if type(code) is not int or not 0 <= code <= 255:
            raise RuntimeError("Invalid CLI exit code in stream outcome.")
        # Only a matched, validated outcome can supply a preserved CLI exit code.
        response = parsed
        files = response["transfer"]["files"]
        if not isinstance(files, dict):
            raise RuntimeError("Invalid stream file manifest.")
        maximum = 2 * 1024 * 1024
        for name, metadata in files.items():
            safe_relative(name)
            size = metadata["size"]
            if type(size) is not int or size < 0:
                raise RuntimeError("Invalid stream manifest size.")
            maximum += size + 8192  # Tar path/directory headers, plus manifest allowance.
        transfer_started = time.monotonic()
        with archive.open("xb") as output:
            archive.chmod(0o600)
            received = 0
            while block := process.stdout.read(64 * 1024):
                received += len(block)
                if received > maximum:
                    raise RuntimeError("Stream archive exceeds its authenticated file manifest.")
                output.write(block)
        if process.wait() != 0:
            raise RuntimeError("SSH stream failed; completed server results can be retrieved by run ID.")
        return response, archive, header_seconds, time.monotonic() - transfer_started
    except Exception as error:
        raise StreamTransferError(f"Stream failed; partial transfer retained in {reserved}. {error}", response) from error
    finally:
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def result_root() -> Path:
    path = Path(os.getenv("UMA_BCT_DEMO_RESULTS", str(Path.home() / "UMA-BCT-Demo/Results"))).expanduser().absolute()
    if path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("UMA_BCT_DEMO_RESULTS must be outside the checkout and its Docker build context.")
    return path


def demo_run(image: Path | None, verification: str, started: float, *, run_id: str | None = None,
             timings: bool = False, backend: str = "gpu", transcription_workers: int = 1,
             archive_results: bool = False, stream_results: bool = False) -> tuple[dict, int]:
    if backend not in {"gpu", "cpu"}:
        raise ValueError("Backend must be gpu or cpu.")
    if transcription_workers not in {1, 2, 4, 6, 8} or (backend == "cpu" and transcription_workers != 1):
        raise ValueError("Two independent transcription runners require GPU demo mode.")
    identifier = run_id or uuid.uuid4().hex
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("--run-id must be the recorded 32-character run identifier.")
    chosen, fingerprint = selected_image(image) if image is not None else (None, None)
    if run_id is None and chosen is None:
        raise ValueError("run requires --image.")
    destination = result_root()
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    if destination.stat().st_mode & 0o077 or not os.access(destination, os.W_OK):
        raise ValueError("Local result root must be private and writable; fix it before running inference.")
    request = ({"action": "retrieve", "run_id": identifier} if run_id else
               {"action": "run", "run_id": identifier, "source_sha256": fingerprint, "verification": verification,
                "timings": timings, "backend": backend, "transcription_workers": transcription_workers})
    print(f"Demo run ID: {identifier}", file=sys.stderr, flush=True)
    rpc_started = time.monotonic()
    local_archive = None
    received_seconds = 0.0
    try:
        if stream_results and not run_id:
            request["action"] = "run-stream-v1"
            response, local_archive, rpc_seconds, received_seconds = stream_rpc(request, destination)
        else:
            if not run_id and backend == "cpu":
                # Old G6 controllers ignore unknown RPC fields. Confirm CPU capability
                # before asking such a controller to perform any fresh inference.
                require_cpu_status(rpc({"action": "status", "backend": "cpu"}))
            if not run_id and transcription_workers > 1:
                prepared = rpc({"action": "status", "backend": "gpu", "transcription_workers": transcription_workers})
                if prepared.get("readiness", {}).get("settings", {}).get("transcription_workers") != transcription_workers:
                    raise RuntimeError("Server did not confirm two independent runners or the selected count. Run prepare first.")
            response = rpc(request)
            rpc_seconds = time.monotonic() - rpc_started
    except Exception as error:
        failed_envelope = error.response.get("envelope", {}) if isinstance(error, StreamTransferError) else {}
        failed_code = failed_envelope.get("cli_exit_code")
        return {"status": "failed", "run_id": identifier, "cli_exit_code": failed_code,
                "cli_status": failed_envelope.get("status"),
                "error": str(error), "elapsed_seconds": time.monotonic() - started}, (
                    failed_code if isinstance(failed_code, int) and 1 <= failed_code <= 255 else 2)
    envelope = response["envelope"]
    cli_code = envelope["cli_exit_code"]
    if not isinstance(cli_code, int) or not 0 <= cli_code <= 255:
        raise ValueError("Remote CLI exit code is invalid.")
    if envelope.get("run_id") != identifier or (not run_id and envelope.get("source", {}).get("source_sha256") != fingerprint):
        return {"status": "failed", "run_id": identifier, "cli_exit_code": cli_code,
                "error": "Remote run/source identity does not match the selected input."}, cli_code or 2
    if not run_id and envelope.get("backend", "gpu") != backend:
        return {"status": "failed", "run_id": identifier, "cli_exit_code": cli_code,
                "error": "Remote backend differs from the explicitly selected backend."}, cli_code or 2
    if not run_id and envelope.get("settings", {}).get("transcription_workers", 1) != transcription_workers:
        return {"status": "failed", "run_id": identifier, "cli_exit_code": cli_code,
                "error": "Remote transcription worker count differs from the selected mode."}, cli_code or 2
    try:
        retrieval_started = time.monotonic()
        if local_archive:
            local = retrieve(response, destination, archive_results=True, local_archive=local_archive)
        else:
            local = retrieve(response, destination, archive_results=archive_results)
        retrieval_seconds = received_seconds + time.monotonic() - retrieval_started
    except Exception as error:
        return {"status": "failed", "run_id": identifier, "cli_exit_code": cli_code,
                "cli_status": envelope.get("status"), "retrieval_error": str(error),
                "elapsed_seconds": time.monotonic() - started}, cli_code or 2
    value = {**envelope, "cli_status": envelope["status"], "local_results": str(local),
             "worker_elapsed_seconds": time.monotonic() - started, "retrieval_complete": True,
             "operation": "retrieve" if run_id else "run", "host": connection()[0]}
    value["result_transport"] = "ssh-stream" if local_archive else ("ssh-tar" if archive_results else "scp")
    if timings:
        value["workflow_timings"] = {"remote_calls_seconds": rpc_seconds,
                                    "retrieval_and_validation_seconds": retrieval_seconds}
    if envelope.get("orchestration_error"):
        value["status"] = "failed"
    # Private evidence, outside the source/build context. Browser opening is manual.
    write_private_json(local / "Demo Run.json", value)
    return value, cli_code or (2 if envelope.get("orchestration_error") else 0)


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    commands = argument_parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "status", "run", "retrieve"):
        command = commands.add_parser(name)
        command.add_argument("--json", action="store_true", help="one JSON summary on stdout; progress stays on stderr")
        if name != "retrieve":
            command.add_argument("--backend", choices=("gpu", "cpu"), default="gpu",
                                 help="gpu by default; select cpu explicitly for M7i functional rehearsal")
            command.add_argument("--transcription-workers", type=int, choices=(1, 2, 4, 6, 8), default=1,
                                 help="opt-in independent GPU runners; default sequential")
        if name in {"prepare", "run"}:
            command.add_argument("--image", type=Path, required=name == "run")
        if name == "prepare":
            command.add_argument("--gpu-mps", action="store_true",
                                 help="opt-in H100 CUDA multiprocess scheduling; normal preparation disables it")
        if name == "run":
            command.add_argument("--verification", choices=("never", "always"), default="never")
            command.add_argument("--timings", action="store_true", help="private per-request timings; inference settings unchanged")
            command.add_argument("--stream-results", action="store_true",
                                 help="run and retrieve complete verified results over one authenticated SSH connection")
        if name in {"run", "retrieve"}:
            command.add_argument("--archive-results", action="store_true",
                                 help="retrieve all unchanged result files as one verified private SSH archive")
        if name == "retrieve":
            command.add_argument("--run-id", required=True)
    return argument_parser


def display(value: dict, *, json_mode: bool) -> None:
    if json_mode:
        print(json.dumps(value))
        return
    print(f"Demo outcome: {value.get('status')}")
    backend = value.get("backend") or value.get("readiness", {}).get("backend", {}).get("mode")
    if backend:
        print(f"Backend: {backend.upper()}" + (" (functional rehearsal)" if backend == "cpu" else ""))
    if "cli_exit_code" in value:
        print(f"CLI exit code: {value['cli_exit_code']}")
    if "cli_elapsed_seconds" in value:
        print(f"Warm EC2 CLI execution: {value['cli_elapsed_seconds']:.3f} seconds")
    if "end_to_end_seconds" in value:
        print(f"End-to-end timed Mac operation: {value['end_to_end_seconds']:.2f} seconds")
    elif "worker_elapsed_seconds" in value:
        print(f"Python worker operation: {value['worker_elapsed_seconds']:.3f} seconds (startup excluded)")
    if "local_results" in value:
        print(f"Local results: {value['local_results']}\nAccuracy: unreviewed; compare every visible box with its evidence.")
    for key in ("error", "orchestration_error", "retrieval_error"):
        if value.get(key):
            print(value[key], file=sys.stderr)


def write_private_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".demo-summary-{uuid.uuid4().hex}")
    try:
        with temporary.open("x") as output:
            json.dump(value, output, indent=2)
            output.write("\n")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def finish_timing(arguments: list[str]) -> int:
    """Join the external timer with the completed worker summary, after retrieval."""
    summary_path, timing_path, json_mode, exit_code = arguments
    code = int(exit_code)
    try:
        value = json.loads(Path(summary_path).read_text())
        if not isinstance(value, dict):
            raise ValueError("Summary is not an object.")
    except (OSError, ValueError):
        value = {"status": "failed", "error": "The demo worker produced no JSON result; inspect stderr."}
        code = code or 2
    try:
        duration = re.search(r"^real\s+([0-9.]+)\s*$", Path(timing_path).read_text(), re.MULTILINE)
    except OSError:
        duration = None
    if duration is None:
        value["status"] = "failed"
        value["error"] = "External wall timer failed; no end-to-end claim is available."
        display(value, json_mode=json_mode == "1")
        return code or 2
    value["end_to_end_seconds"] = float(duration.group(1))
    value["measurement_boundary"] = "external time: fresh wrapper shell/Python startup through verified retrieval; timer/formatter excluded"
    value["timer_resolution_seconds"] = 0.01
    value["timing_target_met_provisionally"] = (code == 0 and value.get("status") == "complete"
                                              and value.get("retrieval_complete") is True and value["end_to_end_seconds"] + 0.01 <= 5)
    if value.get("local_results"):
        try:
            write_private_json(Path(value["local_results"]) / "Demo Run.json", value)
        except OSError:
            value.update(status="failed", error="Could not atomically save the timed summary; prior evidence is retained.",
                         timing_target_met_provisionally=False)
            code = code or 2
    display(value, json_mode=json_mode == "1")
    return code


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    if os.getenv("UMA_BCT_DEMO_STDERR_FD") == "3":
        os.dup2(3, 2)  # Progress goes straight to terminal; external time uses its own stderr.
    arguments = argv if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "_finish":
        return finish_timing(arguments[1:])
    args = parser().parse_args(arguments)
    started = time.monotonic()
    code = 0
    try:
        if args.command == "prepare":
            value = prepare(args.image, backend=args.backend, transcription_workers=args.transcription_workers,
                            gpu_mps=args.gpu_mps)
        elif args.command == "status":
            value = rpc({"action": "status", "backend": args.backend,
                         "transcription_workers": args.transcription_workers})
            if args.backend == "cpu":
                require_cpu_status(value)
            if (args.transcription_workers > 1 and value.get("readiness", {}).get("settings", {}).get("transcription_workers")
                    != args.transcription_workers):
                raise RuntimeError("Server did not confirm two independent runners; prepare this mode first.")
        else:
            value, code = demo_run(getattr(args, "image", None), getattr(args, "verification", "never"), started,
                                   run_id=getattr(args, "run_id", None), timings=getattr(args, "timings", False),
                                   backend=getattr(args, "backend", "gpu"),
                                   transcription_workers=getattr(args, "transcription_workers", 1),
                                   archive_results=getattr(args, "archive_results", False),
                                   stream_results=getattr(args, "stream_results", False))
    except KeyboardInterrupt:
        value, code = {"status": "interrupted", "error": "Interrupted; inspect the recorded run ID before retrying."}, 130
    except Exception as error:
        value, code = {"status": "failed", "error": str(error), "elapsed_seconds": time.monotonic() - started}, 2
    display(value, json_mode=args.json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
