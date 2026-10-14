#!/usr/bin/env python3
"""Prepare two approved EC2 GPUs; live runs use the existing timed demo wrapper."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not __package__:
    sys.path.insert(0, str(ROOT))
client = importlib.import_module("scripts.demo_ec2")


@contextlib.contextmanager
def connection(host: str, key: str):
    saved = {name: os.environ.get(name) for name in ("UMA_BCT_EC2_HOST", "UMA_BCT_EC2_KEY")}
    os.environ.update(UMA_BCT_EC2_HOST=host, UMA_BCT_EC2_KEY=key)
    try:
        client.connection()
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def identity(host: str, key: str) -> dict:
    code = """import json,urllib.request
b='http://169.254.169.254/latest/'
o=urllib.request.build_opener(urllib.request.ProxyHandler({}))
t=o.open(urllib.request.Request(b+'api/token',method='PUT',headers={'X-aws-ec2-metadata-token-ttl-seconds':'60'}),timeout=5).read().decode()
def get(p):return o.open(urllib.request.Request(b+p,headers={'X-aws-ec2-metadata-token':t}),timeout=5).read().decode()
d=json.loads(get('dynamic/instance-identity/document'))
m=get('meta-data/network/interfaces/macs/').splitlines()[0]
print(json.dumps({'instance_id':d['instanceId'],'private_ip':d['privateIp'],'region':d['region'],'vpc':get('meta-data/network/interfaces/macs/'+m+'vpc-id')}))
"""
    with connection(host, key):
        _host, options = client.connection()
        value = subprocess.run(["ssh", "-T", *options, host, "python3 -"], input=code, text=True,
                               capture_output=True, check=True)
        return json.loads(value.stdout)


def certificates(root: Path, address: str) -> None:
    def openssl(*arguments: str) -> None:
        subprocess.run(["openssl", *arguments], cwd=root, check=True, capture_output=True)
    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30", "-keyout", "ca.key", "-out", "ca.crt",
            "-subj", "/CN=UMA BCT temporary worker CA", "-addext", "basicConstraints=critical,CA:TRUE")
    for role in ("server", "client"):
        openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", role + ".key", "-out", role + ".csr",
                "-subj", "/CN=UMA BCT " + role)
        extension = root / (role + ".ext")
        extension.write_text("basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
                             + f"extendedKeyUsage={role}Auth\n"
                             + (f"subjectAltName=IP:{address}\n" if role == "server" else ""))
        openssl("x509", "-req", "-in", role + ".csr", "-CA", "ca.crt", "-CAkey", "ca.key", "-CAcreateserial",
                "-days", "30", "-out", role + ".crt", "-extfile", role + ".ext")
    for path in root.iterdir():
        path.chmod(0o600)


def prepare(args) -> dict:
    started = time.monotonic()
    primary, key = os.environ.get("UMA_BCT_EC2_HOST", ""), os.environ.get("UMA_BCT_EC2_KEY", "")
    peer, peer_key = args.peer_host, args.peer_key or key
    first, second = identity(primary, key), identity(peer, peer_key)
    if first["vpc"] != second["vpc"] or first["region"] != second["region"]:
        raise RuntimeError("This temporary GPU pair requires the same VPC and Region.")
    if first["instance_id"] == second["instance_id"] or first["private_ip"] == second["private_ip"]:
        raise RuntimeError("The GPU pair requires two distinct instances.")
    def node(host: str, node_key: str, image: Path | None) -> dict:
        environment = dict(os.environ, UMA_BCT_EC2_HOST=host, UMA_BCT_EC2_KEY=node_key)
        command = [str(ROOT / "scripts/demo-ec2.sh"), "prepare", "--transcription-workers", "4", "--gpu-mps", "--json"]
        if image:
            command += ["--image", str(image)]
        value = subprocess.run(command, env=environment, stdout=subprocess.PIPE, text=True, check=True)
        return json.loads(value.stdout)
    print("Preparing both GPU services and selected input; outside live-run timing.", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=2) as pool:
        primary_future = pool.submit(node, primary, key, args.image)
        peer_future = pool.submit(node, peer, peer_key, None)
        local, worker = primary_future.result(), peer_future.result()
    if local["application"]["source_sha256"] != worker["application"]["source_sha256"]:
        raise RuntimeError("Source changed between GPU deployments; repeat preparation.")
    with tempfile.TemporaryDirectory(prefix="uma-bct-peer-tls-") as temporary:
        directory = Path(temporary)
        certificates(directory, second["private_ip"])
        upload_id = uuid.uuid4().hex
        for host, node_key, role in ((peer, peer_key, "server"), (primary, key, "client")):
            with connection(host, node_key):
                _host, options = client.connection()
                subprocess.run(["ssh", "-T", *options, host,
                                f"install -d -m 0700 /home/ubuntu/demo_incoming/{upload_id}-tls"], check=True)
                for name in ("ca.crt", role + ".crt", role + ".key"):
                    client.copy(str(directory / name), f"{host}:/home/ubuntu/demo_incoming/{upload_id}-tls/{name}")
                client.copy(str(ROOT / "scripts/demo_ec2_peer.py"),
                            f"{host}:/home/ubuntu/demo_incoming/{upload_id}-peer.py")
                result = client.ssh_json(f"sudo python3 /home/ubuntu/demo_incoming/{upload_id}-peer.py "
                            + ("gateway" if role == "server" else "coordinator"),
                            {"workers": 4, "private_ip": second["private_ip"],
                             "tls_staging": f"/home/ubuntu/demo_incoming/{upload_id}-tls"})
                if result.get("status") != "configured":
                    raise RuntimeError("Private worker configuration did not finish.")
    with connection(primary, key):
        ready = client.rpc({"action": "status", "backend": "gpu", "transcription_workers": 8})
    return {"status": "prepared", "primary": primary, "peer": peer, "readiness": ready,
            "preparation_elapsed_seconds": time.monotonic() - started,
            "note": "Eight independent readers over two GPUs; fresh inference and existing timed retrieval remain required."}


def main() -> int:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare",))
    parser.add_argument("--peer-host", required=True)
    parser.add_argument("--peer-key")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args)))
        return 0
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
