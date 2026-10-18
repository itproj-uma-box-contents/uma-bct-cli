#!/usr/bin/env python3
"""Check Docker's privacy exclusions using only an invented build context."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_WEIGHT_SUFFIXES = (
    "gguf", "GGUF", "GgUf",
    "onnx", "ONNX", "OnNx",
    "pt", "PT", "Pt",
    "pth", "PTH", "PtH",
    "safetensors", "SAFETENSORS", "SafeTensors",
)
PRIVATE_CONTEXT_PATHS = (
    "catalogue.csv",
    "CATALOGUE.CSV",
    "Catalogue.CsV",
    "CATALOGUE.XLSX",
    "nested/catalogue.csv",
    "nested/Catalogue.cSv",
    "nested/catalogue.xlsx",
    "nested/Catalogue.XlSx",
    ".env",
    ".env.local",
    ".EnV",
    ".eNv.local",
    "nested/.env",
    "nested/.env.production",
    "nested/.ENV.production",
    "nested/catalogue.run.json",
    "nested/photo.JpG",
    "nested/photo.PnG",
    "nested/model.GgUf",
    ".ollama/manifests/state",
    "nested/.ollama/manifests/state",
    "nested/deeper/.ollama/manifests/state",
    "models/blobs/sha256-synthetic",
    "nested/models/blobs/sha256-synthetic",
    "nested/deeper/models/blobs/sha256-synthetic",
    ".uma-bct-runtime/state.txt",
    "nested/.uma-bct-runtime/state.txt",
    "Archive box photos/expectations.json",
    "add_repository_photos/notes.txt",
    "Add Photos Here/notes.txt",
    "inputs/source.jpg",
    "Results/Run Details.json",
    "outputs/catalogue.csv",
    # Docker matches complete paths, so check every suffix below several roots.
) + tuple(
    f"{directory}synthetic-weight.{suffix}"
    for directory in ("", "nested/", "nested/deeper/")
    for suffix in MODEL_WEIGHT_SUFFIXES
)


def main() -> None:
    """Ask Docker to copy an isolated synthetic context and inspect its export."""

    with tempfile.TemporaryDirectory(prefix="uma-bct-context-check-") as directory:
        root = Path(directory)
        context, exported = root / "context", root / "exported"
        context.mkdir()
        # Copy only the exclusion rules from the checkout. No source photos,
        # credentials, model state, or generated results enter this context.
        shutil.copyfile(ROOT / ".dockerignore", context / ".dockerignore")
        (context / "Dockerfile").write_text("FROM scratch\nCOPY . /\n", encoding="utf-8")
        (context / "allowed.txt").write_text("synthetic public marker\n", encoding="utf-8")
        for name in PRIVATE_CONTEXT_PATHS:
            path = context / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic excluded content\n", encoding="utf-8")
        subprocess.run(
            ["docker", "build", "--network=none", "--no-cache", "--output", f"type=local,dest={exported}", str(context)],
            check=True,
        )
        if (exported / "allowed.txt").read_text(encoding="utf-8") != "synthetic public marker\n":
            raise RuntimeError("Docker did not export the expected synthetic context.")
        leaked = [name for name in PRIVATE_CONTEXT_PATHS if (exported / name).exists()]
        if leaked:
            raise RuntimeError("Docker privacy exclusions missed synthetic paths: " + ", ".join(leaked))
    print(f"Docker context privacy check passed for {len(PRIVATE_CONTEXT_PATHS)} synthetic paths.")


if __name__ == "__main__":
    main()
