from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import yaml


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    _atomic_write(Path(path), lambda f: json.dump(value, f, indent=2, allow_nan=False))


def atomic_yaml(path: str | Path, value) -> None:
    _atomic_write(Path(path), lambda f: yaml.safe_dump(value, f, sort_keys=False))


def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            write(handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: str | Path, *, compressed: bool = True, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            save = np.savez_compressed if compressed else np.savez
            save(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path, *, trust: bool = False) -> dict:
    """Safe loading by default; never silently fall back to unrestricted pickle."""
    try:
        value = torch.load(path, map_location="cpu", weights_only=not trust)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {path}. For a checkpoint you created and trust, "
            "use --trust-checkpoint. Untrusted pickle checkpoints can execute code."
        ) from exc
    if not isinstance(value, dict) or "state_dict" not in value:
        raise ValueError("Expected a Lightning checkpoint containing state_dict")
    return value


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "gpu":
        name = "cuda"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def source_tree_hash() -> str:
    """Fingerprint implementation files to prevent silently reusing stale evals."""
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
