"""Collect run provenance: environment, versions, git state, config hash (P2 §5.9)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from typing import Optional


def _git(*args) -> Optional[str]:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True).encode()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def collect(config: Optional[dict] = None) -> dict:
    import torch
    import torchvision

    gpu = None
    gpu_mem = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)

    prov = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "gpu": gpu,
        "gpu_memory_gb": gpu_mem,
    }
    if config is not None:
        prov["config_hash"] = config_hash(config)
    return prov
