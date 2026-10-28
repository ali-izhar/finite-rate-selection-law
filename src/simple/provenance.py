"""Provenance block written into every results/simple/*.json."""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT), *args],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _head_from_files() -> str | None:
    """HEAD commit without a git binary (inside a container that has none)."""
    try:
        head = (ROOT / ".git" / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        loose = ROOT / ".git" / ref
        if loose.exists():
            return loose.read_text().strip()
        for line in (ROOT / ".git" / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


def source_hashes() -> dict[str, str]:
    """sha256 of every file of src/simple, so a result names the exact code that made it."""
    out = {}
    for f in sorted((ROOT / "src" / "simple").rglob("*.py")):
        out[str(f.relative_to(ROOT))] = hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return out


def provenance(**seeds) -> dict:
    import numpy as np
    import scipy
    info = dict(
        utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        argv=sys.argv, host=socket.gethostname(), python=platform.python_version(),
        numpy=np.__version__, scipy=scipy.__version__,
        git_head=_git("rev-parse", "HEAD") or _head_from_files(),
        git_dirty=_git("status", "--porcelain", "--untracked-files=all", "src/simple"),
        source_sha256=source_hashes(), seeds=seeds,
    )
    if "torch" in sys.modules or os.environ.get("SIMPLE_WANT_TORCH"):
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return info
