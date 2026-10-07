"""Where a run came from: the box3d-cuda commit, branch, whether the tree had uncommitted changes, and the hash of the
rl/ sources a run used (so a copy without .git, as the GPU job ships it, still identifies itself: rl/PROVENANCE.json is
written by the shipper with the commit it packed)."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def sources_hash():
    h = hashlib.sha256()
    for f in sorted((ROOT / "rl").rglob("*")):
        if f.is_file() and f.suffix in (".py", ".cu", ".cpp", ".h", ".json") and "__pycache__" not in f.parts and f.name != "PROVENANCE.json" \
                and not f.name.startswith("._"):          # (macOS tar metadata)
            h.update(str(f.relative_to(ROOT)).encode())
            h.update(f.read_bytes())
    for f in ("csrc/manifold.cu",):
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:16]


def provenance():
    shipped = ROOT / "rl" / "PROVENANCE.json"
    commit = _git("rev-parse", "HEAD")
    if commit:
        rec = dict(commit=commit, branch=_git("rev-parse", "--abbrev-ref", "HEAD"), dirty=bool(_git("status", "--porcelain", "--", "rl", "csrc")))
        rec["rl_sources"] = sources_hash()
    elif shipped.exists():
        rec = json.loads(shipped.read_text())
        now = sources_hash()
        if rec.get("rl_sources") != now:          # edited after packing
            rec.update(dirty=True, rl_sources_packed=rec.get("rl_sources"), rl_sources=now)
    else:
        rec = dict(commit="unknown", branch="unknown", dirty=None, rl_sources=sources_hash())
    return rec
