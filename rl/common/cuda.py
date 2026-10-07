"""Device policy and native extensions for rl/.

Training never falls back to the CPU on its own: `require_device('cuda')` raises when CUDA is missing, and every
extension load goes through `load_ext`, which

  - names a build by the sha256 of its sources, headers and flags (``<name>_<hash12>``);
  - imports a prebuilt build from ``$B3_PREBUILT`` (the GPU snapshot's ``/opt/b3ext``) when one exists;
  - else, with ``B3_REQUIRE_PREBUILT=1`` (set in the snapshot), fails loudly instead of compiling;
  - else compiles once into ``$B3_BUILD_CACHE`` (default ~/.cache/b3ext) with torch's cpp_extension.

``python -m rl.common.cuda --build-all [--dest DIR]`` compiles every extension any skill declares (rl/build list
below) into DIR: the snapshot's image runs it with TORCH_CUDA_ARCH_LIST set, no GPU needed.
A CPU device is allowed only when a caller asks for it by name (tests, the sim-match on a laptop, World2-side checks).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

RL = Path(__file__).resolve().parents[1]
ROOT = RL.parent
PREBUILT = os.environ.get("B3_PREBUILT", "/opt/b3ext")


class DeviceError(RuntimeError):
    pass


def require_device(device: str) -> torch.device:
    """The device training runs on. 'cuda' must really be CUDA (a visible GPU and a CUDA build of torch)."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise DeviceError(f"CUDA requested but not available (torch {torch.__version__}, built for CUDA {torch.version.cuda}): refusing to train on the CPU")
        torch.zeros(1, device=dev).add_(1)          # a real kernel launch
    return dev


def device_record(dev: torch.device) -> dict:
    """What the manifest records about where a run trained."""
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        return dict(type="cuda", name=p.name, cuda=torch.version.cuda, torch=torch.__version__, capability=f"{p.major}.{p.minor}")
    return dict(type=dev.type, torch=torch.__version__)


# ---------------------------------------------------------------- extensions
EXTENSIONS = {
    # name: (sources relative to the repository root, include dirs, cuda?)
    "b3_manifold": (["rl/ext/manifold_bind.cpp", "csrc/manifold.cu"], [], True),
    "b3_detent": (["rl/ext/detent_bind.cpp", "rl/ext/detent.cu"], [], True),
    "b3_screw": (["rl/ext/screw_joint.cu"], [], True),
    "b3_duck_cuda": (["rl/skills/duck_walk/duck_cuda.cu"], ["rl/skills/duck_walk"], True),
    "b3_duck_cpu": (["rl/skills/duck_walk/duck_cpu.cpp"], ["rl/skills/duck_walk"], False),
}
CFLAGS = ["-O3"]
CUDA_CFLAGS = {"b3_screw": ["-O3"], "b3_duck_cuda": ["-O3", "--use_fast_math", "-lineinfo"]}
CPU_CFLAGS = {"b3_duck_cpu": ["-O2"]}


def _flags(name):
    cuda = EXTENSIONS[name][2]
    return dict(extra_cflags=CPU_CFLAGS.get(name, CFLAGS), extra_cuda_cflags=CUDA_CFLAGS.get(name, ["-O3", "--use_fast_math"]) if cuda else None)


def source_hash(name: str) -> str:
    srcs, incs, cuda = EXTENSIONS[name]
    h = hashlib.sha256(json.dumps([name, _flags(name), torch.__version__.split("+")[0], torch.version.cuda]).encode())
    files = [ROOT / s for s in srcs] + sorted(f for d in incs for f in (ROOT / d).glob("*.h"))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def _import_so(name: str, so: Path):
    spec = importlib.util.spec_from_file_location(name, str(so), loader=importlib.machinery.ExtensionFileLoader(name, str(so)))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_loaded: dict = {}
LOAD_LOG: list = []


def build(name: str, dest: Path, verbose=False):
    """Compile `name` into dest/<name>_<hash>/; returns (the .so path, the loaded module)."""
    from torch.utils.cpp_extension import load
    srcs, incs, cuda = EXTENSIONS[name]
    d = dest / f"{name}_{source_hash(name)}"
    d.mkdir(parents=True, exist_ok=True)
    mod = load(name=name, sources=[str(ROOT / s) for s in srcs], extra_include_paths=[str(ROOT / i) for i in incs], build_directory=str(d),
               verbose=verbose, **{k: v for k, v in _flags(name).items() if v is not None})
    return d / f"{name}.so", mod


def load_ext(name: str):
    """The compiled extension `name` (see EXTENSIONS): prebuilt if present, else built once (unless prebuilt is required)."""
    if name in _loaded:
        return _loaded[name]
    if EXTENSIONS[name][2] and not torch.cuda.is_available():
        raise DeviceError(f"extension {name} is CUDA code and CUDA is not available")
    hsh = source_hash(name)
    so = Path(PREBUILT) / f"{name}_{hsh}" / f"{name}.so"
    t0 = time.time()
    if so.exists():
        mod, how = _import_so(name, so), "prebuilt"
    elif os.environ.get("B3_REQUIRE_PREBUILT") == "1":
        have = sorted(p.name for p in Path(PREBUILT).glob(f"{name}_*")) if Path(PREBUILT).exists() else []
        raise DeviceError(f"extension {name}_{hsh} is not prebuilt in {PREBUILT} (have {have}) and B3_REQUIRE_PREBUILT=1: the snapshot is stale for these sources")
    else:
        cache = Path(os.environ.get("B3_BUILD_CACHE", Path.home() / ".cache" / "b3ext"))
        built = cache / f"{name}_{hsh}" / f"{name}.so"
        mod, how = (_import_so(name, built), "cached") if built.exists() else (build(name, cache)[1], "compiled")
    LOAD_LOG.append(dict(ext=name, hash=hsh, how=how, s=round(time.time() - t0, 1)))
    print(json.dumps(dict(event="ext", **LOAD_LOG[-1])), file=sys.stderr, flush=True)
    _loaded[name] = mod
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-all", action="store_true")
    ap.add_argument("--dest", default=PREBUILT)
    ap.add_argument("--only", default=None, help="comma-separated extension names")
    ap.add_argument("--hashes", action="store_true", help="print each extension's source hash")
    a = ap.parse_args()
    names = a.only.split(",") if a.only else list(EXTENSIONS)
    if a.hashes:
        print(json.dumps({n: source_hash(n) for n in names}))
    if a.build_all:
        for n in names:
            if EXTENSIONS[n][2] and not os.environ.get("TORCH_CUDA_ARCH_LIST") and not torch.cuda.is_available():
                raise SystemExit("set TORCH_CUDA_ARCH_LIST to build CUDA extensions without a GPU")
            t0 = time.time()
            so, _ = build(n, Path(a.dest), verbose=False)
            print(json.dumps(dict(built=n, so=str(so), s=round(time.time() - t0, 1))), flush=True)


if __name__ == "__main__":
    main()
