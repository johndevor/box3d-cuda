from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]

if "box3d_cuda" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "box3d_cuda",
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load box3d_cuda source package")
    package = importlib.util.module_from_spec(spec)
    sys.modules["box3d_cuda"] = package
    spec.loader.exec_module(package)
