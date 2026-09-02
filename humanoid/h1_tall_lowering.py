"""H1-TALL humanoid lowering (family member of the accepted H1.1).

Derived PARAMETRICALLY from humanoid/h1_lowering.py by humanoid/h1_family.py
(no forked tables): thigh and shank +12 % length, torso +5 %, masses of the
scaled links rescaled with length (density preserved; 68.0 -> 71.52 kg),
hip anchors unchanged laterally (+-0.15 m), soles/arms/head/pelvis and
every joint limit, axis, dt, gravity (-20 m/s^2), friction inherited.
Hip height 1.00 -> 1.1032 m, leg length 0.86 -> 0.9632 m.

Gains/caps are AUTHORED here via the Morphology spec (h1_family.H1_TALL,
rationale string) after the plant feasibility checklist
(humanoid/feasibility_check.py) and the executed-validation gate
(humanoid/tests/test_reference_gait.py, per variant).

PROFILE: duckgridwalk.humanoid.h1_tall-v1
Artifacts: humanoid/variants/h1_tall/{include/duck_model.h,
reference_gait.json, bc_init.pt, bc_init_ckpt.pt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import h1_family  # noqa: E402

h1_family.export(h1_family.H1_TALL, globals())
