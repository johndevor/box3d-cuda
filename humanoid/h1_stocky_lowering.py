"""H1-STOCKY humanoid lowering (family member of the accepted H1.1).

Derived PARAMETRICALLY from humanoid/h1_lowering.py by humanoid/h1_family.py
(no forked tables): total mass +20 % distributed proportionally (68.0 ->
81.6 kg), thigh and shank -6 % length, soles +3 cm wider laterally (half-
extent 0.14 -> 0.155 m), every effort cap +15 % (180/140/70 -> 207/161/80.5),
hip anchors unchanged laterally; limits, axes, dt, gravity (-20), friction inherited.
Hip height 1.00 -> 0.9484 m, leg length 0.86 -> 0.8084 m.

Gains/caps are AUTHORED here via the Morphology spec (h1_family.H1_STOCKY,
rationale string) after the plant feasibility checklist
(humanoid/feasibility_check.py) and the executed-validation gate
(humanoid/tests/test_reference_gait.py, per variant).

PROFILE: duckgridwalk.humanoid.h1_stocky-v1
Artifacts: humanoid/variants/h1_stocky/{include/duck_model.h,
reference_gait.json, bc_init.pt, bc_init_ckpt.pt}.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import h1_family  # noqa: E402

h1_family.export(h1_family.H1_STOCKY, globals())
