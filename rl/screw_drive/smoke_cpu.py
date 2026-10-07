"""Local CPU smoke of the env (box3d's Python manifold reference + the screw oracle; slow, 2 worlds).
  python -m rl.screw_drive.smoke_cpu [seed] [ticks]
"""
import sys

import torch

from . import screw_reference as SR
from .bench import scripted
from .env import ScrewDriveBatch

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 3
ticks = int(sys.argv[2]) if len(sys.argv) > 2 else 30
e = ScrewDriveBatch(2, device="cpu", seed=seed)
o = e.obs(update=False)
print(o.shape)
print("belief_err deg", torch.rad2deg(e.p["belief_err"]).tolist(), "thx", torch.rad2deg(e.p["thx"]).tolist(), "modeb", e.p["modeb"].tolist(),
      "size", e.p["size"].tolist(), "eng turns", (e.p["eng"] / e.p["pitch"]).tolist())
for t in range(ticks):
    o, r, d, i = e.step(scripted(o))
    J = e.J
    print(t, "mode", J[:, 0].tolist(), "th", [round(x, 2) for x in J[:, 1].tolist()], "T", [round(x, 3) for x in J[:, SR.J_TORQUE].tolist()],
          "tipz", [round(x * 1000, 3) for x in e._tip()[:, 2].tolist()], "lat", [round(x * 1e3, 3) for x in e._lat_to_insert().tolist()],
          "r", [round(x, 3) for x in r.tolist()], i["code"].tolist(), "Fz", [round(x, 2) for x in o[:, 13].tolist()], flush=True)
    if d.any():
        e.reset(d)
        o = torch.where(d[:, None], e.obs(update=False), o)
