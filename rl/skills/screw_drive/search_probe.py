"""The scripted screw prior vs a scripted tilt search on box3d's crooked_4_6 and nominal sets (no learning): the search
backs out on an early torque rise (the cross-thread binding while the head is still far from its seat) and tilts the next
attempt another way (8 directions at the given magnitude, in the order given), the attempt count read from the
observation's back-out counter.

    python3 -m rl.skills.screw_drive.search_probe [n] [prior|search|both] ['{"mag_deg": 5, "xthread": 0.3, ...}']
"""
import json
import math
import sys

import torch

from rl.common.evaluate import episodic
from rl.common.priors import ScriptedScrewPrior
from rl.common.skill import load_skill

DEV = "cuda" if torch.cuda.is_available() else "cpu"
sk = load_skill("screw_drive")
CROOKED = sk.eval_sets["crooked_4_6"]
prior = ScriptedScrewPrior().to(DEV)
ROT = 0.004
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
MODE = sys.argv[2] if len(sys.argv) > 2 else "both"
P = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
mag = math.radians(P.get("mag_deg", 5.0))
xth = P.get("xthread", 0.3)
order = P.get("order", [0, 180, 90, 270, 45, 225, 135, 315])
dirs = torch.tensor([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in order], device=DEV) * mag


def search(obs, priv):
    a = prior(obs).clone()
    k = obs[:, 36].round().long().clamp(0, len(order) - 1)
    tgt = dirs[k]
    if not P.get("first_tilt", False):
        tgt = torch.where((obs[:, 36] > 0.5)[:, None], tgt, torch.zeros_like(tgt))
    a[:, 2:4] = ((obs[:, 3:5] + tgt) / ROT).clamp(-1, 1)
    torque, tset, z, pitch, turns = obs[:, 19], obs[:, 33], obs[:, 2], obs[:, 29], obs[:, 20]
    prev_spin = obs[:, 46]
    bad = (torque > xth * tset) & (z > 2 * pitch)
    backing = (prev_spin < -0.05) & (turns > P.get("stop_turns", -0.25))
    rev = bad | backing
    a[:, 5] = torch.where(rev, torch.full_like(a[:, 5], -1.0), a[:, 5])
    a[:, 4] = torch.where(rev, torch.full_like(a[:, 4], P.get("rev_push", 0.0)), a[:, 4])
    return a


def scripted(obs, priv):
    return prior(obs)


if __name__ == "__main__":
    for name, fn in (("prior", scripted), ("search", search)):
        if MODE not in ("both", name):
            continue
        for set_name, cfg in (("crooked_4_6", CROOKED), ("nominal", {})):
            r = episodic(sk, fn, cfg, N, 12345, DEV, 380)
            print(json.dumps(dict(policy=name, params=P, set=set_name, success=r["success"], codes=r["codes"], backouts=r.get("backouts_mean"),
                                  belief_over=r.get("belief_over_theta_x"))), flush=True)
