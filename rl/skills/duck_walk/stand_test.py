"""Stand / open-loop checks of the batched duck on the CPU build (compare with World2's traces)."""
import math
import sys
import time

import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from rl.skills.duck_walk.env import DuckWalkEnv, LEGS  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "stand"
import json, os
env = DuckWalkEnv(4, device="cpu", seed=1, dr_on=False, dr=json.loads(os.environ.get("DR", "{}")), substeps=int(sys.argv[2]) if len(sys.argv) > 2 else 4, iterations=int(sys.argv[3]) if len(sys.argv) > 3 else 8)
y0 = env.trunk()["h"][0].item()
print("trunk com y0", round(y0, 4), "mass", env.base_mass[0].item())
t0 = time.time()
for k in range(120):
    a = torch.zeros(4, LEGS)
    if mode == "sine":   # knees and hips: a 1 Hz squat
        s = math.sin(2 * math.pi * 1.0 * k / 40)
        a[:, 2] = -0.5 * s; a[:, 3] = 1.0 * s; a[:, 4] = -0.5 * s
        a[:, 7] = 0.5 * s; a[:, 8] = 1.0 * s; a[:, 9] = -0.5 * s
    obs, priv, r, done, info = env.step(a, autoreset=False)
    if k % 10 == 9:
        T = env.trunk()
        print("   pitch q", " ".join(f"{env._coord()[0, i].item():.3f}" for i in [2, 3, 4, 7, 8, 9])); print(f"knee_q={env._coord()[0,3].item():.4f} theta={env.servo[0, env.jidx[3], 0].item():.4f} tau={env.out[0, 9 + env.jidx[3].item()].item():.3f} ", end=""); print(f"t={env.t[0].item():.2f} h={T['h'][0].item():.4f} tilt={math.degrees(T['tilt'][0].item()):.2f} x={T['p'][0, 0].item():.4f} soles={env.out[0, 7].item():.4f},{env.out[0, 8].item():.4f} "
              f"contact={env.out[0, 0].item():.0f}{env.out[0, 1].item():.0f} F={env.out[0, 2].item():.1f},{env.out[0, 3].item():.1f} gyro={obs[0, 0:3].numpy().round(3)} grav={obs[0, 3:6].numpy().round(3)} q={obs[0, 6:11].numpy().round(3)} done={bool(done[0])}")
print("wall", round(time.time() - t0, 2), "s for", 120 * 4, "env-steps")
from rl.skills.duck_walk.env import tq_rot  # noqa: E402
st = env.state[0]
pa, ca = env.mf[0].reshape(-1, 3), env.mf[1].reshape(-1, 3)
jp, jc = env.mi[0].long(), env.mi[1].long()
wp = st[jp, :3] + tq_rot(st[jp, 3:7], pa); wc = st[jc, :3] + tq_rot(st[jc, 3:7], ca)
print("joint separation mm", ((wc - wp).norm(dim=-1) * 1000).numpy().round(2))
print("coord (servo) rad", env._coord()[0].numpy().round(3))
print("servo theta_m", env.servo[0, env.jidx, 0].numpy().round(3), "goal", env.servo[0, env.jidx, 3].numpy().round(3))
print("tau", env.out[0, 9:].numpy().round(3))
