"""duck_walk: the Open Duck Mini v2 on its own batched solver (duck_sim.h), World2's walk contract (leg-joint-targets).

Recipe: asymmetric PPO in stages, as duck-walk-box3d-v1 -> v2 were made: balance (no pushes or turns), v1 (IMU, encoders, previous action,
command) from scratch, then v2 (+ foot switches and a 2.5 Hz gait clock, a reference-gait reward) warm-started from v1
with the new observation columns at zero weight (transfer below).
Evaluation (World2's judge, tasks/open-duck-walk: 10 s window at 0.15 m/s after a 1 s hold, the program-side heading
hold of gain 1): success = no fall, forward >= 0.5 m, sideways <= 0.5 m, >= 4 lift-offs over 4 mm per foot.
"""
import math

import torch

from rl.common import models
from rl.common.skill import Skill

from .env import ACTION_SCALE, FRAME, HISTORY, LEGS, RATE_HZ, SPEC, DuckWalkEnv

MODEL = dict(actor_hidden=(256, 256, 128), critic_hidden=(512, 256, 128), squash=False, log_std=math.log(0.35), norm_clip=5.0)
PPO = dict(horizon=24, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5, minibatches=4, ent=0.006, vf_coef=1.0, mean_reg=0.001,
           kl="adaptive", kl_target=0.01, log_std_bounds=(math.log(0.15), math.log(1.0)), norm="step", bootstrap_timeouts=True,
           score="return", plateau_iters=0, stats_window=2000, save_every=20)
RECIPE = dict(
    kind="asymmetric", envs=8192, eval_n=64, eval_seed=10_000, ppo=PPO,
    env=dict(substeps=8, iterations=6),
    # From scratch the full v1 task (pushes, yaw-rate commands, a random start heading, a small alive bonus) did not learn
    # (an RTX PRO 6000 run: 490 M steps, episodes ~2.7 s, the action noise growing): duck-walk-box3d-v1 itself came from a
    # chain of runs that began with a larger alive bonus. So a first stage stands and steps without pushes or turns.
    stages=[
        dict(name="balance", env=dict(clock_hz=0.0, alive_bonus=0.5), cfg=dict(push_p=0.0, cmd_wz=0.0, init_yaw=0.0),
             ppo=dict(ent=0.003, max_minutes=15, max_steps=250e6)),
        # (v2 50 min / 900 M steps walked 26/30 in World2 against the shipped v2's 30/30, failing on sideways drift:
        # the shipped one had ~800 M steps of v1 fine-tunes before its 440 M of v2; so v1 and v2 run longer)
        dict(name="v1", env=dict(clock_hz=0.0), ppo=dict(ent=0.004, max_minutes=45, max_steps=800e6)),
        dict(name="v2", env=dict(clock_hz=2.5, clearance_m=0.035), ppo=dict(ent=0.002, max_minutes=90, max_steps=2.0e9)),
    ],
)
CRITERIA = dict(window_s=10.0, vx=0.15, settle_s=1.0, min_forward_m=0.5, max_lateral_m=0.5, min_liftoffs=4, sole_clearance_m=0.004,
                sole_down_m=0.003,   # (World2's judge: on the floor under 1 mm; box3d's sole height reads 1.5-2 mm in stance)
                heading_gain=1.0, heading_max=0.3)


def make_env(n, device="cuda", seed=0, cfg=None, **env):
    cfg = dict(cfg or {})
    dr_on = cfg.pop("dr_on", True)
    return DuckWalkEnv(n, device=device, seed=seed, dr=SPEC.override(cfg) if dr_on else cfg, dr_on=dr_on, **env)


def transfer(prev, env):
    """A previous stage's policy on this stage's interface: unchanged when the observation is the same; else (v1 -> v2:
    foot switches and the gait clock appended to each history frame) the old columns keep their weights and normaliser
    statistics and the new ones start at zero weight (the policy starts as the old one)."""
    s = prev.sizes
    new = models.build(dict(actor_hidden=s["actor_hidden"], critic_hidden=s["critic_hidden"], squash=s["squash"], norm_clip=s["norm_clip"]),
                       env.obs_size, env.obs_size + env.priv_size, s["act"]).to(env.dev)
    if s["actor_in"] == env.obs_size:
        new.load_state_dict(prev.state_dict())
        return new
    old_f, new_f, priv = s["actor_in"] // HISTORY, env.obs_size // HISTORY, env.priv_size
    idx = torch.cat([torch.arange(old_f) + h * new_f for h in range(HISTORY)]).to(env.dev)
    cidx = torch.cat([idx, torch.arange(priv, device=env.dev) + new_f * HISTORY])
    with torch.no_grad():
        new.load_state_dict({k: v for k, v in prev.state_dict().items() if not (k.startswith(("actor.0", "critic.0")) or "norm" in k)}, strict=False)
        a0, c0 = new.actor[0], new.critic[0]
        a0.weight.zero_(); a0.weight[:, idx] = prev.actor[0].weight; a0.bias.copy_(prev.actor[0].bias)
        c0.weight.zero_(); c0.weight[:, cidx] = prev.critic[0].weight; c0.bias.copy_(prev.critic[0].bias)
        for nn_, on, sel in ((new.obs_norm, prev.obs_norm, idx), (new.crit_norm, prev.crit_norm, cidx)):
            nn_.mean.zero_(); nn_.var.fill_(1.0); nn_.mean[sel] = on.mean; nn_.var[sel] = on.var; nn_.count.copy_(on.count)
        new.log_std.copy_(prev.log_std)
    return new


def evaluate(act_fn, cfg, n, seed, device, env=None):
    """World2's judge in the batched env: hold 1 s, then walk 10 s at 0.15 m/s with the heading hold."""
    C = CRITERIA
    cfg = dict(cfg or {})
    dr_on = cfg.pop("dr_on", True)
    over = {**cfg, **dict(push_p=0.0, cmd_zero_p=0.0, cmd_vx=(C["vx"], C["vx"]), episode_s=1e9, init_yaw=0.0, cmd_wz=0.0)}
    e = DuckWalkEnv(n, device=device, seed=seed, dr=SPEC.override(over) if dr_on else over, dr_on=dr_on, **(env or {}))
    e.cmd[:, 0] = C["vx"]
    for _ in range(int(C["settle_s"] * RATE_HZ)):
        e.step(torch.zeros(n, LEGS, device=e.dev), autoreset=False)
    e.prev_action.zero_(); e.t.zero_(); e.hist[:] = e._frame()[:, None]
    T0 = e.trunk()
    x0, z0 = T0["p"][:, 0].clone(), T0["p"][:, 2].clone()
    fell = torch.zeros(n, dtype=torch.bool, device=e.dev)
    on = torch.ones(n, 2, dtype=torch.bool, device=e.dev)
    lifts = torch.zeros(n, 2, device=e.dev)
    obs, priv = e.observe()
    yaw = torch.zeros(n, device=e.dev)
    for _ in range(int(C["window_s"] * RATE_HZ)):
        yaw += obs[:, 1] / RATE_HZ                                   # the gyro's yaw rate (newest frame first)
        e.cmd[:, 2] = (-C["heading_gain"] * yaw).clamp(-C["heading_max"], C["heading_max"])
        obs, priv, r, done, info = e.step(act_fn(obs, priv), autoreset=False)
        T = e.trunk()
        fell |= (T["h"] < 0.7 * e.trunk_y0) | (T["tilt"] > math.radians(40)) | ~torch.isfinite(T["h"])
        h = e.out[:, 7:9]
        lift = on & (h > C["sole_clearance_m"])
        lifts += lift.float()
        on = torch.where(lift, torch.zeros_like(on), on) | (h < C["sole_down_m"])
    T = e.trunk()
    fwd, lat = T["p"][:, 0] - x0, T["p"][:, 2] - z0
    ok = ~fell & (fwd >= C["min_forward_m"]) & (lat.abs() <= C["max_lateral_m"]) & (lifts.min(-1).values >= C["min_liftoffs"])
    q = lambda t: [round(float(x), 3) for x in torch.quantile(t.float().cpu(), torch.tensor([0.1, 0.5, 0.9]))]
    return dict(n=n, success=round(ok.float().mean().item(), 4), falls=int(fell.sum()), forward_m_p10_50_90=q(fwd), lateral_abs_m_p10_50_90=q(lat.abs()),
                liftoffs_min_foot_p10_50_90=q(lifts.min(-1).values), dr=dr_on)


def manifest(spec, env=None):
    clock = (env or {}).get("clock_hz", 0.0)
    return dict(
        skill="walk", rate_hz=RATE_HZ,
        observation=["imu_gyro", "imu_gravity", "leg_joint_pos", "leg_joint_vel", "prev_action", "command"] + (["foot_contact", "gait_clock"] if clock > 0 else []),
        history=HISTORY, **({"gait_clock_hz": clock} if clock > 0 else {}),
        action={"kind": "leg-joint-targets", "scale": {"rad": ACTION_SCALE}}, input="obs", output="action", families=["open-duck-mini-v2"],
        heading_hold={"gain": CRITERIA["heading_gain"], "max": CRITERIA["heading_max"]},
        robot={"model": "parts/open-duck-mini (World2), exported by parts/open-duck-mini/export-cuda-model.mjs",
               "legs": ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]},
        licence="MIT (box3d-cuda); robot model derived from Open_Duck_Mini (Apache-2.0)",
        training_text=dict(engine="box3d-cuda (MIT, github johndevor/box3d-cuda) rl/skills/duck_walk/duck_sim.h: maximal-coordinate rigid bodies, revolute joints and "
                                  "sole-floor contacts by sequential impulses (float32, one CUDA thread per world), World2's bam-m1 servo step and IMU signal chain step "
                                  "for step at 1/120 s"))


SKILL = Skill(
    name="duck_walk", contract_skill="walk", make_env=make_env, spec=SPEC, model=MODEL, recipe=RECIPE,
    eval_sets={"nominal": {"dr_on": False}, "dr": {}}, evaluate=evaluate, manifest=manifest, extensions=("b3_duck_cuda",),
    description="Open Duck Mini v2 walking (World2 walk contract)",
)
SKILL.transfer = transfer


# ---------------------------------------------------------------- sim-match (World2 scripts/sim-match.mjs duck_walk)
PITCH = [2, 3, 4, 7, 8, 9]          # hips, knees, ankles (policy order)


def squat(k):
    s = math.sin(2 * math.pi * k / 40)
    a = torch.zeros(1, LEGS)
    a[0, 2], a[0, 3], a[0, 4], a[0, 7], a[0, 8], a[0, 9] = -0.5 * s, s, -0.5 * s, 0.5 * s, s, -0.5 * s
    return a


def sim_match(w2, device):
    """The open-loop 1 Hz squat after a 1 s stand, the nominal robot (DR off), compared tick by tick with World2's: the
    encoder angles the policy would read, and the trunk's height and tilt (each from its first frame)."""
    import numpy as np
    from rl.common.simmatch import check, lag_ticks, rms
    e = DuckWalkEnv(1, device=device, seed=0, dr_on=False, dr=dict(push_p=0.0, cmd_zero_p=0.0, cmd_vx=(0.0, 0.0), episode_s=1e9), clock_hz=0.0,
                    **RECIPE["env"])
    for _ in range(int(w2["settle_s"] * RATE_HZ)):
        e.step(torch.zeros(1, LEGS, device=e.dev), autoreset=False)
    q, y, up = [], [], []
    obs, _ = e.observe()
    for k in range(w2["ticks"]):
        T = e.trunk()
        q.append(obs[0, 6:16].tolist()); y.append(float(T["h"][0])); up.append(T["up"][0].tolist())
        obs, _, _, _, _ = e.step(squat(k).to(e.dev), autoreset=False)
    W = w2["trace"]
    qw, qb = np.array(W["leg_joint_pos"])[:, PITCH], np.array(q)[:, PITCH]
    yw, yb = np.array(W["trunk_y"]) - W["trunk_y"][0], np.array(y) - y[0]
    u0 = np.array(up[0])
    tb = np.degrees(np.arccos(np.clip(np.array(up) @ u0, -1, 1)))
    tw = np.array(W["trunk_tilt_deg"])
    p2p = lambda x: float(x[40:].max() - x[40:].min())
    off = (qb - qw).mean(0)
    checks = [
        check("pitch joint static offset, worst joint (rad)", float(np.abs(off).max()), 0.03,
              "the mean encoder difference per joint (the standing sag under load, servo statics); 0.03 rad is 12 encoder steps"),
        check("pitch joint dynamic RMS, offsets removed (rad)", rms((qw + off).ravel(), qb.ravel()), 0.012,
              "the motion itself, hips/knees/ankles; a one-tick (25 ms) servo lag in this 0.3 rad 1 Hz squat adds 0.033 rad RMS"),
        check("pitch joint lag box3d vs World2, worst joint (ticks)", max((lag_ticks(qw[:, j], qb[:, j]) for j in range(6)), key=abs), 0.4,
              "cross-correlation peak, sub-tick: |lag| under 10 ms (the servo-lag class of bug: World2's motors once lagged under 16 PGS passes)"),
        check("pitch joint amplitude ratio - 1, worst joint", max((p2p(qb[:, j]) / p2p(qw[:, j]) - 1 for j in range(6)), key=abs), 0.1,
              "peak-to-peak after the first second: servo gain and torque limits (the same bug: hips that barely moved)"),
        check("trunk height RMS (mm)", rms(yw, yb) * 1000, 2.0, "the trunk's rise and fall (about 28 mm peak to peak): gravity, masses, sole contact"),
        check("trunk tilt RMS (deg)", rms(tw, tb), 1.0,
              "the trunk's tilt from its first frame; 1 degree is about 2 sigma of the IMU's attitude bias (accel bias 0.08 m/s2): the policy "
              "cannot tell a smaller difference from its own sensor's"),
    ]
    return checks, dict(joint_pos=q, trunk_y=y, trunk_tilt_deg=tb.round(4).tolist())


SKILL.sim_match = sim_match
