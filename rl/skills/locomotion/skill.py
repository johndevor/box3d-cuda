"""Locomotion for any robot World2 designs (the replicator brain's generic skill): the model is generated from the
robot's World2 design (World2 scripts/robot-model.mjs <design> --out model.json; train with --set env.model_file=...),
the recipe is generic:
  1. prepare: an open-loop probe picks the oscillator prior (rl/common/priors.py OscillatorPrior): random per-joint
     sinusoids (amplitude, phase, offset) at a few clock rates on the nominal robot, judged as World2's task judges
     (forward distance in the window, no flip, little sideways drift); the best becomes the prior and its clock the gait
     clock of the observation;
  2. asymmetric PPO of a residual over that prior, the randomization ramped in from the nominal robot over the first half
     (the duck's lesson: under full randomization from the start a residual learns to cancel its prior);
  3. evaluation by the same judge in the batched env, nominal and randomized.
The robot's own numbers (its speed, action scale, judge) come from the model's `locomotion` block (the World2 design's).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from rl.common.skill import Skill
from rl.skills.duck_walk.env import HISTORY, RATE_HZ
from rl.skills.locomotion.env import SPEC, LocomotionEnv

MODEL = dict(actor_hidden=(128, 128), critic_hidden=(256, 128), squash=False, log_std=math.log(0.15), norm_clip=5.0)
PPO = dict(horizon=24, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=5, minibatches=8, ent=0.003, vf_coef=1.0, mean_reg=0.001,
           kl="adaptive", kl_target=0.01, log_std_bounds=(math.log(0.05), math.log(0.6)), norm="step", bootstrap_timeouts=True,
           score="return", plateau_iters=0, stats_window=2000, save_every=20, max_minutes=4.0, max_steps=3e9)
RECIPE = dict(kind="asymmetric", envs=16384, eval_n=256, eval_seed=10_000, graphs=False, ppo=PPO, model=dict(prior="auto"),
              env=dict(substeps=8, iterations=6, dr_ramp=0.5), probe=dict(patterns=2048, clocks=(1.0, 1.5, 2.0), refine=16, refine_n=64))
DEFAULTS = dict(v_ref=0.05, action_scale=1.0, window_s=10.0, settle_s=1.0, min_forward_m=0.25, max_lateral_m=0.25, max_tilt_deg=45.0, fall_height_frac=0.0)


def robot_of(env_kw):
    m = json.loads(Path(env_kw["model_file"]).read_text())
    return m, dict(DEFAULTS, **(m.get("locomotion") or {}))


def env_settings(env_kw):
    """The env's settings from the model's locomotion block, under the recipe's."""
    _, L = robot_of(env_kw)
    base = dict(v_ref=L["v_ref"], action_scale=L["action_scale"], max_tilt_deg=L["max_tilt_deg"], fall_height_frac=L["fall_height_frac"])
    return dict(base, **env_kw)


def make_env(n, device="cuda", seed=0, cfg=None, **env):
    cfg = dict(cfg or {})
    dr_on = cfg.pop("dr_on", True)
    return LocomotionEnv(n, device=device, seed=seed, dr=SPEC.override(cfg) if dr_on else cfg, dr_on=dr_on, **env_settings(env))


# ---------------------------------------------------------------- the judge, batched (World2's task check)
def judge_rollout(e, act, L, dr_on):
    """Settle, then the window at the command v_ref: forward distance, sideways drift, no flip; act(obs, priv, k) -> action."""
    n = e.E
    e.cmd[:, 0] = L["v_ref"]; e.cmd[:, 1:] = 0
    for _ in range(int(L["settle_s"] * RATE_HZ)):
        e.step(torch.zeros(n, e.legs, device=e.dev), autoreset=False)
    e.prev_action.zero_(); e.t.zero_(); e.hist[:] = e._frame()[:, None]
    T0 = e.trunk()
    x0, z0 = T0["p"][:, 0].clone(), T0["p"][:, 2].clone()
    flip = torch.zeros(n, dtype=torch.bool, device=e.dev)
    obs, priv = e.observe()
    for k in range(int(L["window_s"] * RATE_HZ)):
        obs, priv, r, done, info = e.step(act(obs, priv, k), autoreset=False)
        T = e.trunk()
        flip |= (T["tilt"] > math.radians(L["max_tilt_deg"])) | ~torch.isfinite(T["h"])
    T = e.trunk()
    fwd, lat = T["p"][:, 0] - x0, T["p"][:, 2] - z0
    ok = ~flip & (fwd >= L["min_forward_m"]) & (lat.abs() <= L["max_lateral_m"])
    return ok, fwd, lat, flip


def evaluate(act_fn, cfg, n, seed, device, env=None):
    cfg = dict(cfg or {})
    dr_on = cfg.pop("dr_on", True)
    env = {k: v for k, v in (env or {}).items() if k != "dr_ramp"}
    _, L = robot_of(env)
    over = dict(cfg, push_p=0.0, init_yaw=0.0)
    e = LocomotionEnv(n, device=device, seed=seed, dr=SPEC.override(over) if dr_on else over, dr_on=dr_on, **env_settings(env))
    ok, fwd, lat, flip = judge_rollout(e, lambda o, p, k: act_fn(o, p), L, dr_on)
    q = lambda t: [round(float(x), 3) for x in torch.quantile(t.float().cpu(), torch.tensor([0.1, 0.5, 0.9]))]
    return dict(n=n, success=round(ok.float().mean().item(), 4), flips=int(flip.sum()), forward_m_p10_50_90=q(fwd), lateral_abs_m_p10_50_90=q(lat.abs()), dr=dr_on)


# ---------------------------------------------------------------- the prior probe
def probe(env_kw, device, P, log=print, seed=0):
    """Random open-loop oscillators on the nominal robot (one world each), per clock; the best `refine` re-judged on
    refine_n randomized worlds (the ramp ends at full randomization). -> (prior args, clock_hz, report)."""
    m, L = robot_of(env_kw)
    g = torch.Generator().manual_seed(seed)
    best = []
    for hz in P["clocks"]:
        N = int(P["patterns"])
        e = LocomotionEnv(N, device=device, seed=seed, dr={}, dr_on=False, **dict(env_settings(env_kw), clock_hz=hz, dr_ramp=0.0))
        Lj = e.legs
        amp = torch.rand(N, Lj, generator=g) * 0.8
        ph = torch.rand(N, Lj, generator=g) * 2 * math.pi
        off = (torch.rand(N, Lj, generator=g) - 0.5) * 0.6
        S, C, O = (amp * torch.cos(ph)).to(e.dev), (amp * torch.sin(ph)).to(e.dev), off.to(e.dev)
        ci = e.clock_idx
        act = lambda o, p, k: (S * o[:, ci:ci + 1] + C * o[:, ci + 1:ci + 2] + O).clamp(-1, 1)
        ok, fwd, lat, flip = judge_rollout(e, act, L, False)
        score = torch.where(flip, torch.full_like(fwd, -9.0), fwd - 0.5 * lat.abs())
        top = torch.topk(score, k=min(int(P["refine"]), N)).indices.tolist()
        for i in top:
            best.append(dict(hz=hz, S=S[i].tolist(), C=C[i].tolist(), O=O[i].tolist(), fwd=round(float(fwd[i]), 3), lat=round(float(lat[i]), 3), score=float(score[i])))
        log(dict(event="probe_clock", clock_hz=hz, patterns=N, passing_nominal=int(ok.sum()), best_forward_m=round(float(fwd[top[0]]), 3)))
    best.sort(key=lambda x: -x["score"])
    best = best[: int(P["refine"])]
    # refine: each candidate on randomized robots
    for b in best:
        n = int(P["refine_n"])
        e = LocomotionEnv(n, device=device, seed=seed + 1, dr=SPEC.override(dict(push_p=0.0, init_yaw=0.0)), dr_on=True, **dict(env_settings(env_kw), clock_hz=b["hz"], dr_ramp=0.0))
        S, C, O = (torch.tensor(b[k], device=e.dev)[None] for k in ("S", "C", "O"))
        ci = e.clock_idx
        ok, fwd, lat, flip = judge_rollout(e, lambda o, p, k: (S * o[:, ci:ci + 1] + C * o[:, ci + 1:ci + 2] + O).clamp(-1, 1), L, True)
        b.update(dr_success=round(ok.float().mean().item(), 3), dr_forward_median=round(float(fwd.median()), 3), dr_flips=int(flip.sum()))
    best.sort(key=lambda x: (-x["dr_success"], -x["dr_forward_median"]))
    w = best[0]
    log(dict(event="probe_done", chosen=w, runner_up=best[1] if len(best) > 1 else None))
    return w, best


def prepare(R, cfg, out, log):
    """Settle the prior (and the gait clock) by the probe when the recipe asks for it (model.prior 'auto')."""
    if (R.get("model") or {}).get("prior") != "auto":
        return
    env_kw = dict(R.get("env", {}))
    w, best = probe(env_kw, R["device"], R.get("probe", RECIPE["probe"]), log=log)
    e = LocomotionEnv(1, device="cpu" if R["device"] != "cuda" else R["device"], seed=0, dr={}, dr_on=False, **dict(env_settings(env_kw), clock_hz=w["hz"], dr_ramp=0.0))
    frame = e.frame
    R["env"]["clock_hz"] = w["hz"]
    R["model"]["prior"] = dict(kind="oscillator", args=dict(n_in=frame * HISTORY, n_act=e.legs, sin_idx=e.clock_idx, cos_idx=e.clock_idx + 1, cmd_vx_idx=e.cmd_vx_idx,
                                                          S=w["S"], C=w["C"], O=w["O"], v_ref=e.v_ref))
    (Path(out) / "probe.json").write_text(json.dumps(dict(chosen=w, candidates=best), indent=1))


def manifest(spec, env=None):
    env = env or {}
    m, L = robot_of(env)
    import hashlib
    raw = Path(env["model_file"]).read_bytes()
    legs = int(m.get("policy_joints", len(m["joints"]))); nsw = int(m.get("n_switches", 0))
    return dict(
        skill="walk", rate_hz=RATE_HZ,
        observation=["imu_gyro", "imu_gravity", "leg_joint_pos", "leg_joint_vel", "prev_action", "command"] + (["foot_contact"] if nsw else []) + ["gait_clock"],
        history=HISTORY, gait_clock_hz=env.get("clock_hz", 1.5), **({"foot_switches": nsw} if nsw else {}),
        action={"kind": "leg-joint-targets", "size": legs, "scale": {"rad": L["action_scale"]}}, input="obs", output="action", families=[m.get("robot", "robot")],
        robot={"model": f"generated from its World2 design by skills/robot-model.mjs ({m.get('generated_by', '')})", "robot": m.get("robot"),
               "model_sha256": hashlib.sha256(raw).hexdigest(), "bodies": len(m["bodies"]), "joints": [j["name"] for j in m["joints"]],
               "driven": m["policy_order"][:legs], "contact_boxes": len(m["feet"]), "mass_kg": m.get("mass_kg"), "locomotion": L},
        licence="project",
        basis="trained in box3d-cuda on the robot's model generated from its World2 design (bodies, joints, servos, contact boxes, IMU read back from World2), sim-matched against World2 before training",
    )


# ---------------------------------------------------------------- sim-match (World2 scripts/sim-match.mjs locomotion --robot <design>)
def sim_match(w2, device, env=None):
    """World2's open-loop probe (after the settle, each driven joint a 1 Hz sinusoid of the probe's amplitude, phases a
    quarter turn apart) replayed on the generated model, the nominal robot on a 0.8 floor: the encoder angles, the trunk's
    height, tilt and its travel (the friction of the contact boxes shows in the travel)."""
    import numpy as np
    from rl.common.simmatch import check, lag_ticks, rms
    env = dict(env or {})
    P = w2["oscillator"]
    e = LocomotionEnv(1, device=device, seed=0, dr=dict(push_p=0.0), dr_on=False, **dict(env_settings(env), clock_hz=P["hz"], dr_ramp=0.0))
    for _ in range(int(w2["settle_s"] * RATE_HZ)):
        e.step(torch.zeros(1, e.legs, device=e.dev), autoreset=False)
    e.t.zero_()
    S, C, O = (torch.tensor([P[k]], dtype=torch.float32, device=e.dev) for k in ("S", "C", "O"))
    q, y, up, x = [], [], [], []
    obs, _ = e.observe()
    L = e.legs
    for k in range(w2["ticks"]):
        T = e.trunk()
        q.append(obs[0, 6:6 + L].tolist()); y.append(float(T["h"][0])); up.append(T["up"][0].tolist()); x.append(T["p"][0].tolist())
        ph = 2 * math.pi * P["hz"] * k / RATE_HZ
        a = (S * math.sin(ph) + C * math.cos(ph) + O).clamp(-1, 1)
        obs, _, _, _, _ = e.step(a, autoreset=False)
    W = w2["trace"]
    qw, qb = np.array(W["leg_joint_pos"]), np.array(q)
    yw, yb = np.array(W["trunk_y"]) - W["trunk_y"][0], np.array(y) - y[0]
    u0 = np.array(up[0])
    tb = np.degrees(np.arccos(np.clip(np.array(up) @ u0, -1, 1)))
    tw = np.array(W["trunk_tilt_deg"])
    xb = np.array(x); xw = np.array(W["trunk_xz"])
    travel_b, travel_w = float(xb[-1, 0] - xb[0, 0]), float(xw[-1][0] - xw[0][0])
    p2p = lambda v: float(v[40:].max() - v[40:].min())
    off = (qb - qw).mean(0)
    checks = [
        check("joint static offset, worst joint (rad)", float(np.abs(off).max()), 0.04, "the servos' statics under the robot's own load"),
        check("joint dynamic RMS, offsets removed (rad)", rms((qw + off).ravel(), qb.ravel()), 0.03, "the motion; a one-tick lag of a 1 Hz, 0.4 rad swing is 0.044 rad RMS"),
        check("joint lag box3d vs World2, worst joint (ticks)", max((lag_ticks(qw[:, j], qb[:, j]) for j in range(L)), key=abs), 0.5, "servo lag"),
        check("joint amplitude ratio - 1, worst joint", max((p2p(qb[:, j]) / max(p2p(qw[:, j]), 1e-6) - 1 for j in range(L)), key=abs), 0.12, "servo gain, torque limits, loads"),
        check("trunk height RMS (mm)", rms(yw, yb) * 1000, 3.0, "the trunk's rise and fall: masses, contacts"),
        check("trunk tilt RMS (deg)", rms(tw, tb), 2.0, "the trunk's pitch and roll from its first frame"),
        check("trunk travel along +X (m, box3d - World2)", travel_b - travel_w, max(0.03, 0.3 * abs(travel_w)),
              "where the probe moves the robot: contact friction (each box's own, averaged with the floor), normal loads, slip (3 cm or 30 %)"),
    ]
    return checks, dict(joint_pos=q, trunk_y=y, trunk_tilt_deg=tb.round(4).tolist(), travel_m=travel_b, world2_travel_m=travel_w)


SKILL = Skill(
    name="locomotion", contract_skill="walk", make_env=make_env, spec=SPEC, model=MODEL, recipe=RECIPE,
    eval_sets={"nominal": {"dr_on": False}, "dr": {}}, evaluate=evaluate, manifest=manifest, extensions=("b3_duck_cuda",),
    description="locomotion of any robot designed in World2 (its model generated from the design)",
)
SKILL.prepare = prepare
SKILL.sim_match = sim_match
