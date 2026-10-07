"""Batched Open Duck Mini v2 walking environment on box3d-cuda (duck_sim.h), matched to World2's tasks/open-duck-walk.

The model (model/cuda-model.json) is exported from World2 itself (parts/open-duck-mini/export-cuda-model.mjs): the same
rigid groups, masses, inertias, joints, end stops, soles and IMU mount; the servo (STS3215, World2's bam-m1 servo step)
and IMU (World2's imu kind and signal chain) are reproduced step for step at World2's 1/120 s physics step.

Policy interface (the student; contract world2-learned-skill/1 skill 'walk', see export_onnx.py):
  observation per frame (39): imu_gyro (3, rad/s, sensor frame), imu_gravity (3, the IMU's onboard down estimate),
    leg_joint_pos (10, rad from the standing pose, from the servos' encoders), leg_joint_vel (10, finite difference of
    those over the control period, x 0.1), prev_action (10), command (3: vx m/s, vy m/s, yaw rate rad/s)
  history: the last HISTORY frames, newest first (HISTORY*39 inputs)
  action (10, [-1, 1]): leg joint targets = standing pose + ACTION_SCALE * a (URDF joint convention), clipped to limits
Control at 40 Hz (3 physics steps of 1/120 s).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from rl.common.cuda import load_ext
from rl.common.randomization import Spec, U
from rl.common.skill import EnvBase

HERE = Path(__file__).resolve().parent
MODEL_FILE = HERE / "model" / "cuda-model.json"
RATE_HZ = 40
N_OUTER = 3
ACTION_SCALE = 0.3
HISTORY = 3
FRAME = 39
OBS_SIZE = FRAME * HISTORY
LEGS = 10
STEPS_PER_RAD = 4096 / (2 * math.pi)


def load_duck_ext(device):
    return load_ext("b3_duck_cuda" if device.type == "cuda" else "b3_duck_cpu")


# ---------------------------------------------------------------- quaternions (xyzw), python floats / torch
def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


def qconj(q):
    return [-q[0], -q[1], -q[2], q[3]]


def qrot(q, v):
    x, y, z, w = q
    t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])]
    return [v[0] + w * t[0] + y * t[2] - z * t[1], v[1] + w * t[1] + z * t[0] - x * t[2], v[2] + w * t[2] + x * t[1] - y * t[0]]


def tq_rot(q, v):  # torch, last dim
    u, w = q[..., :3], q[..., 3:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def tq_mul(a, b):
    ax, ay, az, aw = a.unbind(-1); bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


# Randomization (rl/common/randomization.py). The duck samples these itself with a CPU generator (the spec's ranges);
# goal_extra_delay (extra servo command steps, integers lo..hi) and imu_delay (steps, rounded) are integer ranges.
SPEC = Spec(dict(
    mass_scale=U(0.85, 1.15), trunk_com_shift_m=0.01, friction=U(0.4, 1.2), damping_lin=U(0.5, 1.5), damping_ang=U(1.0, 3.0),
    kt=U(0.85, 1.15), R=U(0.85, 1.15), vin=U(6.8, 8.2), gain=U(0.85, 1.15), armature=U(0.75, 1.3), friction_base=U(0.5, 1.6),
    friction_viscous=U(0.5, 1.6), backlash_rad=U(0.0, 0.02), gear_stiffness=U(0.4, 1.3), gear_damping=U(0.6, 3.0), max_velocity=U(0.9, 1.1),
    goal_extra_delay=(0, 1), imu_delay=(0, 1), imu_noise=U(0.5, 2.0), imu_bias=U(0.0, 2.0), calibration_steps=3,
    push_interval_s=U(2.0, 5.0), push_dv=0.25, push_p=1.0, init_yaw=math.pi, init_vel=0.05,
    cmd_vx=U(0.0, 0.25), cmd_zero_p=0.15, episode_s=20.0, cmd_wz=0.4, cmd_wz_p=0.7, cmd_wz_change_s=3.0,
))
# DR off: the nominal robot (the sim-match and nominal evaluations)
NOMINAL = dict(mass_scale=(1, 1), trunk_com_shift_m=0, friction=(0.8, 0.8), damping_lin=(1, 1), damping_ang=(2, 2),
               kt=(1, 1), R=(1, 1), vin=(7.4, 7.4), gain=(1, 1), armature=(1, 1), friction_base=(1, 1), friction_viscous=(1, 1),
               backlash_rad=(0.0087, 0.0087), gear_stiffness=(1, 1), gear_damping=(1, 1), max_velocity=(1, 1), goal_extra_delay=(0, 0), imu_delay=(1, 1),
               imu_noise=(1, 1), imu_bias=(1, 1), calibration_steps=0, push_p=0.0, init_yaw=0.0, init_vel=0.0, cmd_wz=0.0)


def default_dr(**over):
    return SPEC.override(over)


class DuckWalkEnv(EnvBase):
    """E batched duck worlds (rl/common/skill.py env protocol). step(action) -> obs, priv, reward, done, info."""

    def __init__(self, n_envs, device="cpu", seed=0, dr=None, substeps=4, iterations=8, model_file=MODEL_FILE, dr_on=True, clock_hz=0.0, clearance_m=0.03, alive_bonus=0.1):
        # clock_hz > 0: the v2 interface: the frame adds foot_contact (2: the foot switches) and gait_clock (2: sin, cos of
        # 2*pi*clock_hz*t from the episode's start), and the reward follows a periodic reference gait (each foot swings up
        # to clearance_m in its half of the cycle, stands in the other; both stand when the command is zero)
        self.clock_hz, self.clearance, self.alive_bonus = float(clock_hz), float(clearance_m), float(alive_bonus)   # (alive_bonus: a standing-first curriculum stage)
        self.frame = FRAME + (4 if self.clock_hz > 0 else 0)
        self.obs_size = self.frame * HISTORY
        self.E, self.device = n_envs, torch.device(device)
        self.dr = (dr if isinstance(dr, Spec) else default_dr(**(dr or {}))) if dr_on else default_dr(**NOMINAL).override(dr if isinstance(dr, dict) else None)
        self.n = n_envs
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.gs = torch.Generator(device=self.device).manual_seed(seed + 7919)    # (the steps' randomness, on the device)
        self.ext = load_duck_ext(self.device)
        self.substeps, self.iterations = substeps, iterations
        m = json.loads(Path(model_file).read_text())
        self.m = m
        self._build_model(m)
        self._alloc()
        self.reset()

    # -------------------------------------------------------------- model
    def _build_model(self, m):
        d, B = self.device, m["bodies"]
        self.NB, self.NJ = len(B), len(m["joints"])
        com = [b["com"] for b in B]; fr = [b["frame"] for b in B]
        def local(i, p): return qrot(qconj(fr[i]), [p[k] - com[i][k] for k in range(3)])
        J = m["joints"]
        pa = [local(j["parent"], j["anchor"]) for j in J]; ca = [local(j["child"], j["anchor"]) for j in J]
        axis = [qrot(qconj(fr[j["parent"]]), j["axis"]) for j in J]
        ref = [qmul(qconj(fr[j["parent"]]), fr[j["child"]]) for j in J]
        lo = [min(j["stops_rad"]) for j in J]; hi = [max(j["stops_rad"]) for j in J]
        F = m["feet"]
        foot_c = [local(f["body"], f["center"]) for f in F]; foot_h = [f["half"] for f in F]; foot_q = [qmul(qconj(fr[f["body"]]), f["quaternion"]) for f in F]
        imu = m["imu"]
        imu_p = local(imu["body"], imu["position"]); imu_q = qmul(qconj(fr[imu["body"]]), imu["quaternion"])
        T = lambda x: torch.tensor(x, dtype=torch.float32, device=d).reshape(-1).contiguous()
        I = lambda x: torch.tensor(x, dtype=torch.int32, device=d).reshape(-1).contiguous()
        self.mf = [T(pa), T(ca), T(axis), T(ref), T(lo), T(hi), T(foot_c), T(foot_h), T(foot_q), T(imu_p), T(imu_q)]
        self.mi = [I([j["parent"] for j in J]), I([j["child"] for j in J]), I([f["body"] for f in F])]
        self.imu_b = imu["body"]
        self.dt = m["dt"]
        self.mass0 = torch.tensor([b["mass"] for b in B], device=d)
        self.inertia0 = torch.tensor([b["principal"] for b in B], device=d)
        self.com0 = torch.tensor(com, device=d); self.frame0 = torch.tensor(fr, device=d)
        # the trunk object's orientation from its body's principal frame: q_obj = q_body * q_po
        self.q_po = torch.tensor(qmul(qconj(fr[0]), B[0]["quaternion"]), device=d)
        self.trunk_y0 = com[0][1]
        # policy order -> joint index; signs, homes, limits (URDF convention)
        order = m["policy_order"]
        self.jidx = torch.tensor([next(i for i, j in enumerate(J) if j["name"] == n) for n in order], device=d)
        js = [next(j for j in J if j["name"] == n) for n in order]
        self.sign = torch.tensor([j["sign"] for j in js], dtype=torch.float32, device=d)
        self.home = torch.tensor([j["home"] for j in js], dtype=torch.float32, device=d)
        # URDF limits from the stops about the standing pose: urdf = home + sign*servo
        s_lo = torch.tensor([min(j["stops_rad"]) for j in js], device=d); s_hi = torch.tensor([max(j["stops_rad"]) for j in js], device=d)
        self.lim_lo = self.home + torch.minimum(self.sign * s_lo, self.sign * s_hi)
        self.lim_hi = self.home + torch.maximum(self.sign * s_lo, self.sign * s_hi)
        self.home_steps = torch.round(self.sign * self.home * STEPS_PER_RAD)
        sv = m["servo"]
        self.sp0 = torch.tensor([sv["vin"], sv["kt"], sv["R"], sv["kp_fw"] * sv["error_gain"] * sv["error_gain_ratio"], sv["max_pwm"], sv["armature"], sv["friction_base"],
                                 sv["friction_viscous"], sv["backlash_rad"], sv["gear_stiffness"], sv["gear_damping"], sv["gear_max_torque"], sv["max_velocity"], float(__import__("os").environ.get("DUCK_MOTOR_LATE", "0"))], device=d)
        P = imu["profile"]
        self.imu_prof = P

    def _alloc(self):
        E, d, NB, NJ = self.E, self.device, self.NB, self.NJ
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=d)
        self.state, self.inv_mass, self.inv_inertia = z(E, NB, 13), z(E, NB), z(E, NB, 3)
        self.servo, self.sp, self.jl, self.cl = z(E, NJ, 8), z(E, NJ, 14), z(E, NJ, 8), z(E, 2, 8, 3)
        self.mu, self.damp, self.imu, self.imu_par, self.imu_bias = z(E), z(E, 2), z(E, 32), z(E, 8), z(E, 6)
        self.goal_new, self.push, self.out = z(E, NJ), z(E, 6), z(E, 9 + NJ)
        self.rng = torch.randint(1, 2**31 - 1, (E,), generator=self.g, dtype=torch.int32).to(d)
        self.goal_delay = torch.ones(E, dtype=torch.int32, device=d)
        self.torque_on = torch.ones(E, dtype=torch.uint8, device=d)
        self.cal = z(E, NJ)                       # encoder calibration error (steps), policy order
        self.prev_action, self.q_obs_prev, self.cmd = z(E, LEGS), z(E, LEGS), z(E, 3)
        self.hist = z(E, HISTORY, self.frame)
        self.switch = torch.zeros(E, 2, dtype=torch.bool, device=d)
        self.t, self.next_push, self.air, self.contact_prev = z(E), z(E), z(E, 2), z(E, 2)
        self.stance = z(E, 2)
        self.yaw_ref = z(E)
        self.ep_ret, self.ep_len = z(E), z(E)
        self.last_touch_x = z(E, 2)
        self.base_mass, self.fric_now = z(E), z(E)

    def _u(self, lo, hi, n):
        return (lo + (hi - lo) * torch.rand(n, generator=self.g)).to(self.device)

    # -------------------------------------------------------------- reset
    def reset(self, mask=None):
        if mask is None: mask = torch.ones(self.E, dtype=torch.bool, device=self.device)
        idx = mask.nonzero().flatten()
        n = len(idx)
        if n == 0: return self.obs()
        dr, d = self.dr, self.device
        # initial pose: the standing pose, turned by a random yaw about the trunk, small random velocities
        yaw = self._u(-dr["init_yaw"], dr["init_yaw"], n)
        qy = torch.stack([torch.zeros_like(yaw), torch.sin(yaw / 2), torch.zeros_like(yaw), torch.cos(yaw / 2)], -1)
        pivot = self.com0[0].clone(); pivot[1] = 0
        rel = self.com0 - pivot
        p = tq_rot(qy[:, None].expand(n, self.NB, 4), rel[None].expand(n, -1, -1)) + pivot
        q = tq_mul(qy[:, None].expand(n, self.NB, 4), self.frame0[None].expand(n, -1, -1))
        v = (torch.randn(n, 1, 3, generator=self.g).to(d) * dr["init_vel"]).expand(n, self.NB, 3).clone(); v[..., 1] = 0
        st = torch.cat([p, q, v, torch.zeros(n, self.NB, 3, device=d)], -1)
        self.state[idx] = st
        self.yaw_ref[idx] = 0
        self.yaw_ref[idx] = self.trunk()["yaw"][idx]
        # masses and inertias
        ms = self._u(*dr["mass_scale"], n * self.NB).reshape(n, self.NB)
        mass = self.mass0[None] * ms
        self.inv_mass[idx] = 1.0 / mass
        self.inv_inertia[idx] = 1.0 / (self.inertia0[None] * ms[..., None])
        self.base_mass[idx] = mass.sum(-1)
        # servo params
        sp = self.sp0[None, None].repeat(n, self.NJ, 1)
        for k, key in [(1, "kt"), (2, "R"), (3, "gain"), (5, "armature"), (6, "friction_base"), (7, "friction_viscous"), (9, "gear_stiffness"), (12, "max_velocity")]:
            sp[..., k] *= self._u(*dr[key], n * self.NJ).reshape(n, self.NJ)
        sp[..., 0] = self._u(*dr["vin"], n)[:, None]            # one battery per duck
        sp[..., 8] = self._u(*dr["backlash_rad"], n * self.NJ).reshape(n, self.NJ)
        sp[..., 10] *= self._u(*dr["gear_damping"], n * self.NJ).reshape(n, self.NJ)
        sp[..., 10] = torch.maximum(sp[..., 10], sp[..., 9] * self.dt / 1.25)   # (World2's ring check: k·DT/c <= 1.25)
        self.sp[idx] = sp
        self.servo[idx] = 0; self.jl[idx] = 0; self.cl[idx] = 0
        self.mu[idx] = self._u(*dr["friction"], n)
        self.fric_now[idx] = self.mu[idx]
        self.damp[idx] = torch.stack([self._u(*dr["damping_lin"], n), self._u(*dr["damping_ang"], n)], -1)
        # imu: World2 signal chain parameters per unit
        P = self.imu_prof
        sg, sa = P["gyro"]["sigma"] * self._u(*dr["imu_noise"], n), P["accel"]["sigma"] * self._u(*dr["imu_noise"], n)
        ag = 1 - math.exp(-2 * math.pi * P["gyro"]["bandwidth_hz"] * self.dt); aa = 1 - math.exp(-2 * math.pi * P["accel"]["bandwidth_hz"] * self.dt)
        isg, isa = sg * math.sqrt((2 - ag) / ag), sa * math.sqrt((2 - aa) / aa)
        delay = torch.round(self._u(dr["imu_delay"][0] - 0.49, dr["imu_delay"][1] + 0.49, n))
        self.imu_par[idx] = torch.stack([isg, isa, torch.full_like(sg, ag), torch.full_like(sg, aa), delay, torch.full_like(sg, P["fusion_tau_s"]), sg, sa], -1)
        bscale = self._u(*dr["imu_bias"], n)[:, None]
        self.imu_bias[idx] = torch.randn(n, 6, generator=self.g).to(d) * torch.tensor([P["gyro"]["bias_sigma"]] * 3 + [P["accel"]["bias_sigma"]] * 3, device=d) * bscale
        self.imu[idx] = 0
        self.cal[idx] = torch.round((torch.rand(n, self.NJ, generator=self.g).to(d) - 0.5) * 2 * dr["calibration_steps"])
        # commands
        vx = self._u(*dr["cmd_vx"], n)
        vx = torch.where(torch.rand(n, generator=self.g).to(d) < dr["cmd_zero_p"], torch.zeros_like(vx), vx)
        wz = self._u(-dr["cmd_wz"], dr["cmd_wz"], n) * (torch.rand(n, generator=self.g).to(d) < dr["cmd_wz_p"]).float()
        self.cmd[idx] = torch.stack([vx, torch.zeros_like(vx), wz], -1)
        self.prev_action[idx] = 0; self.t[idx] = 0; self.air[idx] = 0; self.stance[idx] = 0; self.contact_prev[idx] = 0
        self.next_push[idx] = self._u(*dr["push_interval_s"], n)
        self.ep_ret[idx] = 0; self.ep_len[idx] = 0
        # the servos hold the standing pose; the IMU's fusion starts from the true down direction (as after power-on rest)
        self.servo[idx, :, 3] = self._goal_from_target(self.home[None].expand(n, -1), idx)
        self.servo[idx, :, 2] = 0
        qs = tq_mul(self.state[idx, self.imu_b, 3:7], self.mf[10][None].expand(n, 4))
        qsc = torch.cat([-qs[:, :3], qs[:, 3:]], -1)
        self.imu[idx, 10:13] = tq_rot(qsc, torch.tensor([0.0, -1.0, 0.0], device=d).expand(n, 3))
        self.imu[idx, 13] = 1
        self.q_obs_prev[idx] = self._q_obs()[idx]
        fr = self._frame()
        self.hist[idx] = fr[idx][:, None].expand(-1, HISTORY, -1)
        return self.obs()

    # -------------------------------------------------------------- conversions (World2's program: duck.mjs toSteps / toRad)
    def _goal_from_target(self, target, idx=None):
        """URDF joint targets (policy order, all 14) -> servo goals (rad, servo sense from the standing pose), joint order."""
        cal = self.cal if idx is None else self.cal[idx]
        steps = torch.round(self.sign * target * STEPS_PER_RAD)
        g_pol = (steps - self.home_steps - cal) / STEPS_PER_RAD
        out = torch.zeros(target.shape[0], self.NJ, device=self.device)
        out[:, self.jidx] = g_pol
        return out

    def _coord(self):
        """servo-sense joint angles from the standing pose (policy order) from the bodies."""
        st = self.state
        J = self.m["joints"]
        # recompute like the kernel's joint_geometry coord, in torch
        jp, jc = self.mi[0].long(), self.mi[1].long()
        qp, qc = st[:, jp, 3:7], st[:, jc, 3:7]
        ref = self.mf[3].reshape(-1, 4)
        refc = torch.cat([-ref[:, :3], ref[:, 3:]], -1)
        qpc = torch.cat([-qp[..., :3], qp[..., 3:]], -1)
        qd = tq_mul(tq_mul(qpc, qc), refc[None].expand_as(qc))
        qd = torch.where(qd[..., 3:] < 0, -qd, qd)
        s = qd[..., :3].norm(dim=-1, keepdim=True)
        ang = 2 * torch.atan2(s, qd[..., 3:])
        rv = torch.where(s > 1e-9, qd[..., :3] / s.clamp_min(1e-12) * ang, 2 * qd[..., :3])
        coord = (rv * self.mf[2].reshape(-1, 3)[None]).sum(-1)
        return coord[:, self.jidx]

    def _q_obs(self):
        """the program's leg joint angles from the encoders (URDF convention minus the standing pose), policy order (legs)."""
        coord = self._coord()
        present = torch.round(2048 + self.home_steps + self.cal + coord * STEPS_PER_RAD) + torch.randint(-1, 2, coord.shape, generator=self.gs, device=self.device)
        q = self.sign * (present - 2048) / STEPS_PER_RAD
        return (q - self.home)[:, :LEGS]

    def _frame(self):
        gyro, grav = self.imu[:, 28:31], self.imu[:, 10:13]
        q = self._q_obs()
        qd = (q - self.q_obs_prev) * RATE_HZ * 0.1
        return torch.cat([gyro, grav, q, qd, self.prev_action, self.cmd] + self._extra(), -1)

    def _extra(self):
        if self.clock_hz <= 0:
            return []
        # foot switches (World2 contact_switch: closes over 2 N, opens under 1.5 N; force sigma 0.2 N) and the gait clock
        f = self.out[:, 2:4] + 0.2 * torch.randn(self.E, 2, generator=self.gs, device=self.device)
        self.switch = torch.where(self.switch, f > 1.5, f > 2.0)
        ph = 2 * math.pi * self.clock_hz * self.t
        return [self.switch.float(), torch.stack([torch.sin(ph), torch.cos(ph)], -1)]

    def _extra_after_tick(self):
        # (the clock of the observation taken after a tick: the phase the next action is chosen at, t already advanced)
        return self._extra()

    def obs(self):
        return self.hist.reshape(self.E, -1).clone()    # (a copy: hist is updated in place)

    def _step_physics(self):
        ef = [self.state, self.inv_mass, self.inv_inertia, self.servo, self.sp, self.jl, self.cl, self.mu, self.damp, self.imu, self.imu_par, self.imu_bias, self.goal_new, self.push, self.out]
        self.ext.step(self.mf, self.mi, self.imu_b, self.dt, self.substeps, self.iterations, 0.2, 0.0005, 0.2, 0.0005, 0.5, ef, self.rng, self.goal_delay, self.torque_on, N_OUTER)

    # -------------------------------------------------------------- trunk measurements (privileged)
    def trunk(self):
        st = self.state[:, 0]
        q_obj = tq_mul(st[:, 3:7], self.q_po[None].expand(self.E, 4))
        up = tq_rot(q_obj, torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(self.E, 3))
        fwd = tq_rot(q_obj, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.E, 3))
        yaw = torch.atan2(-fwd[:, 2], fwd[:, 0])
        # velocities in the command frame: the heading the duck had at the episode's start (a circle earns no forward
        # tracking: an earlier policy walked 116 deg round in 10 s with tracking in the current heading frame)
        c, s = torch.cos(self.yaw_ref), torch.sin(self.yaw_ref)
        v = st[:, 7:10]; w = st[:, 10:13]
        vx = c * v[:, 0] - s * v[:, 2]; vy = s * v[:, 0] + c * v[:, 2]   # x forward, y = the duck's right (World2 +z at yaw 0)
        tilt = torch.acos(up[:, 1].clamp(-1, 1))
        return dict(p=st[:, :3], v=v, w=w, up=up, yaw=yaw, vx=vx, vy=vy, wz=w[:, 1], tilt=tilt, h=st[:, 1])

    def priv(self):
        T = self.trunk()
        return torch.cat([T["v"], T["w"], T["up"], T["h"][:, None] - self.trunk_y0, self.out[:, 0:2], self.out[:, 7:9], self.mu[:, None], self.base_mass[:, None] - 2.0,
                          self.air, self.sp[:, :1, 0] / 7.4 - 1], -1)

    PRIV_SIZE = 3 + 3 + 3 + 1 + 2 + 2 + 1 + 1 + 2 + 1
    priv_size, act_size = PRIV_SIZE, LEGS

    # -------------------------------------------------------------- the shared pipeline's protocol
    dev = property(lambda self: self.device)

    def observe(self):
        return self.obs(), self.priv()

    def iteration_info(self):
        T = self.trunk()
        return dict(vx=round(T["vx"].mean().item(), 3), cmd=round(self.cmd[:, 0].mean().item(), 3))

    # -------------------------------------------------------------- step
    def step(self, action, autoreset=True):
        obs, priv, reward, done, info = self._run_step(action)
        if autoreset and done.any():
            self.reset(done)
            obs, priv = self.obs(), self.priv()
        return obs, priv, reward, done, info

    def _step(self, action):
        E, d, dr = self.E, self.device, self.dr
        a = action.clamp(-1, 1)
        target = self.home[None].repeat(E, 1)
        target[:, :LEGS] = (self.home[None, :LEGS] + ACTION_SCALE * a).clamp(self.lim_lo[None, :LEGS], self.lim_hi[None, :LEGS])
        self.goal_new.copy_(self._goal_from_target(target))
        # (the step's randomness from a generator on the device, no host branches: a captured CUDA graph replays it)
        R = lambda: torch.rand(E, generator=self.gs, device=d)
        self.goal_delay.copy_((1 + torch.randint(dr["goal_extra_delay"][0], dr["goal_extra_delay"][1] + 1, (E,), generator=self.gs, device=d)).int())
        # pushes: a random horizontal velocity change of the trunk (impulse = dv * duck mass)
        self.push.zero_()
        due = (self.t >= self.next_push) & (R() < dr["push_p"])
        ang, dv = R() * 2 * math.pi, R() * dr["push_dv"]
        imp = (dv * self.base_mass)[:, None] * torch.stack([torch.cos(ang), torch.zeros_like(ang), torch.sin(ang)], -1)
        self.push[:, :3] = torch.where(due[:, None], imp, torch.zeros_like(imp))
        lo, hi = dr["push_interval_s"]
        self.next_push = torch.where(due, self.t + lo + (hi - lo) * R(), self.next_push)
        self._step_physics()
        self.t += 1.0 / RATE_HZ
        # a yaw-rate command (a heading controller on the robot sends these): the command frame turns with it, and it
        # changes now and then
        self.yaw_ref = self.yaw_ref + self.cmd[:, 2] / RATE_HZ
        if dr["cmd_wz"] > 0:
            ch = R() < 1.0 / (dr["cmd_wz_change_s"] * RATE_HZ)
            nw = (-dr["cmd_wz"] + 2 * dr["cmd_wz"] * R()) * (R() < dr["cmd_wz_p"]).float()
            self.cmd[:, 2] = torch.where(ch, nw, self.cmd[:, 2])
        q_obs = self._q_obs()
        fr = torch.cat([self.imu[:, 28:31], self.imu[:, 10:13], q_obs, (q_obs - self.q_obs_prev) * RATE_HZ * 0.1, a, self.cmd] + self._extra_after_tick(), -1)
        self.q_obs_prev = q_obs
        self.hist = torch.cat([fr[:, None], self.hist[:, :-1]], 1)
        reward, terms, done, timeout = self._reward(a)
        self.prev_action = a.clone()
        self.ep_ret += reward; self.ep_len += 1
        info = dict(terms=terms, timeout=timeout, ep_ret=self.ep_ret.clone(), ep_len=self.ep_len.clone())
        return self.obs(), self.priv(), reward, done, info

    def _reward(self, a):
        T, dt = self.trunk(), 1.0 / RATE_HZ
        cmd = self.cmd
        contact = self.out[:, 0:2]
        # gait bookkeeping on real lifts: a foot is in the air only with its whole sole over 6 mm (World2's judge counts 4 mm)
        lifted = (self.out[:, 7:9] > 0.006).float()
        touch = (lifted < 0.5) & (self.contact_prev > 0.5)        # contact_prev holds the previous `lifted`
        self.air += dt * lifted
        moving = (cmd[:, 0].abs() > 0.02).float()
        r_air = ((self.air - 0.15).clamp(max=0.3) * touch.float()).sum(-1) * moving
        self.air = torch.where(touch, torch.zeros_like(self.air), self.air)
        self.stance = torch.where(lifted > 0.5, torch.zeros_like(self.stance), self.stance + dt)
        self.contact_prev = lifted.clone()
        single = (lifted.sum(-1) == 1).float()
        swing_h = (self.out[:, 7:9] * lifted).clamp(max=0.03)
        terms = dict(
            track_vx=2.0 * torch.exp(-(T["vx"] - cmd[:, 0]) ** 2 / 0.01),
            track_vy=1.0 * torch.exp(-(T["vy"] - cmd[:, 1]) ** 2 / 0.01),
            track_wz=1.5 * torch.exp(-(T["wz"] - cmd[:, 2]) ** 2 / 0.02),
            alive=torch.full_like(T["vx"], self.alive_bonus),
            upright=-2.0 * (1 - T["up"][:, 1]),
            height=-2000.0 * (T["h"] - self.trunk_y0).clamp(max=0) ** 2,
            air=3.0 * r_air,
            single=0.6 * single * moving,
            stance=-1.0 * (self.stance > 0.6).float().sum(-1) * moving,
            clearance=10.0 * swing_h.sum(-1) * moving,
            still=-0.5 * (1 - moving) * (self.out[:, 4:6].sum(-1) + (contact < 0.5).float().sum(-1) * 0.5),
            action_rate=-0.02 * ((a - self.prev_action) ** 2).sum(-1),
            power=-0.002 * self.out[:, 6],
            slip=-0.5 * (self.out[:, 4:6] * contact).sum(-1),
            ang_vel=-0.05 * (T["w"][:, 0] ** 2 + T["w"][:, 2] ** 2),
            hip_pose=-0.3 * (a[:, [0, 1, 5, 6]] ** 2).sum(-1),
            heading=-2.0 * torch.atan2(torch.sin(T["yaw"] - self.yaw_ref), torch.cos(T["yaw"] - self.yaw_ref)) ** 2,
        )
        if self.clock_hz > 0:
            ph = 2 * math.pi * self.clock_hz * (self.t - dt)
            swing = torch.stack([torch.sin(ph), torch.sin(ph + math.pi)], -1).clamp(min=0) * moving[:, None]
            h_ref = self.clearance * swing
            h = self.out[:, 7:9].clamp(min=0)
            stance_ref = swing <= 0
            for k in ("air", "single", "clearance", "stance"):
                terms.pop(k, None)
            terms["foot_height"] = 2.0 * torch.exp(-((h - h_ref) ** 2).sum(-1) / 0.008 ** 2)
            terms["contact_match"] = 0.6 * (stance_ref == (contact > 0.5)).float().mean(-1)
        reward = sum(terms.values()) * dt * 10
        fell = (T["h"] < 0.7 * self.trunk_y0) | (T["tilt"] > math.radians(40)) | ~torch.isfinite(self.state).all(-1).all(-1)
        timeout = self.t >= self.dr["episode_s"]
        reward = torch.where(fell, torch.full_like(reward, -2.0), reward)
        reward = torch.nan_to_num(reward, nan=-5.0)
        done = fell | timeout
        return reward, terms, done, timeout & ~fell
