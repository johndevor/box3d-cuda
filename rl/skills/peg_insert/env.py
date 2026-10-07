"""Batched peg-in-hole insertion on box3d-cuda, built to World2's learned-skill contract `world2-learned-skill/1`.

Crude on purpose: a square box peg (side = the family's diameter) into a square hole of static box walls with 45 degree
chamfer boxes, all box3d-cuda OBBs with persistent manifolds (``manifold_step``, one kernel call per controller
substep). The contract's contact-tier controller (World2 operations/compliant-insert.mjs) is applied here in torch
between kernel calls: a 2 kg virtual mass driven at its leading point (the tip) by F = k e - 2 sqrt(k m) v_tip, a
rotational spring-damper T = k_rot theta - 2 sqrt(k_rot I) w, the force acting at the tip (so it also turns the body
about its centre), gravity compensated.

Observations are exactly those of research/skills-rl/skillsrl/insert_env.py (World2's runtime packs the same 63
values), including the beliefs' errors and the wrist F/T model (noise, low-pass, delay, gain error, drift, tare). The
teacher additionally sees privileged truth (``priv``).

Physics runs in millimetres (lengths x 1000, so box3d's absolute epsilons sit at nanometres); the controller is SI.
Frames: z up, insertion axis -z, mouth at z = 0, target (tip at full depth) at z = -depth. Task frame (contract, z-up
exports): x_t = (0,-1,0), y_t = (-1,0,0), z_t = (0,0,-1).
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch

from rl.common.cuda import load_ext
from rl.common.randomization import Bern, IntU, LogU, Spec, U
from rl.common.skill import EnvBase, cvec, InPlaceDict

ROOT = Path(__file__).resolve().parents[3]

OBS_FIELDS = ["target_pos", "target_rot", "part_vel", "wrench", "wrench_sigma", "target_sigma", "in_hand_sigma",
              "features", "progress", "depth", "prev_action", "time"]
OBS_SIZE = 57
PRIV_SIZE = 23
ACTION_SCALE = dict(pos_m=0.001, rot_rad=0.01, k_trans_n_per_m=(300.0, 8000.0), k_rot_nm_per_rad=(0.5, 10.0))
LIMITS = dict(max_lateral_m=0.004, max_axial_m=0.003, max_tilt_rad=0.06)
RATE_HZ = 30
VMASS, VINERT = 2.0, 0.002
# wrist F/T profiles (World2 operations/sensors/wrist-ft.mjs): sigma F, sigma T, bandwidth Hz, delay s, range
FT_PROFILES = torch.tensor([[2.5, 0.1, 30.0, 0.008, 100.0],      # ur-e-series-ur12e
                            [1.75, 0.1, 30.0, 0.008, 50.0],      # ur-e-series-ur5e
                            [0.1, 0.005, 40.0, 0.012, 300.0]])   # robotiq-ft300s
N_STATIC = 13
B = 1 + N_STATIC


SPEC = Spec(dict(
    d_mm=LogU(3.0, 16.0), length_over_d=U(2.5, 6.0), depth_over_d=U(0.8, 2.0),
    clearance_mm=LogU(0.02, 1.0),            # radial
    host_chamfer_mm=U(0.2, 1.2), part_chamfer_mm=U(0.1, 1.0),
    friction=U(0.1, 0.6), friction_groups=8,
    density=U(1200.0, 8500.0),
    start_height_mm=U(1.0, 15.0), start_lateral_mm=U(0.0, 3.0), start_tilt_deg=U(0.0, 3.5),
    target_sigma_mm=U(0.1, 1.0), target_sigma_rot_deg=U(0.05, 1.5), in_hand_sigma_mm=U(0.05, 0.5),
    target_err_z_mm=U(0.02, 0.2), in_hand_err_z_mm=U(0.02, 0.15), in_hand_rot_deg=0.2,
    max_force_n=U(20.0, 60.0),
    ft_profile=IntU(0, 3),                   # FT_PROFILES row
    ft_tare_offset=1.0,                      # the filter's state at the tare (one noisy reading: an offset of ~sigma); 0 in the sim-match
    ft_extra_delay_s=U(0.0, 0.012), ft_gain_err=U(0.005, 0.01), ft_noise_mult=U(1.0, 1.5), flange_h=U(0.08, 0.2),
    action_latency_p=Bern(0.3), motor_lag_s=U(0.0, 0.03), vmass=U(1.7, 2.3),
    substeps_per_world_step=4, solver_iterations=8, timeout_s=6.0,
    pose_scale=1.0,                          # multiplies start lateral/tilt and target error (harder eval sets)
    belief_error_scale=1.0,                  # multiplies the in-hand error and the target's axial error (0 in the sim-match)
))


def default_cfg(**over):
    return SPEC.override(over)


# ---------------------------------------------------------------- quaternion helpers (xyzw, last dim)
def qmul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qconj(q):
    return torch.cat([-q[..., :3], q[..., 3:]], -1)


def qrot(q, v):
    u, w = q[..., :3], q[..., 3:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def rotvec_of(q):
    q = torch.where(q[..., 3:] < 0, -q, q)
    s = q[..., :3].norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(s, q[..., 3:])
    return torch.where(s > 1e-9, q[..., :3] / s.clamp_min(1e-12) * ang, 2 * q[..., :3])


def quat_of(rv):
    a = rv.norm(dim=-1, keepdim=True)
    k = torch.where(a > 1e-9, torch.sin(a / 2) / a.clamp_min(1e-12), torch.full_like(a, 0.5))
    return torch.cat([rv * k, torch.cos(a / 2)], -1)


def rotvec_between(qf, qt):
    return rotvec_of(qmul(qt, qconj(qf)))


def to_task(v):  # its own inverse
    return torch.stack([-v[..., 1], -v[..., 0], -v[..., 2]], -1)


# ---------------------------------------------------------------- physics backends
def load_cuda_ext():
    return load_ext("b3_manifold")


def load_cpu_reference():
    spec = importlib.util.spec_from_file_location("b3ref", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b3ref"] = mod
    spec.loader.exec_module(mod)
    import importlib as il
    return il.import_module("b3ref.manifold_reference"), il.import_module("b3ref.sat_reference")


def uni(g, n, lo_hi, device):
    lo, hi = lo_hi
    return lo + (hi - lo) * torch.rand(n, generator=g, device=device)


def loguni(g, n, lo_hi, device):
    lo, hi = lo_hi
    return torch.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * torch.rand(n, generator=g, device=device))


class PegInsertBatch(EnvBase):
    """N independent worlds (rl/common/skill.py env protocol). step(action[N,9]) -> obs[N,57], priv[N,PRIV], reward[N],
    done[N], info (dict of [N])."""
    obs_size, priv_size, act_size = OBS_SIZE, PRIV_SIZE, 9

    def __init__(self, n, device="cuda", seed=0, cfg=None):
        self.n, self.dev = n, torch.device(device)
        self.cfg = cfg if isinstance(cfg, Spec) else default_cfg(**(cfg or {}))
        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(seed)
        self.per = 120 // RATE_HZ
        self.sub = self.cfg["substeps_per_world_step"]
        self.h = 1.0 / 120 / self.sub
        if self.dev.type == "cuda":
            self.ext = load_cuda_ext()
        else:
            self.mref, self.sref = load_cpu_reference()
        f = lambda *s: torch.zeros(*s, device=self.dev)
        self.state = f(n, B, 13)
        self.state[:, :, 6] = 1
        self.inv_mass = f(n, B)
        self.half = f(n, B, 3) + 1.0
        self.inv_inertia = f(n, B, 3)
        self.pairs = torch.tensor([[0, j] for j in range(1, B)], dtype=torch.int64, device=self.dev)
        self.cache_ids = torch.zeros(n, B - 1, 4, dtype=torch.int64, device=self.dev)
        self.cache_imp = f(n, B - 1, 4, 3)
        G = self.cfg["friction_groups"]
        self.group_of = torch.arange(n, device=self.dev) * G // n
        self.group_slices = [(i * n // G, (i + 1) * n // G) for i in range(G)]
        self.group_mu = self.cfg.draw("friction", self.g, G, self.dev)
        self.group_mu_f = self.group_mu.tolist()       # (host floats for the kernel calls: no sync inside a captured step)
        self.p = InPlaceDict()
        self.ft_buf = f(n, 4, 6)
        self.ft_ptr = 0
        self.all = torch.ones(n, dtype=torch.bool, device=self.dev)
        self.reset(self.all)

    # ------------------------------------------------------------ sampling and geometry
    def resample_friction(self):
        self.group_mu = self.cfg.draw("friction", self.g, len(self.group_slices), self.dev)
        self.group_mu_f = self.group_mu.tolist()
        self.graph_stale()

    # ------------------------------------------------------------ the shared pipeline's protocol
    code_names = property(lambda self: CODE_NAMES)

    def on_iteration(self, phase=None, it=0, frac=0.0):
        self.resample_friction()

    def observe(self):
        return self.obs(update=False), self.priv()

    def step(self, action, autoreset=True):
        obs, r, done, info = self._run_step(action)
        if autoreset and done.any():
            self.reset(done)
            obs = torch.where(done[:, None], self.obs(update=False), obs)
        return obs, self.priv(), r, done, info

    def _sample(self, idx):
        c, g, d, m = self.cfg, self.g, self.dev, len(idx)
        D = lambda k: c.draw(k, g, m, d)
        P = {}
        P["d"] = D("d_mm") * 1e-3
        P["L"] = P["d"] * D("length_over_d")
        P["clear"] = D("clearance_mm") * 1e-3
        P["depth"] = torch.minimum(P["d"] * D("depth_over_d"), 0.6 * P["L"])
        P["hch"] = torch.minimum(D("host_chamfer_mm") * 1e-3, 0.4 * P["depth"])
        P["pch"] = torch.minimum(D("part_chamfer_mm") * 1e-3, 0.2 * P["d"])
        P["mass"] = D("density") * math.pi * P["d"] ** 2 / 4 * P["L"]
        P["vmass"] = D("vmass")
        ps = c["pose_scale"]
        P["h0"] = D("start_height_mm") * 1e-3
        P["lat0"] = D("start_lateral_mm") * 1e-3 * ps
        P["tilt0"] = torch.deg2rad(D("start_tilt_deg")) * ps
        P["tsp"] = D("target_sigma_mm") * 1e-3
        P["tsr"] = torch.deg2rad(D("target_sigma_rot_deg"))
        P["tsz"] = D("target_sigma_mm") * 1e-3
        P["ihs"] = D("in_hand_sigma_mm") * 1e-3
        P["ihz"] = D("in_hand_sigma_mm") * 1e-3
        P["maxF"] = D("max_force_n")
        P["ft"] = D("ft_profile")
        P["ftd"] = D("ft_extra_delay_s")
        P["gain_err"] = D("ft_gain_err")
        P["noise_mult"] = D("ft_noise_mult")
        P["flange"] = D("flange_h")
        P["latency"] = D("action_latency_p")
        P["lag"] = D("motor_lag_s")
        # beliefs' errors
        rn = lambda *s: torch.randn(*s, generator=g, device=d)
        tez = torch.minimum(P["tsz"], D("target_err_z_mm") * 1e-3)
        es = c["belief_error_scale"]
        P["terr"] = torch.stack([rn(m) * P["tsp"] * ps, rn(m) * P["tsp"] * ps, rn(m) * tez * es], -1)
        P["qT_b"] = quat_of(rn(m, 3) * P["tsr"][:, None] * ps)
        ihz = torch.minimum(P["ihz"], D("in_hand_err_z_mm") * 1e-3)
        P["iherr"] = torch.stack([rn(m) * P["ihs"], rn(m) * P["ihs"], rn(m) * ihz], -1) * es
        P["ihq"] = quat_of(rn(m, 3) * math.radians(c["in_hand_rot_deg"]) * es)
        ang = uni(g, m, (0, 2 * math.pi), d)
        P["tip0"] = torch.stack([P["lat0"] * torch.cos(ang), P["lat0"] * torch.sin(ang), P["h0"]], -1)
        ta = uni(g, m, (0, 2 * math.pi), d)
        P["q0"] = quat_of(torch.stack([torch.cos(ta), torch.sin(ta), torch.zeros_like(ta)], -1) * P["tilt0"][:, None])
        return P

    def _geometry(self, idx):
        """Static boxes (mm) for worlds idx: per side a lower wall, an upper wall and a 45 degree chamfer box; a floor."""
        P = self.p
        mm = 1000.0
        s, c, D, ch = P["d"][idx] * mm, P["clear"][idx] * mm, P["depth"][idx] * mm, P["hch"][idx] * mm
        a = s / 2 + c
        T = torch.full_like(a, 5.0)
        W = a + ch + T
        m = len(idx)
        pos = torch.zeros(m, N_STATIC, 3, device=self.dev)
        half = torch.zeros(m, N_STATIC, 3, device=self.dev)
        quat = torch.zeros(m, N_STATIC, 4, device=self.dev)
        quat[..., 3] = 1
        r2 = math.sqrt(0.5)
        for k in range(4):
            phi = k * math.pi / 2
            cz, sz = math.cos(phi), math.sin(phi)
            qz = cvec([0, 0, math.sin(phi / 2), math.cos(phi / 2)], self.dev).expand(m, 4)
            # template for the +x side (x outward), rotated about z by phi
            local = [
                (torch.stack([a + (ch + T) / 2, 0 * a, -(D + ch) / 2], -1), torch.stack([(ch + T) / 2, W, (D - ch) / 2], -1), None),
                (torch.stack([a + ch + T / 2, 0 * a, -ch / 2], -1), torch.stack([T / 2, W, ch / 2], -1), None),
                (torch.stack([a + ch, 0 * a, -ch], -1), torch.stack([ch * r2, W, ch * r2], -1),
                 cvec([0, math.sin(math.pi / 8), 0, math.cos(math.pi / 8)], self.dev).expand(m, 4)),
            ]
            for j, (p, hf, q) in enumerate(local):
                i = 3 * k + j
                pos[:, i] = torch.stack([cz * p[:, 0] - sz * p[:, 1], sz * p[:, 0] + cz * p[:, 1], p[:, 2]], -1)
                half[:, i] = hf
                quat[:, i] = qmul(qz, q) if q is not None else qz
        pos[:, 12] = torch.stack([0 * a, 0 * a, -D - T / 2], -1)
        half[:, 12] = torch.stack([W, W, T / 2], -1)
        st = self.state
        st[idx, 1:, 0:3] = pos
        st[idx, 1:, 3:7] = quat
        st[idx, 1:, 7:] = 0
        self.half[idx, 1:] = half
        self.inv_mass[idx, 1:] = 0
        self.inv_inertia[idx, 1:] = 0

    def reset(self, mask):
        idx = mask.nonzero().squeeze(-1)
        if len(idx) == 0:
            return
        P = self._sample(idx)
        for k, v in P.items():
            if k not in self.p:
                self.p[k] = torch.zeros((self.n,) + tuple(v.shape[1:]), dtype=v.dtype, device=self.dev)
            self.p[k][idx] = v
        self._geometry(idx)
        p, m = self.p, len(idx)
        mm = 1000.0
        # the peg: dynamic, side d, length L, centre L/2 above the tip along its axis
        q0, tip0 = p["q0"][idx], p["tip0"][idx]
        L = p["L"][idx]
        ctr = tip0 + qrot(q0, torch.stack([0 * L, 0 * L, L / 2], -1))
        self.state[idx, 0, 0:3] = ctr * mm
        self.state[idx, 0, 3:7] = q0
        self.state[idx, 0, 7:] = 0
        hs = torch.stack([p["d"][idx] / 2, p["d"][idx] / 2, L / 2], -1)
        self.half[idx, 0] = hs * mm
        M = p["vmass"][idx]
        # inertia: the part's own box inertia about its centre + the contract's virtual 0.002 kg m^2
        Ib = p["mass"][idx, None] / 3 * (hs[:, [1, 0, 0]] ** 2 + hs[:, [2, 2, 1]] ** 2) + VINERT
        self.I = torch.zeros(self.n, device=self.dev) if not hasattr(self, "I") else self.I
        self.I[idx] = Ib.max(-1).values
        self.inv_mass[idx, 0] = 1.0 / M
        self.inv_inertia[idx, 0] = 1.0 / (Ib * 1e6)
        self.cache_ids[idx] = 0
        self.cache_imp[idx] = 0
        # beliefs
        ax = cvec([0.0, 0.0, -1.0], self.dev)
        self.ax = ax
        D = p["depth"][idx]
        true_target = torch.stack([0 * D, 0 * D, -D], -1)
        self._set("b_target", idx, true_target + p["terr"][idx])
        self._set("b_entry", idx, true_target + p["terr"][idx] - ax * D[:, None])
        self._set("ref_tip", idx, tip0.clone())
        self._set("ref_q", idx, q0.clone())
        self._set("ref_eff", idx, tip0.clone())
        self._set("k", idx, torch.full((m,), 1500.0, device=self.dev))
        self._set("krot", idx, torch.full((m,), 3.0, device=self.dev))
        btip, bq = self._believed(idx)
        self._set("last_btip", idx, btip)
        self._set("last_bq", idx, bq)
        r = btip - self.v["b_entry"][idx]
        self._set("lat0", idx, r - ax * (r @ ax)[:, None])
        self._set("prev_action", idx, torch.zeros(m, 9, device=self.dev))
        self._set("pending", idx, torch.zeros(m, 9, device=self.dev))
        self._set("t", idx, torch.zeros(m, device=self.dev))
        self._set("over_since", idx, torch.full((m,), -1.0, device=self.dev))
        self._set("stall_p", idx, torch.full((m,), 1e9, device=self.dev))
        self._set("stall_since", idx, torch.zeros(m, device=self.dev))
        self._set("peak_push", idx, torch.zeros(m, device=self.dev))
        self._set("contact_w", idx, torch.zeros(m, 6, device=self.dev))
        # F/T sensor
        prof = FT_PROFILES.to(self.dev)[p["ft"][idx]]
        dt = 1 / 120
        alpha = 1 - torch.exp(-2 * math.pi * prof[:, 2] * dt)
        self._set("ft_alpha", idx, alpha)
        sw = torch.stack([prof[:, 0]] * 3 + [prof[:, 1]] * 3, -1) * torch.sqrt((2 - alpha) / alpha)[:, None] * p["noise_mult"][idx, None]
        self._set("ft_sw", idx, sw)
        self._set("ft_gain", idx, 1 + torch.randn(m, 6, generator=self.g, device=self.dev) * p["gain_err"][idx, None])
        self._set("ft_delay", idx, torch.round((prof[:, 3] + p["ftd"][idx]) / dt).long().clamp(1, 3))
        self._set("ft_range", idx, prof[:, 4])
        self._set("ft_sig", idx, prof[:, :2])
        self._set("ft_f", idx, torch.randn(m, 6, generator=self.g, device=self.dev) * prof[:, [0, 0, 0, 1, 1, 1]] * self.cfg["ft_tare_offset"])
        self._set("ft_drift", idx, torch.zeros(m, 6, device=self.dev))
        self._set("ft_tare", idx, self.v["ft_f"][idx].clone())
        self.ft_buf[idx] = 0
        self._set("reading", idx, torch.zeros(m, 6, device=self.dev))
        self._set("prog_b", idx, torch.zeros(m, device=self.dev))
        self._set("true_prog_last", idx, self._true_progress(idx))
        self._set("lat_err_last", idx, self._true_lateral(idx))

    def _set(self, k, idx, val):
        if not hasattr(self, "v"):
            self.v = InPlaceDict()
        if k not in self.v:
            self.v[k] = torch.zeros((self.n,) + tuple(val.shape[1:]), dtype=val.dtype, device=self.dev)
        self.v[k][idx] = val

    # ------------------------------------------------------------ kinematic helpers (SI)
    def _tip(self, idx=slice(None)):
        st = self.state[idx, 0]
        L = self.p["L"][idx]
        return st[:, 0:3] / 1000.0 + qrot(st[:, 3:7], torch.stack([0 * L, 0 * L, -L / 2], -1))

    def _q(self, idx=slice(None)):
        return self.state[idx, 0, 3:7]

    def _believed(self, idx=slice(None)):
        return self._tip(idx) + self.p["iherr"][idx], qmul(self._q(idx), self.p["ihq"][idx])

    def _true_progress(self, idx=slice(None)):
        D = self.p["depth"][idx]
        return -self._tip(idx)[:, 2]  # past the mouth (z = 0) along -z

    def _true_lateral(self, idx=slice(None)):
        return self._tip(idx)[:, :2].norm(dim=-1)

    # ------------------------------------------------------------ one controller substep + kernel call
    def _substep(self, ref, nq, k, krot):
        v, p, st = self.v, self.p, self.state
        h = self.h
        # motor lag: the reference the spring sees follows the commanded one (first order, tau = lag)
        lag_a = torch.where(p["lag"] > 1e-4, 1 - torch.exp(-h / p["lag"].clamp_min(1e-4)), torch.ones_like(p["lag"]))
        v["ref_eff"] = v["ref_eff"] + (ref - v["ref_eff"]) * lag_a[:, None]
        q = st[:, 0, 3:7]
        com = st[:, 0, 0:3] / 1000.0
        vel = st[:, 0, 7:10] / 1000.0
        w = st[:, 0, 10:13]
        tip = self._tip()
        rT = tip - com
        vT = vel + torch.cross(w, rT, dim=-1)
        M = p["vmass"]
        F = k[:, None] * (v["ref_eff"] - tip) - (2 * torch.sqrt(k * M))[:, None] * vT
        I = self.I
        Tq = krot[:, None] * rotvec_between(q, nq) - (2 * torch.sqrt(krot * I))[:, None] * w
        Ttot = Tq + torch.cross(rT, F, dim=-1)
        dv = F / M[:, None] * h
        dw = Ttot / I[:, None] * h
        v_pre = vel + dv
        w_pre = w + dw
        st[:, 0, 7:10] = v_pre * 1000.0
        st[:, 0, 10:13] = w_pre
        self._kernel()
        # contact wrench on the part (momentum balance; privileged)
        fc = (self.state[:, 0, 7:10] / 1000.0 - v_pre) * M[:, None] / h
        tc = (self.state[:, 0, 10:13] - w_pre) * I[:, None] / h
        # the arm's wrench reaction at the flange (what the wrist sensor reads, before its errors)
        flange = tip + qrot(q, torch.stack([0 * M, 0 * M, p["L"] + p["flange"]], -1))
        Tf = Tq + torch.cross(tip - flange, F, dim=-1)
        return -torch.cat([F, Tf], -1), torch.cat([fc, tc], -1)

    def _kernel(self):
        if self.dev.type == "cuda":
            for gi, (a, b) in enumerate(self.group_slices):
                if b <= a:          # (fewer worlds than friction groups: the sim-match runs one)
                    continue
                out = self.ext.manifold_step(self.state[a:b], self.inv_mass[a:b], self.half[a:b], self.inv_inertia[a:b], self.pairs,
                                             self.cache_ids[a:b], self.cache_imp[a:b], self.h, 1, 0.0, 0.0, self.group_mu_f[gi],
                                             1e-3, 0.2, 0.0, self.cfg["solver_iterations"], 1e-4)
                self.state[a:b] = out[0]
                self.cache_ids[a:b] = out[3]
                self.cache_imp[a:b] = out[4]
        else:
            cfgc = self.sref.SATConfig
            for gi, (a, b) in enumerate(self.group_slices):
                if b <= a:
                    continue
                c = cfgc(dt=self.h, substeps=1, gravity_y=0.0, restitution=0.0, friction=self.group_mu_f[gi], position_slop=1e-3,
                         position_correction=0.2, angular_damping=0.0, solver_iterations=self.cfg["solver_iterations"], sat_epsilon=1e-4)
                out = self.mref.step_manifold_reference(self.state[a:b].tolist(), self.inv_mass[a:b].tolist(), self.half[a:b].tolist(),
                                                        self.inv_inertia[a:b].tolist(), self.pairs.tolist(), self.cache_ids[a:b].tolist(),
                                                        self.cache_imp[a:b].tolist(), c)
                self.state[a:b] = torch.tensor(out[0], dtype=torch.float32)
                self.cache_ids[a:b] = torch.tensor(out[3], dtype=torch.int64)
                self.cache_imp[a:b] = torch.tensor(out[4], dtype=torch.float32)

    def _ft_sample(self, true6):
        v = self.v
        self.ft_ptr = (self.ft_ptr + 1) % 4
        self.ft_buf[:, self.ft_ptr] = true6 * v["ft_gain"]
        rd = (self.ft_ptr - v["ft_delay"]) % 4
        x = self.ft_buf[torch.arange(self.n, device=self.dev), rd]
        x = x + torch.randn(self.n, 6, generator=self.g, device=self.dev) * v["ft_sw"]
        v["ft_f"] = v["ft_f"] + v["ft_alpha"][:, None] * (x - v["ft_f"])
        v["ft_drift"] = v["ft_drift"] + torch.randn(self.n, 6, generator=self.g, device=self.dev) * cvec([0.002] * 3 + [0.0001] * 3, self.dev)
        out = v["ft_f"] + v["ft_drift"] - v["ft_tare"]
        r = v["ft_range"][:, None]
        return torch.maximum(torch.minimum(out, r), -r)

    # ------------------------------------------------------------ observation
    def _features(self):
        p = self.p
        oh = torch.zeros(self.n, 8, device=self.dev)
        oh[:, 1] = 1.0
        return torch.cat([oh, torch.stack([p["d"], p["L"], p["d"] + 2 * p["clear"], p["clear"], p["pch"], p["hch"], p["depth"], p["mass"]], -1)], -1)

    def obs(self, update=True):
        """update=False: the observation of freshly reset worlds (their velocities zero) without touching the rest."""
        v, p = self.v, self.p
        btip, bq = self._believed()
        dt = 1 / RATE_HZ
        v_lin = (btip - v["last_btip"]) / dt
        v_ang = rotvec_between(v["last_bq"], bq) / dt
        prog = (btip - v["b_entry"]) @ self.ax
        if update:
            v["last_btip"], v["last_bq"] = btip, bq
            v["prog_b"] = prog
        qT_b = p["qT_b"]
        one = torch.ones(self.n, 1, device=self.dev)
        parts = [to_task(v["b_target"] - btip), to_task(rotvec_between(bq, qT_b)), to_task(v_lin), to_task(v_ang),
                 to_task(v["reading"][:, :3]), to_task(v["reading"][:, 3:]), v["ft_sig"],
                 torch.stack([p["tsp"], p["tsp"], p["tsz"], p["tsr"], p["tsr"], p["tsr"]], -1),
                 torch.stack([p["ihs"], p["ihs"], p["ihz"]], -1), self._features(), prog[:, None], p["depth"][:, None],
                 v["prev_action"], v["t"][:, None]]
        o = torch.cat(parts, -1)
        return o

    def priv(self):
        v, p = self.v, self.p
        tip, q = self._tip(), self._q()
        D = p["depth"]
        tt = torch.stack([0 * D, 0 * D, -D], -1)
        st = self.state[:, 0]
        return torch.cat([to_task(tt - tip) * 1000.0, to_task(rotvec_of(qconj(q))) * 10.0, to_task(st[:, 7:10] / 1000.0) * 100.0,
                          v["contact_w"][:, :3] / 20.0, v["contact_w"][:, 3:] / 0.5,
                          (p["clear"] * 1e4)[:, None], self.group_mu[self.group_of][:, None], (p["hch"] * 1e3)[:, None],
                          (p["lag"] * 50)[:, None], p["latency"][:, None], (p["vmass"] - 2)[:, None],
                          ((tt - tip)[:, :2].norm(dim=-1) * 1000)[:, None], (v["k"] / 8000)[:, None]], -1)

    # ------------------------------------------------------------ step
    def _step(self, action):
        v, p, c = self.v, self.p, self.cfg
        a = torch.nan_to_num(action.float()).clamp(-1, 1)
        lat = p["latency"][:, None] > 0.5
        a_now = torch.where(lat, v["pending"], a)
        v["pending"] = torch.where(lat, a, v["pending"])
        a = a_now
        S = ACTION_SCALE
        lo, hi = S["k_trans_n_per_m"]
        k = lo * (hi / lo) ** ((a[:, 6] + 1) / 2)
        lo, hi = S["k_rot_nm_per_rad"]
        krot = lo * (hi / lo) ** ((a[:, 7] + 1) / 2)
        terminate = (a[:, 8] > 0) & (v["t"] >= 0.1)
        v["prev_action"] = a.clone()
        v["k"] = k
        depth, maxF = p["depth"], p["maxF"]
        btip, _ = self._believed()
        ax = self.ax
        nt = v["ref_tip"] + to_task(a[:, 0:3] * S["pos_m"])
        rel = nt - v["b_entry"]
        along = (rel @ ax).clamp(min=-0.03) .minimum(depth + LIMITS["max_axial_m"])
        latv = rel - ax * (rel @ ax)[:, None]
        off = latv - v["lat0"]
        on = off.norm(dim=-1, keepdim=True)
        latv = torch.where(on > LIMITS["max_lateral_m"], v["lat0"] + off * LIMITS["max_lateral_m"] / on.clamp_min(1e-12), latv)
        nt = v["b_entry"] + latv + ax * along[:, None]
        lead = nt - btip
        ln = lead.norm(dim=-1, keepdim=True)
        cap = (maxF / k)[:, None]
        nt = torch.where(ln > cap, btip + lead * cap / ln.clamp_min(1e-12), nt)
        nq = qmul(quat_of(to_task(a[:, 3:6] * S["rot_rad"])), v["ref_q"])
        tilt = rotvec_between(p["qT_b"], nq)
        tn = tilt.norm(dim=-1, keepdim=True)
        nq = torch.where(tn > LIMITS["max_tilt_rad"], qmul(quat_of(tilt * LIMITS["max_tilt_rad"] / tn.clamp_min(1e-12)), p["qT_b"]), nq)
        frm = v["ref_tip"]
        cw = torch.zeros(self.n, 6, device=self.dev)
        for s in range(1, self.per + 1):
            ref_s = frm + (nt - frm) * s / self.per
            acc = torch.zeros(self.n, 6, device=self.dev)
            for _ in range(self.sub):
                arm_w, con_w = self._substep(ref_s, nq, k, krot)
                acc = acc + arm_w
                cw = cw + con_w
            v["reading"] = self._ft_sample(acc / self.sub)
        v["contact_w"] = cw / (self.per * self.sub)
        v["ref_tip"], v["ref_q"] = nt, nq
        v["t"] = v["t"] + 1 / RATE_HZ
        obs = self.obs()
        # ---- termination: the runtime's rules on believed progress and the sensed force
        pb = v["prog_b"]
        fW = v["reading"][:, :3]
        push = -(fW @ ax)
        v["peak_push"] = torch.maximum(v["peak_push"], push)
        t = v["t"]
        code = torch.zeros(self.n, dtype=torch.long, device=self.dev)  # 0 running
        # codes: 1 depth, 2 policy, 3 POLICY_STOPPED_SHORT, 4 TIMEOUT, 5 FORCE_LIMIT, 6 stop, 7 NOT_FOUND, 8 JAMMED
        fn = fW.norm(dim=-1)
        over = fn > 1.25 * maxF
        v["over_since"] = torch.where(over, torch.where(v["over_since"] < 0, t, v["over_since"]), torch.full_like(t, -1.0))
        moved = (pb - v["stall_p"]).abs() > 5e-5
        v["stall_since"] = torch.where(moved, t, v["stall_since"])
        v["stall_p"] = torch.where(moved, pb, v["stall_p"])
        still = t - v["stall_since"]

        def put(cond, val):
            nonlocal code
            code = torch.where((code == 0) & cond, torch.full_like(code, val), code)
        put(pb >= depth, 1)
        put(terminate & (pb >= depth - 3e-4), 2)
        put(terminate, 3)
        put(t > c["timeout_s"], 4)
        put(over & (v["over_since"] >= 0) & (t - v["over_since"] > 0.1), 5)
        put((~moved) & (pb >= depth - 3e-4) & (still > 0.25), 6)
        put((~moved) & (still > 3) & (push > 0.25 * maxF) & (pb < 3e-4), 7)
        put((~moved) & (still > 3) & (push > 0.25 * maxF), 8)
        # ---- reward (truth)
        tp, tl = self._true_progress(), self._true_lateral()
        r = 4.0 * (torch.minimum(tp, depth) - torch.minimum(v["true_prog_last"], depth)) / depth
        r = r + 1.0 * (v["lat_err_last"] - tl) / 1e-3 * (tp < 0).float()
        v["true_prog_last"], v["lat_err_last"] = tp, tl
        r = r - 0.01 - 0.05 * (fn / maxF - 0.6).clamp_min(0)
        done = code > 0
        success = done & ((code == 1) | (code == 2) | (code == 6)) & (tp >= depth - 3e-4)
        r = r + torch.where(done, torch.where(success, torch.full_like(r, 10.0), torch.full_like(r, -3.0)), torch.zeros_like(r))
        info = dict(code=code, success=success, timeout=code == 4, true_depth=tp, clear=p["clear"].clone(), d=p["d"].clone())
        nonfinite = ~torch.isfinite(self.state).all(-1).all(-1)
        if True:  # (a solver blow-up ends the episode as a failure; unconditional masked updates: no host sync)
            done = done | nonfinite
            code = torch.where(nonfinite, torch.full_like(code, 9), code)
            info["code"] = code
            r = torch.where(nonfinite, torch.full_like(r, -3.0), r)
            blank = torch.zeros_like(self.state)
            blank[:, :, 6] = 1
            self.state.copy_(torch.where(nonfinite[:, None, None], blank, self.state))
        return obs, r, done, info


CODE_NAMES = {1: "depth", 2: "policy", 3: "POLICY_STOPPED_SHORT", 4: "INSERT_TIMEOUT", 5: "INSERT_FORCE_LIMIT", 6: "stop",
              7: "INSERT_NOT_FOUND", 8: "INSERT_JAMMED", 9: "NONFINITE"}
