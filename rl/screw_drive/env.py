"""Batched screw driving on box3d-cuda, built to World2's learned-skill contract (docs/learned-skills.md section 11,
action kind `screw-drive`, skill `screw`).

Crude on purpose. The screw is a square box (side d, length L) on a compliant bit; the cover plate's clearance hole is
box walls with 45 degree chamfer boxes (rl/peg_insert's host geometry), the threaded insert's top a slab under it (its
bore is smaller than the screw: the tip rests on it until the thread starts). Contacts: box3d's manifold_step. The
thread: rl/screw_drive/screw_joint.cu (helical joint + engagement + failure state; CPU oracle screw_reference.py), run
after manifold_step each substep. The contract's controller (World2 operations/compliant-insert.mjs, contact tier) is
applied in torch between kernel calls: a 2 kg virtual mass driven at the tip by F = k e - 2 sqrt(k m) v_tip, k 2000 N/m
(actual x0.6-1.5: the bit's compliance), a rotational spring 2 N m/rad about the tip toward the reference orientation
composed with the screw's wobble on the bit.

Beliefs as World2's runtime forms them: the believed orientation is the commanded reference (the wrist's encoders),
the believed tip the true tip + an in-hand error, the head bearing point tip - axis_ref L, the seat and hole axis the
program's estimate (errors drawn from the stated sigma). While the thread holds the screw the wrist reading is World2's:
the axial push (commanded) and the spindle reaction torque about the believed axis, nothing lateral.

Physics in millimetres (box3d), controller SI. z up, plate top at z = 0, the plate hole's axis is world -z.
"""
from __future__ import annotations

import math

import torch

from ..peg_insert.env import (FT_PROFILES, load_cpu_reference, load_cuda_ext as load_manifold_ext, loguni, qconj, qmul, qrot,
                              quat_of, rotvec_between, rotvec_of, uni)
from . import screw_reference as SR

OBS_FIELDS = ["seat_pos", "screw_tilt", "part_vel", "wrench", "wrench_sigma", "spindle", "target_sigma", "screw_spec",
              "attempt", "prev_action", "time"]
N_ACT = 7
OBS_SIZE = 3 + 2 + 6 + 6 + 2 + 3 + 6 + 8 + 5 + N_ACT + 1   # 49
PRIV_SIZE = 16
ACTION_SCALE = dict(pos_m=0.0005, rot_rad=0.004, force_n=(2.0, 60.0), descend_m_s=0.012, lift_m_s=0.01, max_rpm=400.0,
                    k_n_per_m=2000.0, k_rot_nm_per_rad=2.0)
LIMITS = dict(max_lateral_m=0.002, max_tilt_rad=0.1, timeout_s=12.0)
RATE_HZ = 30
VINERT = 0.002
N_STATIC = 13
B = 1 + N_STATIC
DOWN = (0.0, 0.0, -1.0)
# ISO coarse sizes: d, pitch, set torque range (N m), Phillips r_eff (PH0/PH1/PH2), hex key test torque (ISO 2936 class)
SIZES = torch.tensor([[2.0e-3, 0.40e-3, 0.20, 0.35, 0.7e-3, 0.8],
                      [2.5e-3, 0.45e-3, 0.30, 0.60, 1.0e-3, 1.9],
                      [3.0e-3, 0.50e-3, 0.40, 1.00, 1.0e-3, 3.8],
                      [4.0e-3, 0.70e-3, 1.00, 2.50, 1.5e-3, 6.6],
                      [5.0e-3, 0.80e-3, 2.00, 5.00, 1.5e-3, 16.0]])
CODE_NAMES = {1: "seated", 2: "SCREW_CROSS_THREAD", 3: "SCREW_EARLY_TORQUE", 4: "SCREW_STRIPPED", 5: "SCREW_STRIPPED_DECLARED",
              6: "SCREW_CAM_OUT", 7: "SCREW_BIT_SLIP", 8: "SCREW_ABORTED", 9: "SCREW_TIMEOUT", 10: "NONFINITE",
              11: "SCREW_FAST_SEAT"}
SEAT_RPM_MAX = 120.0   # World2 verify-by-sensing needs torque-angle samples from snug to the set torque (the scripted drive seats at 60 rpm)


def default_cfg(**over):
    c = dict(
        plate_mm=(1.5, 6.0), clearance_mm=(0.1, 0.35), plate_chamfer_mm=(0.1, 0.5), engage_over_d=(1.5, 3.0),
        trun_per_m=(3.0, 12.0), kj=(2e6, 5e7), nut_k=(0.15, 0.25), strip_p=0.15, strip_rel=(0.7, 1.3),
        phillips_p=0.5, flank_deg=(7.0, 13.0), flank_mu=(0.08, 0.14), hex_fraction=(0.6, 1.0), clutch_scatter=0.01,
        kx_rel=(0.25, 0.8), theta_x_deg=(1.8, 3.5), capture_over_d=(0.1, 0.2), mode_b_p=0.3,
        insert_tilt_deg=(0.0, 4.0), insert_offset_mm=(0.0, 0.25),
        rot_sigma_deg=(0.3, 2.5), seat_sigma_mm=(0.1, 0.5), seat_sigma_z_mm=(0.05, 0.2), in_hand_mm=0.05,
        start_height_mm=(1.0, 5.0), wobble_deg=(0.0, 1.0), stiff_mult=(0.6, 1.5),
        torque_noise_nm=(0.005, 0.02), torque_delay_p=0.5,
        ft_extra_delay_s=(0.0, 0.012), ft_gain_err=(0.005, 0.01), ft_noise_mult=(1.0, 1.5), flange_h=(0.08, 0.2),
        action_latency_p=0.3, motor_lag_s=(0.0, 0.03), vmass=(1.7, 2.3), friction=(0.1, 0.5), friction_groups=8,
        substeps_per_world_step=4, solver_iterations=8, timeout_s=LIMITS["timeout_s"],
        pose_scale=1.0,                       # multiplies the belief errors and offsets (hard sets)
    )
    c.update(over)
    return c


def load_screw_ext():
    from pathlib import Path
    from torch.utils.cpp_extension import load
    return load(name="box3d_screw_joint_ext", sources=[str(Path(__file__).with_name("screw_joint.cu"))],
                extra_cuda_cflags=["-O3"], verbose=False)


def qbetween(p, q):
    c = (p * q).sum(-1, keepdim=True)
    x = torch.cross(p, q, dim=-1)
    w = 1 + c
    o = torch.cat([x, w], -1)
    o = o / o.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    bad = (w < 1e-9).squeeze(-1)
    if bad.any():
        o[bad] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=o.device)
    return o


class ScrewDriveBatch:
    """N independent worlds. step(action[N,7]) -> obs[N,49], reward[N], done[N], info."""

    def __init__(self, n, device="cuda", seed=0, cfg=None):
        self.n, self.dev = n, torch.device(device)
        self.cfg = default_cfg(**(cfg or {}))
        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(seed)
        self.per = 120 // RATE_HZ
        self.sub = self.cfg["substeps_per_world_step"]
        self.h = 1.0 / 120 / self.sub
        if self.dev.type == "cuda":
            self.mext, self.sext = load_manifold_ext(), load_screw_ext()
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
        self.P = f(n, SR.NP)
        self.J = f(n, SR.NJ)
        G = self.cfg["friction_groups"]
        self.group_of = torch.arange(n, device=self.dev) * G // n
        self.group_slices = [(i * n // G, (i + 1) * n // G) for i in range(G)]
        self.group_mu = uni(self.g, G, self.cfg["friction"], self.dev)
        self.p, self.v = {}, {}
        self.ft_buf = f(n, 4, 6)
        self.ft_ptr = 0
        self.reset(torch.ones(n, dtype=torch.bool, device=self.dev))

    def resample_friction(self):
        self.group_mu = uni(self.g, len(self.group_slices), self.cfg["friction"], self.dev)

    def _set(self, k, idx, val):
        if k not in self.v:
            self.v[k] = torch.zeros((self.n,) + tuple(val.shape[1:]), dtype=val.dtype, device=self.dev)
        self.v[k][idx] = val

    # ------------------------------------------------------------ sampling
    def _sample(self, idx):
        c, g, d, m = self.cfg, self.g, self.dev, len(idx)
        rn = lambda *s: torch.randn(*s, generator=g, device=d)
        ru = lambda: torch.rand(m, generator=g, device=d)
        ps = c["pose_scale"]
        P = {}
        si = torch.randint(0, 5, (m,), generator=g, device=d)
        S = SIZES.to(d)[si]
        P["size"] = si
        P["d"], P["pitch"] = S[:, 0], S[:, 1]
        P["tp"] = uni(g, m, c["plate_mm"], d) * 1e-3
        P["clear"] = uni(g, m, c["clearance_mm"], d) * 1e-3
        P["pch"] = torch.minimum(uni(g, m, c["plate_chamfer_mm"], d) * 1e-3, 0.4 * P["tp"])
        P["eng"] = P["d"] * uni(g, m, c["engage_over_d"], d)
        P["L"] = P["tp"] + P["eng"]
        P["trun"] = uni(g, m, c["trun_per_m"], d) * P["d"]
        P["tset_nom"] = S[:, 2] + (S[:, 3] - S[:, 2]) * ru()
        P["tset"] = P["tset_nom"] * (1 + c["clutch_scatter"] * rn(m))
        P["kj"] = loguni(g, m, c["kj"], d)
        P["nutk"] = uni(g, m, c["nut_k"], d)
        strip = ru() < c["strip_p"]
        P["tstrip"] = torch.where(strip, P["tset_nom"] * uni(g, m, c["strip_rel"], d), torch.full((m,), 1e3, device=d))
        P["phil"] = (ru() < c["phillips_p"]).float()
        tb = torch.tan(torch.deg2rad(uni(g, m, c["flank_deg"], d)))
        mu = uni(g, m, c["flank_mu"], d)
        camc = torch.where(tb > mu + 1e-3, S[:, 4] * (1 + mu * tb) / (tb - mu).clamp_min(1e-3), torch.full((m,), 1e3, device=d))
        # (a Phillips screw whose set torque needs more axial force than the arm's 60 N limit allows at 0.9 x is driven hex:
        # an infeasible episode teaches nothing; cam-out stays possible for a light push)
        P["phil"] = P["phil"] * (P["tset_nom"] <= camc * 54.0).float()
        P["camc"] = torch.where(P["phil"] > 0.5, camc, torch.full((m,), 1e3, device=d))
        P["hexcap"] = torch.where(P["phil"] > 0.5, torch.full((m,), 1e3, device=d), S[:, 5] * uni(g, m, c["hex_fraction"], d))
        P["tmot"] = 1.875 * S[:, 3]
        P["kx"] = uni(g, m, c["kx_rel"], d) * P["tset_nom"]
        P["thx"] = torch.deg2rad(uni(g, m, c["theta_x_deg"], d))
        P["rcap"] = uni(g, m, c["capture_over_d"], d) * P["d"]
        P["modeb"] = (ru() < c["mode_b_p"]).float()
        az = uni(g, m, (0, 2 * math.pi), d)
        tau = torch.deg2rad(uni(g, m, c["insert_tilt_deg"], d))
        down = torch.tensor(DOWN, device=d).expand(m, 3)
        qi = quat_of(torch.stack([torch.cos(az), torch.sin(az), 0 * az], -1) * tau[:, None])
        P["uins"] = qrot(qi, down)
        az2 = uni(g, m, (0, 2 * math.pi), d)
        off = uni(g, m, c["insert_offset_mm"], d) * 1e-3 * ps
        P["mouth"] = torch.stack([off * torch.cos(az2), off * torch.sin(az2), -P["tp"]], -1)   # m
        # beliefs: the hole axis and the seat from an estimate with a stated sigma (errors drawn from it)
        P["rsig"] = torch.deg2rad(uni(g, m, c["rot_sigma_deg"], d))
        e = torch.cat([rn(m, 2) * P["rsig"][:, None] * ps, torch.zeros(m, 1, device=d)], -1)
        P["bu"] = qrot(quat_of(e), P["uins"])
        P["belief_err"] = torch.acos((P["bu"] * P["uins"]).sum(-1).clamp(-1, 1))
        P["ssig"] = uni(g, m, c["seat_sigma_mm"], d) * 1e-3
        P["ssigz"] = uni(g, m, c["seat_sigma_z_mm"], d) * 1e-3
        P["seat"] = P["mouth"] - P["uins"] * P["tp"][:, None]
        P["bseat"] = P["seat"] + torch.stack([rn(m) * P["ssig"], rn(m) * P["ssig"], rn(m) * P["ssigz"]], -1) * ps
        P["iherr"] = torch.stack([rn(m) * c["in_hand_mm"], rn(m) * c["in_hand_mm"], rn(m) * 0.6 * c["in_hand_mm"]], -1) * 1e-3
        wa = uni(g, m, (0, 2 * math.pi), d)
        wob = torch.deg2rad(uni(g, m, c["wobble_deg"], d))
        P["wq"] = quat_of(torch.stack([torch.cos(wa), torch.sin(wa), 0 * wa], -1) * wob[:, None])
        P["h0"] = uni(g, m, c["start_height_mm"], d) * 1e-3
        P["kmul"] = uni(g, m, c["stiff_mult"], d)
        P["krmul"] = uni(g, m, c["stiff_mult"], d)
        P["vmass"] = uni(g, m, c["vmass"], d)
        P["tnoise"] = uni(g, m, c["torque_noise_nm"], d)
        P["tdelay"] = (ru() < c["torque_delay_p"]).float()
        P["ft"] = torch.randint(0, 3, (m,), generator=g, device=d)
        P["ftd"] = uni(g, m, c["ft_extra_delay_s"], d)
        P["gain_err"] = uni(g, m, c["ft_gain_err"], d)
        P["noise_mult"] = uni(g, m, c["ft_noise_mult"], d)
        P["flange"] = uni(g, m, c["flange_h"], d)
        P["latency"] = (ru() < c["action_latency_p"]).float()
        P["lag"] = uni(g, m, c["motor_lag_s"], d)
        return P

    def _geometry(self, idx):
        """The plate's square clearance hole (per side a lower wall, an upper wall, a 45 degree chamfer box) and the
        insert's top slab under it (top at z = -plate), all mm (rl/peg_insert's host geometry with its floor)."""
        P, mm, dev = self.p, 1000.0, self.dev
        s, c, D, ch = P["d"][idx] * mm, P["clear"][idx] * mm, P["tp"][idx] * mm, P["pch"][idx] * mm
        a = s / 2 + c
        T = torch.full_like(a, 5.0)
        W = a + ch + T
        m = len(idx)
        pos = torch.zeros(m, N_STATIC, 3, device=dev)
        half = torch.zeros(m, N_STATIC, 3, device=dev)
        quat = torch.zeros(m, N_STATIC, 4, device=dev)
        quat[..., 3] = 1
        r2 = math.sqrt(0.5)
        for k in range(4):
            phi = k * math.pi / 2
            cz, sz = math.cos(phi), math.sin(phi)
            qz = torch.tensor([0, 0, math.sin(phi / 2), math.cos(phi / 2)], device=dev).expand(m, 4)
            local = [
                (torch.stack([a + (ch + T) / 2, 0 * a, -(D + ch) / 2], -1), torch.stack([(ch + T) / 2, W, (D - ch) / 2], -1), None),
                (torch.stack([a + ch + T / 2, 0 * a, -ch / 2], -1), torch.stack([T / 2, W, ch / 2], -1), None),
                (torch.stack([a + ch, 0 * a, -ch], -1), torch.stack([ch * r2, W, ch * r2], -1),
                 torch.tensor([0, math.sin(math.pi / 8), 0, math.cos(math.pi / 8)], device=dev).expand(m, 4)),
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
        Pn = self._sample(idx)
        for k, val in Pn.items():
            if k not in self.p:
                self.p[k] = torch.zeros((self.n,) + tuple(val.shape[1:]), dtype=val.dtype, device=self.dev)
            self.p[k][idx] = val
        self._geometry(idx)
        p, m, dev, mm = self.p, len(idx), self.dev, 1000.0
        down = torch.tensor(DOWN, device=dev).expand(m, 3)
        # task frame: z the believed axis, x/y the contract's basis for a downward axis turned with it
        qbu = qbetween(down, p["bu"][idx])
        X = qrot(qbu, torch.tensor([0.0, -1.0, 0.0], device=dev).expand(m, 3))
        Y = qrot(qbu, torch.tensor([-1.0, 0.0, 0.0], device=dev).expand(m, 3))
        self._set("frame", idx, torch.stack([X, Y, p["bu"][idx]], 1))   # rows x, y, z
        self._set("qbu", idx, qbu)
        # start: the wrist over the believed seat, the tip h0 above it along the believed axis; the screw wobbles on the bit
        L = p["L"][idx]
        tip0 = p["bseat"][idx] - p["bu"][idx] * p["h0"][idx, None] + torch.randn(m, 3, generator=self.g, device=dev) * 5e-5
        q0 = qmul(p["wq"][idx], qbu)
        ctr = tip0 - qrot(q0, down) * (L / 2)[:, None]
        self.state[idx, 0, 0:3] = ctr * mm
        self.state[idx, 0, 3:7] = q0
        self.state[idx, 0, 7:] = 0
        hs = torch.stack([p["d"][idx] / 2, p["d"][idx] / 2, L / 2], -1)
        self.half[idx, 0] = hs * mm
        M = p["vmass"][idx]
        mass = 7800 * p["d"][idx] ** 2 * L
        Ib = mass[:, None] / 3 * (hs[:, [1, 0, 0]] ** 2 + hs[:, [2, 2, 1]] ** 2) + VINERT
        if not hasattr(self, "I"):
            self.I = torch.zeros(self.n, device=dev)
        self.I[idx] = Ib.max(-1).values
        self.inv_mass[idx, 0] = 1.0 / M
        self.inv_inertia[idx, 0] = 1.0 / (Ib * 1e6)
        self.cache_ids[idx] = 0
        self.cache_imp[idx] = 0
        # the thread's parameters and joint state (layouts: screw_reference.py)
        Pt = torch.zeros(m, SR.NP, device=dev)
        Pt[:, SR.P_PITCH], Pt[:, SR.P_D], Pt[:, SR.P_RCAP], Pt[:, SR.P_THX] = p["pitch"][idx], p["d"][idx], p["rcap"][idx], p["thx"][idx]
        Pt[:, SR.P_TRUN], Pt[:, SR.P_KX], Pt[:, SR.P_TSET], Pt[:, SR.P_KJ] = p["trun"][idx], p["kx"][idx], p["tset"][idx], p["kj"][idx]
        Pt[:, SR.P_K], Pt[:, SR.P_TSTRIP], Pt[:, SR.P_PHIL], Pt[:, SR.P_CAMC] = p["nutk"][idx], p["tstrip"][idx], p["phil"][idx], p["camc"][idx]
        Pt[:, SR.P_HEXCAP], Pt[:, SR.P_TMOT], Pt[:, SR.P_TRAVEL], Pt[:, SR.P_MODEB] = p["hexcap"][idx], p["tmot"][idx], p["eng"][idx], p["modeb"][idx]
        Pt[:, SR.P_MOUTH:SR.P_MOUTH + 3] = p["mouth"][idx] * mm
        Pt[:, SR.P_UINS:SR.P_UINS + 3] = p["uins"][idx]
        Pt[:, SR.P_DROP] = 0.4 * p["pitch"][idx]
        Pt[:, SR.P_L] = L
        self.P[idx] = Pt
        Jt = torch.zeros(m, SR.NJ, device=dev)
        Jt[:, SR.J_ARMED] = 1.0
        Jt[:, SR.J_THSEAT] = -1.0
        Jt[:, SR.J_PHASE] = torch.rand(m, generator=self.g, device=dev) * 2 * math.pi
        self.J[idx] = Jt
        # controller reference (world, SI): tip and wrist orientation
        self._set("ref_tip", idx, tip0.clone())
        self._set("ref_eff", idx, tip0.clone())
        self._set("ref_q", idx, qbu.clone())
        r = tip0 - p["bseat"][idx]
        bz = p["bu"][idx]
        self._set("lat0", idx, r - bz * (r * bz).sum(-1, keepdim=True))
        self._set("fcmd", idx, torch.zeros(m, device=dev))
        self._set("wcmd", idx, torch.zeros(m, device=dev))
        self._set("prev_action", idx, torch.zeros(m, N_ACT, device=dev))
        self._set("pending", idx, torch.zeros(m, N_ACT, device=dev))
        self._set("t", idx, torch.zeros(m, device=dev))
        self._set("strip_t", idx, torch.full((m,), -1.0, device=dev))
        self._set("backouts", idx, torch.zeros(m, device=dev))
        self._set("last_fail_tilt", idx, torch.zeros(m, 2, device=dev))
        self._set("prev_mode", idx, torch.zeros(m, device=dev))
        self._set("seat_rpm", idx, torch.zeros(m, device=dev))
        self._set("treading", idx, torch.zeros(m, device=dev))
        self._set("tprev", idx, torch.zeros(m, device=dev))
        # F/T sensor (rl/peg_insert's model of World2's operations/sensors/wrist-ft.mjs)
        prof = FT_PROFILES.to(dev)[p["ft"][idx]]
        dt = 1 / 120
        alpha = 1 - torch.exp(-2 * math.pi * prof[:, 2] * dt)
        self._set("ft_alpha", idx, alpha)
        sw = torch.stack([prof[:, 0]] * 3 + [prof[:, 1]] * 3, -1) * torch.sqrt((2 - alpha) / alpha)[:, None] * p["noise_mult"][idx, None]
        self._set("ft_sw", idx, sw)
        self._set("ft_gain", idx, 1 + torch.randn(m, 6, generator=self.g, device=dev) * p["gain_err"][idx, None])
        self._set("ft_delay", idx, torch.round((prof[:, 3] + p["ftd"][idx]) / dt).long().clamp(1, 3))
        self._set("ft_range", idx, prof[:, 4])
        self._set("ft_sig", idx, prof[:, :2])
        self._set("ft_f", idx, torch.randn(m, 6, generator=self.g, device=dev) * prof[:, [0, 0, 0, 1, 1, 1]])
        self._set("ft_drift", idx, torch.zeros(m, 6, device=dev))
        self._set("ft_tare", idx, self.v["ft_f"][idx].clone())
        self.ft_buf[idx] = 0
        self._set("reading", idx, torch.zeros(m, 6, device=dev))
        btip = tip0 + p["iherr"][idx]
        self._set("last_btip", idx, btip)
        self._set("last_bq", idx, qbu.clone())
        self._set("lat_last", idx, self._lat_to_insert(idx))
        self._set("tilt_last", idx, self._tilt_err(idx))
        self._set("theta_last", idx, torch.zeros(m, device=dev))

    # ------------------------------------------------------------ helpers (SI)
    def _axis(self, idx=slice(None)):
        return qrot(self.state[idx, 0, 3:7], torch.tensor(DOWN, device=self.dev).expand(self.state[idx, 0].shape[0], 3))

    def _tip(self, idx=slice(None)):
        return self.state[idx, 0, 0:3] / 1000.0 + self._axis(idx) * (self.p["L"][idx] / 2)[:, None]

    def _lat_to_insert(self, idx=slice(None)):
        r = self._tip(idx) - self.p["mouth"][idx]
        u = self.p["uins"][idx]
        return (r - u * (r * u).sum(-1, keepdim=True)).norm(dim=-1)

    def _tilt_err(self, idx=slice(None)):
        return torch.acos((self._axis(idx) * self.p["uins"][idx]).sum(-1).clamp(-1, 1))

    def to_task(self, v):
        return torch.einsum("nij,nj->ni", self.v["frame"], v)

    def to_world(self, a):
        return torch.einsum("nij,ni->nj", self.v["frame"], a)

    # ------------------------------------------------------------ physics substep
    def _substep(self, ref, nq):
        v, p, st = self.v, self.p, self.state
        h = self.h
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
        k = ACTION_SCALE["k_n_per_m"] * p["kmul"]
        krot = ACTION_SCALE["k_rot_nm_per_rad"] * p["krmul"]
        F = k[:, None] * (v["ref_eff"] - tip) - (2 * torch.sqrt(k * M))[:, None] * vT
        Tq = krot[:, None] * rotvec_between(q, qmul(p["wq"], nq)) - (2 * torch.sqrt(krot * self.I))[:, None] * w
        Ttot = Tq + torch.cross(rT, F, dim=-1)
        st[:, 0, 7:10] = (vel + F / M[:, None] * h) * 1000.0
        st[:, 0, 10:13] = w + Ttot / self.I[:, None] * h
        ax = self._axis()
        fax = (F * ax).sum(-1).clamp_min(0)
        self._manifold()
        C = torch.stack([v["wcmd"], fax], -1)
        self._screw(C)
        flange = tip + qrot(q, torch.stack([0 * M, 0 * M, p["L"] + p["flange"]], -1))
        Tf = Tq + torch.cross(tip - flange, F, dim=-1)
        T = self.J[:, SR.J_TORQUE]
        free = torch.cat([-F, -Tf - ax * T[:, None]], -1)
        bu = p["bu"]
        held = torch.cat([-bu * v["fcmd"][:, None], -bu * T[:, None]], -1)
        eng = (self.J[:, SR.J_MODE] >= 0.5)[:, None]
        return torch.where(eng, held, free)

    def _manifold(self):
        if self.dev.type == "cuda":
            for gi, (a, b) in enumerate(self.group_slices):
                out = self.mext.manifold_step(self.state[a:b], self.inv_mass[a:b], self.half[a:b], self.inv_inertia[a:b], self.pairs,
                                              self.cache_ids[a:b], self.cache_imp[a:b], self.h, 1, 0.0, 0.0, float(self.group_mu[gi]),
                                              1e-3, 0.2, 0.0, self.cfg["solver_iterations"], 1e-4)
                self.state[a:b] = out[0]
                self.cache_ids[a:b] = out[3]
                self.cache_imp[a:b] = out[4]
        else:
            cfgc = self.sref.SATConfig
            for gi, (a, b) in enumerate(self.group_slices):
                if b <= a:
                    continue
                c = cfgc(dt=self.h, substeps=1, gravity_y=0.0, restitution=0.0, friction=float(self.group_mu[gi]), position_slop=1e-3,
                         position_correction=0.2, angular_damping=0.0, solver_iterations=self.cfg["solver_iterations"], sat_epsilon=1e-4)
                out = self.mref.step_manifold_reference(self.state[a:b].tolist(), self.inv_mass[a:b].tolist(), self.half[a:b].tolist(),
                                                        self.inv_inertia[a:b].tolist(), self.pairs.tolist(), self.cache_ids[a:b].tolist(),
                                                        self.cache_imp[a:b].tolist(), c)
                self.state[a:b] = torch.tensor(out[0], dtype=torch.float32)
                self.cache_ids[a:b] = torch.tensor(out[3], dtype=torch.int64)
                self.cache_imp[a:b] = torch.tensor(out[4], dtype=torch.float32)

    def _screw(self, C):
        if self.dev.type == "cuda":
            out = self.sext.screw_joint_step(self.state, self.P, self.J, C.float().contiguous(), self.h)
            self.state, self.J = out[0], out[1]
        else:
            st, J = SR.step_reference(self.state.tolist(), self.P.tolist(), self.J.tolist(), C.tolist(), self.h)
            self.state = torch.tensor(st, dtype=torch.float32)
            self.J = torch.tensor(J, dtype=torch.float32)

    def _ft_sample(self, true6):
        v = self.v
        self.ft_ptr = (self.ft_ptr + 1) % 4
        self.ft_buf[:, self.ft_ptr] = true6 * v["ft_gain"]
        rd = (self.ft_ptr - v["ft_delay"]) % 4
        x = self.ft_buf[torch.arange(self.n, device=self.dev), rd]
        x = x + torch.randn(self.n, 6, generator=self.g, device=self.dev) * v["ft_sw"]
        v["ft_f"] = v["ft_f"] + v["ft_alpha"][:, None] * (x - v["ft_f"])
        v["ft_drift"] = v["ft_drift"] + torch.randn(self.n, 6, generator=self.g, device=self.dev) * torch.tensor([0.002] * 3 + [0.0001] * 3, device=self.dev)
        out = v["ft_f"] + v["ft_drift"] - v["ft_tare"]
        r = v["ft_range"][:, None]
        return torch.maximum(torch.minimum(out, r), -r)

    # ------------------------------------------------------------ observation
    def _tilt_offset(self, q):
        """(x, y) task components of the rotation from the believed hole axis to orientation q's axis."""
        down = torch.tensor(DOWN, device=self.dev).expand(self.n, 3)
        rv = rotvec_between(self.v["qbu"], qbetween(down, qrot(q, down)))
        return self.to_task(rv)[:, :2]

    def obs(self, update=True):
        v, p = self.v, self.p
        down = torch.tensor(DOWN, device=self.dev).expand(self.n, 3)
        btip = self._tip() + p["iherr"]
        bq = v["ref_q"]
        bax = qrot(bq, down)
        head = btip - bax * p["L"][:, None]
        dt = 1 / RATE_HZ
        v_lin = (btip - v["last_btip"]) / dt
        v_ang = rotvec_between(v["last_bq"], bq) / dt
        if update:
            v["last_btip"], v["last_bq"] = btip, bq.clone()
        # screw_tilt: the believed screw axis onto the believed hole axis
        tilt = -self._tilt_offset(bq)
        J = self.J
        one = torch.ones(self.n, device=self.dev)
        spec = torch.stack([p["d"], p["pitch"], p["L"], 1.8 * p["d"], p["tp"], p["tset_nom"], p["phil"], p["eng"] / p["pitch"]], -1)
        att = torch.cat([v["backouts"][:, None], self._tilt_offset(bq), v["last_fail_tilt"]], -1)
        spindle = torch.stack([v["treading"], J[:, SR.J_SPIN] / (2 * math.pi), J[:, SR.J_OMEGA] / (2 * math.pi)], -1)
        parts = [self.to_task(p["bseat"] - head), tilt, self.to_task(v_lin), self.to_task(v_ang),
                 self.to_task(v["reading"][:, :3]), self.to_task(v["reading"][:, 3:]), v["ft_sig"], spindle,
                 torch.stack([p["ssig"], p["ssig"], p["ssigz"], p["rsig"], p["rsig"], p["rsig"]], -1), spec, att,
                 v["prev_action"], v["t"][:, None]]
        return torch.cat(parts, -1)

    def priv(self):
        v, p, J = self.v, self.p, self.J
        down = torch.tensor(DOWN, device=self.dev).expand(self.n, 3)
        rv = rotvec_between(qbetween(down, self._axis()), qbetween(down, p["uins"]))
        r = self._tip() - p["mouth"]
        mode = J[:, SR.J_MODE]
        theta = J[:, SR.J_THETA]
        travel_left = p["eng"] - p["pitch"] * theta / (2 * math.pi) * (mode >= 0.5).float()
        cap = torch.where(p["phil"] > 0.5, 60 * p["camc"], p["hexcap"]) / p["tset"]
        bind = J[:, SR.J_CROSSED] * p["kx"] * theta / (2 * math.pi) / p["tset"]
        return torch.cat([self.to_task(rv)[:, :2] * 30, self.to_task(r) * 1000.0,
                          torch.stack([(mode == 0).float(), (mode == 1).float(), (mode == 2).float(), J[:, SR.J_CROSSED], p["modeb"],
                                       torch.rad2deg(p["thx"]) / 3, travel_left * 100, J[:, SR.J_STRIPPED], J[:, SR.J_ARMED],
                                       bind, cap.clamp(0, 5)], -1)], -1)

    # ------------------------------------------------------------ step
    def step(self, action):
        v, p, c, S = self.v, self.p, self.cfg, ACTION_SCALE
        a = torch.nan_to_num(action.float()).clamp(-1, 1)
        lat_on = p["latency"][:, None] > 0.5
        a_now = torch.where(lat_on, v["pending"], a)
        v["pending"] = torch.where(lat_on, a, v["pending"])
        a = a_now
        v["prev_action"] = a.clone()
        stop = (a[:, 6] > 0) & (v["t"] >= 0.1)
        # ---- the reference: lateral and tilt moves, the axial channel (push with a force or lift), the spindle
        bz, bseat = p["bu"], p["bseat"]
        btip = self._tip() + p["iherr"]
        knom = S["k_n_per_m"]
        nt = v["ref_tip"] + self.to_world(torch.stack([a[:, 0], a[:, 1], 0 * a[:, 0]], -1) * S["pos_m"])
        rel = nt - bseat
        rz = (rel * bz).sum(-1)
        bz_tip = ((btip - bseat) * bz).sum(-1)
        push = a[:, 4] >= 0
        F = S["force_n"][0] * (S["force_n"][1] / S["force_n"][0]) ** a[:, 4].clamp_min(0)
        rz_push = torch.minimum(rz + S["descend_m_s"] / RATE_HZ, bz_tip + F / knom)
        rz_lift = torch.minimum(rz, bz_tip) + a[:, 4] * S["lift_m_s"] / RATE_HZ
        rz = torch.where(push, rz_push, rz_lift).clamp(-0.03, 0.03)
        v["fcmd"] = torch.where(push, F, torch.zeros_like(F))
        latv = rel - bz * (rel * bz).sum(-1, keepdim=True)
        off = latv - v["lat0"]
        on = off.norm(dim=-1, keepdim=True)
        latv = torch.where(on > LIMITS["max_lateral_m"], v["lat0"] + off * LIMITS["max_lateral_m"] / on.clamp_min(1e-12), latv)
        nt = bseat + latv + bz * rz[:, None]
        nq = qmul(quat_of(self.to_world(torch.stack([a[:, 2], a[:, 3], 0 * a[:, 2]], -1) * S["rot_rad"])), v["ref_q"])
        tl = rotvec_between(v["qbu"], nq)
        tn = tl.norm(dim=-1, keepdim=True)
        nq = torch.where(tn > LIMITS["max_tilt_rad"], qmul(quat_of(tl * LIMITS["max_tilt_rad"] / tn.clamp_min(1e-12)), v["qbu"]), nq)
        sp = a[:, 5]
        v["wcmd"] = torch.where(sp.abs() < 0.02, torch.zeros_like(sp), sp * S["max_rpm"] * 2 * math.pi / 60)
        frm = v["ref_tip"]
        mode0 = self.J[:, SR.J_MODE].clone()
        for s in range(1, self.per + 1):
            ref_s = frm + (nt - frm) * s / self.per
            acc = torch.zeros(self.n, 6, device=self.dev)
            for _ in range(self.sub):
                acc = acc + self._substep(ref_s, nq)
            v["reading"] = self._ft_sample(acc / self.sub)
        v["ref_tip"], v["ref_q"] = nt, nq
        J = self.J
        mode = J[:, SR.J_MODE]
        # a back-out: the screw rides the bit again; the wrist's reference restarts where the tip is (World2: a new
        # compliance session)
        released = (mode0 >= 0.5) & (mode0 < 2.5) & (mode == 0)
        if released.any():
            v["backouts"] = v["backouts"] + released.float()
            v["last_fail_tilt"] = torch.where(released[:, None], self._tilt_offset(nq), v["last_fail_tilt"])
            v["ref_tip"] = torch.where(released[:, None], self._tip(), v["ref_tip"])
            v["ref_eff"] = torch.where(released[:, None], self._tip(), v["ref_eff"])
        # the driver's torque reading (noise, 0-1 tick delay)
        tr = J[:, SR.J_TORQUE] + torch.randn(self.n, generator=self.g, device=self.dev) * p["tnoise"]
        v["treading"] = torch.where(p["tdelay"] > 0.5, v["tprev"], tr)
        v["tprev"] = tr
        v["t"] = v["t"] + 1 / RATE_HZ
        t = v["t"]
        # the spindle speed when the head seated (this tick's command: it is constant across the tick)
        seated_now = (mode >= 1.5) & (mode < 2.5) | (J[:, SR.J_SUCCESS] > 0.5)
        newly = seated_now & ~((mode0 >= 1.5) & (mode0 < 2.5))
        v["seat_rpm"] = torch.where(newly, v["wcmd"].abs() * 60 / (2 * math.pi), v["seat_rpm"])
        obs = self.obs()
        # ---- termination
        stripped = J[:, SR.J_STRIPPED] > 0.5
        v["strip_t"] = torch.where(stripped & (v["strip_t"] < 0), t, v["strip_t"])
        fail = J[:, SR.J_FAIL]
        code = torch.zeros(self.n, dtype=torch.long, device=self.dev)

        def put(cond, val):
            nonlocal code
            code = torch.where((code == 0) & cond, torch.full_like(code, val), code)
        put((J[:, SR.J_SUCCESS] > 0.5) & (v["seat_rpm"] > SEAT_RPM_MAX + 1e-3), 11)
        put(J[:, SR.J_SUCCESS] > 0.5, 1)
        put((fail == SR.FAIL_CLUTCH) & (J[:, SR.J_CROSSED] > 0.5), 2)
        put(fail == SR.FAIL_CLUTCH, 3)
        put(stripped & stop, 5)
        put(stripped & (t - v["strip_t"] > 0.5), 4)
        put(fail == SR.FAIL_CAM, 6)
        put(fail == SR.FAIL_SLIP, 7)
        put(stop, 8)
        put(t > c["timeout_s"], 9)
        # ---- reward (truth)
        lat, tilt = self._lat_to_insert(), self._tilt_err()
        free = (mode0 < 0.5) & (mode < 0.5)
        r = torch.full((self.n,), -0.005, device=self.dev)
        r = r + free.float() * (0.5 * (v["lat_last"] - lat) / 1e-3 + 0.3 * torch.rad2deg(v["tilt_last"] - tilt))
        theta = J[:, SR.J_THETA]
        eng_now = (mode >= 0.5) & (mode0 >= 0.5) & (mode < 2.5)
        crossed = J[:, SR.J_CROSSED] > 0.5
        dturn = (theta - v["theta_last"]) / (2 * math.pi)
        r = r + (eng_now & ~crossed & ~stripped).float() * 0.3 * dturn.clamp(-1, 1)
        bind = p["kx"] * theta / (2 * math.pi) / p["tset"]
        r = r - (eng_now & crossed).float() * 0.05 * bind.clamp_min(0)
        v["lat_last"], v["tilt_last"], v["theta_last"] = lat, tilt, theta * (mode >= 0.5).float()
        done = code > 0
        term = torch.zeros_like(r)
        for val, rw in ((1, 10.0), (2, -5.0), (3, -5.0), (4, -5.0), (5, -1.0), (6, -5.0), (7, -5.0), (8, -2.0), (9, -5.0), (11, -3.0)):
            term = torch.where(code == val, torch.full_like(r, rw), term)
        r = r + term
        success = code == 1
        info = dict(code=code, success=success, belief_err=p["belief_err"].clone(), thx=p["thx"].clone(), backouts=v["backouts"].clone(),
                    size=p["size"].clone(), phil=p["phil"].clone(), modeb=p["modeb"].clone())
        nonfinite = ~torch.isfinite(self.state).all(-1).all(-1) | ~torch.isfinite(obs).all(-1)
        if nonfinite.any():
            done = done | nonfinite
            code = torch.where(nonfinite, torch.full_like(code, 10), code)
            info["code"] = code
            info["success"] = success & ~nonfinite
            r = torch.where(nonfinite, torch.full_like(r, -3.0), r)
            obs = torch.nan_to_num(obs)
            self.state[nonfinite] = 0
            self.state[nonfinite, :, 6] = 1
        v["prev_mode"] = mode.clone()
        return obs, r, done, info
