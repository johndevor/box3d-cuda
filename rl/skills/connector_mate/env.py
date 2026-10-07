"""Batched connector mating on box3d-cuda, to World2's learned-skill contract `world2-learned-skill/1` (insert skill).

The connector is modelled as its interaction, not its geometry (rl/skills/peg_insert/env.py is the base: the same contract
observation, action, controller law, runtime limits and termination):
  - the housing's sliding mate and lead-in funnel: a box plug (w x t cross-section) into a rectangular pocket of static
    box walls with 45-degree funnel boxes, turned by a random yaw about the axis (box3d-cuda manifold_step contacts);
  - the latch: rl/ext/detent.cu (an optional extension), one prismatic detent per world stepped each substep before the contact solve:
    force rise to a peak, the click (drop), sliding friction, the withdrawal hold once latched, jam above a tilt angle,
    a bent pin when pushed hard while tilted (oracle: detent_reference.py);
  - stubbing on an offset: the plug's face meets the housing's face outside the funnel (geometry).
Domain randomisation per episode: the curve (peak 5-30 N, click position, drop width, hold, friction), funnel width,
clearance, sizes, Coulomb friction (groups), pose error from a D405-class sigma, in-hand error, wrist F/T noise / delay /
gain / drift, one-tick action latency, motor lag, virtual mass, and the compliance tier: World2's contact tier (the
policy's stiffness) or its physical tier (a passive RCC nose: lateral 5 N/mm, axial 50 N/mm, rocking 5 N m/rad, the
policy's stiffness ignored). The design data World2 gives for a socket has no pocket size: in a fraction of episodes the
feature size, clearance and chamfer read 0, as World2's design proxy reports them.

Success: the runtime ended the skill as seated (depth / policy within tolerance / stop) with the latch closed, the plug
within 0.3 mm of the planned depth and no pin bent.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from rl.common.cuda import load_ext
from rl.common.randomization import Bern, IntU, LogU, Spec, U
from rl.ext import detent_reference as DR
from rl.skills.peg_insert.env import (B, N_STATIC, LIMITS, ACTION_SCALE, RATE_HZ, OBS_FIELDS, OBS_SIZE, PegInsertBatch, loguni, uni,
                                      qmul, qconj, qrot, quat_of, rotvec_between, rotvec_of, to_task, ROOT)

PRIV_SIZE = 33
NOSE = dict(k_lat=5000.0, k_ax=5.0e4, k_rot=5.0)


SPEC = Spec(dict(
    w_mm=U(4.0, 14.0), t_mm=U(2.5, 8.0), length_mm=U(10.0, 25.0), depth_mm=U(3.0, 9.0), floor_extra_mm=U(0.0, 1.0), floor_flush_p=Bern(0.3),
    clearance_mm=LogU(0.03, 0.6), funnel_mm=U(0.2, 1.5), yaw_err_deg=1.0,
    peak_n=U(5.0, 30.0), click_before_depth_mm=U(0.2, 1.5), ramp_mm=U(0.5, 2.0), drop_mm=U(0.08, 0.4), res_n=U(0.2, 2.0),
    hold_over_peak=U(1.5, 4.0), jam_deg=U(2.0, 6.0), jam_engage_mm=U(0.3, 1.0), jam_n=U(40.0, 120.0), bend_n=U(8.0, 40.0),
    pin_engage_mm=U(0.5, 2.0), max_over_peak=U(1.2, 2.0), max_force_n=U(20.0, 60.0),
    friction=U(0.1, 0.6), friction_groups=8, density=U(1100.0, 1800.0),
    start_height_mm=U(1.0, 15.0), start_lateral_mm=U(0.0, 3.0), start_tilt_deg=U(0.0, 3.5),
    target_sigma_mm=U(0.1, 1.0), target_sigma_rot_deg=U(0.05, 1.5), in_hand_sigma_mm=U(0.05, 0.5),
    target_err_z_mm=U(0.02, 0.2), in_hand_err_z_mm=U(0.02, 0.15), in_hand_rot_deg=0.2,
    ft_profile=IntU(0, 3), ft_extra_delay_s=U(0.0, 0.012), ft_gain_err=U(0.005, 0.01), ft_noise_mult=U(1.0, 1.5), flange_h=U(0.08, 0.2),
    action_latency_p=Bern(0.3), motor_lag_s=U(0.0, 0.03), vmass=U(1.7, 2.3), physical_p=Bern(0.35), feature_known_p=Bern(0.5),
    detent=1.0,                        # 0: no detent force (the sim-match's sliding-fit probe; World2 latches by rule)
    chain_states=None, chain_sigma_mm=0.3, chain_present_mm=(0.0, 0.5),   # the chain test's step 2 (chain.py)
    substeps_per_world_step=4, solver_iterations=8, timeout_s=6.0, pose_scale=1.0, belief_error_scale=1.0,
    # (unused by this env, drawn by the base class)
    d_mm=LogU(3.0, 16.0), length_over_d=U(2.5, 6.0), depth_over_d=U(0.8, 2.0), host_chamfer_mm=U(0.2, 1.2), part_chamfer_mm=U(0.1, 1.0),
))


def default_cfg(**over):
    return SPEC.override(over)


def qyaw(a):
    return torch.stack([torch.zeros_like(a), torch.zeros_like(a), torch.sin(a / 2), torch.cos(a / 2)], -1)


def pocket_statics(ax_, ay_, D, ch, yaw):
    """13 static boxes (mm) of a rectangular pocket, mouth at z = 0, floor at z = -D, half-widths ax_ (x) and ay_ (y),
    a 45-degree lead-in of ch at the mouth, all turned by yaw about z: per side a lower wall, an upper wall and a funnel
    box; a floor. Returns pos [m,13,3], half [m,13,3], quat [m,13,4]."""
    dev, m = ax_.device, len(ax_)
    T = torch.full_like(ax_, 5.0)
    W = torch.maximum(ax_, ay_) + ch + T
    pos = torch.zeros(m, N_STATIC, 3, device=dev)
    half = torch.zeros(m, N_STATIC, 3, device=dev)
    quat = torch.zeros(m, N_STATIC, 4, device=dev)
    quat[..., 3] = 1
    r2 = math.sqrt(0.5)
    qy = qyaw(yaw)
    qc = torch.tensor([0, math.sin(math.pi / 8), 0, math.cos(math.pi / 8)], device=dev).expand(m, 4)
    for k in range(4):
        phi = k * math.pi / 2
        a = ax_ if k % 2 == 0 else ay_
        qz = qmul(qy, torch.tensor([0, 0, math.sin(phi / 2), math.cos(phi / 2)], device=dev).expand(m, 4))
        local = [
            (torch.stack([a + (ch + T) / 2, 0 * a, -(D + ch) / 2], -1), torch.stack([(ch + T) / 2, W, (D - ch) / 2], -1), None),
            (torch.stack([a + ch + T / 2, 0 * a, -ch / 2], -1), torch.stack([T / 2, W, ch / 2], -1), None),
            (torch.stack([a + ch, 0 * a, -ch], -1), torch.stack([ch * r2, W, ch * r2], -1), qc),
        ]
        for j, (p, hf, q) in enumerate(local):
            i = 3 * k + j
            pos[:, i] = qrot(qz, p)
            half[:, i] = hf
            quat[:, i] = qmul(qz, q) if q is not None else qz
    pos[:, 12] = torch.stack([0 * D, 0 * D, -D - T / 2], -1)
    half[:, 12] = torch.stack([W, W, T / 2], -1)
    quat[:, 12] = qy
    return pos, half, quat


class ConnectorMateBatch(PegInsertBatch):
    """N independent worlds. step(action[N,9]) -> obs[N,57], reward[N], done[N], info; priv() -> [N, PRIV_SIZE]."""

    priv_size = PRIV_SIZE

    def __init__(self, n, device="cuda", seed=0, cfg=None):
        if torch.device(device).type == "cuda":
            self.dext = load_ext("b3_detent")
        super().__init__(n, device=device, seed=seed, cfg=cfg if isinstance(cfg, Spec) else default_cfg(**(cfg or {})))

    code_names = property(lambda self: CODE_NAMES)

    # ------------------------------------------------------------ sampling and geometry
    def _sample(self, idx):
        P = super()._sample(idx)
        c, g, d, m = self.cfg, self.g, self.dev, len(idx)
        D = lambda k: c.draw(k, g, m, d)
        mm = 1e-3
        P["w"] = D("w_mm") * mm
        P["t"] = torch.minimum(D("t_mm") * mm, P["w"])
        P["d"] = torch.minimum(P["w"], P["t"])                      # (the base class's lead size)
        P["L"] = D("length_mm") * mm
        P["depth"] = torch.minimum(D("depth_mm") * mm, 0.6 * P["L"])
        flush = D("floor_flush_p") > 0.5
        P["floor"] = P["depth"] + torch.where(flush, torch.zeros(m, device=d), D("floor_extra_mm") * mm)
        P["clear"] = D("clearance_mm") * mm
        P["hch"] = torch.minimum(D("funnel_mm") * mm, 0.4 * P["depth"])
        P["pch"] = torch.zeros(m, device=d)
        P["mass"] = D("density") * P["w"] * P["t"] * P["L"]
        P["yaw"] = uni(g, m, (0.0, math.pi), d)
        # the detent's curve and failure thresholds
        P["fpk"] = D("peak_n")
        xc = torch.clamp(P["depth"] - D("click_before_depth_mm") * mm, min=0.6 * mm)
        P["xclick"] = xc
        P["xramp"] = torch.clamp(xc - D("ramp_mm") * mm, min=0.2 * mm)
        P["dropw"] = torch.minimum(D("drop_mm") * mm, torch.clamp(P["floor"] - xc - 0.02 * mm, min=0.03 * mm))
        P["fres"] = D("res_n")
        P["fwd"] = P["fpk"] * D("hold_over_peak")
        P["thjam"] = torch.deg2rad(D("jam_deg"))
        P["xjam"] = D("jam_engage_mm") * mm
        P["fjam"] = D("jam_n")
        P["fbend"] = D("bend_n")
        P["xpin"] = D("pin_engage_mm") * mm
        P["maxF"] = torch.clamp(P["fpk"] * D("max_over_peak"), min=c["max_force_n"][0], max=c["max_force_n"][1])
        P["fpk"] = torch.minimum(P["fpk"], P["maxF"] / 1.15)
        P["phys"] = D("physical_p")
        P["fknown"] = D("feature_known_p")
        # the target turned by the socket's yaw; the plug presented at that yaw plus a small error; beliefs as the base's
        qy = qyaw(P["yaw"])
        P["qT"] = qy
        P["qT_b"] = qmul(P["qT_b"], qy)
        yerr = torch.deg2rad(torch.randn(m, generator=g, device=d) * c["yaw_err_deg"] * c["pose_scale"])
        P["q0"] = qmul(P["q0"], qyaw(P["yaw"] + yerr))
        if c.get("chain_states"):
            self._chain(P, idx)
        return P

    def _chain(self, P, idx):
        """Step 2 of the chain test: the socket's true pose is a placed housing's (step 1's end states, restored from the
        snapshot file): the program believes the nest's layout pose, so the target error is minus the placement's offset
        and the plug, presented at the believed yaw, is off by the placement's yaw. The stated sigma is the nest's."""
        c, g, d, m = self.cfg, self.g, self.dev, len(idx)
        if not hasattr(self, "_chain_db"):
            db = torch.load(c["chain_states"], map_location="cpu")
            ok = db["ok"]
            self._chain_db = (db["off_mm"][ok].to(d) * 1e-3, db["yaw_rad"][ok].to(d))
        off, yaw = self._chain_db
        j = torch.randint(0, len(off), (m,), generator=g, device=d)
        o, dy = off[j], yaw[j]
        # (the socket at the housing's centre: the world frame is the true socket's; the belief is the layout's)
        P["terr"] = torch.stack([-o[:, 0], -o[:, 1], P["terr"][:, 2]], -1)
        sig = c.get("chain_sigma_mm", 0.3) * 1e-3
        P["tsp"] = torch.full((m,), sig, device=d)
        # the plug presented over the believed target (the in-hand error and the presentation as sampled, about the belief)
        ang = uni(g, m, (0, 2 * math.pi), d)
        r = uni(g, m, c.get("chain_present_mm", (0.0, 0.5)), d) * 1e-3
        P["tip0"] = torch.stack([-o[:, 0] + r * torch.cos(ang), -o[:, 1] + r * torch.sin(ang), P["tip0"][:, 2]], -1)
        P["q0"] = qmul(P["q0"], qyaw(-dy))
        P["qT_b"] = qmul(P["qT_b"], qyaw(-dy))

    def _geometry(self, idx):
        """Static boxes (mm): the socket's pocket (pocket_statics), turned by the yaw."""
        P, mmf = self.p, 1000.0
        pos, half, quat = pocket_statics((P["w"][idx] / 2 + P["clear"][idx]) * mmf, (P["t"][idx] / 2 + P["clear"][idx]) * mmf,
                                         P["floor"][idx] * mmf, P["hch"][idx] * mmf, P["yaw"][idx])
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
        super().reset(mask)
        p = self.p
        hs = torch.stack([p["w"][idx] / 2, p["t"][idx] / 2, p["L"][idx] / 2], -1)
        self.half[idx, 0] = hs * 1000.0
        Ib = p["mass"][idx, None] / 3 * (hs[:, [1, 0, 0]] ** 2 + hs[:, [2, 2, 1]] ** 2) + 0.002
        self.I[idx] = Ib.max(-1).values
        self.inv_inertia[idx, 0] = 1.0 / (Ib * 1e6)
        m = len(idx)
        self._set("det_state", idx, torch.zeros(m, 3, device=self.dev))
        self._set("det_params", idx, torch.stack([p[k][idx] for k in ("xramp", "xclick", "dropw", "fpk", "fres", "fwd", "thjam", "xjam", "fjam", "fbend", "xpin")], -1))
        self._set("clicked", idx, torch.zeros(m, device=self.dev))
        self._set("click_t", idx, torch.full((m,), -1.0, device=self.dev))

    # ------------------------------------------------------------ one controller substep: the impedance law, the detent, the contacts
    def _tilt(self):
        a = qrot(self.state[:, 0, 3:7], torch.tensor([0.0, 0.0, -1.0], device=self.dev).expand(self.n, 3))
        return torch.atan2(a[:, :2].norm(dim=-1), -a[:, 2])

    def _substep(self, ref, nq, k, krot):
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
        phys = p["phys"] > 0.5
        k_lat = torch.where(phys, torch.full_like(k, NOSE["k_lat"]), k)
        k_ax = torch.where(phys, torch.full_like(k, NOSE["k_ax"]), k)
        kr = torch.where(phys, torch.full_like(krot, NOSE["k_rot"]), krot)
        e = v["ref_eff"] - tip
        kv = torch.stack([k_lat, k_lat, k_ax], -1)
        cv = 2 * torch.sqrt(kv * M[:, None])
        F = kv * e - cv * vT
        I = self.I
        Tq = kr[:, None] * rotvec_between(q, nq) - (2 * torch.sqrt(kr * I))[:, None] * w
        Ttot = Tq + torch.cross(rT, F, dim=-1)
        v_pre = vel + F / M[:, None] * h
        w_pre = w + Ttot / I[:, None] * h
        # the detent (axis -z): progress, inward velocity of the tip, tilt, the controller's inward push
        vT_pre = v_pre + torch.cross(w_pre, rT, dim=-1)
        x = -tip[:, 2]
        if self.cfg["detent"] <= 0:          # (the sim-match's sliding-fit probe: no latch force)
            J, ds, clk = torch.zeros_like(x), v["det_state"], torch.zeros_like(x)
        elif self.dev.type == "cuda":
            J, ds, clk = self.dext.detent_step(x, -vT_pre[:, 2], self._tilt(), -F[:, 2], M, v["det_params"], v["det_state"], h)
        else:
            J, ds, clk = DR.detent_step(x, -vT_pre[:, 2], self._tilt(), -F[:, 2], M, v["det_params"], v["det_state"], h)
        v["det_state"] = ds
        v["clicked"] = torch.maximum(v["clicked"], clk)
        Jv = torch.stack([torch.zeros_like(J), torch.zeros_like(J), J], -1)            # outward = +z
        v_pre = v_pre + Jv / M[:, None]
        w_pre = w_pre + torch.cross(rT, Jv, dim=-1) / I[:, None]
        st[:, 0, 7:10] = v_pre * 1000.0
        st[:, 0, 10:13] = w_pre
        self._kernel()
        fc = (self.state[:, 0, 7:10] / 1000.0 - v_pre) * M[:, None] / h + Jv / h
        tc = (self.state[:, 0, 10:13] - w_pre) * I[:, None] / h
        flange = tip + qrot(q, torch.stack([0 * M, 0 * M, p["L"] + p["flange"]], -1))
        Tf = Tq + torch.cross(tip - flange, F, dim=-1)
        return -torch.cat([F, Tf], -1), torch.cat([fc, tc], -1)

    # ------------------------------------------------------------ observation
    def _features(self):
        p = self.p
        oh = torch.zeros(self.n, 8, device=self.dev)
        oh[:, 0] = 1.0                                              # box
        kn = p["fknown"]
        return torch.cat([oh, torch.stack([p["d"], p["L"], kn * (p["t"] + 2 * p["clear"]), kn * p["clear"], p["pch"], kn * p["hch"], p["depth"], p["mass"]], -1)], -1)

    def priv(self):
        v, p = self.v, self.p
        tip, q = self._tip(), self._q()
        D = p["depth"]
        tt = torch.stack([0 * D, 0 * D, -D], -1)
        st = self.state[:, 0]
        x = -tip[:, 2]
        ds = v["det_state"]
        return torch.cat([to_task(tt - tip) * 1000.0, to_task(rotvec_between(q, p["qT"])) * 10.0, to_task(st[:, 7:10] / 1000.0) * 100.0,
                          v["contact_w"][:, :3] / 20.0, v["contact_w"][:, 3:] / 0.5,
                          (p["clear"] * 1e4)[:, None], self.group_mu[self.group_of][:, None], (p["hch"] * 1e3)[:, None],
                          (p["lag"] * 50)[:, None], p["latency"][:, None], (p["vmass"] - 2)[:, None],
                          ((tt - tip)[:, :2].norm(dim=-1) * 1000)[:, None], (v["k"] / 8000)[:, None],
                          ds, (p["fpk"] / 30)[:, None], ((p["xclick"] - x) * 1000)[:, None], ((p["xramp"] - x) * 1000)[:, None],
                          (self._tilt() / p["thjam"])[:, None], p["phys"][:, None], (p["maxF"] / 60)[:, None], ((p["floor"] - x) * 1000)[:, None]], -1)

    # ------------------------------------------------------------ step (the base's, with the latch in the outcome)
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
        along = (rel @ ax).clamp(min=-0.03).minimum(depth + LIMITS["max_axial_m"])
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
        clicked_before = v["clicked"].clone()
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
        pb = v["prog_b"]
        fW = v["reading"][:, :3]
        push = -(fW @ ax)
        v["peak_push"] = torch.maximum(v["peak_push"], push)
        t = v["t"]
        code = torch.zeros(self.n, dtype=torch.long, device=self.dev)
        fn = fW.norm(dim=-1)
        over = fn > 1.25 * maxF
        v["over_since"] = torch.where(over, torch.where(v["over_since"] < 0, t, v["over_since"]), torch.full_like(t, -1.0))
        moved = (pb - v["stall_p"]).abs() > 5e-5
        v["stall_since"] = torch.where(moved, t, v["stall_since"])
        v["stall_p"] = torch.where(moved, pb, v["stall_p"])
        still = t - v["stall_since"]
        ds = v["det_state"]
        latched, jammed, bent = ds[:, 0] > 0.5, ds[:, 1] > 0.5, ds[:, 2] > 0.5

        def put(cond, val):
            nonlocal code
            code = torch.where((code == 0) & cond, torch.full_like(code, val), code)
        put(bent, 10)
        put(pb >= depth, 1)
        put(terminate & (pb >= depth - 3e-4), 2)
        put(terminate, 3)
        put(t > c["timeout_s"], 4)
        put(over & (v["over_since"] >= 0) & (t - v["over_since"] > 0.1), 5)
        put((~moved) & (pb >= depth - 3e-4) & (still > 0.25), 6)
        put((~moved) & (still > 3) & (push > 0.25 * maxF) & (pb < 3e-4), 7)
        put((~moved) & (still > 3) & (push > 0.25 * maxF), 8)
        tp, tl = self._true_progress(), self._true_lateral()
        seated_code = (code == 1) | (code == 2) | (code == 6)
        success = seated_code & latched & (tp >= depth - 3e-4) & ~bent
        code = torch.where(seated_code & ~success, torch.full_like(code, 11), code)
        # ---- reward (truth): progress, centring before the mouth, the click, force and jam penalties, the outcome
        r = 4.0 * (torch.minimum(tp, depth) - torch.minimum(v["true_prog_last"], depth)) / depth
        r = r + 1.0 * (v["lat_err_last"] - tl) / 1e-3 * (tp < 0).float()
        v["true_prog_last"], v["lat_err_last"] = tp, tl
        newclick = (v["clicked"] > 0.5) & (clicked_before < 0.5)
        r = r + 2.0 * newclick.float() - 0.01 - 0.05 * (fn / maxF - 0.8).clamp_min(0) - 0.05 * jammed.float()
        done = code > 0
        r = r + torch.where(done, torch.where(success, torch.full_like(r, 10.0), torch.where(bent, torch.full_like(r, -6.0), torch.full_like(r, -3.0))), torch.zeros_like(r))
        info = dict(code=code, success=success, timeout=code == 4, true_depth=tp, clear=p["clear"].clone(), d=p["d"].clone(),
                    latched=latched, phys=p["phys"] > 0.5, peak=p["fpk"].clone())
        nonfinite = ~torch.isfinite(self.state).all(-1).all(-1)
        if nonfinite.any():
            done = done | nonfinite
            code = torch.where(nonfinite, torch.full_like(code, 9), code)
            info["code"] = code
            info["success"] = info["success"] & ~nonfinite
            r = torch.where(nonfinite, torch.full_like(r, -3.0), r)
            self.state[nonfinite] = 0
            self.state[nonfinite, :, 6] = 1
        return obs, r, done, info


CODE_NAMES = {1: "depth", 2: "policy", 3: "POLICY_STOPPED_SHORT", 4: "INSERT_TIMEOUT", 5: "INSERT_FORCE_LIMIT", 6: "stop",
              7: "INSERT_NOT_FOUND", 8: "INSERT_JAMMED", 9: "NONFINITE", 10: "PIN_BENT", 11: "NOT_LATCHED"}
