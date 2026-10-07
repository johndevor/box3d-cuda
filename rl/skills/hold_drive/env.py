"""Hold and drive on box3d-cuda: one robot holds a part (a plate with a clearance hole) while another inserts a pin through
it into a base's bore — World2's tasks/hold-and-drive bracket-pin case (operations/hold-session.mjs, compliant-insert.mjs).

The driver is World2's scripted force-controlled insertion, here rl/common/priors.py ScriptedInsertPrior on rl/skills/
peg_insert's 57 insert observations: the peg (pin) a dynamic box on the contract's impedance controller, the base's bore
box3d-cuda's static chamfered box walls (manifold_step, exact). The held part is reduced-order, in torch: a plate of
thickness tp lying over the bore's mouth (a seeded gap), its clearance hole (square, half-width d/2 + c2) off the bore by the
holder's grasp error (seeded, unknown to both); the plate moves (3 translations) under its holder's impedance — World2's
hold session: a spring-damper to the holder's anchor at the holder's stiffness, the reflected mass, the anchor moved and
the stiffness scaled by the holding policy; the grip's friction caps what the spring carries (beyond it the part slips in
the grip, past 3 mm it drops). The pin meets the plate by a stiff penalty contact: across the axis inside the hole's
span (the hole's walls), along it on the plate's top outside the hole.

The policy is the HOLDER's (World2 contract action kind `hold-compliance`, 5 values in [-1, 1]): the anchor's offset
(x, y across the joint axis ×1.5 mm, along it ×1 mm) and the stiffness scales across and along the axis (log-interpolated
between 0.02 and 1 of the holder's). The scripted coordinated hold (the prior) holds still and stiff: [0, 0, 0, 1, 1].
Observations (16): the holder's force reading (3, task frame, /10 N), its anchor offset (3, mm), its stiffness scales
(2, log), the driver's axial push (1, /10 N) and believed progress (1, mm/5), the time (1), the previous action (5).

Task frame as rl/skills/peg_insert: z up in the physics, insertion axis -z, the bore's mouth at z = 0.
"""
from __future__ import annotations

import math

import torch

from rl.common.priors import ScriptedInsertPrior
from rl.common.randomization import Bern, IntU, LogU, Spec, U
from rl.common.skill import cvec
from rl.skills.peg_insert.env import FT_PROFILES, SPEC as PEG_SPEC, PegInsertBatch, to_task, uni

N_ACT = 5
OBS_FIELDS = ["hold_wrench", "hold_offset", "hold_stiffness", "driver_push", "driver_progress", "time", "prev_action"]
OBS_SIZE = 3 + 3 + 2 + 1 + 1 + 1 + N_ACT
ACTION_SCALE = dict(offset_lat_m=0.0015, offset_ax_m=0.001, k_scale=(0.02, 1.0))
PRIOR_ACTION = (0.0, 0.0, 0.0, 1.0, 1.0)

# The bracket-pin family: World2's case (Ø4 pin, Ø4.4 hole in a 4 mm flange, Ø4.05 bore) inside wider ranges.
SPEC = PEG_SPEC.override(dict(
    d_mm=U(3.0, 6.0), length_over_d=U(8.0, 14.0), depth_over_d=U(2.0, 3.5), clearance_mm=LogU(0.015, 0.08),
    host_chamfer_mm=U(0.0, 0.3), start_height_mm=U(2.0, 5.0), start_lateral_mm=U(0.0, 0.4), start_tilt_deg=U(0.0, 0.5),
    target_sigma_mm=U(0.1, 0.5), max_force_n=U(30.0, 50.0), timeout_s=8.0,
))
SPEC = Spec({**SPEC.e, **dict(
    plate_t_mm=U(3.0, 6.0), plate_clear_mm=U(0.12, 0.3),      # the held part's thickness and its hole's radial clearance
    misalign_mm=U(0.0, 1.5),                                  # the grasp error: the held hole off the bore (unknown)
    gap_mm=U(0.0, 0.3),                                       # the held part over the bore's mouth
    hold_k_lat=LogU(3e4, 3e5), hold_k_ax=LogU(3e4, 3e5),      # the holder's stiffness at the grip (UR12e ~5e4, gantry ~2e5)
    hold_mass=U(1.5, 3.0), grip_force_n=U(80.0, 235.0), grip_mu=U(0.4, 0.6),
    hold_ft_noise_n=U(0.2, 1.0),
)})
PEN_K, PEN_ZETA = 2e5, 0.7


class HoldDriveBatch(PegInsertBatch):
    obs_size, act_size = OBS_SIZE, N_ACT
    priv_extra = 12

    def __init__(self, n, device="cuda", seed=0, cfg=None):
        self.prior = None
        super().__init__(n, device=device, seed=seed, cfg=cfg if isinstance(cfg, Spec) else SPEC.override(cfg or {}))
        self.prior = ScriptedInsertPrior().to(self.dev)
        self.priv_size = super().priv_size + self.priv_extra

    # ---------------------------------------------------------------- sampling, reset
    def _sample(self, idx):
        P = super()._sample(idx)
        c, g, d, m = self.cfg, self.g, self.dev, len(idx)
        D = lambda k: c.draw(k, g, m, d)
        P["tp"] = D("plate_t_mm") * 1e-3
        P["c2"] = D("plate_clear_mm") * 1e-3
        ang = uni(g, m, (0, 2 * math.pi), d)
        mis = D("misalign_mm") * 1e-3 * c["pose_scale"]
        P["mis"] = torch.stack([mis * torch.cos(ang), mis * torch.sin(ang)], -1)
        P["gap"] = D("gap_mm") * 1e-3
        P["klat"], P["kax"] = D("hold_k_lat"), D("hold_k_ax")
        P["hm"] = D("hold_mass")
        P["gripF"], P["gmu"] = D("grip_force_n"), D("grip_mu")
        P["hnoise"] = D("hold_ft_noise_n")
        # the pin starts over the held part's hole as the driver believes it: over the bore (its belief), a few mm up
        h = P["gap"] + P["tp"] + P["h0"]
        P["tip0"] = torch.stack([P["tip0"][:, 0], P["tip0"][:, 1], h], -1)
        return P

    def reset(self, mask):
        super().reset(mask)
        idx = mask.nonzero().squeeze(-1)
        if len(idx) == 0:
            return
        m, d = len(idx), self.dev
        z3 = torch.zeros(m, 3, device=d)
        self._set("pl_x", idx, z3.clone())            # plate displacement from where it was put (SI)
        self._set("pl_v", idx, z3.clone())
        self._set("anchor", idx, z3.clone())          # the holder's anchor (the plate's rest place + the policy's offset + slip)
        self._set("slip", idx, torch.zeros(m, device=d))
        self._set("hf", idx, z3.clone())              # the hold force on the plate (spring), last substep
        self._set("hreading", idx, z3.clone())
        self._set("hact", idx, torch.tensor(PRIOR_ACTION, device=d).expand(m, N_ACT).clone())
        self._set("pen_acc", idx, torch.zeros(m, device=d))
        self._set("prev_hact", idx, torch.tensor(PRIOR_ACTION, device=d).expand(m, N_ACT).clone())
        self._set("off_last", idx, z3.clone())
        self._set("dobs", idx, super().obs(update=False)[idx])

    # ---------------------------------------------------------------- the pin vs the held plate (penalty), the plate's motion
    def _plate_contact(self):
        """The penalty force on the pin (SI, world) and its point: the hole's walls across the axis where the pin passes
        the plate's span; the plate's top along the axis where the tip is on it outside the hole."""
        p, v = self.p, self.v
        st = self.state[:, 0]
        q, com = st[:, 3:7], st[:, 0:3] / 1000.0
        ax = cvec([0.0, 0.0, -1.0], self.dev).expand(self.n, 3)
        from rl.skills.peg_insert.env import qrot
        axis = qrot(q, cvec([0.0, 0.0, -1.0], self.dev).expand(self.n, 3))           # the pin's own axis, tip-ward
        tip = self._tip()
        zb, zt = p["gap"] + v["pl_x"][:, 2], p["gap"] + p["tp"] + v["pl_x"][:, 2]       # the plate's bottom and top
        hole = p["mis"] + v["pl_x"][:, :2]
        half_free = p["c2"]                                                            # radial clearance of the hole
        s = p["d"] / 2
        # the pin's axis at the plate's mid-height (if the pin spans it)
        zm = (zb + zt) / 2
        tz = ((zm - tip[:, 2]) / (-axis[:, 2]).clamp(min=1e-3)).clamp(min=0.0)          # back along the axis from the tip
        tz = torch.minimum(tz, p["L"])
        pm = tip - axis * tz[:, None]
        spans = (tip[:, 2] < zt) & (tip[:, 2] + p["L"] * 0.99 > zb)
        r = pm[:, :2] - hole
        pen = (r.abs() - half_free[:, None]).clamp(min=0.0) * torch.sign(r)
        inside = (r.abs() <= half_free[:, None] + s[:, None]).all(-1)                  # the pin's section over the hole
        lat_on = spans & inside
        f_lat = -PEN_K * pen * lat_on[:, None].float()
        # along the axis: the tip on the plate's top, outside the hole's opening
        rt = tip[:, :2] - hole
        over = (rt.abs() > half_free[:, None]).any(-1)
        dz = (zt - tip[:, 2]).clamp(min=0.0)
        top_on = over & (dz > 0) & (dz < 0.002)
        f_ax = PEN_K * dz * top_on.float()
        F = torch.cat([f_lat, f_ax[:, None]], -1)
        point = torch.where(top_on[:, None], tip, pm)
        # damping on the relative velocity at the point
        vel = st[:, 7:10] / 1000.0
        w = st[:, 10:13]
        vp = vel + torch.cross(w, point - com, dim=-1) - v["pl_v"]
        M = p["vmass"]
        active = torch.cat([lat_on[:, None].expand(-1, 2), top_on[:, None]], -1).float()
        F = F - 2 * PEN_ZETA * torch.sqrt(PEN_K * M)[:, None] * vp * active
        return F, point, (pen.abs().max(-1).values * lat_on.float() + dz * top_on.float())

    def _substep(self, ref, nq, k, krot):
        p, v = self.p, self.v
        h = self.h
        F, point, pen = self._plate_contact()
        # on the pin: the penalty force at its point (velocity change before the controller and the kernel)
        st = self.state
        M, I = p["vmass"], self.I
        com = st[:, 0, 0:3] / 1000.0
        st[:, 0, 7:10] = st[:, 0, 7:10] + (F / M[:, None] * h) * 1000.0
        st[:, 0, 10:13] = st[:, 0, 10:13] + torch.cross(point - com, F, dim=-1) / I[:, None] * h
        v["pen_acc"] = torch.maximum(v["pen_acc"], pen)
        out = super()._substep(ref, nq, k, krot)
        # the plate: its holder's spring-damper to the anchor (per direction), the pin's reaction, its seat under it
        x, vel = v["pl_x"], v["pl_v"]
        a = v["hact"]
        lo, hi = ACTION_SCALE["k_scale"]
        ks_lat = lo * (hi / lo) ** ((a[:, 3] + 1) / 2)
        ks_ax = lo * (hi / lo) ** ((a[:, 4] + 1) / 2)
        kl, ka, mh = p["klat"] * ks_lat, p["kax"] * ks_ax, p["hm"]
        Kv = torch.stack([kl, kl, ka], -1)
        e = v["anchor"] - x
        fs = Kv * e - 2 * torch.sqrt(Kv * mh[:, None]) * vel
        # the grip's capacity across the jaws (the joint's axis lies along the pads' normals: the push along it is carried
        # by the pads' normal force up to twice the grip force)
        cap_lat = 2 * p["gmu"] * p["gripF"]
        flat = fs[:, :2].norm(dim=-1)
        over = flat > cap_lat
        sc = torch.where(over, cap_lat / flat.clamp_min(1e-9), torch.ones_like(flat))
        slip_d = (e[:, :2] * (1 - sc[:, None])).norm(dim=-1)
        v["anchor"] = torch.cat([v["anchor"][:, :2] - e[:, :2] * (1 - sc[:, None]), v["anchor"][:, 2:]], -1)
        v["slip"] = v["slip"] + slip_d
        fs = torch.cat([fs[:, :2] * sc[:, None], fs[:, 2:]], -1)
        # the plate's seat: it cannot go down into the base (its bottom at the mouth)
        zb = p["gap"] + x[:, 2]
        fseat = torch.where(zb < 0, -zb * PEN_K - 2 * PEN_ZETA * torch.sqrt(PEN_K * mh) * vel[:, 2].clamp(max=0), torch.zeros_like(zb))
        ftot = fs - F + torch.stack([torch.zeros_like(zb), torch.zeros_like(zb), fseat], -1)
        vel = vel + ftot / mh[:, None] * h
        v["pl_v"] = vel
        v["pl_x"] = x + vel * h
        v["hf"] = fs
        return out

    # ---------------------------------------------------------------- the holder's observation
    def obs(self, update=True):
        dobs = super().obs(update)
        if self.prior is None:            # (the parent's constructor, before the holder exists)
            return dobs
        v, p = self.v, self.p
        if update:
            v["dobs"] = dobs
        a = v["hact"]
        lo, hi = ACTION_SCALE["k_scale"]
        off = torch.stack([a[:, 0] * ACTION_SCALE["offset_lat_m"], a[:, 1] * ACTION_SCALE["offset_lat_m"], a[:, 2] * ACTION_SCALE["offset_ax_m"]], -1)
        kls = torch.stack([(a[:, 3] + 1) / 2 * math.log(hi / lo) + math.log(lo), (a[:, 4] + 1) / 2 * math.log(hi / lo) + math.log(lo)], -1)
        push = dobs[:, 14:15]                                   # the driver's wrench reading, task z (the push is −F_z)
        prog = dobs[:, 45:46]
        return torch.cat([to_task(v["hreading"]) / 10.0, to_task(off) * 1000.0, kls, -push / 10.0, prog * 1000.0 / 5.0, v["t"][:, None], v["prev_hact"]], -1)

    def priv(self):
        v, p = self.v, self.p
        base = super().priv()
        return torch.cat([base, p["mis"] * 1000.0, (p["gap"] * 1000.0)[:, None], v["pl_x"] * 1000.0, (v["slip"] * 1000.0)[:, None],
                          (p["klat"] / 1e5)[:, None], (p["kax"] / 1e5)[:, None], (p["gripF"] / 100)[:, None], (p["c2"] * 1e4)[:, None], (p["tp"] * 1e3)[:, None]], -1)

    # ---------------------------------------------------------------- step: the holder's action, the scripted driver
    def _step(self, action):
        v, p = self.v, self.p
        a = torch.nan_to_num(action.float()).clamp(-1, 1)
        v["prev_hact"] = v["hact"].clone()
        v["hact"] = a
        off = torch.stack([a[:, 0] * ACTION_SCALE["offset_lat_m"], a[:, 1] * ACTION_SCALE["offset_lat_m"], a[:, 2] * ACTION_SCALE["offset_ax_m"]], -1)
        # the anchor: where the plate was put + the policy's offset (+ the slip so far, kept in the anchor's history)
        base = v["anchor"] - v["off_last"]
        v["anchor"] = base + off
        v["off_last"] = off
        slip0 = v["slip"].clone()
        driver = self.prior(v["dobs"])
        obs, r, done, info = super()._step(driver)
        # the holder's reading: the spring's force on the plate, reacting on the holder, with noise
        v["hreading"] = -v["hf"] + torch.randn(self.n, 3, generator=self.g, device=self.dev) * p["hnoise"][:, None]
        dropped = v["slip"] > 0.003
        code = torch.where(dropped & ~done, torch.full_like(info["code"], 10), info["code"])
        done = done | dropped
        success = info["success"] & ~dropped
        r = r - 300.0 * (v["slip"] - slip0) - 0.002 * (v["hf"].norm(dim=-1) / 10.0)
        r = torch.where(dropped, r - 3.0, r)
        info.update(code=code, success=success, misalign=p["mis"].norm(dim=-1).clone(), slip_mm=v["slip"] * 1000.0)
        obs = self.obs(update=False)
        return obs, r, done, info


CODE_NAMES = {1: "depth", 2: "policy", 3: "POLICY_STOPPED_SHORT", 4: "INSERT_TIMEOUT", 5: "INSERT_FORCE_LIMIT", 6: "stop",
              7: "INSERT_NOT_FOUND", 8: "INSERT_JAMMED", 9: "NONFINITE", 10: "HOLD_SLIP"}
HoldDriveBatch.code_names = property(lambda self: CODE_NAMES)
