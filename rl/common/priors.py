"""Priors: fixed controllers the policy corrects (the policy outputs a residual on top). They are part of the exported
graph, so World2 runs prior + correction through its unchanged runtime. Written with operators World2's pure-JS ONNX
evaluator has (MatMul, Mul, Add, Sub, Abs, Relu, Sigmoid, Tanh, Clip): observation components are picked by constant
selector matrices, not slices.

GaitPrior (locomotion, reusable): a phase clock and parametric stepping. From the observation's gait clock (sin, cos of
  the phase) and command: each side's swing weight s_L = relu(sin), s_R = relu(-sin) (each leg swings in its half of the
  cycle), scaled by how much the robot is asked to move, a = clip(|vx|/v_ref + |wz|/w_ref, 0, 1); the action is
  amp * a * (lift_L s_L + lift_R s_R) + stride * a * vx/v_ref * (stride_L s_L + stride_R s_R): per-joint coefficients
  for lifting a foot (hip, knee, ankle flexion) and for carrying it forward. A robot gives its joints' coefficients.
OscillatorPrior (locomotion, any robot): one sinusoid per driven joint on the gait clock, a = clip(|vx|/v_ref, 0, 1);
  action_j = a * (S_j sin + C_j cos) + O_j (amplitude and phase per joint, an offset): the open-loop pattern a generated
  robot's probe picks (rl/skills/locomotion: World2's judge on random patterns of the nominal robot).
ScriptedInsertPrior (World2's insert contract): the scripted insertion re-expressed per tick: centre on the believed
  target and descend at fixed stiffness; when stuck at the mouth (axial push over contact_n before the hole), a spiral
  search: the previous lateral action turned by a fixed angle and grown, kept within the search radius by the clip.
ScriptedScrewPrior (World2's screw-drive contract): the scripted drive: centre the tip on the believed seat, align the
  screw with the believed axis, push, spin forward, slow to final_rpm two pitches before the seat; never stop.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def selector(n_in, cols):
    """A constant [n_in, len(cols)] matrix picking columns (obs @ S)."""
    S = torch.zeros(n_in, len(cols))
    for j, c in enumerate(cols):
        S[c, j] = 1.0
    return S


class GaitPrior(nn.Module):
    def __init__(self, n_in, n_act, sin_idx, cmd_vx_idx, cmd_wz_idx, lift_l, lift_r, stride_l=None, stride_r=None, amp=0.5, stride=0.0, v_ref=0.15, w_ref=0.4):
        super().__init__()
        self.args = dict(n_in=n_in, n_act=n_act, sin_idx=sin_idx, cmd_vx_idx=cmd_vx_idx, cmd_wz_idx=cmd_wz_idx, lift_l=list(lift_l), lift_r=list(lift_r),
                         stride_l=list(stride_l or [0.0] * n_act), stride_r=list(stride_r or [0.0] * n_act), amp=amp, stride=stride, v_ref=v_ref, w_ref=w_ref)
        self.register_buffer("S", selector(n_in, [sin_idx, cmd_vx_idx, cmd_wz_idx]))
        self.register_buffer("lift_l", torch.tensor(lift_l, dtype=torch.float32)[None] * amp)
        self.register_buffer("lift_r", torch.tensor(lift_r, dtype=torch.float32)[None] * amp)
        self.register_buffer("stride_l", torch.tensor(self.args["stride_l"], dtype=torch.float32)[None] * stride)
        self.register_buffer("stride_r", torch.tensor(self.args["stride_r"], dtype=torch.float32)[None] * stride)
        self.v_ref, self.w_ref = v_ref, w_ref

    def forward(self, obs):
        z = obs @ self.S                                   # [N, 3]: sin, vx, wz
        s, vx, wz = z[:, 0:1], z[:, 1:2], z[:, 2:3]
        sl, sr = torch.relu(s), torch.relu(-s)
        a = torch.clamp(torch.abs(vx) * (1 / self.v_ref) + torch.abs(wz) * (1 / self.w_ref), 0.0, 1.0)
        fwd = vx * (1 / self.v_ref)
        return a * (sl * self.lift_l + sr * self.lift_r) + a * fwd * (sl * self.stride_l + sr * self.stride_r)


class OscillatorPrior(nn.Module):
    def __init__(self, n_in, n_act, sin_idx, cos_idx, cmd_vx_idx, S, C, O=None, v_ref=0.15):
        super().__init__()
        self.args = dict(n_in=n_in, n_act=n_act, sin_idx=sin_idx, cos_idx=cos_idx, cmd_vx_idx=cmd_vx_idx, S=list(S), C=list(C), O=list(O or [0.0] * n_act), v_ref=v_ref)
        self.register_buffer("Sel", selector(n_in, [sin_idx, cos_idx, cmd_vx_idx]))
        self.register_buffer("W", torch.tensor([list(S), list(C)], dtype=torch.float32))          # [2, n_act]
        self.register_buffer("off", torch.tensor([self.args["O"]], dtype=torch.float32))
        self.v_ref = v_ref

    def forward(self, obs):
        z = obs @ self.Sel
        a = torch.clamp(torch.abs(z[:, 2:3]) * (1 / self.v_ref), 0.0, 1.0)
        return a * (z[:, 0:2] @ self.W) + self.off


class ScriptedInsertPrior(nn.Module):
    """Insert contract (57 obs, 9 actions): target_pos 0:3, target_rot 3:6, wrench 12:18, progress 45, depth 46,
    prev_action 47:56 (lateral 47, 48), time 56."""

    def __init__(self, n_in=57, n_act=9, pos_m=0.001, rot_rad=0.01, descend=0.5, k_act=0.0, contact_n=3.0, spiral_turn_deg=25.0, spiral_grow=0.06,
                 spiral_seed=0.08, gain=0.6):
        super().__init__()
        self.args = dict(n_in=n_in, n_act=n_act, pos_m=pos_m, rot_rad=rot_rad, descend=descend, k_act=k_act, contact_n=contact_n, spiral_turn_deg=spiral_turn_deg,
                         spiral_grow=spiral_grow, spiral_seed=spiral_seed, gain=gain)
        self.register_buffer("Sxy", selector(n_in, [0, 1]) * (gain / pos_m))
        self.register_buffer("Srot", selector(n_in, [3, 4, 5]) * (gain / rot_rad))
        self.register_buffer("Spush", selector(n_in, [14]))            # wrench task z: the push is -F_z
        self.register_buffer("Sprog", selector(n_in, [45, 46]))
        c, s = math.cos(math.radians(spiral_turn_deg)), math.sin(math.radians(spiral_turn_deg))
        R = torch.tensor([[c, s], [-s, c]]) * (1 + spiral_grow)        # (row vector times R: turned and grown)
        self.register_buffer("Sprev", selector(n_in, [47, 48]) @ R)
        self.register_buffer("seed", torch.tensor([[spiral_seed, 0.0]]))
        self.register_buffer("const", torch.tensor([[0, 0, descend, 0, 0, 0, k_act, 0, -1.0]]))
        self.register_buffer("Exy", torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0, 0]]))
        self.register_buffer("Erot", torch.tensor([[0, 0, 0, 1.0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1.0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1.0, 0, 0, 0]]))
        self.contact_n = contact_n

    def forward(self, obs):
        centre = torch.clamp(obs @ self.Sxy, -0.5, 0.5)
        push = -(obs @ self.Spush)                                    # [N, 1]
        pd = obs @ self.Sprog                                         # progress, depth
        # stuck at the mouth: pushing over contact_n before 30 % of the depth (smooth gates)
        stuck = torch.sigmoid((push - self.contact_n) * 4.0) * torch.sigmoid((0.3 * pd[:, 1:2] - pd[:, 0:1]) * 4000.0)
        spiral = torch.clamp(obs @ self.Sprev + self.seed, -1.0, 1.0)
        lat = centre * (1 - stuck) + spiral * stuck
        rot = torch.clamp(obs @ self.Srot, -0.5, 0.5)
        return self.const + lat @ self.Exy + rot @ self.Erot


class ScriptedScrewPrior(nn.Module):
    """Screw-drive contract (49 obs, 7 actions): seat_pos 0:3 (z: distance to the seat along the axis), screw_tilt 3:5,
    screw_spec 28:36 (pitch 29), prev_action 41:48, time 48."""

    def __init__(self, n_in=49, n_act=7, pos_m=0.0005, rot_rad=0.004, push=0.5, run=1.0, final=0.25, slow_pitches=2.0, gain=0.5):
        super().__init__()
        self.args = dict(n_in=n_in, n_act=n_act, pos_m=pos_m, rot_rad=rot_rad, push=push, run=run, final=final, slow_pitches=slow_pitches, gain=gain)
        self.register_buffer("Sxy", selector(n_in, [0, 1]) * (gain / pos_m))
        self.register_buffer("Stilt", selector(n_in, [3, 4]) * (1.0 / rot_rad))
        self.register_buffer("Sz", selector(n_in, [2, 29]) @ torch.tensor([[1.0], [-slow_pitches]]))   # z - slow_pitches * pitch
        self.register_buffer("const", torch.tensor([[0, 0, 0, 0, push, final, -1.0]]))
        self.register_buffer("Exy", torch.tensor([[1.0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0]]))
        self.register_buffer("Etilt", torch.tensor([[0, 0, 1.0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0, 0]]))
        self.register_buffer("Espin", torch.tensor([[0, 0, 0, 0, 0, 1.0, 0]]) * (run - final))

    def forward(self, obs):
        lat = torch.clamp(obs @ self.Sxy, -0.5, 0.5)
        tilt = torch.clamp(obs @ self.Stilt, -1.0, 1.0)
        far = torch.sigmoid((obs @ self.Sz) * 1e4)                   # 1 more than two pitches from the seat, else 0
        return self.const + lat @ self.Exy + tilt @ self.Etilt + far @ self.Espin


PRIORS = dict(gait=GaitPrior, oscillator=OscillatorPrior, scripted_insert=ScriptedInsertPrior, scripted_screw=ScriptedScrewPrior)


def build_prior(spec):
    return PRIORS[spec["kind"]](**spec["args"]) if spec else None
