"""Generic legged/crawling locomotion on box3d-cuda for any robot World2 designs: the articulated kernel of the duck
(rl/skills/duck_walk/duck_sim.h: maximal-coordinate bodies, revolute servo joints with World2's STS3215 bam-m1 servo
step, the IMU signal chain, box contacts with the floor) on a model generated from the robot's World2 design
(World2 skills/robot-model.mjs: the robot powered there and its bodies, joints, servos, contact boxes and IMU read back).

Policy interface (World2's walk contract, skills/duck-walk.mjs walker, any number of driven joints L):
  frame: imu_gyro 3, imu_gravity 3, leg_joint_pos L, leg_joint_vel L (x 0.1), prev_action L, command 3,
         foot_contact (the model's switches, if any), gait_clock 2 (sin, cos at clock_hz); HISTORY frames newest first
  action L in [-1, 1]: joint targets = design pose + action_scale * a, clipped to the joints' limits; 40 Hz.
Reward (generic): track the forward command in the heading frame of the episode's start, keep upright and on heading,
smooth and low power; an episode ends on a flip (tilt past max_tilt_deg) or a fall (trunk under fall_height_frac of its
design height, when set). Each contact box carries its own friction (averaged with the floor's, Rapier's default).
"""
from __future__ import annotations

import math

import torch

from rl.common.randomization import Spec, U
from rl.skills.duck_walk.env import NOMINAL, RATE_HZ, DuckWalkEnv

SPEC = Spec(dict(
    mass_scale=U(0.85, 1.15), trunk_com_shift_m=0.01, friction=U(0.6, 1.0), damping_lin=U(0.5, 1.5), damping_ang=U(1.0, 3.0),
    kt=U(0.85, 1.15), R=U(0.85, 1.15), vin=U(6.8, 8.2), gain=U(0.85, 1.15), armature=U(0.75, 1.3), friction_base=U(0.5, 1.6),
    friction_viscous=U(0.5, 1.6), backlash_rad=U(0.0, 0.02), gear_stiffness=U(0.4, 1.3), gear_damping=U(0.6, 3.0), max_velocity=U(0.9, 1.1),
    goal_extra_delay=(0, 1), imu_delay=(0, 1), imu_noise=U(0.5, 2.0), imu_bias=U(0.0, 2.0), calibration_steps=3,
    push_interval_s=U(2.0, 5.0), push_dv=0.05, push_p=0.5, init_yaw=math.pi, init_vel=0.0,
    cmd_vx=U(0.6, 1.4), cmd_zero_p=0.0, episode_s=12.0, cmd_wz=0.0, cmd_wz_p=0.0, cmd_wz_change_s=3.0,
))
# (cmd_vx: a multiple of the robot's v_ref; the nominal robot as the duck's, the floor at World2's default 0.8)
LOCO_NOMINAL = dict(NOMINAL, cmd_vx=(1.0, 1.0))


class LocomotionEnv(DuckWalkEnv):
    box_friction = True

    def __init__(self, n_envs, device="cpu", seed=0, dr=None, model_file=None, dr_on=True, clock_hz=1.5, action_scale=1.0, v_ref=0.05,
                 max_tilt_deg=45.0, fall_height_frac=0.0, substeps=8, iterations=6, dr_ramp=0.0, alive_bonus=0.1):
        assert model_file, "a generated robot model (World2 scripts/robot-model.mjs) is needed"
        self.action_scale, self.v_ref = float(action_scale), float(v_ref)
        self.max_tilt, self.fall_frac = math.radians(max_tilt_deg), float(fall_height_frac)
        if dr_on:
            dr = dr if isinstance(dr, Spec) else SPEC.override(dr or {})
        else:
            dr = SPEC.override(LOCO_NOMINAL).override(dr if isinstance(dr, dict) else None)
        super().__init__(n_envs, device=device, seed=seed, dr=dr, substeps=substeps, iterations=iterations, model_file=model_file, dr_on=True,
                         clock_hz=clock_hz, alive_bonus=alive_bonus, dr_ramp=dr_ramp if dr_on else 0.0)
        if self.dr_ramp > 0:
            self.dr.nom = SPEC.override(LOCO_NOMINAL)

    def reset(self, mask=None):
        out = super().reset(mask)
        # (the command is a multiple of v_ref: the spec's cmd_vx range is relative)
        if mask is None:
            mask = torch.ones(self.E, dtype=torch.bool, device=self.device)
        self.cmd[mask, 0] = self.cmd[mask, 0] * self.v_ref
        return out

    # frame indices (newest frame first): the command's vx and the gait clock
    @property
    def cmd_vx_idx(self):
        return 6 + 3 * self.legs

    @property
    def clock_idx(self):
        return 6 + 3 * self.legs + 3 + self.nsw

    def _reward(self, a):
        T, dt, cmd = self.trunk(), 1.0 / RATE_HZ, self.cmd
        v = self.v_ref
        moving = (cmd[:, 0].abs() > 1e-4).float()
        terms = dict(
            track_vx=2.0 * torch.exp(-(T["vx"] - cmd[:, 0]) ** 2 / (0.5 * v) ** 2),
            forward=1.0 * torch.clamp(T["vx"] / v, -1.0, 1.5) * moving,
            lateral=-0.5 * (T["vy"] / v) ** 2 * 0.1,
            alive=torch.full_like(T["vx"], self.alive_bonus),
            upright=-2.0 * (1 - T["up"][:, 1]),
            heading=-2.0 * torch.atan2(torch.sin(T["yaw"] - self.yaw_ref), torch.cos(T["yaw"] - self.yaw_ref)) ** 2,
            ang_vel=-0.05 * (T["w"][:, 0] ** 2 + T["w"][:, 2] ** 2),
            action_rate=-0.02 * ((a - self.prev_action) ** 2).sum(-1),
            power=-0.002 * self.o_power,
        )
        reward = sum(terms.values()) * dt * 10
        fell = (T["tilt"] > self.max_tilt) | ~torch.isfinite(self.state).all(-1).all(-1)
        if self.fall_frac > 0:
            fell = fell | (T["h"] < self.fall_frac * self.trunk_y0)
        timeout = self.t >= self.dr["episode_s"]
        reward = torch.where(fell, torch.full_like(reward, -2.0), reward)
        reward = torch.nan_to_num(reward, nan=-5.0)
        done = fell | timeout
        return reward, terms, done, timeout & ~fell
