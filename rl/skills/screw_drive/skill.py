"""screw_drive: a box screw on a compliant bit, manifold_step contacts plus the screw-joint extension (rl/ext/screw_joint.cu),
World2's screw-drive contract (docs/learned-skills.md section 11: 49 obs, 7 actions)."""
import copy

from rl.common.evaluate import rate
from rl.common.skill import Skill
from rl.skills.peg_insert.skill import MANIP_RECIPE

from .env import ACTION_SCALE, CODE_NAMES, LIMITS, OBS_FIELDS, RATE_HZ, SEAT_RPM_MAX, SPEC, ScrewDriveBatch

# start pushing (~19 N), turning forward (~115 rpm: under the seat-speed limit) and never stopping (a[6] > 0 ends the skill);
# little noise on the spindle speed (noise over the 120 rpm seat limit would end most runs as FAST_SEAT)
MODEL = dict(actor_hidden=(256, 256), critic_hidden=(256, 256), squash=True, log_std=-0.5, norm_clip=10.0,
             init_bias={4: 0.8, 5: 0.3, 6: -3.0}, init_log_std={6: -2.0, 5: -1.6})
RECIPE = copy.deepcopy(MANIP_RECIPE)
RECIPE["ppo"].update(horizon=64, gamma=0.995)
RECIPE["teacher"]["max_minutes"] = 40
RECIPE["finetune_log_std_at"] = {6: -2.5}
RECIPE["eval_ticks"] = 380


def splits(p0, succ, final):
    berr, thx, phil, size, modeb = p0["belief_err"], p0["thx"], p0["phil"], p0["size"], p0["modeb"]
    backouts = final.get("backouts", succ.float() * 0)
    over = berr > thx      # the believed hole axis off by more than the cross-thread angle: only a back-out and a new tilt save it
    return dict(belief_over_theta_x=dict(n=int(over.sum()), success=rate(succ, over),
                                         mean_backouts_in_successes=round(backouts[over & succ].mean().item(), 3) if (over & succ).any() else None),
                belief_within=dict(n=int((~over).sum()), success=rate(succ, ~over)),
                by_size={f"M{[2, 2.5, 3, 4, 5][k]}": rate(succ, size == k) for k in range(5)}, phillips=rate(succ, phil > 0.5), hex=rate(succ, phil < 0.5),
                catch_mode_b=rate(succ, modeb > 0.5), catch_mode_a=rate(succ, modeb < 0.5), backouts_mean=round(backouts.mean().item(), 3))


def manifest(spec, env=None):
    S = ACTION_SCALE
    return dict(
        skill="screw", rate_hz=RATE_HZ, observation=OBS_FIELDS,
        action={"kind": "screw-drive", "scale": {"pos_m": S["pos_m"], "rot_rad": S["rot_rad"], "force_n": list(S["force_n"]), "descend_m_s": S["descend_m_s"],
                                                "lift_m_s": S["lift_m_s"], "max_rpm": S["max_rpm"], "k_n_per_m": S["k_n_per_m"], "k_rot_nm_per_rad": S["k_rot_nm_per_rad"]}},
        input="obs", output="action", families=["screw"], limits=dict(LIMITS), termination={"min_s": 0.1, "verify": "screw", "seat_rpm_max": SEAT_RPM_MAX},
        licence="project",
        basis="PPO in box3d-cuda (crude box screw, a helical joint with engagement and failure state, heavy domain randomisation: M2-M5 pitch, run-down torque, "
              "set torque, joint stiffness, strip, Phillips/hex, cross-thread angle and binding, belief tilt/offset (D405-class sigma), bit compliance, wobble, "
              "torque and F/T noise/delay/drift, action latency, motor lag); World2's Rapier referees",
        training_text=dict(
            engine="box3d-cuda (MIT, github johndevor/box3d-cuda) manifold_step (persistent OBB manifolds, float32) + rl/ext/screw_joint.cu (optional extension: "
                   "helical joint, thread engagement, cross-thread / strip / cam-out state; CPU oracle rl/ext/screw_reference.py), one CUDA thread per world",
            model="the interaction, not the threads: a square box screw on a compliant bit through a plate's square clearance hole (box walls, chamfer boxes) "
                  "onto an insert slab; the thread starts within the capture (0.1-0.2 d) of the insert axis (catch mode A: forward spin through the thread start, "
                  "a back-turn clicks; mode B (30 %): on capture, as World2's referee); over theta_x of tilt at the catch it is cross-threaded (binding k_x per turn); "
                  "helical joint (pitch per turn); quasi-static spindle (speed command, clutch, motor limit, Phillips cam-out, hex bit slip); run-down, seat, clamp "
                  "K d F, strip; reverse backs it out and releases it past the thread start"))


def make_env(n, device="cuda", seed=0, cfg=None, **env):
    return ScrewDriveBatch(n, device=device, seed=seed, cfg=SPEC.override(cfg))


SKILL = Skill(
    name="screw_drive", contract_skill="screw", make_env=make_env, spec=SPEC, model=MODEL, recipe=RECIPE,
    eval_sets={"nominal": {}, "hard_1.5x": {"pose_scale": 1.5}}, eval_splits=splits, extensions=("b3_manifold", "b3_screw"), eval_ticks=380,
    manifest=manifest, description="screw driving (World2 screw-drive contract)",
)
SKILL.code_names = CODE_NAMES


# ---------------------------------------------------------------- sim-match (World2 scripts/sim-match.mjs screw_drive)
def sim_match(w2, device):
    """World2's constant push-and-spin probe (100 rpm, ~11 N) on an upright, on-axis M3 insert, replayed with every
    randomization pinned to its case: the run-down (advance per turn), the time to seat, the run-down and final torque."""
    import math
    import numpy as np
    import torch
    from rl.common.simmatch import check
    from rl.ext import screw_reference as SR
    c, mm = w2["case"], 1000.0
    cfg = dict(size=2, plate_mm=c["plate_m"] * mm, clearance_mm=c["clearance_m"] * mm, plate_chamfer_mm=c["plate_chamfer_m"] * mm,
               engage_over_d=(c["L_m"] - c["plate_m"]) / c["d_m"], trun_per_m=8.33, kj=1e7, nut_k=0.2, strip_p=0.0, phillips_p=0.0 if c["drive"] != "phillips" else 1.0,
               hex_fraction=1.0, clutch_scatter=0.0, kx_rel=0.4, theta_x_deg=c["theta_x_deg"], capture_over_d=c["capture_over_d"], mode_b_p=1.0,
               insert_tilt_deg=0.0, insert_offset_mm=0.0, rot_sigma_deg=math.degrees(c["axis_sigma_rad"]), seat_sigma_mm=c["seat_sigma_m"] * mm,
               seat_sigma_z_mm=c["seat_sigma_m"] * mm, in_hand_mm=0.0, start_height_mm=c["start_height_m"] * mm, wobble_deg=0.0, stiff_mult=1.0,
               torque_noise_nm=c["torque_sigma_nm"], torque_delay_p=0.0, ft_profile=1, ft_extra_delay_s=0.0, ft_gain_err=0.0, ft_noise_mult=1.0,
               action_latency_p=0.0, motor_lag_s=0.0, vmass=2.0, friction=0.3, pose_scale=0.0, set_torque_nm=c["torque_nm"], curriculum_frac=0.0)
    env = make_env(1, device=device, seed=0, cfg=cfg)
    a = torch.tensor([w2["action"]], dtype=torch.float32, device=env.dev)
    obs, _ = env.observe()
    T = dict(t=[], torque_nm=[], turns=[], seat_z_mm=[])
    end = None
    for k in range(int(env.cfg["timeout_s"] * 30) + 2):
        T["t"].append(round(k / 30, 3)); T["torque_nm"].append(float(obs[0, 19])); T["turns"].append(float(obs[0, 20]))
        T["seat_z_mm"].append(float((env._tip()[0, 2] + env.p["L"][0]) * mm))
        obs, _, r, done, info = env.step(a, autoreset=False)
        if bool(done[0]):
            T["t"].append(round((k + 1) / 30, 3)); T["torque_nm"].append(float(obs[0, 19])); T["turns"].append(float(obs[0, 20]))
            T["seat_z_mm"].append(float((env._tip()[0, 2] + env.p["L"][0]) * mm))
            end = dict(code=CODE_NAMES.get(int(info["code"][0])), success=bool(info["success"][0]), torque_nm=round(float(env.J[0, SR.J_TORQUE]), 4))
            break
    W = w2["trace"]

    def rundown(tr):
        z, n, q = np.array(tr["seat_z_mm"]), np.array(tr["turns"]), np.array(tr["torque_nm"])
        m = (z > 0.5) & (z < 4.0)
        if m.sum() < 5:
            return None, None
        return float(np.polyfit(n[m], z[m], 1)[0]), float(q[m].mean())

    def seated_at(tr):
        z = np.array(tr["seat_z_mm"])
        i = next((i for i, x in enumerate(z) if x <= 0.05), len(z) - 1)
        return tr["t"][i]

    sw, qw = rundown(W)
    sb, qb = rundown(T)
    checks = [
        check("seated in both", bool(w2.get("end", {}).get("ok")) and bool(end and end["code"] == "seated"), ("equal", True), "World2: seated and verified at torque; box3d: code seated"),
        check("run-down advance per spindle turn (mm, box3d - World2)", (sb - sw) if sb is not None and sw is not None else None, 0.05,
              "the helix: 0.5 mm per turn for M3; a slip or a wrong pitch shows here"),
        check("time to seat (s, box3d - World2)", seated_at(T) - seated_at(W), 0.5, "approach, catch and ten turns at 100 rpm (0.5 s is under one turn)"),
        check("mean run-down torque (N m, box3d - World2)", (qb - qw) if qb is not None and qw is not None else None, 0.03,
              "the prevailing torque the driver reads while running down (reading noise 0.01 N m)"),
        check("final torque reading (N m, box3d - World2)", T["torque_nm"][-1] - W["torque_nm"][-1], 0.1,
              "both stop at the clutch's set torque (0.6 N m) with the reading noise"),
    ]
    return checks, dict(trace=T, end=end, cfg=cfg)


SKILL.sim_match = sim_match
