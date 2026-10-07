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


# The fast recipe (default): a residual over the scripted drive (rl/common/priors.py ScriptedScrewPrior), asymmetric PPO,
# trained in World2's opt-in interaction (thread catch mode A, the full driving wrench) and its crooked-insert cases (the
# program believes the layout's axis half the time; inserts to 6.5 degrees, 0.6 mm off: World2's hard set inside).
RESIDUAL = dict(kind="asymmetric", envs=131072, eval_n=2048, eval_seed=10_000, eval_ticks=380, graphs=False,
                cfg=dict(mode_b_p=0.0, full_wrench=1.0, layout_belief_p=0.5, insert_tilt_deg=(0.0, 6.5), insert_offset_mm=(0.0, 0.6), curriculum_frac=0.0),
                model=dict(prior=dict(kind="scripted_screw", args={}), residual_scale=1.0, log_std=-1.2, init_bias={}, init_log_std={}),
                ppo=dict(horizon=64, lr=3e-4, gamma=0.995, lam=0.95, clip=0.2, epochs=4, minibatches=4, ent=0.0, kl="stop", kl_target=0.03,
                         score="success", min_episodes=4000, plateau_iters=30, min_iters=20, max_minutes=6.0, max_steps=3e9))


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
    name="screw_drive", contract_skill="screw", make_env=make_env, spec=SPEC, model=MODEL, recipe=RESIDUAL,
    recipes={"teacher_student": RECIPE},
    eval_sets={"nominal": {}, "hard_1.5x": {"pose_scale": 1.5}, "crooked_4_6": {"insert_tilt_deg": (4.0, 6.0), "insert_offset_mm": (0.45, 0.6), "layout_belief_p": 1.0}},
    eval_splits=splits, extensions=("b3_manifold", "b3_screw"), eval_ticks=380,
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
    def replay(cfg_, act, seed=0):
        env = make_env(1, device=device, seed=seed, cfg=cfg_)
        obs, _ = env.observe()
        T = dict(t=[], torque_nm=[], turns=[], seat_z_mm=[], moment_xy_nm=[], mode=[], clicked=[])
        end = None
        rec = lambda k, o: (T["t"].append(round(k / 30, 3)), T["torque_nm"].append(float(o[0, 19])), T["turns"].append(float(o[0, 20])),
                            T["seat_z_mm"].append(float((env._tip()[0, 2] + env.p["L"][0]) * mm)), T["moment_xy_nm"].append(float(math.hypot(o[0, 14], o[0, 15]))),
                            T["mode"].append(int(env.J[0, SR.J_MODE])), T["clicked"].append(float(env.J[0, SR.J_CLICKED])))
        for k in range(int(env.cfg["timeout_s"] * 30) + 2):
            rec(k, obs)
            obs, _, r, done, info = env.step(act(obs).to(env.dev), autoreset=False)
            if bool(done[0]):
                rec(k + 1, obs)
                end = dict(code=CODE_NAMES.get(int(info["code"][0])), success=bool(info["success"][0]), torque_nm=round(float(env.J[0, SR.J_TORQUE]), 4),
                           clicked=max(T["clicked"]) > 0.5, crossed=bool(env.J[0, SR.J_CROSSED] > 0.5))
                break
        return T, end

    a = torch.tensor([w2["action"]], dtype=torch.float32)
    T, end = replay(cfg, lambda o: a)
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
    out = dict(trace=T, end=end, cfg=cfg)
    I = w2.get("interaction")
    if I:
        # World2's opt-in thread catch and full driving wrench (operations/interaction.mjs) on a crooked insert, the back-turn
        # schedule: box3d catch mode A (mode_b_p 0) and the full wrench, the insert pinned to the same tilt
        cfg2 = dict(cfg, mode_b_p=0.0, insert_tilt_deg=I["case"]["insert_tilt_deg"], full_wrench=1.0, layout_belief_p=1.0, flange_h=I["case"].get("flange_to_tcp_m") or 0.1,
                    **({"ft_noise_mult": 0.0, "ft_tare_offset": 0.0, "torque_noise_nm": 0.0} if (I.get("ft") or {}).get("noise") == 0 else {}))
        base = torch.tensor([I["schedule"]["base"]], dtype=torch.float32)

        def sched(o):
            x = base.clone()
            x[0, 5] = float(min(0.25, max(-1.0, 2.5 * float(o[0, 48]) - 3.0)))     # clamp(2.5 t - 3, -1, 0.25)
            return x
        T2, end2 = replay(cfg2, sched)
        Wi = I["trace"]
        start_b = next((T2["t"][i] for i, m in enumerate(T2["mode"]) if m >= 1), None)
        start_w = (I.get("thread_start") or {}).get("skill_s")

        def moment(tr, t0):
            if t0 is None:
                return None
            z, t, m = np.array(tr["seat_z_mm"]), np.array(tr["t"]), np.array(tr["moment_xy_nm"])
            sel = (t > t0 + 0.2) & (z > 0.3)
            return float(m[sel].mean()) if sel.any() else None
        # (the moment's size depends on the tilt's direction against the spin, drawn per world and not recorded by
        # World2: box3d's is the median over 8 directions (seeds; CPU: 0.147-0.247 N m against World2's 0.123), the first seed's trace kept for the catch checks)
        mbs = [moment(T2, start_b)]
        for sd in range(1, 8):
            Tk, _ = replay(cfg2, sched, seed=sd)
            mbs.append(moment(Tk, next((Tk["t"][i] for i, m in enumerate(Tk["mode"]) if m >= 1), None)))
        mbs = [x for x in mbs if x is not None]
        mw, mb = moment(Wi, start_w), (float(np.median(mbs)) if mbs else None)
        checks += [
            check("[catch] clicked by the back-turn in both", bool(I.get("clicked")) and bool(end2 and end2["clicked"]), ("equal", True),
                  "turning back inside the capture drops the lead thread into the start"),
            check("[catch] thread start time (s, box3d - World2)", (start_b - start_w) if start_b is not None and start_w is not None else None, 0.15,
                  "the forward turn after the click catches at once (forward resumes at 1.2 s)"),
            check("[catch] clean start on the 2-degree insert in both", (not (I.get("thread_start") or {}).get("cross_threaded", True)) and bool(end2 and not end2["crossed"]), ("equal", True),
                  "2 degrees is under the cross-thread angle plus the click's 1 degree"),
            check("[catch] seated in both", bool(I.get("end", {}).get("ok")) and bool(end2 and end2["code"] == "seated"), ("equal", True), ""),
            check("[wrench] mean lateral moment while running down (N m, box3d - World2)", (mb - mw) if mb is not None and mw is not None else None, max(0.05, 0.3 * (mw or 0)),
                  "the full driving wrench: the bit's moment of the tilt between the tool and the screw on the crooked axis, with the lateral spring about the flange (0.05 N m or 30 %)"),
        ]
        out["interaction"] = dict(trace=T2, end=end2, cfg=cfg2, start_s=start_b, moment_mean_nm=mb, moment_by_seed_nm=mbs, world2_moment_mean_nm=mw)
    return checks, out


SKILL.sim_match = sim_match
