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
