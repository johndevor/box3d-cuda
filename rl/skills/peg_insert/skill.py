"""peg_insert: a square box peg into a chamfered box-wall hole on manifold_step, World2's insert contract (57 obs, 9 actions)."""
from rl.common.evaluate import rate
from rl.common.skill import Skill

from .env import ACTION_SCALE, CODE_NAMES, LIMITS, OBS_FIELDS, RATE_HZ, SPEC, PegInsertBatch

MODEL = dict(actor_hidden=(256, 256), critic_hidden=(256, 256), squash=True, log_std=-0.5, norm_clip=10.0,
             init_bias={2: 0.3, 8: -3.0}, init_log_std={8: -2.0})    # start pushing in gently and never terminating (a[8] > 0 ends the skill)

MANIP_RECIPE = dict(
    kind="teacher_student", envs=32768, eval_n=4096, eval_seed=10_000,
    ppo=dict(horizon=32, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4, ent=0.0, kl="stop", kl_target=0.03,
             score="success", min_episodes=2000, plateau_iters=40, max_steps=3e9),
    teacher=dict(max_minutes=40, min_iters=30),
    distill=dict(iters=150),
    finetune=dict(max_minutes=30, min_iters=20),
    student_log_std=-1.5, finetune_log_std=-1.5, finetune_log_std_at={8: -2.5},
)


# The fast recipe: a residual over the scripted insertion (rl/common/priors.py ScriptedInsertPrior), asymmetric PPO (the
# policy on its observations, the critic on the privileged state too), many worlds. The connector's default; the peg's
# alternative ("residual"): in 4 minutes it reached box3d hard 76 % but World2's peg sets 24/30 nominal, 0/90 hard (it
# pushes 30-45 N at the mouth and never finds the bore), so the peg keeps the teacher-student recipe by default.
def residual_recipe(minutes=4.0, envs=131072, **ppo):
    return dict(kind="asymmetric", envs=envs, eval_n=2048, eval_seed=10_000, graphs=False,
                model=dict(prior=dict(kind="scripted_insert", args={}), residual_scale=1.0, log_std=-1.2, init_bias={}, init_log_std={}),
                ppo=dict(horizon=32, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4, ent=0.0, kl="stop", kl_target=0.03,
                         score="success", min_episodes=4000, plateau_iters=30, min_iters=20, max_minutes=minutes, max_steps=3e9, **ppo))


def clearance_bins(p0, succ, final):
    c = p0["clear"]
    return dict(by_clearance={f"{int(lo * 1e6)}-{int(hi * 1e6)}um": rate(succ, (c >= lo) & (c < hi)) for lo, hi in ((0, 50e-6), (50e-6, 200e-6), (200e-6, 1.1e-3))})


def insert_manifest(families, basis, model_text):
    def manifest(spec, env=None):
        S = ACTION_SCALE
        return dict(
            skill="insert", rate_hz=RATE_HZ, observation=OBS_FIELDS,
            action={"kind": "delta-pose-impedance", "scale": {"pos_m": S["pos_m"], "rot_rad": S["rot_rad"], "k_trans_n_per_m": list(S["k_trans_n_per_m"]),
                                                             "k_rot_nm_per_rad": list(S["k_rot_nm_per_rad"])}},
            input="obs", output="action", families=families, limits=dict(LIMITS), termination={"min_s": 0.1, "verify": "insertion"},
            licence="project", basis=basis, training_text=model_text)
    return manifest


def make_env(n, device="cuda", seed=0, cfg=None, **env):
    return PegInsertBatch(n, device=device, seed=seed, cfg=SPEC.override(cfg))


SKILL = Skill(
    name="peg_insert", contract_skill="insert", make_env=make_env, spec=SPEC, model=MODEL, recipe=MANIP_RECIPE,
    recipes={"residual": residual_recipe()},
    eval_sets={"nominal": {}, "hard_1.5x": {"pose_scale": 1.5}}, eval_splits=clearance_bins, extensions=("b3_manifold",), eval_ticks=200,
    manifest=insert_manifest(["peg-hole"],
        "PPO in box3d-cuda (crude box peg/hole, heavy domain randomisation: clearance, pose error, friction, mass, F/T noise/delay/drift/gain, action latency, motor lag); World2's Rapier referees",
        dict(engine="box3d-cuda (MIT, github johndevor/box3d-cuda) manifold_step: persistent OBB manifolds, float32, one CUDA thread per world",
             geometry="crude on purpose: a square box peg (side = the family's diameter) into a square hole of box walls with 45-degree chamfer boxes; no part chamfer; cylinder features reported")),
    description="peg-in-hole insertion (World2 insert contract)",
)
SKILL.code_names = CODE_NAMES


# ---------------------------------------------------------------- sim-match (rl/common/simmatch.py; World2 scripts/sim-match.mjs peg_insert)
def pinned_insert_cfg(w2):
    """Every randomization pinned to World2's probe case: no belief errors, no latency or lag, the 2 kg virtual mass, its
    F/T sensor profile, its friction."""
    import math
    c, mm = w2["case"], 1000.0
    sig = w2.get("estimate_sigma", [0.0005] * 3 + [0.005] * 3)
    ft = w2.get("ft") or {}
    prof = {"ur-e-series-ur12e": 0, "ur-e-series-ur5e": 1, "robotiq-ft300s": 2}.get(ft.get("profile"), 0)   # FT_PROFILES rows
    quiet = 0.0 if ft.get("noise") == 0 else 1.0           # World2's sensor noise-free: so is this one (no noise, no tare offset)
    return dict(start_height_mm=w2["start_offset_m"][1] * mm, start_lateral_mm=0.0, start_tilt_deg=0.0,
                target_sigma_mm=sig[0] * mm, target_sigma_rot_deg=math.degrees(sig[3]), in_hand_sigma_mm=w2.get("in_hand_sigma_m", 0.0005) * mm,
                target_err_z_mm=0.0, in_hand_err_z_mm=0.0, max_force_n=w2.get("max_force_n", 40.0), ft_profile=prof, ft_extra_delay_s=0.0,
                ft_gain_err=0.0, ft_noise_mult=quiet, ft_tare_offset=quiet, flange_h=0.1, action_latency_p=0.0, motor_lag_s=0.0, vmass=2.0, friction=c["friction"],
                pose_scale=0.0, belief_error_scale=0.0, clearance_mm=c["clearance_m"] * mm, host_chamfer_mm=c["host_chamfer_m"] * mm)


def sim_match(w2, device):
    from rl.common.simmatch import insert_checks, insert_descend
    c, mm = w2["case"], 1000.0
    d, L, D = c["d_m"] * mm, c["L_m"] * mm, c["depth_m"] * mm
    cfg = dict(pinned_insert_cfg(w2), d_mm=d, length_over_d=L / d, depth_over_d=D / d, part_chamfer_mm=c.get("part_chamfer_m", 0.0) * mm,
               density=c["mass_kg"] / (c["d_m"] ** 2 * c["L_m"] * 3.141592653589793 / 4))
    env = make_env(1, device=device, seed=0, cfg=cfg)
    T, end = insert_descend(env, w2)
    return insert_checks(w2, T, end, D), dict(trace=T, end=end, cfg=cfg)


SKILL.sim_match = sim_match
