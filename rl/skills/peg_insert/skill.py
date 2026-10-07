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
    eval_sets={"nominal": {}, "hard_1.5x": {"pose_scale": 1.5}}, eval_splits=clearance_bins, extensions=("b3_manifold",), eval_ticks=200,
    manifest=insert_manifest(["peg-hole"],
        "PPO in box3d-cuda (crude box peg/hole, heavy domain randomisation: clearance, pose error, friction, mass, F/T noise/delay/drift/gain, action latency, motor lag); World2's Rapier referees",
        dict(engine="box3d-cuda (MIT, github johndevor/box3d-cuda) manifold_step: persistent OBB manifolds, float32, one CUDA thread per world",
             geometry="crude on purpose: a square box peg (side = the family's diameter) into a square hole of box walls with 45-degree chamfer boxes; no part chamfer; cylinder features reported")),
    description="peg-in-hole insertion (World2 insert contract)",
)
SKILL.code_names = CODE_NAMES
