"""connector_mate: a box plug into a funnel pocket plus the detent extension (rl/ext/detent.cu), World2's insert contract."""
import copy

from rl.common.evaluate import rate
from rl.common.skill import Skill
from rl.skills.peg_insert.skill import MANIP_RECIPE, MODEL, clearance_bins, insert_manifest

from .env import CODE_NAMES, SPEC, ConnectorMateBatch

RECIPE = copy.deepcopy(MANIP_RECIPE)
RECIPE["teacher"]["max_minutes"] = 45


def splits(p0, succ, final):
    phys, fpk = p0["phys"] > 0.5, p0["fpk"]
    out = clearance_bins(p0, succ, final)
    out["split"] = dict(contact_tier=rate(succ, ~phys), physical_tier=rate(succ, phys), peak_5_15n=rate(succ, fpk < 15), peak_15_30n=rate(succ, fpk >= 15))
    return out


def make_env(n, device="cuda", seed=0, cfg=None, **env):
    return ConnectorMateBatch(n, device=device, seed=seed, cfg=SPEC.override(cfg))


SKILL = Skill(
    name="connector_mate", contract_skill="insert", make_env=make_env, spec=SPEC, model=MODEL, recipe=RECIPE,
    eval_sets={"nominal": {}, "hard_1.5x": {"pose_scale": 1.5}}, eval_splits=splits, extensions=("b3_manifold", "b3_detent"), eval_ticks=200,
    manifest=insert_manifest(["box-pocket", "connector"],
        "PPO in box3d-cuda (box plug/pocket + detent latch model, heavy domain randomisation: latch curve, funnel, clearance, sizes, friction, pose error, F/T noise/delay/drift/gain, action latency, motor lag, contact or physical-tier compliance); World2's Rapier referees",
        dict(engine="box3d-cuda (MIT, github johndevor/box3d-cuda) manifold_step: persistent OBB manifolds, float32, one CUDA thread per world; rl/ext/detent.cu (optional extension)",
             geometry="interaction model, not geometry: a box plug into a rectangular pocket of box walls with 45-degree funnel boxes (random yaw about the axis), plus a prismatic detent per world (rl/ext/detent.cu): latch force rise to a peak, click (drop), sliding friction, withdrawal hold, jam above a tilt angle, bent pin when pushed hard while tilted; box features reported, the pocket size/clearance/chamfer unknown (0) in half the episodes as World2's design proxy gives them for a socket")),
    description="connector mating with a latch (World2 insert contract)",
)
SKILL.code_names = CODE_NAMES
