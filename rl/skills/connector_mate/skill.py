"""connector_mate: a box plug into a funnel pocket plus the detent extension (rl/ext/detent.cu), World2's insert contract."""
import copy

from rl.common.evaluate import rate
from rl.common.skill import Skill
from rl.skills.peg_insert.skill import MANIP_RECIPE, MODEL, clearance_bins, insert_manifest, residual_recipe

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
    name="connector_mate", contract_skill="insert", make_env=make_env, spec=SPEC, model=MODEL, recipe=residual_recipe(minutes=5.0),
    recipes={"teacher_student": RECIPE},
    eval_sets={"nominal": {}, "hard_1.5x": {"pose_scale": 1.5}}, eval_splits=splits, extensions=("b3_manifold", "b3_detent"), eval_ticks=200,
    manifest=insert_manifest(["box-pocket", "connector"],
        "PPO in box3d-cuda (box plug/pocket + detent latch model, heavy domain randomisation: latch curve, funnel, clearance, sizes, friction, pose error, F/T noise/delay/drift/gain, action latency, motor lag, contact or physical-tier compliance); World2's Rapier referees",
        dict(engine="box3d-cuda (MIT, github johndevor/box3d-cuda) manifold_step: persistent OBB manifolds, float32, one CUDA thread per world; rl/ext/detent.cu (optional extension)",
             geometry="interaction model, not geometry: a box plug into a rectangular pocket of box walls with 45-degree funnel boxes (random yaw about the axis), plus a prismatic detent per world (rl/ext/detent.cu): latch force rise to a peak, click (drop), sliding friction, withdrawal hold, jam above a tilt angle, bent pin when pushed hard while tilted; box features reported, the pocket size/clearance/chamfer unknown (0) in half the episodes as World2's design proxy gives them for a socket")),
    description="connector mating with a latch (World2 insert contract)",
)
SKILL.code_names = CODE_NAMES


# ---------------------------------------------------------------- sim-match (World2 scripts/sim-match.mjs connector_mate)
def sim_match(w2, device):
    """The sliding fit and the controller: World2 latches a connector by rule (a push at depth), so the detent is off here
    (detent=0) and the check is the same insert probe as the peg's on the plug-in-pocket geometry."""
    from rl.common.simmatch import insert_checks, insert_descend
    from rl.skills.peg_insert.skill import pinned_insert_cfg
    c, mm = w2["case"], 1000.0
    cfg = dict(pinned_insert_cfg(w2), w_mm=c["w_m"] * mm, t_mm=c["t_m"] * mm, length_mm=c["L_m"] * mm, depth_mm=c["depth_m"] * mm, floor_extra_mm=0.0,
               floor_flush_p=1.0, funnel_mm=c["host_chamfer_m"] * mm, yaw_err_deg=0.0, max_force_n=w2.get("max_force_n", 40.0), physical_p=0.0,
               feature_known_p=1.0, detent=0.0, density=c["mass_kg"] / (c["w_m"] * c["t_m"] * c["L_m"]))
    cfg.pop("host_chamfer_mm")
    env = make_env(1, device=device, seed=0, cfg=cfg)
    T, end = insert_descend(env, w2)
    checks = insert_checks(w2, T, end, c["depth_m"] * mm, seated_codes=("depth", "policy", "stop", "NOT_LATCHED"), need_success=False)
    out = dict(trace=T, end=end, cfg=cfg, note="default probe: detent off (World2 latches by rule); the interaction probe: World2's opt-in click curve against the detent")
    W = w2.get("interaction")
    if W and W.get("click"):
        # World2's opt-in connector click (operations/interaction.mjs) against the detent, pinned to its curve
        import numpy as np
        from rl.common.simmatch import check
        cv = W["click"]["curve"]
        cfg2 = dict(cfg, detent=1.0, peak_n=cv["peak"], click_before_depth_mm=(cv["depth"] - cv["x_click"]) * mm, ramp_mm=(cv["x_click"] - cv["x_ramp"]) * mm,
                    drop_mm=cv["drop"] * mm, res_n=cv["res"], hold_over_peak=cv["hold"] / cv["peak"], jam_deg=89.0, bend_n=1e6, max_over_peak=10.0)
        env2 = make_env(1, device=device, seed=0, cfg=cfg2)
        T2, end2 = insert_descend(env2, W)
        latched_b = bool(env2.v["det_state"][0, 0] > 0.5) or bool(end2 and end2["code"] in ("depth", "stop", "policy") and end2["success"])
        wp, wf = np.array(W["trace"]["progress_mm"]), np.array(W["trace"]["push_n"])
        bp, bf = np.array(T2["progress_mm"]), np.array(T2["push_n"])
        zone = lambda p: (p > (cv["x_ramp"] - 0.0005) * mm) & (p < (cv["depth"] - 0.05e-3) * mm)
        pk_w, pk_b = (wf[zone(wp)].max() if zone(wp).any() else None), (bf[zone(bp)].max() if zone(bp).any() else None)
        at_w, at_b = (wp[zone(wp)][wf[zone(wp)].argmax()] if pk_w is not None else None), (bp[zone(bp)][bf[zone(bp)].argmax()] if pk_b is not None else None)
        checks += [
            check("[click] peak push through the click (N, box3d - World2)", (pk_b - pk_w) if pk_w is not None and pk_b is not None else None, max(1.0, 0.15 * cv["peak"]),
                  "the latch's curve felt through the same impedance controller: rise to the peak, the drop (1 N or 15 % of the peak)"),
            check("[click] depth at the peak push (mm, box3d - World2)", (at_b - at_w) if at_w is not None and at_b is not None else None, 0.6,
                  "where the click happens (the probe advances 0.5 mm per 30 Hz tick)"),
            check("[click] latched in both", bool(W["click"]["latched"]) and latched_b, ("equal", True), "World2: the click state at the seat; box3d: the detent's latch"),
        ]
        out["interaction"] = dict(trace=T2, end=end2, cfg=cfg2)
    return checks, out


SKILL.sim_match = sim_match
