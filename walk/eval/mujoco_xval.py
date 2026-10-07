"""Cross-simulator validation: run a trained flat-floor duck policy UNCHANGED
inside real MuJoCo and score it with the strict gait evaluator.

This is sim-to-sim transfer evidence (not proof): the policy was trained on
our native lowering (walk/env/flat.py over walk/env/native_lane.py); here the
exact same observation/action contract is replicated over the SOURCE MuJoCo
model that lowering came from (duck_model/model/open_duck_mini_v2.xml +
scene_flat_terrain.xml), and the resulting per-tick trace is scored by
walk/eval/gait.evaluate_episode.

Contract replicated from walk/env/flat.py (constants imported, not copied):
  - obs 58: joint q offset from HOME, qdot*0.05, prev action, gravity in body
    frame, root ang vel (body), root lin vel (body), command 3-vector, foot
    contacts, phase clock sin/cos (PHASE_HZ_BASE/PER_MPS; phase0 = 0 for
    evaluation);
  - actions: requested = HOME + 0.25*a, slew-limited 0.1048 rad per 0.02 s
    policy step, joint-limit clipped, held for 10 x 0.002 s MuJoCo steps;
  - actuation: the model's OWN <position> servos are used (verified at load to
    be exactly the kp=13.37, kv=0, force cap 3.23 servos our native lane's PD
    constants came from); ctrl = effective (slew + limit clipped) targets.
    MuJoCo additionally models the source joint damping 0.56 / frictionloss
    0.068 / armature 0.027 declared in the XML.
  - reset: the model's home keyframe with the minimal floor-clear root lift
    (+Z so the lowest sole vertex clears the floor by exactly 1 mm), i.e. the
    same closed-form reset the native lane pins (root z 0.16788827542191784).

Foot telemetry for the evaluator (native-lane conventions, see
walk/env/native_lane.py and the voxel-gate open_duck_mujoco_admission /
floor_clear_reset scripts): whole-sole height = min world z over the compiled
foot_bottom_tpu collision-mesh vertices (18 per foot); contact flag = a
MuJoCo contact between that foot geom and the floor plane in the manifold
used by the just-solved 2 ms step; foot position = foot-assembly body COM
(data.xipos), the analog of the lane's principal-COM body positions.

Usage:
  .venv/bin/python -B -m walk.eval.mujoco_xval --actor <pt> --command 0.15 \
      [--seconds 8] [--seed 4242] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from walk.env.contract import ACT, OBS
from walk.env.flat import (ACTION_SCALE, CONTROL_DT, HOME,
                           MAX_TARGET_INCREMENT, MAX_TILT_RAD,
                           MIN_HEIGHT_FRACTION, PHASE_HZ_BASE,
                           PHASE_HZ_PER_MPS, QDOT_OBS_SCALE, SIM_DT,
                           TICKS_PER_STEP)
from walk.env.native_lane import quat_to_rot
from walk.eval import capture as capture_mod
from walk.eval.gait import evaluate_episode

PINNED_MUJOCO_VERSION = "3.12.0"   # uv pip install -p .venv/bin/python mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "duck_model" / "model"
# The XMLs in duck_model/model/ are the source of truth, but the meshes they
# reference (meshdir="assets") only exist next to the byte-identical evidence
# copies. resolve_scene_path() verifies the SHA equality before falling back.
EVIDENCE_MODEL_DIR = Path(
    "/Users/john/Code/box3d-cuda-voxel-gate-c1/evidence/"
    "open-duck-mujoco-cpu-r3/model")
ROBOT_XML = "open_duck_mini_v2.xml"
SCENE_XML = "scene_flat_terrain.xml"

# Canonical plain-14 joint order == walk/env/flat.py HOME order (verified
# against duck_model/open_duck_plain14_candidate.py JOINTS and the XML).
# All model lookups below go by these names, never by index assumption.
JOINT_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee",
    "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee",
    "right_ankle",
)
FREE_JOINT = "floating_base"
FOOT_GEOMS = ("left_foot_bottom_tpu", "right_foot_bottom_tpu")   # (L, R)
FOOT_BODIES = ("foot_assembly", "foot_assembly_2")               # (L, R)
FLOOR_GEOM = "floor"
KP, KV, EFFORT_CAP = 13.37, 0.0, 3.23     # must equal the model's servos
CLEARANCE_M = 0.001                        # pinned floor-clear reset clearance
SOLE_VERTICES = 18                         # per foot, as baked by the lowering


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_scene_path() -> tuple[Path, str]:
    """Prefer duck_model/model; fall back to the byte-identical evidence copy.

    duck_model/model/ holds the source-of-truth XMLs but (currently) not the
    meshes; the evidence directory holds identical XMLs plus assets/. Never
    silently substitute a different model: the fallback requires byte-equal
    XML hashes.
    """
    if (MODEL_DIR / "assets").is_dir():
        return MODEL_DIR / SCENE_XML, "duck_model/model"
    for name in (ROBOT_XML, SCENE_XML):
        ours, theirs = MODEL_DIR / name, EVIDENCE_MODEL_DIR / name
        if not theirs.is_file():
            raise FileNotFoundError(
                f"model assets missing under {MODEL_DIR} and no evidence copy "
                f"at {theirs}")
        if _sha256(ours) != _sha256(theirs):
            raise ValueError(
                f"{name}: evidence copy differs from duck_model/model source "
                "of truth; refusing to run on an unverified model")
    return EVIDENCE_MODEL_DIR / SCENE_XML, str(EVIDENCE_MODEL_DIR)


class MujocoDuckLane:
    """One duck in real MuJoCo, exposing exactly what the xval loop needs."""

    def __init__(self, scene_path: Path | None = None):
        import mujoco  # deferred: only this module needs the pinned wheel
        self.mujoco = mujoco
        if scene_path is None:
            scene_path, self.model_source = resolve_scene_path()
        else:
            self.model_source = str(scene_path)
        self.scene_path = Path(scene_path)
        m = self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(m)
        self._verify_and_index()
        self.floor_clear_root_z: float | None = None  # set by reset()

    # -- model admission ---------------------------------------------------
    def _require(self, ok: bool, what: str) -> None:
        if not ok:
            raise ValueError(f"MuJoCo model mismatch vs the pinned lowering "
                             f"contract: {what} ({self.scene_path})")

    def _verify_and_index(self) -> None:
        mujoco, m = self.mujoco, self.model

        def name2id(kind, name):
            i = int(mujoco.mj_name2id(m, kind, name))
            self._require(i >= 0, f"missing {name}")
            return i

        self._require(abs(float(m.opt.timestep) - SIM_DT) == 0.0,
                      f"timestep {m.opt.timestep} != {SIM_DT}")
        free = name2id(mujoco.mjtObj.mjOBJ_JOINT, FREE_JOINT)
        self._require(int(m.jnt_type[free]) == int(mujoco.mjtJoint.mjJNT_FREE)
                      and int(m.jnt_qposadr[free]) == 0
                      and int(m.jnt_dofadr[free]) == 0,
                      "floating_base must be the leading free joint")
        # Explicit by-name joint/actuator mapping (canonical plain-14 order).
        self.qadr = np.empty(ACT, int)
        self.vadr = np.empty(ACT, int)
        self.act_ids = np.empty(ACT, int)
        self.joint_limits = np.empty((ACT, 2))
        for k, name in enumerate(JOINT_NAMES):
            j = name2id(mujoco.mjtObj.mjOBJ_JOINT, name)
            self._require(int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE),
                          f"{name} must be a hinge")
            self.qadr[k] = int(m.jnt_qposadr[j])
            self.vadr[k] = int(m.jnt_dofadr[j])
            self.joint_limits[k] = m.jnt_range[j]
            a = name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.act_ids[k] = a
            # Native actuation decision: use the model's own position servos,
            # but only after proving they ARE the PD law our lane replicated.
            self._require(int(m.actuator_trnid[a, 0]) == j,
                          f"actuator {name} drives a different joint")
            self._require(int(m.actuator_dyntype[a]) == int(mujoco.mjtDyn.mjDYN_NONE)
                          and int(m.actuator_gaintype[a]) == int(mujoco.mjtGain.mjGAIN_FIXED)
                          and int(m.actuator_biastype[a]) == int(mujoco.mjtBias.mjBIAS_AFFINE),
                          f"actuator {name} is not a stateless position servo")
            gain, bias = m.actuator_gainprm[a], m.actuator_biasprm[a]
            self._require(abs(float(gain[0]) - KP) < 1e-9
                          and abs(float(bias[1]) + KP) < 1e-9
                          and abs(float(bias[2]) + KV) < 1e-9,
                          f"actuator {name} gains != kp={KP}, kv={KV}")
            self._require(np.allclose(m.actuator_forcerange[a],
                                      (-EFFORT_CAP, EFFORT_CAP), atol=1e-12),
                          f"actuator {name} force range != +/-{EFFORT_CAP}")
        self._require(len(set(self.qadr.tolist())) == ACT, "duplicate joints")
        # Home keyframe must be exactly walk/env/flat.py HOME.
        key = name2id(mujoco.mjtObj.mjOBJ_KEY, "home")
        self.key_id = key
        self._require(np.allclose(m.key_qpos[key][self.qadr], HOME, atol=1e-12),
                      "home keyframe joints != flat.py HOME")
        self._require(np.allclose(m.key_qpos[key][:7],
                                  (0., 0., .15, 1., 0., 0., 0.), atol=1e-12),
                      "home keyframe root pose changed")
        # Feet + floor (the only colliders; sole/contact conventions).
        self.floor_geom = name2id(mujoco.mjtObj.mjOBJ_GEOM, FLOOR_GEOM)
        self._require(int(m.geom_type[self.floor_geom])
                      == int(mujoco.mjtGeom.mjGEOM_PLANE), "floor not a plane")
        self.foot_geoms = tuple(name2id(mujoco.mjtObj.mjOBJ_GEOM, g)
                                for g in FOOT_GEOMS)
        self.foot_bodies = tuple(name2id(mujoco.mjtObj.mjOBJ_BODY, b)
                                 for b in FOOT_BODIES)
        self.foot_mesh_verts = []
        for g in self.foot_geoms:
            self._require(int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH),
                          "foot collider not a mesh")
            mesh = int(m.geom_dataid[g])
            adr, cnt = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            self._require(cnt == SOLE_VERTICES,
                          f"foot mesh has {cnt} vertices, expected "
                          f"{SOLE_VERTICES} (sole-height convention)")
            self.foot_mesh_verts.append(m.mesh_vert[adr:adr + cnt].astype(float))

    # -- reset ---------------------------------------------------------------
    def reset(self) -> None:
        """Home keyframe + minimal floor-clear root lift (1 mm clearance)."""
        mujoco, m, d = self.mujoco, self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, self.key_id)
        mujoco.mj_kinematics(m, d)
        lift = max(0.0, CLEARANCE_M - min(self._sole_heights()))
        d.qpos[2] += lift
        self.floor_clear_root_z = float(d.qpos[2])
        d.ctrl[self.act_ids] = np.clip(HOME, self.joint_limits[:, 0],
                                       self.joint_limits[:, 1])
        mujoco.mj_forward(m, d)

    @property
    def home_root_height(self) -> float:
        if self.floor_clear_root_z is None:
            raise RuntimeError("reset() first")
        return self.floor_clear_root_z

    # -- reads ----------------------------------------------------------------
    def _sole_heights(self) -> list[float]:
        """Whole-sole height per foot: min world z over the mesh vertices.

        Callers must have refreshed kinematics (mj_kinematics/forward/step)."""
        d = self.data
        out = []
        for g, verts in zip(self.foot_geoms, self.foot_mesh_verts):
            world = verts @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            out.append(float(world[:, 2].min()))
        return out

    def _foot_contacts(self) -> list[bool]:
        """Foot-vs-floor flags from the manifold of the just-solved step."""
        d = self.data
        flags = [False, False]
        for i in range(d.ncon):
            pair = {int(d.contact[i].geom1), int(d.contact[i].geom2)}
            for f, g in enumerate(self.foot_geoms):
                flags[f] |= pair == {self.floor_geom, g}
        return flags

    def tick_state(self) -> "TickState":
        """LaneState-alike ([1, ...] arrays) for capture._append_tick reuse."""
        d = self.data
        qpos = d.qpos
        # native-lane layout: root xyz, root quat XYZW, joint q (canonical).
        q = np.concatenate([qpos[0:3],
                            [qpos[4], qpos[5], qpos[6], qpos[3]],  # wxyz->xyzw
                            qpos[self.qadr]])
        rot = quat_to_rot(q[3:7])                      # body -> world
        # free joint qvel: world linear, BODY-frame angular -> world angular.
        v = np.concatenate([d.qvel[0:3], rot @ d.qvel[3:6], d.qvel[self.vadr]])
        return TickState(
            q=q[None, :], v=v[None, :], time=np.array([float(d.time)]),
            foot_contact=np.array([self._foot_contacts()], bool),
            foot_pos=np.array([[d.xipos[b].copy() for b in self.foot_bodies]]),
            sole_height=np.array([self._sole_heights()]))

    # -- stepping ---------------------------------------------------------------
    def policy_step(self, effective_targets: np.ndarray, on_tick=None) -> "TickState":
        """Hold clipped targets on the native servos for 10 x 2 ms steps."""
        mujoco, m, d = self.mujoco, self.model, self.data
        d.ctrl[self.act_ids] = effective_targets
        state = None
        for _ in range(TICKS_PER_STEP):
            mujoco.mj_step(m, d)
            # scratch-only kinematics refresh at the post-step pose (contacts
            # stay the just-solved step's manifold, as in the native lane).
            mujoco.mj_kinematics(m, d)
            state = self.tick_state()
            if on_tick is not None:
                on_tick(state)
        return state


class TickState:
    """Duck-typed native_lane.LaneState subset used by capture/_observe."""

    def __init__(self, q, v, time, foot_contact, foot_pos, sole_height):
        self.q, self.v, self.time = q, v, time
        self.foot_contact, self.foot_pos = foot_contact, foot_pos
        self.sole_height = sole_height

    def finite(self) -> np.ndarray:
        return np.isfinite(self.q).all(1) & np.isfinite(self.v).all(1)


# -- observation (mirror of FlatFloorDuckEnv._observe, E == 1) -----------------
def observe(state: TickState, prev_action: np.ndarray, command: float,
            t_steps: int, phase0: float = 0.0) -> np.ndarray:
    obs = np.zeros((1, OBS), np.float32)
    rot = quat_to_rot(state.q[:, 3:7])                       # body -> world
    obs[:, 0:14] = state.q[:, 7:] - HOME
    obs[:, 14:28] = QDOT_OBS_SCALE * state.v[:, 6:]
    obs[:, 28:42] = prev_action
    obs[:, 42:45] = -rot[:, 2, :]                            # gravity, body
    obs[:, 45:48] = np.einsum("eji,ej->ei", rot, state.v[:, 3:6])
    obs[:, 48:51] = np.einsum("eji,ej->ei", rot, state.v[:, 0:3])
    obs[:, 51] = command
    obs[:, 54:56] = state.foot_contact
    phase = phase0 + 2.0 * math.pi \
        * (PHASE_HZ_BASE + PHASE_HZ_PER_MPS * command) * t_steps * CONTROL_DT
    obs[:, 56] = np.sin(phase)
    obs[:, 57] = np.cos(phase)
    return obs


def _tilt_rad(q_xyzw: np.ndarray) -> float:
    up = 1.0 - 2.0 * (q_xyzw[0] ** 2 + q_xyzw[1] ** 2)
    return math.acos(max(-1.0, min(1.0, up)))


# -- episode runner --------------------------------------------------------------
def run_episode(lane: MujocoDuckLane, policy, command: float,
                seconds: float = 8.0, seed=None) -> dict:
    """Deterministic single-episode rollout recording duckgridwalk.episode/1.

    `policy(obs [1, 58] f32) -> action [1, 14]`. Reuses capture.py's private
    trace builders so the tick schema can never drift from the evaluator's
    input contract. phase0 is 0 for evaluation (matches capture over the env
    only up to the env's random per-episode phase draw; the policy contract
    itself is identical).
    """
    lane.reset()
    trace = capture_mod._new_trace(command, seed, 0)
    min_height = MIN_HEIGHT_FRACTION * lane.home_root_height
    targets = HOME.astype(float).copy()
    prev_action = np.zeros((1, ACT))
    state = lane.tick_state()
    steps = int(round(seconds / CONTROL_DT))

    def on_tick(s):
        capture_mod._append_tick(trace, s, 0)

    for t in range(steps):
        obs = observe(state, prev_action, command, t)
        a = np.clip(np.asarray(policy(obs), float).reshape(1, ACT), -1.0, 1.0)
        requested = HOME + ACTION_SCALE * a[0]
        targets = np.clip(requested, targets - MAX_TARGET_INCREMENT,
                          targets + MAX_TARGET_INCREMENT)
        effective = np.clip(targets, lane.joint_limits[:, 0],
                            lane.joint_limits[:, 1])
        state = lane.policy_step(effective, on_tick=on_tick)
        prev_action = a
        fell = (state.q[0, 2] < min_height
                or _tilt_rad(state.q[0, 3:7]) > MAX_TILT_RAD
                or not bool(state.finite()[0]))
        if fell:
            trace["terminated"] = True
            break
    else:
        trace["truncated_at_horizon"] = True
    return trace


# -- actor loading ------------------------------------------------------------------
def load_actor(path: str | Path):
    """Load a trained Actor; plain state_dict, {"arch","state_dict"} and
    training-checkpoint {"actor": ...} formats. Feedforward only for now."""
    import torch
    from walk.train.ppo import Actor
    obj = torch.load(path, map_location="cpu", weights_only=False)
    arch = None
    if isinstance(obj, dict) and "state_dict" in obj:
        arch, sd = obj.get("arch"), obj["state_dict"]
    elif isinstance(obj, dict) and "actor" in obj \
            and not any(str(k).startswith("mu_net") for k in obj):
        sd = obj["actor"]
    else:
        sd = obj
    if arch is not None:
        kind = arch if isinstance(arch, str) else \
            str(arch.get("kind", arch.get("type", "ff")))
        if any(x in kind.lower() for x in ("gru", "recurrent", "rnn", "lstm")):
            raise ValueError(
                f"actor arch {kind!r} is recurrent; mujoco_xval only supports "
                "feedforward (ff) actors for now")
    if any("gru" in str(k).lower() or "rnn" in str(k).lower() for k in sd):
        raise ValueError("state_dict contains recurrent (gru/rnn) parameters; "
                         "mujoco_xval only supports feedforward actors")
    # Infer hidden widths from the mu_net Linear stack so non-default sizes load.
    widths = [sd[k].shape[0] for k in sorted(
        (k for k in sd if k.startswith("mu_net.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]))]
    if not widths or widths[-1] != ACT:
        raise ValueError(f"unrecognized actor state_dict (mu_net widths "
                         f"{widths}); expected an ff Actor({OBS}, {ACT})")
    actor = Actor(OBS, ACT, hidden=tuple(widths[:-1]))
    actor.load_state_dict(sd)
    actor.eval()
    return actor


def _json_default(o):
    """Evaluator details can carry numpy scalars; keep the JSON plain."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


# -- CLI ----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--actor", required=True)
    ap.add_argument("--command", type=float, required=True,
                    help="commanded forward velocity, m/s (e.g. 0.15)")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=None, help="write trace + result JSON here")
    a = ap.parse_args(argv)

    import mujoco
    import torch
    if mujoco.__version__ != PINNED_MUJOCO_VERSION:
        print(f"WARNING: mujoco {mujoco.__version__} != pinned "
              f"{PINNED_MUJOCO_VERSION}; results are not the pinned evidence "
              "configuration")

    actor = load_actor(a.actor)

    @torch.no_grad()
    def policy(obs):
        return actor.deterministic(
            torch.from_numpy(np.ascontiguousarray(obs))).numpy()

    lane = MujocoDuckLane()
    trace = run_episode(lane, policy, a.command, seconds=a.seconds, seed=a.seed)
    result = evaluate_episode(trace)

    qualified = [f for f in result.get("footfalls", []) if f.get("qualified")]
    left = sum(1 for f in qualified if f["foot"] == "left")
    print(f"mujoco {mujoco.__version__}  model: {lane.model_source}")
    print(f"floor-clear root z: {lane.home_root_height:.17g}")
    print(f"actor: {a.actor}")
    print(f"command {a.command:+.2f} m/s, {a.seconds:g} s, "
          f"{len(trace['ticks']['time_s'])} ticks recorded")
    for name, c in result["criteria"].items():
        print(f"  [{'PASS' if c['pass'] else 'fail'}] {name}: {c['detail']}")
    print(f"qualified footfalls: {len(qualified)} "
          f"(L{left}/R{len(qualified) - left})")
    print("MUJOCO XVAL "
          + ("PASSED" if result["passed"] else
             "rejected" if result.get("rejected") else "not passed"))

    if a.out is not None:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        tp = out / f"mujoco-xval-episode-cmd{a.command:.2f}.json"
        tp.write_text(json.dumps(trace) + "\n")
        rp = out / f"mujoco-xval-result-cmd{a.command:.2f}.json"
        rp.write_text(json.dumps({
            "schema": "duckgridwalk.mujoco_xval/1",
            "mujoco_version": mujoco.__version__,
            "mujoco_version_pinned": PINNED_MUJOCO_VERSION,
            "model_source": lane.model_source,
            "scene_path": str(lane.scene_path),
            "actuation": "native MuJoCo position servos (verified kp=13.37, "
                         "kv=0, force cap 3.23); ctrl = effective targets",
            "floor_clear_root_z": lane.home_root_height,
            "actor": str(a.actor), "seed": a.seed,
            "command_mps": a.command, "seconds": a.seconds,
            "trace_file": tp.name,
            "evaluation": result}, indent=2, sort_keys=True,
            default=_json_default) + "\n")
        print(f"wrote {tp}\nwrote {rp}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
