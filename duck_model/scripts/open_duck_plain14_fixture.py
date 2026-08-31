"""Freeze the plain-14 source candidate; read text only, run no upstream code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import open_duck_plain14_candidate as candidate
from scripts.audit_open_duck_v2 import COMMIT, PREFIX, PINS, read_local_blobs, build_report

EXTRA_PINS = {
    PREFIX + "mujoco_infer_base.py": "4c8fd11e4659126d49985a9c048e1cb5a5cfde5f4d54cdfe433e4b165d3b0ab1",
    "playground/common/poly_reference_motion.py": "cc2430d82a9e41bf2081a4a9bed45731a43aad8b860e5b168f5850eb3748cf21",
    "playground/common/poly_reference_motion_numpy.py": "585e783722ffb403a14288aafc1fcdf86c8a5d1b39949171a3b2d918028c4406",
    PREFIX + "constants.py": "8fafc240bfb8e70c93b722d89148cffb03688882fd4e6250735eccdf53363374",
    PREFIX + "runner.py": "f4ee421a2edb83fb7ebba10f9f331c9553d15987e608eac995fb5317cab486f4",
    "playground/common/runner.py": "226faa0a16b5dc013dd1006f1f9b1893ef8940868d67d6942d46dbcb28ed3659",
}


def require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError("candidate/source mismatch: " + label)


def source_binding(blobs):
    """Compare candidate constants to pinned original text, not candidate goldens."""
    audit = build_report(blobs)
    plain = audit["plain"]
    model = ET.fromstring(blobs[PREFIX + "xmls/open_duck_mini_v2.xml"])
    scene = ET.fromstring(blobs[PREFIX + "xmls/scene_flat_terrain.xml"])
    require_equal(scene.find("include").attrib["file"], "open_duck_mini_v2.xml", "model include")
    require_equal(tuple(a["joint"] for a in plain["actions"]), candidate.JOINTS, "action order")
    require_equal(plain["free_joint_names"], [candidate.FREE_JOINT_NAME], "free joint")
    home = tuple(map(float, scene.find("./keyframe/key[@name='home']").attrib["ctrl"].split()))
    require_equal(home, candidate.HOME, "home")
    limits = {j["name"]: tuple(map(float, j["range"].split())) for j in plain["joints"]}
    require_equal(tuple(limits[name] for name in candidate.JOINTS), candidate.JOINT_LIMITS, "limits")
    for field in ("damping", "frictionloss", "armature"):
        require_equal(float(plain["loaded_joint_default"][field]),
                      candidate.LOADED_MOTOR_PROPERTIES[field], field)
    for field in ("kp", "kv"):
        require_equal(float(plain["loaded_position_default"][field]),
                      candidate.LOADED_MOTOR_PROPERTIES[field], field)
    force = tuple(map(float, plain["loaded_position_default"]["forcerange"].split()))
    limit = candidate.LOADED_MOTOR_PROPERTIES["force_limit"]
    require_equal(force, (-limit, limit), "force limit")
    require_equal(audit["missing_literal_joint_lookups"], ["trunk_assembly_freejoint"], "known upstream mismatch")
    return {
        "schema": "box3d.open_duck_plain14.source_binding/v1",
        "candidate_schema": candidate.SCHEMA,
        "repository": "https://github.com/apirrone/Open_Duck_Playground",
        "source_commit": COMMIT, "source_sha256": PINS | EXTRA_PINS,
        "model": {"scene": PREFIX + "xmls/scene_flat_terrain.xml",
                  "robot": PREFIX + "xmls/open_duck_mini_v2.xml",
                  "free_joint_name": candidate.FREE_JOINT_NAME,
                  "authored_mass_kg": plain["authored_mass_kg"],
                  "action_order": candidate.JOINTS, "home": home,
                  "joint_limits": candidate.JOINT_LIMITS,
                  "motor_properties": candidate.LOADED_MOTOR_PROPERTIES,
                  "option": plain["solver_option"],
                  "option_flags": dict(model.find("./option/flag").attrib),
                  "imu_site": dict(model.find(".//site[@name='imu']").attrib)},
        "observation_layout": audit["actor_layout_from_pinned_source"],
        "corrections": [
            {"field": "joint lookup", "original": "trunk_assembly_freejoint",
             "candidate": "floating_base", "reason": "exact declared source name"},
            {"field": "CPU playback accelerometer X offset", "original": 1.3,
             "candidate": 0.0, "reason": "match effective training computation, not discarded .at.set"},
            {"field": "raw action observation history", "original_cpu_playback": "latest completed action first",
             "candidate": "PRE-shift frames from advance.observation_history",
             "reason": "match training observation built before raw-history rotation"},
        ],
        "controller": {"control_dt_s": candidate.CONTROL_DT,
                       "simulation_dt_s": candidate.SIMULATION_DT, "substeps": 10,
                       "action_scale_rad": candidate.ACTION_SCALE,
                       "target_slew_rad_per_s": candidate.TARGET_SPEED_LIMIT,
                       "target_increment_rad": candidate.MAX_TARGET_INCREMENT,
                       "delay_frames": [0, 1, 2], "delay_sequence_owner": "caller, explicit",
                       "observation_targets": "slewed targets BEFORE inherited actuator-range clipping",
                       "actuator_input": "effective_controls AFTER inherited range clipping",
                       "phase_owner": "caller; exact reference period unverified"},
        "comparison_contract": {"scalar_oracle_dtype": "float64", "fixed_vector_tolerance": 1e-12,
                                "future_float32_tolerance_proposal": 2e-6,
                                "noise": "disabled for first proposed comparison",
                                "randomization_and_pushes": "disabled for first proposed comparison",
                                "normalizer": "must be verified embedded in matching policy; not applied here",
                                "reset_phase": [0, 0], "phase_period_steps": None},
        "released_policy_contract_established": False,
        "missing_policy_contract": ["matching 14-action/101-input ONNX artifact and full SHA",
                                    "input dtype/names/order and normalizer provenance",
                                    "matching model/controller source checkpoint identity",
                                    "reference phase period/fps in safe metadata, not deserialized pickle"],
        "upstream_source_executed": False, "weights_executed": False,
        "physics_executed": False, "native_import_accepted": False,
    }


def freeze(git_dir):
    blobs = read_local_blobs(git_dir)
    env = dict(os.environ, GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0")
    for path, digest in EXTRA_PINS.items():
        data = subprocess.check_output(["git", "--git-dir=" + str(git_dir), "show", COMMIT + ":" + path],
                                       env=env, timeout=15)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("additional source SHA mismatch: " + path)
    return source_binding(blobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = (json.dumps(freeze(args.git_dir), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with args.output.open("xb") as output:
        output.write(data)
    print(json.dumps({"output": str(args.output.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                      "physics_executed": False, "released_policy_contract_established": False}))


if __name__ == "__main__":
    main()
