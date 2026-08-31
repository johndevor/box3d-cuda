"""Pinned Open Duck coordinate/geometry translation reference, not an importer.

Stdlib only. Reads recorded CPU states; never loads a simulator or steps physics.
Native-like poses use COM/principal frames and xyzw quaternion storage. Exact
offset convex support and an infinite floor are explicitly NOT native OBB support.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts import export_open_duck_recorded_view as cad
from scripts.mujoco337_inertia_numeric import compiler_eig3

ROOT = Path(__file__).resolve().parent
RECORD_SHA = "a6d578064b433e730612d7144742b706471e63a37e3c81bcbc24acb7a7203a58"
SOURCE_PINS = {
    "scripts/export_open_duck_recorded_view.py": "94df276acf6c763e900f01eb43a92d3f97172efb3d6be1b1fca357c1c6eef411",
    "scripts/mujoco337_inertia_numeric.py": "e51215603d607e0b57464a81272d8c7c6422a8959a8d40e3235de718d3c6ecdf",
}
POSITION_TOLERANCE = 1e-9
ANGLE_TOLERANCE = 1e-9
SUPPORT_TOLERANCE = 1e-9
require = cad.require


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def conjugate(q):
    return q[0], -q[1], -q[2], -q[3]


def inverse(pose):
    q = conjugate(pose[1])
    return cad.rotate(q, tuple(-x for x in pose[0])), q


def xyzw(q):
    return q[1], q[2], q[3], q[0]


def wxyz(q):
    return q[3], q[0], q[1], q[2]


def flatten_pose(pose):
    return (*pose[0], *xyzw(pose[1]))


def expand_pose(row):
    return tuple(row[:3]), wxyz(row[3:])


def rotvec(q):
    q = cad.normalized(q)
    if q[0] < 0:
        q = tuple(-x for x in q)
    length = math.hypot(*q[1:])
    if length <= 1e-12:
        return tuple(2*x for x in q[1:])
    angle = 2*math.atan2(length, max(0., q[0]))
    return tuple(x*angle/length for x in q[1:])


def joint_geometry(parent, child, joint):
    """Same coordinate formula as joint.cu geometry, evaluated in Python f64."""
    for pose in (parent, child):
        require(len(pose) == 2 and len(pose[0]) == 3 and len(pose[1]) == 4 and
                all(type(x) in (int, float) and math.isfinite(x) for row in pose for x in row), "finite joint body poses required")
    for field, length in (("reference_xyzw", 4), ("axis_parent", 3), ("parent_anchor", 3), ("child_anchor", 3)):
        require(len(joint[field]) == length and all(type(x) in (int, float) and math.isfinite(x) for x in joint[field]), "finite joint parameters required")
    require(abs(math.hypot(*joint["axis_parent"])-1) <= 1e-12, "unit parent axis required")
    qr = cad.qmul(conjugate(parent[1]), child[1])
    qd = cad.qmul(qr, conjugate(wxyz(joint["reference_xyzw"])))
    rotation = rotvec(qd)
    axis = joint["axis_parent"]
    coordinate = dot(rotation, axis)
    angular = math.hypot(*(rotation[i]-coordinate*axis[i] for i in range(3)))
    parent_anchor = cad.transform(parent, joint["parent_anchor"])
    child_anchor = cad.transform(child, joint["child_anchor"])
    result = coordinate, math.dist(parent_anchor, child_anchor), angular
    require(all(math.isfinite(x) for x in result), "nonfinite joint geometry")
    return result


def tensor_from_principal(diagonal, q):
    columns = [cad.rotate(q, tuple(float(i == j) for i in range(3))) for j in range(3)]
    return [[sum(columns[k][i]*diagonal[k]*columns[k][j] for k in range(3))
             for j in range(3)] for i in range(3)]


def matrix_error(a, b):
    return math.hypot(*(a[i][j]-b[i][j] for i in range(3) for j in range(3)))


def convex_support(vertices, direction):
    require(len(vertices) > 0 and len(direction) == 3, "support dimensions")
    require(all(math.isfinite(x) for x in direction) and math.hypot(*direction) > 0, "support direction")
    require(all(len(v) == 3 and all(math.isfinite(x) for x in v) for v in vertices), "support vertices")
    # First source vertex wins an exact tie; value itself is permutation invariant.
    values = [dot(v, direction) for v in vertices]
    require(all(math.isfinite(x) for x in values), "support projection overflow")
    index = max(range(len(vertices)), key=lambda i: values[i])
    return index, tuple(vertices[index]), values[index]


def plane_support(vertices, world_pose, origin=(0., 0., 0.), normal=(0., 0., 1.)):
    require(len(origin) == len(normal) == 3 and all(math.isfinite(x) for x in (*origin, *normal)), "finite plane required")
    require(abs(math.hypot(*normal)-1) <= 1e-12, "unit plane normal required")
    direction = cad.rotate(conjugate(world_pose[1]), tuple(-x for x in normal))
    index, vertex, _ = convex_support(vertices, direction)
    point = cad.transform(world_pose, vertex)
    return {"vertex_index": index, "support_world": point, "distance_m": dot(sub(point, origin), normal)}


def load_inputs(root=ROOT):
    for name, expected in SOURCE_PINS.items():
        require(cad.sha((root/name).read_bytes()) == expected, "helper identity changed: " + name)
    directory = root / "evidence/open-duck-zero-hold-cpu-v1"
    record = cad.read_record(directory/"cpu-result.json", RECORD_SHA)
    setup = cad.read_record(directory/"setup.json", cad.SETUP_SHA)
    require(record["source_commit"] == cad.SOURCE_COMMIT == setup["source_commit"], "source identity")
    require(record["status"] == "passed-bounded-cpu-zero-action-health-only", "recorded outcome")
    require(len(record["frames"]) == 501, "501 recorded states required")
    require(record["compiled_model"]["accepted"] is True, "compiled source admission")
    require(record["compiled_model"]["action_order"] == list(cad.JOINTS), "joint order")
    for name, expected in setup["asset_files"].items():
        data = (directory/"model"/name).read_bytes()
        require(len(data) == expected["bytes"] and cad.sha(data) == expected["sha256"], "source asset identity: " + name)
    xml_path = directory/"model/open_duck_mini_v2.xml"
    bodies, geometry, identities = cad.load_model(xml_path)
    xml_bodies = {b.attrib["name"]: b for b in ET.parse(xml_path).getroot().findall(".//worldbody//body")}
    return record, bodies, geometry, identities, xml_bodies


def make_mapping(record, source_bodies, xml_bodies):
    saved = {b["name"]: b for b in record["compiled_model"]["bodies"]}
    require(saved["base"]["mass"] == 0 and saved["base"]["principal_inertia"] == [0., 0., 0.], "virtual root must remain massless")
    require(source_bodies[1]["name"] == "trunk_assembly" and source_bodies[1]["joint"] is None and source_bodies[1]["rest"] == ((0., 0., 0.), (1., 0., 0., 0.)), "alias requires rigid identity welded trunk")
    native_bodies = [{"id": 0, "name": "floor", "motion": "fixed", "mass": 0., "inverse_mass": 0.,
                      "inverse_inertia_local": (0., 0., 0.), "source_COM": (0., 0., 0.),
                      "inertial_quaternion_wxyz": (1., 0., 0., 0.), "source_index": None}]
    inertia_checks = []
    accepted = {c["body"]: c for c in record["compiled_model"]["inertia_numerical_comparisons"]}
    for index, source in enumerate(source_bodies[1:], start=1):
        b = saved[source["name"]]
        inertial = xml_bodies[b["name"]].find("inertial")
        require(inertial is not None and b["mass"] > 0, "every physical link has authored mass")
        source_com = cad.numbers(inertial.attrib["pos"], 3)
        require(math.dist(source_com, b["COM"]) <= POSITION_TOLERANCE, "authored COM mismatch")
        require(float(inertial.attrib["mass"]) == b["mass"], "authored mass mismatch")
        xx, yy, zz, ij, ik, jk = cad.numbers(inertial.attrib["fullinertia"], 6)
        tensor = [[xx, ij, ik], [ij, yy, jk], [ik, jk, zz]]
        diagonal = b["principal_inertia"]
        require(all(x > 0 and math.isfinite(x) for x in diagonal), "positive principal inertia required")
        q = cad.normalized(b["inertial_quaternion_wxyz"])
        actual = tensor_from_principal(diagonal, q)
        # Reuse frozen source-derived acceptance bounds; do not call the numpy path.
        old = accepted[b["name"]]
        reproduced = compiler_eig3(tensor)
        reference = reproduced["pre_sort_reconstructed"]
        source_error = matrix_error(actual, tensor)
        reference_error = matrix_error(actual, reference)
        require(old["accepted"] is True and source_error <= old["source_to_actual_bound"], "full tensor/source bound mismatch")
        require(reference_error <= old["supplemental_bound"], "principal frame/compiler reproduction mismatch")
        require(all(abs(actual[i][j]-reference[i][j]) <= 1e-10+1e-8*abs(reference[i][j]) for i in range(3) for j in range(3)), "original inertia element bounds")
        native_bodies.append({"id": index, "name": b["name"], "source_index": index, "motion": "dynamic",
                              "mass": b["mass"], "inverse_mass": 1/b["mass"],
                              "source_COM": source_com, "inertial_quaternion_wxyz": q,
                              "principal_inertia": diagonal, "inverse_inertia_local": [1/x for x in diagonal],
                              "authored_full_tensor": tensor})
        inertia_checks.append({"body": b["name"], "source_frobenius_error": source_error,
                               "source_to_actual_bound": old["source_to_actual_bound"],
                               "reference_error": reference_error, "supplemental_bound": old["supplemental_bound"]})
    require(len(native_bodies) == 16, "15 physical links plus floor")
    require(abs(sum(b["mass"] for b in native_bodies)-2.1071407) <= 1e-12, "mass must not be invented or removed")
    zero = {"base_pose": [0., 0., 0., 1., 0., 0., 0.], "joint_q": [0.]*14}
    zero_source = cad.forward_kinematics(source_bodies, zero)
    zero_native = principal_poses(native_bodies, zero_source)
    ids = {b["name"]: b["id"] for b in native_bodies}
    ids["base"] = ids["trunk_assembly"]
    joints = []
    for index, body in enumerate(source_bodies):
        joint = body["joint"]
        if joint is None:
            continue
        child = ids[body["name"]]
        parent = ids[source_bodies[body["parent"]]["name"]]
        world_anchor = cad.transform(zero_source[index], joint["position"])
        world_axis = cad.rotate(zero_source[index][1], joint["axis"])
        native_axis = cad.normalized(cad.rotate(conjugate(zero_native[parent][1]), world_axis))
        limits = cad.numbers(xml_bodies[body["name"]].find("joint").attrib["range"], 2)
        joints.append({"id": len(joints), "name": joint["name"], "kind": 1, "parent": parent, "child": child,
                       "parent_anchor": cad.transform(inverse(zero_native[parent]), world_anchor),
                       "child_anchor": cad.transform(inverse(zero_native[child]), world_anchor),
                       "axis_parent": native_axis,
                       "reference_xyzw": xyzw(cad.normalized(cad.qmul(conjugate(zero_native[parent][1]), zero_native[child][1]))),
                       "lower": limits[0], "upper": limits[1], "reference_source_q": 0.})
    require(tuple(j["name"] for j in joints) == tuple(cad.JOINTS), "joint mapping order")
    colliders = []
    vertices = {g["name"]: g["compiled_vertices"] for g in record["reset_candidate"]["after"]}
    for saved_collider in record["compiled_model"]["colliders"]:
        name = saved_collider["name"]
        if name == "floor":
            colliders.append({"id": 2, "name": name, "body": 0, "shape": "infinite_plane", "origin": (0., 0., 0.), "normal": (0., 0., 1.)})
            continue
        bid = ids[saved_collider["body"]]
        b = native_bodies[bid]
        inertia_pose = (b["source_COM"], b["inertial_quaternion_wxyz"])
        geom_pose = (saved_collider["compiled_position"], cad.normalized(saved_collider["compiled_quaternion_wxyz"]))
        native_geom = cad.compose(inverse(inertia_pose), geom_pose)
        colliders.append({"id": len(colliders), "name": name, "body": bid, "shape": "offset_convex_vertices",
                          "local_pose_xyzw": flatten_pose(native_geom), "vertices": vertices[name],
                          "support_tie_rule": "first vertex in saved compiler order", "source_geom_id": saved_collider["id"]})
    pairs = [{"id": 0, "colliders": (0, 2), "sliding_friction": .6, "provenance": "authored floor priority1 overrides foot priority0"},
             {"id": 1, "colliders": (1, 2), "sliding_friction": .6, "provenance": "authored floor priority1 overrides foot priority0"},
             {"id": 2, "colliders": (0, 1), "sliding_friction": 1., "provenance": "both authored feet have identical priority0/friction1"}]
    return {"bodies": native_bodies, "joints": joints, "colliders": colliders, "pairs": pairs,
            "virtual_alias": {"base": "trunk_assembly"}, "inertia_checks": inertia_checks,
            "source_name_to_native_id": ids, "world_up": "+Z", "floor_rotation_for_native_y_up": "not applied; no native import is emitted",
            "mass_total_kg": sum(b["mass"] for b in native_bodies)}


def principal_poses(bodies, source_poses):
    poses = [((0., 0., 0.), (1., 0., 0., 0.))]
    for b in bodies[1:]:
        poses.append(cad.compose(source_poses[b["source_index"]], (b["source_COM"], b["inertial_quaternion_wxyz"])))
    return poses


def validate_states(record, source_bodies, mapping):
    frames, max_q, max_anchor, max_angular, max_support, max_geom = [], 0., 0., 0., 0., 0.
    for ordinal, frame in enumerate(record["frames"]):
        require(frame["frame"] == ordinal and frame["finite"] is True, "ordered finite frames required")
        source = cad.forward_kinematics(source_bodies, frame)
        poses = principal_poses(mapping["bodies"], source)
        q, anchors, angular = [], [], []
        for index, joint in enumerate(mapping["joints"]):
            value, anchor, error = joint_geometry(poses[joint["parent"]], poses[joint["child"]], joint)
            max_q = max(max_q, abs(value-frame["joint_q"][index]))
            max_anchor, max_angular = max(max_anchor, anchor), max(max_angular, error)
            q.append(value); anchors.append(anchor); angular.append(error)
        supports = []
        for collider, captured in zip(mapping["colliders"][:2], frame["post_kinematics"]["foot_clearance"]):
            require(collider["name"] == captured["name"], "foot capture order")
            require(math.isfinite(captured["distance_m"]) and
                    all(math.isfinite(x) for x in captured["geom_position"]) and
                    all(math.isfinite(x) for row in captured["geom_rotation"] for x in row), "finite captured geometry required")
            world_geom = cad.compose(poses[collider["body"]], expand_pose(collider["local_pose_xyzw"]))
            measured = plane_support(collider["vertices"], world_geom)
            discrepancy = abs(measured["distance_m"]-captured["distance_m"])
            max_support = max(max_support, discrepancy)
            max_geom = max(max_geom, math.dist(world_geom[0], captured["geom_position"]))
            for axis in range(3):
                basis = tuple(float(i == axis) for i in range(3))
                actual = cad.rotate(world_geom[1], basis)
                expected = [captured["geom_rotation"][i][axis] for i in range(3)]
                max_geom = max(max_geom, math.dist(actual, expected))
            supports.append({"name": collider["name"], **measured, "captured_distance_m": captured["distance_m"]})
        frames.append({"frame": ordinal, "time_s": frame["time_s"], "principal_poses_xyzw": [flatten_pose(p) for p in poses],
                       "joint_coordinates": q, "anchor_error_m": anchors, "angular_error_rad": angular, "foot_supports": supports})
    require(max_q <= ANGLE_TOLERANCE and max_angular <= ANGLE_TOLERANCE, "native signed joint convention mismatch")
    require(max_anchor <= POSITION_TOLERANCE, "native principal-frame anchor mismatch")
    require(max_support <= SUPPORT_TOLERANCE and max_geom <= POSITION_TOLERANCE, "exact compiled convex/floor mapping mismatch")
    return frames, {"states": len(frames), "joint_coordinate_comparisons": len(frames)*14,
                    "foot_support_comparisons": len(frames)*2, "maximum_coordinate_error_rad": max_q,
                    "maximum_anchor_error_m": max_anchor, "maximum_angular_error_rad": max_angular,
                    "maximum_support_error_m": max_support, "maximum_geom_transform_error": max_geom}


def negative_controls(record, source_bodies, mapping, frames):
    reset_poses = [expand_pose(p) for p in frames[0]["principal_poses_xyzw"]]
    wrong_com = list(reset_poses)
    b = mapping["bodies"][1]
    source_reset = cad.forward_kinematics(source_bodies, record["frames"][0])
    wrong_com[1] = (source_reset[b["source_index"]][0], reset_poses[1][1])
    com_error = max(joint_geometry(wrong_com[j["parent"]], wrong_com[j["child"]], j)[1] for j in mapping["joints"])
    axis_error = reference_error = 0.
    for index, j in enumerate(mapping["joints"]):
        reversed_axis = dict(j, axis_parent=[-x for x in j["axis_parent"]])
        axis_error = max(axis_error, abs(joint_geometry(reset_poses[j["parent"]], reset_poses[j["child"]], reversed_axis)[0]-record["frames"][0]["joint_q"][index]))
        home_ref = xyzw(cad.qmul(conjugate(reset_poses[j["parent"]][1]), reset_poses[j["child"]][1]))
        wrong_ref = dict(j, reference_xyzw=home_ref)
        reference_error = max(reference_error, abs(joint_geometry(reset_poses[j["parent"]], reset_poses[j["child"]], wrong_ref)[0]-record["frames"][0]["joint_q"][index]))
    worst_drop = max(math.hypot(*(b["authored_full_tensor"][i][j] for i in range(3) for j in range(3) if i != j)) /
                     mapping["inertia_checks"][b["id"]-1]["source_to_actual_bound"] for b in mapping["bodies"][1:])
    collider = mapping["colliders"][0]
    vertices = collider["vertices"]
    lower = [min(v[i] for v in vertices) for i in range(3)]
    upper = [max(v[i] for v in vertices) for i in range(3)]
    worst_obb, worst_direction = 0., None
    for integers in itertools.product((-1., 0., 1.), repeat=3):
        if not any(integers):
            continue
        direction = cad.normalized(integers)
        exact = convex_support(vertices, direction)[2]
        obb = sum((upper[i] if direction[i] > 0 else lower[i])*direction[i] for i in range(3))
        if obb-exact > worst_obb:
            worst_obb, worst_direction = obb-exact, direction
    controls = [
        {"name": "drop_COM_offset", "observed_error": com_error, "reject_above": POSITION_TOLERANCE, "unit": "m anchor separation"},
        {"name": "reverse_parent_axis", "observed_error": axis_error, "reject_above": ANGLE_TOLERANCE, "unit": "rad signed coordinate"},
        {"name": "use_home_reference_instead_of_zero_q", "observed_error": reference_error, "reject_above": ANGLE_TOLERANCE, "unit": "rad coordinate offset"},
        {"name": "drop_authored_offdiagonal_tensor", "observed_error": worst_drop, "reject_above": 1., "unit": "accepted source-bound multiples"},
        {"name": "substitute_compiled_frame_bounding_OBB", "observed_error": worst_obb, "reject_above": SUPPORT_TOLERANCE, "unit": "m support excess", "direction": worst_direction},
        {"name": "invent_1kg_virtual_root", "observed_error": 1., "reject_above": 1e-12, "unit": "kg total mass error"},
    ]
    for control in controls:
        require(math.isfinite(control["observed_error"]) and control["observed_error"] > control["reject_above"], "nondiscriminating negative control")
        control["rejected"] = True
    return controls


def build(root=ROOT):
    record, source_bodies, geometry, identities, xml_bodies = load_inputs(root)
    mapping = make_mapping(record, source_bodies, xml_bodies)
    frames, summary = validate_states(record, source_bodies, mapping)
    controls = negative_controls(record, source_bodies, mapping, frames)
    # Retain the independent source-STL/reset compiled-support comparison too.
    source_reset = cad.forward_kinematics(source_bodies, record["frames"][0])
    source_support = cad.support_checks(geometry, source_reset, record["reset_candidate"]["after"])
    goldens = {"schema": "box3d.open-duck.native-geometry-goldens/v1", "record_sha256": RECORD_SHA,
               "mapping": mapping, "frames": frames, "negative_controls": controls,
               "geometry_only": True, "physics_executed": False, "native_import_accepted": False}
    goldens_bytes = (json.dumps(goldens, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    report = {"schema": "box3d.open-duck.native-geometry-report/v1", "status": "passed-offline-coordinate-and-support-reference-only",
              "record_sha256": RECORD_SHA, "setup_sha256": cad.SETUP_SHA, "source_commit": cad.SOURCE_COMMIT,
              "source_files": identities, "helper_sha256": SOURCE_PINS, "goldens_sha256": cad.sha(goldens_bytes),
              "counts": {"B": 16, "J": 14, "P": 3, "massive_links": 15, "virtual_massless_aliases": 1},
              "state_validation": summary, "negative_controls": controls, "source_STL_reset_support": source_support,
              "tolerances": {"position_m": POSITION_TOLERANCE, "angle_rad": ANGLE_TOLERANCE, "compiled_support_m": SUPPORT_TOLERANCE,
                             "raw_STL_vs_compiled_reset_m": cad.TOLERANCE, "inertia": "unchanged per-body saved algorithm-derived bounds"},
              "physics_executed": False, "native_import_accepted": False, "cuda_execution": False, "new_simulation": False,
              "unimplemented_native_requirements": ["offset convex vertex colliders", "infinite plane collider", "explicit per-pair material binding"],
              "scope": "f64 coordinate/support algebra over saved501 CPU states; not f32 solver parity, dynamics, contact response, native import, training, or walking",
              "material_scope": "explicit sliding coefficients only; no invented restitution or MuJoCo-soft-contact/native-solver equivalence"}
    return goldens, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    goldens, report = build()
    if args.output_directory is None:
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    else:
        directory = args.output_directory
        require(directory.is_absolute(), "absolute artifact directory required")
        paths = [directory/"geometry-goldens.json", directory/"geometry-report.json"]
        require(not any(p.exists() for p in paths), "refusing to overwrite geometry evidence")
        directory.mkdir(parents=True, exist_ok=True)
        for path, value in zip(paths, (goldens, report)):
            with path.open("x") as stream:
                stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
