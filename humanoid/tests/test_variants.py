"""Humanoid FAMILY gates: builder identity, variant morphology, headers,
plant feasibility checklist, analytic CRBA, discrete PD stability.

Run: .venv/bin/python -B humanoid/tests/test_variants.py

Gates:
- IDENTITY: h1_family.build(H1) reproduces humanoid/h1_lowering.py's tables
  BIT-IDENTICALLY and emits the committed humanoid/include/duck_model.h
  byte-for-byte (the refactor-safety proof for the parametrized builder);
- per variant: the authored deltas (leg/torso length, masses, sole width,
  effort caps, PROFILE), anchors close, FK at reset reproduces the derived
  centers (av1 evaluate), soles exactly on the floor, hip anchors +-0.15;
- header drift per variant (committed == fresh regeneration, carrying the
  VARIANT's reference table) and reference-json drift per variant;
- feasibility checklist: H1.1 reference row pins (values measured here),
  every variant DELIVERABLE (absolute bound, or baseline-parity where the
  accepted H1.1 misses the absolute bound itself), (e) sign per member;
- the analytic CRBA mass matrix == av1's (joint block 1e-12, eigenvalues);
- discrete PD stability at SIM_DT with the variant's per-joint tables
  (spectral radius <= 1 + 1e-9), the H1 oracle test's method.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "humanoid"))
sys.path.insert(0, str(ROOT / "walk" / "env"))

import feasibility_check as fc  # noqa: E402
import h1_family as fam  # noqa: E402
import h1_lowering as h1  # noqa: E402
import native_lane  # noqa: E402

TOOLS = ROOT / "experimental" / "duck_cuda" / "tools"
VARIANTS = ("h1_tall", "h1_stocky")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuilderIdentity(unittest.TestCase):
    def test_identity_build_is_bit_identical_to_h1_lowering(self):
        ns = fam.build(fam.H1)
        self.assertTrue(fam.tables_equal(ns, h1))
        self.assertEqual(ns.PROFILE, h1.PROFILE)
        self.assertEqual(ns.TOTAL_DYNAMIC_MASS, h1.TOTAL_DYNAMIC_MASS)
        self.assertEqual(ns.LEG_LENGTH_M, 0.86)
        self.assertIs(fam.load_lowering("h1"), h1)
        self.assertIs(fam.load_lowering(None), h1)

    def test_identity_build_emits_committed_h1_header_byte_identical(self):
        gen = _load("generate_model_humanoid", TOOLS / "generate_model_humanoid.py")
        committed = (ROOT / "humanoid" / "include" / "duck_model.h").read_text()
        self.assertEqual(gen.emit(fam.build(fam.H1)), committed)
        self.assertEqual(gen.emit_variant("h1"), committed)
        self.assertEqual(gen.emit(gen.load_lowering()), committed)

    def test_registry(self):
        self.assertEqual(fam.variant_names(), ("h1", "h1_tall", "h1_stocky"))
        with self.assertRaises(ValueError):
            fam.load_lowering("h9")
        self.assertEqual(fam.header_path("h1"),
                         ROOT / "humanoid" / "include" / "duck_model.h")
        self.assertEqual(fam.header_path("h1_tall"),
                         ROOT / "humanoid" / "variants" / "h1_tall" / "include"
                         / "duck_model.h")


class VariantMorphology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = native_lane._native()
        cls.lib = cls.native.library(str(native_lane.build_library()))
        cls.lw = {v: fam.load_lowering(v) for v in VARIANTS}

    def test_tall_deltas(self):
        lw = self.lw["h1_tall"]
        g, g0 = lw.GEOMETRY, fam.leg_geometry(h1)
        self.assertAlmostEqual(g["thigh_m"], 1.12 * g0["thigh_m"], places=12)
        self.assertAlmostEqual(g["shank_m"], 1.12 * g0["shank_m"], places=12)
        self.assertAlmostEqual(g["leg_length_m"], 0.9632, places=12)
        self.assertAlmostEqual(g["hip_height_m"], 1.1032, places=12)
        torso = lw.BODIES[h1.BODY_NAMES.index("torso")]
        self.assertAlmostEqual(torso[2][1], 0.42, places=12)      # 0.40 * 1.05
        self.assertAlmostEqual(torso[3], 21.0, places=12)         # 20 * 1.05
        thigh = lw.BODIES[h1.BODY_NAMES.index("left_upper_leg")]
        self.assertAlmostEqual(thigh[3], 6.5 * 1.12, places=12)   # density kept
        self.assertAlmostEqual(lw.TOTAL_DYNAMIC_MASS, 71.52, places=9)
        self.assertEqual(lw.EFFORT, h1.EFFORT)                    # caps unchanged
        self.assertEqual(g["sole_half_width_m"], 0.14)
        self.assertEqual(lw.PROFILE, "duckgridwalk.humanoid.h1_tall-v1")

    def test_stocky_deltas(self):
        lw = self.lw["h1_stocky"]
        g = lw.GEOMETRY
        self.assertAlmostEqual(g["leg_length_m"], 0.86 * 0.94, places=12)
        self.assertAlmostEqual(lw.TOTAL_DYNAMIC_MASS, 68.0 * 1.2, places=9)
        for b, b0 in zip(lw.BODIES[1:], h1.BODIES[1:]):
            self.assertAlmostEqual(b[3], 1.2 * b0[3], places=12, msg=b[0])
        for f in lw.FOOT_BODIES:
            self.assertAlmostEqual(lw.BODIES[f][2][2], 0.155, places=12)
            self.assertEqual(lw.BODIES[f][2][:2], (0.23, 0.07))
        self.assertEqual(lw.EFFORT, tuple(round(e * 1.15, 12) for e in h1.EFFORT))
        self.assertEqual(lw.EFFORT[:2], (207.0, 80.5))
        self.assertEqual(lw.PROFILE, "duckgridwalk.humanoid.h1_stocky-v1")

    def test_shared_structure_and_hip_anchors(self):
        for v, lw in self.lw.items():
            self.assertEqual((lw.J, lw.B, lw.N, lw.Q), (14, 16, 20, 21), v)
            self.assertEqual(lw.JOINT_NAMES, h1.JOINT_NAMES)
            self.assertEqual(lw.BODY_NAMES, h1.BODY_NAMES)
            for jt, jt0 in zip(lw.JOINTS, h1.JOINTS):
                self.assertEqual(jt[:2], jt0[:2], v)          # name, parent
                self.assertEqual(jt[4:6], jt0[4:6], v)        # limits
                self.assertEqual(jt[7], jt0[7], v)            # axis
            jn = list(lw.JOINT_NAMES)
            for side, sign in (("left", 1.0), ("right", -1.0)):
                ap = lw.JOINTS[jn.index(f"{side}_hip_roll")][2]
                self.assertEqual(ap, (0.0, -0.15, sign * 0.15), v)
            self.assertEqual(lw.GRAVITY, 20.0)
            self.assertEqual(lw.SIM_DT, 0.002)

    def test_fk_reset_reproduces_derived_centers_and_soles_on_floor(self):
        for v, lw in self.lw.items():
            f = lw.fixture()
            rc, ev = f.evaluate(self.lib, lw.reset_qpos(), np.zeros(lw.N),
                                gravity=(0.0, 0.0, -lw.GRAVITY))
            self.assertEqual(rc, 0, v)
            for b in range(1, lw.B):
                np.testing.assert_allclose(
                    ev.pose[b, :3], lw.y_up_to_z_up(lw.BODIES[b][1]),
                    atol=1e-12, err_msg=f"{v} {lw.BODY_NAMES[b]}")
                np.testing.assert_allclose(ev.pose[b, 3:], lw.QX90, atol=1e-12)
            verts = np.array(lw.foot_vertices())
            for foot in lw.FOOT_BODIES:
                rot = native_lane.quat_to_rot(ev.pose[foot, 3:][None])[0]
                z = (ev.pose[foot, :3] + verts @ rot.T)[:, 2]
                self.assertAlmostEqual(float(z.min()), 0.0, places=12, msg=v)
            self.assertAlmostEqual(
                float(lw.reset_qpos()[2]), lw.GEOMETRY["hip_height_m"] + 0.15,
                places=12)


class ArtifactDrift(unittest.TestCase):
    def test_variant_headers_match_regeneration(self):
        gen = _load("generate_model_humanoid", TOOLS / "generate_model_humanoid.py")
        for v in VARIANTS:
            path = fam.header_path(v)
            self.assertTrue(path.is_file(), f"{path} missing: regenerate with "
                            f"generate_model_humanoid.py --lowering {v}")
            self.assertEqual(gen.emit_variant(v), path.read_text(),
                             f"{v} header drifted; regenerate")
            text = path.read_text()
            self.assertIn(f"PROFILE duckgridwalk.humanoid.{v}-v1", text)
            self.assertIn("humanoid/tests/test_humanoid_serial_parity.py", text)

    def test_variant_reference_json_match_regeneration(self):
        author = _load("author_reference_gait",
                       ROOT / "humanoid" / "author_reference_gait.py")
        base = json.loads((ROOT / "humanoid" / "reference_gait.json").read_text())
        for v in VARIANTS:
            lw = fam.load_lowering(v)
            path = fam.reference_gait_path(v)
            fresh = json.dumps(author.payload(lw), indent=1) + "\n"
            self.assertEqual(fresh, path.read_text(), f"{v} reference drifted")
            data = json.loads(fresh)
            self.assertEqual(data["variant"], v)
            self.assertEqual(data["constants"]["leg_length_m"], lw.LEG_LENGTH_M)
            self.assertNotEqual(data["table"], base["table"])
            table = np.asarray(data["table"])
            self.assertEqual(table.shape, (64, 14))
            lim = np.array([(j[4], j[5]) for j in lw.JOINTS])
            self.assertTrue((table >= lim[:, 0]).all() and (table <= lim[:, 1]).all())
            self.assertLessEqual(float(np.abs(table).max()), author.ACTION_BOX + 1e-12)
        # H1 payload unchanged by the parametrization
        self.assertEqual(json.dumps(author.payload(), indent=1) + "\n",
                         (ROOT / "humanoid" / "reference_gait.json").read_text())

    def test_header_ref_gait_is_the_variant_table(self):
        for v in VARIANTS:
            text = fam.header_path(v).read_text()
            block = text[text.index("DW_REF_GAIT"):]
            block = block[block.index("{") + 1:block.index(";") - 1]
            rows = [r.strip("{}").split(",") for r in block.split("},{")]
            header_table = np.array([[float(x) for x in row] for row in rows])
            table = np.asarray(json.loads(
                fam.reference_gait_path(v).read_text())["table"])
            np.testing.assert_array_equal(header_table, table, v)


class FeasibilityChecklist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = {v: fc.run(fam.load_lowering(v)) for v in fam.variant_names()}

    def _row(self, v, check, joint=""):
        for r in self.rows[v]:
            if r.check == check and (not joint or r.joint == joint):
                return r
        raise KeyError((v, check, joint))

    def test_h1_reference_row_pins(self):
        """The accepted H1.1 measured by the checklist (documents which
        absolute bars the walker itself meets)."""
        r = self._row("h1", "a_sagittal_hold")
        self.assertAlmostEqual(r.value, 42.75, delta=0.05)
        self.assertTrue(r.ok)
        roll = self._row("h1", "b_bandwidth", "left_hip_roll")
        self.assertAlmostEqual(roll.value, 2.695, delta=0.005)   # PHASE2 2.7 Hz
        self.assertFalse(roll.ok)                                # 1.6x < 3x
        for jn in ("left_hip", "left_knee", "left_ankle"):
            self.assertTrue(self._row("h1", "b_bandwidth", jn).ok, jn)
        kg = self._row("h1", "c_grav_stiffness", "left_hip_roll")
        self.assertAlmostEqual(kg.bound / fc.STIFFNESS_MARGIN, 388.4, delta=0.5)
        self.assertTrue(kg.ok)
        knee = self._row("h1", "c_grav_stiffness", "left_knee")
        self.assertAlmostEqual(knee.bound / fc.STIFFNESS_MARGIN, 1028.3, delta=0.5)
        self.assertFalse(knee.ok)                                # 800 < 1.2*1028
        self.assertAlmostEqual(self._row("h1", "d_ankle_cop_authority").value,
                               0.448, delta=0.002)
        lat = self._row("h1", "e_lateral_static_margin")
        self.assertAlmostEqual(lat.value, -0.010, places=9)      # active lean only
        self.assertFalse(lat.ok)

    def test_variants_deliverable(self):
        ref = self.rows["h1"]
        for v in VARIANTS:
            verdict = fc.verdict(self.rows[v], ref)
            self.assertTrue(all(ok for _, ok, _ in verdict),
                            (v, [(r.check, r.joint, why) for r, ok, why in verdict
                                 if not ok]))
            # every gated ratio at least the accepted baseline's
            for r, q in zip(fc.gated(self.rows[v]), fc.gated(ref)):
                if r.check != "a_sagittal_hold":         # (a) absolute-only
                    self.assertGreaterEqual(r.ratio, q.ratio - 1e-9,
                                            (v, r.check, r.joint))
            self.assertTrue(self._row(v, "a_sagittal_hold").ok, v)
            self.assertTrue(self._row(v, "c_grav_stiffness", "left_hip_roll").ok, v)

    def test_lateral_margin_sign_per_member(self):
        self.assertLess(self._row("h1_tall", "e_lateral_static_margin").value, 0)
        s = self._row("h1_stocky", "e_lateral_static_margin")
        self.assertGreater(s.value, 0)             # +3 cm soles: statically stable
        self.assertAlmostEqual(s.value, 0.005, places=9)

    def test_results_are_value_bound_pass_tuples(self):
        for v, rows in self.rows.items():
            self.assertEqual(len(rows), 10, v)
            for r in rows:
                value, bound, ok = r.as_tuple()
                self.assertTrue(np.isfinite(value), (v, r.check))
                self.assertIsInstance(ok, bool)


class AnalyticDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = native_lane._native()
        cls.lib = cls.native.library(str(native_lane.build_library()))

    def test_crba_matches_av1_mass_matrix(self):
        for v in fam.variant_names():
            lw = fam.load_lowering(v)
            M = fc.Home(lw).mass_matrix()
            rc, ev = lw.fixture().evaluate(self.lib, lw.reset_qpos(),
                                           np.zeros(lw.N),
                                           gravity=(0.0, 0.0, -lw.GRAVITY))
            self.assertEqual(rc, 0, v)
            np.testing.assert_allclose(M[6:, 6:], ev.mass[6:, 6:], atol=1e-12)
            np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(M)),
                                       np.sort(np.linalg.eigvalsh(ev.mass)),
                                       atol=1e-10)
            ieff = fc.Home(lw).effective_inertia()
            np.testing.assert_allclose(
                ieff, 1.0 / np.diag(np.linalg.inv(ev.mass))[6:], rtol=1e-9)
        # the codebase's quoted H1 hip-roll I_eff
        self.assertAlmostEqual(fc.Home(h1).effective_inertia()[2], 1.743, delta=0.001)

    def test_discrete_pd_stability_with_variant_tables(self):
        """Spectral radius of the linearized one-tick map at SIM_DT with the
        member's per-joint KP/KV tables (test_humanoid_oracle's method)."""
        for v in fam.variant_names():
            lw = fam.load_lowering(v)
            rc, ev = lw.fixture().evaluate(self.lib, lw.reset_qpos(),
                                           np.zeros(lw.N),
                                           gravity=(0.0, 0.0, -lw.GRAVITY))
            self.assertEqual(rc, 0, v)
            minv = np.linalg.inv(ev.mass)
            n = lw.N
            kp, kd = np.zeros((n, n)), np.zeros((n, n))
            for j in range(lw.J):
                kp[6 + j, 6 + j] = lw.KP_TABLE[j]
                kd[6 + j, 6 + j] = lw.KV_TABLE[j]
            dt = lw.SIM_DT
            a = np.block([[np.eye(n) - dt * minv @ kd, -dt * minv @ kp],
                          [dt * (np.eye(n) - dt * minv @ kd),
                           np.eye(n) - dt * dt * minv @ kp]])
            radius = float(np.abs(np.linalg.eigvals(a)).max())
            self.assertLessEqual(radius, 1.0 + 1e-9, (v, radius))
            ieff = fc.Home(lw).effective_inertia()
            self.assertLess(float((np.array(lw.KV_TABLE) * dt / ieff).max()), 2.0, v)


class RewardOverrides(unittest.TestCase):
    """Per-variant reward-constant overrides (h1_family FAMILY LAWS): the
    swing-clearance margin for TALL, pinned into its header as
    DW_RW_CLEARANCE_M, applied by the python reward, absent for the base."""

    def test_tall_clearance_margin_pinned_in_header(self):
        import re
        tall = fam.load_lowering("h1_tall")
        self.assertEqual(tall.REWARD_OVERRIDES, {"CLEARANCE_M": 0.045})
        self.assertEqual(getattr(h1, "REWARD_OVERRIDES", {}), {})
        self.assertEqual(fam.build(fam.H1).REWARD_OVERRIDES, {})

        def macro(path):
            m = re.search(r"#define DW_RW_CLEARANCE_M ([^ \n]+)", path.read_text())
            return float(m.group(1))
        self.assertEqual(macro(fam.header_path("h1_tall")), 0.045)
        self.assertEqual(macro(fam.header_path("h1")), 0.03)
        self.assertEqual(macro(fam.header_path("h1_stocky")), 0.03)
        self.assertEqual(fam.CLEARANCE_MARGIN_FACTOR, 1.5)

    def test_reward_override_changes_only_the_clearance_bar(self):
        from walk.env import humanoid_reward as hr
        E = 1
        tracker_a, tracker_b = hr.GaitTracker(E), hr.GaitTracker(E)
        z = np.zeros((E, 3))
        prev = {"root_lin_vel": z, "root_ang_vel": z, "foot_contact": np.array([[True, True]]),
                "sole_height": np.zeros((E, 2)), "action": np.zeros((E, 14)),
                "torque": np.zeros((E, 14)), "foot_x": np.zeros((E, 2)),
                "joint_q": np.zeros((E, 14)), "phase": np.zeros(E)}
        cur = dict(prev, foot_contact=np.array([[True, False]]),
                   sole_height=np.array([[0.0, 0.038]]))   # 38 mm: above 30, below 45
        a = hr.reward(prev, cur, np.zeros((E, 14)), np.array([0.5]), tracker_a)
        b = hr.reward(prev, cur, np.zeros((E, 14)), np.array([0.5]), tracker_b,
                      overrides={"CLEARANCE_M": 0.045})
        self.assertAlmostEqual(float(a[0] - b[0]), hr.W_CLEARANCE, places=6)
        c = hr.reward(prev, cur, np.zeros((E, 14)), np.array([0.5]), hr.GaitTracker(E),
                      overrides=None)
        self.assertEqual(float(c[0]), float(a[0]))

    def test_variant_env_applies_override(self):
        from walk.env import humanoid_flat as hf
        from walk.env.humanoid_cuda_lane import CudaHumanoidLane
        env = hf.FlatFloorHumanoidEnv(
            environments=1, seed=3, variant="h1_tall",
            lane_factory=lambda E, off: CudaHumanoidLane(E, joint_offsets=off,
                                                         variant="h1_tall"))
        try:
            self.assertEqual(env._reward_overrides, {"CLEARANCE_M": 0.045})
        finally:
            env.close()
        env = hf.FlatFloorHumanoidEnv(environments=1, seed=3)
        try:
            self.assertIsNone(env._reward_overrides)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
