"""Cube-grid walker fault-corpus regression for dwv1 (numpy + ctypes only).

Replays the grid-backend solver_fault artifacts recorded while the accepted
flat-floor walker ran zero-shot on the cube grid (runs/faults/20260901T18*,
backend "duck_world_v1"). Every phase-2 artifact was a hard DWV1_CONTACT
geometry fault in dwv1_geometry.h's convex_contact: near-parallel grazing
foot-edge-on-cube configurations starve the SAT edge-edge support band
(empty band / far-only band / premature narrow-band break), although a valid
witness within reach = depth + 2e-4 exists among all edge pairs. The repair
widens the CANDIDATE band progressively and finishes with an exhaustive
edge-pair scan while keeping the unchanged acceptance certificate (witness
distance <= reach), so these gates must hold forever:

  * every phase-2 (geometry) artifact steps 10 ticks with zero faults from a
    fresh scene at the recorded qpos/velocity over the recorded grid
    (geometry faults are pose-pure: 20/20 originally reproduced this way);
  * momentum residual stays <= 1e-8 on every accepted tick;
  * phase-3 artifacts (civ1 no-convergence on the foot-straddling-two-
    coplanar-cubes block) may still stall at the pinned 1e-8 -- that is a
    documented civ1 certificate floor (they converge at 1e-7 in ~20 sweeps),
    NOT a geometry regression -- but they must never fault in phase 2.
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
FAULTS = ROOT / "runs" / "faults"
sys.path.insert(0, str(ROOT))

from walk.env import world  # noqa: E402


def load_corpus():
    files = []
    for f in sorted(FAULTS.glob("20260901T18*.json")):
        artifact = json.loads(f.read_text())
        if artifact.get("backend") == "duck_world_v1":
            files.append((f.name, artifact))
    return files


class GridFaultCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not FAULTS.is_dir():
            raise unittest.SkipTest("fault corpus not present: " + str(FAULTS))
        cls.corpus = load_corpus()
        if not cls.corpus:
            raise unittest.SkipTest("no duck_world_v1 fault artifacts")
        cls.lib_path = world.build()
        cls.lib = world.library(cls.lib_path)
        cls.fixture = world.av.duck()
        cls.cm = world.contact.Model(world.contact.library(str(cls.lib_path)))
        cls.limits = world.av.limits(cls.fixture)

    def _scene(self, artifact):
        st, g = artifact["state"], artifact["grid"]
        spec = world.grid_spec(
            nx=g["nx"], nz=g["nz"], cube_size=g["cube_size"],
            spacing=g["spacing"], base_height=g["base_height"],
            height_jitter=g["height_jitter"], origin_x=g["origin_x"],
            origin_y=g["origin_y"], dynamic=g["dynamic"],
            cube_mass=g["cube_mass"], friction=g["friction"], seed=g["seed"])
        return world.Scene(self.lib, self.fixture, np.array([st["qpos"]]),
                           np.array([st["velocity"]]), self.cm.shapes,
                           self.cm.pairs, np.tile(self.cm.mu, (1, 1)), spec,
                           limits=self.limits)

    def test_geometry_fault_states_step_10_ticks_clean(self):
        phase2 = [(n, a) for n, a in self.corpus
                  if a["diagnostics"] and a["diagnostics"][0]["phase"] == 2]
        self.assertGreaterEqual(len(phase2), 20, "corpus shrank unexpectedly")
        failures, worst_mr = [], 0.0
        for name, artifact in phase2:
            scene = self._scene(artifact)
            try:
                for tick in range(10):
                    rc, diag = scene.step(
                        dt=artifact["dt"], target=[artifact["effective_targets"]],
                        max_iterations=16384, tolerance=artifact["tolerance"],
                        jtol=artifact.get("jtol", 1e-8))
                    if rc:
                        failures.append((name, tick, rc, diag[0]["phase"]))
                        break
                    worst_mr = max(worst_mr, diag[0]["momentum_residual"])
                else:
                    x = scene.read()
                    if not (np.isfinite(x.q).all() and np.isfinite(x.v).all()):
                        failures.append((name, "nonfinite", 0, 0))
            finally:
                scene.close()
        self.assertEqual(failures, [],
                         f"{len(failures)}/{len(phase2)} geometry states faulted")
        self.assertLessEqual(worst_mr, 1e-8, "momentum residual contract")
        print(f"grid_fault_corpus phase2 {len(phase2)}/{len(phase2)} clean, "
              f"momentum<= {worst_mr:.2e}", file=sys.stderr)

    def test_solver_fault_states_never_fault_in_geometry(self):
        phase3 = [(n, a) for n, a in self.corpus
                  if a["diagnostics"] and a["diagnostics"][0]["phase"] == 3]
        if not phase3:
            raise unittest.SkipTest("no phase-3 artifacts in corpus")
        clean = 0
        for name, artifact in phase3:
            scene = self._scene(artifact)
            try:
                rc, diag = scene.step(
                    dt=artifact["dt"], target=[artifact["effective_targets"]],
                    max_iterations=16384, tolerance=artifact["tolerance"],
                    jtol=artifact.get("jtol", 1e-8))
                if rc == 0:
                    clean += 1
                else:  # documented civ1 certificate floor; never geometry
                    self.assertEqual(diag[0]["phase"], 3, name)
            finally:
                scene.close()
        print(f"grid_fault_corpus phase3 {clean}/{len(phase3)} clean fresh; "
              "rest stall in civ1 only (documented floor)", file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
