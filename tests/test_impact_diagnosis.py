from __future__ import annotations

import copy
import unittest

from box3d_cuda.contracts.coupling import CUDA_BACKEND, PHYSX_BACKEND
from box3d_cuda.contracts.impact import (
    IMPACT_SAMPLED_WORLDS,
    IMPACT_STEPS,
    diagnose_impact_reports,
    validated_impact_trace,
)


def report(backend: str, *, shift: int = 0) -> dict:
    samples = []
    for step in IMPACT_STEPS:
        phase = max(0, step - 220 - shift)
        samples.append({
            "control_step": step,
            "joint_positions_rad": [[0.4 - phase * 0.001, -0.8 + phase * 0.002] for _ in range(IMPACT_SAMPLED_WORLDS)],
            "joint_velocities_rad_s": [[-0.12, 0.24] for _ in range(IMPACT_SAMPLED_WORLDS)],
            "drive_efforts_nm": [[2.0, -1.0] for _ in range(IMPACT_SAMPLED_WORLDS)],
            "payload_position_m": [[0.4 + phase * 0.0005, 0.14, 0.0] for _ in range(IMPACT_SAMPLED_WORLDS)],
            "payload_linear_velocity_mps": [[0.06 if phase else 0.0, 0.0, 0.0] for _ in range(IMPACT_SAMPLED_WORLDS)],
            "link2_position_m": [[0.25 + phase * 0.0005, 0.14, 0.0] for _ in range(IMPACT_SAMPLED_WORLDS)],
            "link2_linear_velocity_mps": [[0.06, 0.0, 0.0] for _ in range(IMPACT_SAMPLED_WORLDS)],
            "link2_payload_contact": [step >= 220 + shift for _ in range(IMPACT_SAMPLED_WORLDS)],
            "link2_payload_impulse_magnitude_ns": [0.04 if step in range(220 + shift, 224 + shift) else 0.0 for _ in range(IMPACT_SAMPLED_WORLDS)],
        })
    return {
        "backend": backend,
        "impact_trace": {
            "sampled_worlds": IMPACT_SAMPLED_WORLDS,
            "sampled_world_indices": list(range(IMPACT_SAMPLED_WORLDS)),
            "sampled_steps": list(IMPACT_STEPS),
            "pair_role": "link2_payload",
            "samples": samples,
        },
    }


class ImpactDiagnosisTests(unittest.TestCase):
    def test_dense_trace_reports_contact_timing_and_best_phase_shift(self):
        result = diagnose_impact_reports([
            report(PHYSX_BACKEND),
            report(CUDA_BACKEND, shift=2),
        ])
        self.assertEqual(result["authority"], "descriptive_only_strict_output_parity_remains_authoritative")
        self.assertEqual(result["maximum_contact_onset_error_control_steps"], 2)
        self.assertEqual(result["mean_contact_onset_error_control_steps"], 2)
        self.assertEqual(result["cuda_contact_onset_minus_physx_histogram_control_steps"]["2"], 64)
        self.assertEqual(result["first_observed_contact_control_step"], 220)
        self.assertEqual(result["worlds_missing_contact_onset"], [])
        self.assertEqual(result["best_phase_shift_histogram_control_steps"]["2"], 64)
        self.assertAlmostEqual(result["maximum_cumulative_link2_payload_impulse_error_ns"], 0.0)
        self.assertEqual(result["precontact_maximum_errors"]["joint_position_rad"], 0.0)
        self.assertIsNone(result["first_parity_threshold_crossing_control_steps"]["joint_position_rad"])

    def test_trace_validator_fails_closed_on_shape_type_and_window_changes(self):
        valid = report(CUDA_BACKEND)
        validated_impact_trace(valid)
        mutations = []
        wrong_step = copy.deepcopy(valid)
        wrong_step["impact_trace"]["sampled_steps"][0] -= 1
        mutations.append(wrong_step)
        wrong_shape = copy.deepcopy(valid)
        wrong_shape["impact_trace"]["samples"][0]["payload_position_m"].pop()
        mutations.append(wrong_shape)
        wrong_type = copy.deepcopy(valid)
        wrong_type["impact_trace"]["samples"][0]["link2_payload_contact"][0] = 1
        mutations.append(wrong_type)
        nonfinite = copy.deepcopy(valid)
        nonfinite["impact_trace"]["samples"][0]["joint_positions_rad"][0][0] = float("nan")
        mutations.append(nonfinite)
        for index, candidate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                validated_impact_trace(candidate)


if __name__ == "__main__":
    unittest.main()
