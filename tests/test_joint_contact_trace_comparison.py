from __future__ import annotations

import copy
import math
import unittest

from box3d_cuda.contracts.coupling_compare import (
    SAMPLED_STEPS,
    SAMPLED_WORLD_INDICES,
    SAMPLED_WORLDS,
    compare_coupling_reports,
)
from box3d_cuda.contracts.coupling import (
    BENCHMARK_STEPS,
    BODY_COUNT,
    CONTRACT_ID,
    CUDA_BACKEND,
    CUDA_SOLVER_CONFIGURATION,
    DEFAULT_SEED,
    GATE_THRESHOLDS,
    PHYSX_BACKEND,
    PHYSX_PGS_SOLVER_CONFIGURATION,
    SPEC,
    WORLDS,
    initial_joint_positions_rad,
    target_positions_rad,
    target_batch_rad,
    target_scale,
)


def test_shared_target_scale_matches_contract_targets() -> None:
    for step in (0, 59, 60, 61, 180, 300, 719):
        initial = initial_joint_positions_rad(17)
        expected = tuple(value * target_scale(step) for value in initial)
        assert target_positions_rad(step, 17) == expected


def test_matched_batch_replicates_world_zero_initial_targets() -> None:
    targets = target_batch_rad(64, 120)

    assert len(targets) == 64
    assert all(target == targets[0] for target in targets)
    assert SPEC.metadata()["initial_state_batch_layout"] == "replicated_world_zero"


def measured_report(backend: str, *, rate: float) -> dict:
    correctness = {
        "passed": True,
        "measured_runtime_evidence": True,
        "synthetic": False,
        **SPEC.metadata(seed=DEFAULT_SEED),
        "gate_thresholds": dict(GATE_THRESHOLDS),
        "replicated_initial_state_passed": True,
        "maximum_initial_state_replica_error": 0.0,
        "finite_joint_and_body_state": True,
        "normalized_body_quaternions": True,
        "deterministic_replay_passed": True,
        "world_isolation_passed": True,
        "joint_limits_respected": True,
        "drive_effort_clamped": True,
        "real_contact_impulses_observed": True,
        "friction_negative_control_passed": True,
        "no_attachment_or_teleportation": True,
        "no_hidden_force_injection": True,
        "maximum_joint_limit_excess_rad": 0.0,
        "maximum_drive_effort_ratio": 1.0,
        "maximum_joint_anchor_error_m": 0.0001,
        "maximum_penetration_m": 0.001,
        "maximum_quaternion_norm_error": 1.0e-7,
        "maximum_uncommanded_energy_increase_j": 0.01,
        "link2_payload_contact_frames": 20,
        "floor_payload_contact_frames": 700,
        "payload_forward_displacement_m": 0.20,
        "friction_negative_control_tail_speed_delta_mps": 0.08,
    }
    samples = []
    for step in SAMPLED_STEPS:
        progress = step / (BENCHMARK_STEPS - 1)
        samples.append({
            "control_step": step,
            "joint_positions_rad": [[0.55 * (1.0 - progress), -1.0 * (1.0 - progress)] for _ in range(SAMPLED_WORLDS)],
            "joint_velocities_rad_s": [[-0.1, 0.15] for _ in range(SAMPLED_WORLDS)],
            "joint_targets_rad": [[0.0, 0.0] for _ in range(SAMPLED_WORLDS)],
            "drive_efforts_nm": [[1.0, -0.5] for _ in range(SAMPLED_WORLDS)],
            "body_positions_m": [
                [[body * 0.1 + (0.2 * progress if body == 4 else 0.0), 0.14, 0.0] for body in range(BODY_COUNT)]
                for _ in range(SAMPLED_WORLDS)
            ],
            "body_quaternions_xyzw": [
                [[0.0, 0.0, 0.0, 1.0] for _ in range(BODY_COUNT)]
                for _ in range(SAMPLED_WORLDS)
            ],
            "body_linear_velocities_mps": [
                [[0.02 if body == 4 else 0.0, 0.0, 0.0] for body in range(BODY_COUNT)]
                for _ in range(SAMPLED_WORLDS)
            ],
            "body_angular_velocities_rad_s": [
                [[0.0, 0.0, 0.0] for _ in range(BODY_COUNT)]
                for _ in range(SAMPLED_WORLDS)
            ],
            "pair_contact": [[True, False, False, False, step >= 60] for _ in range(SAMPLED_WORLDS)],
            "pair_contact_impulse_magnitude_ns": [[0.08, 0.0, 0.0, 0.0, 0.04 if step >= 60 else 0.0] for _ in range(SAMPLED_WORLDS)],
        })
    return {
        "backend": backend,
        "contract_id": CONTRACT_ID,
        "worlds": WORLDS,
        "bodies_per_world": BODY_COUNT,
        "steps": BENCHMARK_STEPS,
        "world_steps_per_second": rate,
        "capabilities": {"articulated_joints": True, "rigid_body_contacts": True},
        "solver_configuration": (
            PHYSX_PGS_SOLVER_CONFIGURATION
            if backend == PHYSX_BACKEND
            else CUDA_SOLVER_CONFIGURATION
        ),
        "correctness": correctness,
        "parity_trace": {
            "sampled_steps": list(SAMPLED_STEPS),
            "sampled_worlds": SAMPLED_WORLDS,
            "sampled_world_indices": list(SAMPLED_WORLD_INDICES),
            "samples": samples,
        },
    }


class CouplingTraceComparisonTests(unittest.TestCase):
    def setUp(self):
        self.physx = measured_report(PHYSX_BACKEND, rate=100.0)
        self.cuda = measured_report(CUDA_BACKEND, rate=250.0)

    def test_identical_bounded_traces_produce_one_validated_speedup(self):
        comparison = compare_coupling_reports([self.physx, self.cuda])
        self.assertEqual(len(comparison["speedups"]), 1)
        row = comparison["speedups"][0]
        self.assertEqual(row["world_step_speedup"], 2.5)
        self.assertEqual(row["output_parity"]["contact_state_agreement_ratio"], 1.0)
        self.assertEqual(row["output_parity"]["maximum_body_position_error_m"], 0.0)

    def test_quaternion_sign_is_invariant_but_real_angular_error_is_gated(self):
        for sample in self.cuda["parity_trace"]["samples"]:
            sample["body_quaternions_xyzw"][0][4] = [0.0, 0.0, 0.0, -1.0]
        comparison = compare_coupling_reports([self.physx, self.cuda])
        self.assertEqual(comparison["speedups"][0]["output_parity"]["maximum_body_orientation_error_rad"], 0.0)
        angle = 0.03
        self.cuda["parity_trace"]["samples"][1]["body_quaternions_xyzw"][0][4] = [
            0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0)
        ]
        with self.assertRaisesRegex(RuntimeError, "maximum_body_orientation_error_rad"):
            compare_coupling_reports([self.physx, self.cuda])

    def test_every_output_metric_is_computed_and_gated(self):
        mutations = (
            ("joint_positions_rad", (0, 0), 0.1, "maximum_joint_position_error_rad"),
            ("joint_velocities_rad_s", (0, 0), 1.0, "maximum_joint_velocity_error_rad_s"),
            ("joint_targets_rad", (0, 0), 0.01, "maximum_joint_target_error_rad"),
            ("drive_efforts_nm", (0, 0), 1.0, "maximum_drive_effort_error_nm"),
            ("body_positions_m", (0, 4, 1), 0.1, "maximum_body_position_error_m"),
            ("body_linear_velocities_mps", (0, 4, 0), 1.0, "maximum_body_velocity_error_mps"),
            ("body_angular_velocities_rad_s", (0, 4, 2), 1.0, "maximum_body_angular_velocity_error_rad_s"),
            ("pair_contact_impulse_magnitude_ns", (0, 0), 1.0, "maximum_pair_contact_impulse_magnitude_error_ns"),
        )
        for field, indices, delta, message in mutations:
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.cuda)
                target = candidate["parity_trace"]["samples"][1][field]
                for index in indices[:-1]:
                    target = target[index]
                target[indices[-1]] += delta
                with self.assertRaisesRegex(RuntimeError, message):
                    compare_coupling_reports([self.physx, candidate])

    def test_contact_agreement_and_payload_displacement_are_gated(self):
        candidate = copy.deepcopy(self.cuda)
        for sample in candidate["parity_trace"]["samples"]:
            for world in range(SAMPLED_WORLDS):
                sample["pair_contact"][world] = [False, True, True, True, False]
        with self.assertRaisesRegex(RuntimeError, "contact-state parity"):
            compare_coupling_reports([self.physx, candidate])
        candidate = copy.deepcopy(self.cuda)
        candidate["parity_trace"]["samples"][0]["body_positions_m"][0][4][0] -= 0.006
        candidate["parity_trace"]["samples"][-1]["body_positions_m"][0][4][0] += 0.006
        with self.assertRaisesRegex(RuntimeError, "maximum_payload_displacement_error_m"):
            compare_coupling_reports([self.physx, candidate])

    def test_sampling_shapes_types_and_finite_values_fail_closed(self):
        mutations = []
        wrong_steps = copy.deepcopy(self.cuda)
        wrong_steps["parity_trace"]["sampled_steps"][-1] = 718
        mutations.append(wrong_steps)
        wrong_worlds = copy.deepcopy(self.cuda)
        wrong_worlds["parity_trace"]["sampled_world_indices"][-1] = 64
        mutations.append(wrong_worlds)
        wrong_shape = copy.deepcopy(self.cuda)
        wrong_shape["parity_trace"]["samples"][0]["joint_positions_rad"].pop()
        mutations.append(wrong_shape)
        nonfinite = copy.deepcopy(self.cuda)
        nonfinite["parity_trace"]["samples"][0]["drive_efforts_nm"][0][0] = float("nan")
        mutations.append(nonfinite)
        wrong_bool = copy.deepcopy(self.cuda)
        wrong_bool["parity_trace"]["samples"][0]["pair_contact"][0][0] = 1
        mutations.append(wrong_bool)
        bad_quaternion = copy.deepcopy(self.cuda)
        bad_quaternion["parity_trace"]["samples"][0]["body_quaternions_xyzw"][0][0] = [0.0, 0.0, 0.0, 2.0]
        mutations.append(bad_quaternion)
        for index, candidate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                compare_coupling_reports([self.physx, candidate])

    def test_report_validity_duplicates_and_nonpositive_rates_fail_before_speedup(self):
        invalid = copy.deepcopy(self.cuda)
        invalid["correctness"]["synthetic"] = True
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            compare_coupling_reports([self.physx, invalid])
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            compare_coupling_reports([self.cuda, copy.deepcopy(self.cuda)])
        invalid = copy.deepcopy(self.cuda)
        invalid["world_steps_per_second"] = 0.0
        with self.assertRaises(RuntimeError):
            compare_coupling_reports([self.physx, invalid])


if __name__ == "__main__":
    unittest.main()
