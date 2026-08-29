from pathlib import Path
import unittest

from box3d_cuda.manifold_reference import (
    SAT_AXIS_TIE_EPSILON_M,
    make_manifold_stack_state,
    step_manifold_reference,
)
from box3d_cuda.benchmark_manifold import _timed_final_gate
from box3d_cuda.sat_reference import SATConfig


ROOT = Path(__file__).resolve().parents[1]
CUDA = (ROOT / "csrc" / "manifold.cu").read_text()
BINDINGS = (ROOT / "csrc" / "bindings.cpp").read_text()
EXTENSION = (ROOT / "extension.py").read_text()
BENCHMARK = (ROOT / "benchmark_manifold.py").read_text()
REFERENCE = (ROOT / "manifold_reference.py").read_text()


def function_source(signature: str, next_signature: str) -> str:
    start = CUDA.index(signature)
    end = CUDA.index(next_signature, start)
    return CUDA[start:end]


class ManifoldCudaStaticContractTests(unittest.TestCase):
    def test_both_tangent_vectors_are_normalized(self):
        source = function_source("void tangents", "int vertex_bit")
        self.assertEqual(source.count("rsqrtf("), 2)
        cross_t2 = source.index("cross3(n,t1,t2)")
        self.assertIn("dot3(t2,t2)", source[cross_t2:])

    def test_manifold_reduction_deduplicates_topology_ids(self):
        source = function_source("bool manifold", "void point_v")
        self.assertGreaterEqual(source.count("has_feature("), 2)
        self.assertNotIn("used[MAXCLIP]", source)

    def test_parity_critical_path_remains_float32(self):
        self.assertIn("kernel(float*state", CUDA)
        self.assertIn("data_ptr<float>()", CUDA)
        self.assertGreaterEqual(BINDINGS.count("torch::kFloat32"), 5)
        self.assertIn("torch.float32", EXTENSION)
        self.assertIn("torch.float32", BENCHMARK)
        self.assertIn('name="factory_box3d_cuda_v11"', EXTENSION)

    def test_redundant_axis_selection_is_tie_stable_at_control_step_two(self):
        self.assertEqual(SAT_AXIS_TIE_EPSILON_M, 1.0e-6)
        self.assertIn("SAT_AXIS_TIE_EPSILON_M=1.0e-6f", CUDA)
        state, mass, half, inertia, pairs, ids, impulses = make_manifold_stack_state(1, seed=41)
        _, _, _, feature_ids, _, counts = step_manifold_reference(
            state, mass, half, inertia, pairs, ids, impulses, SATConfig(), steps=2
        )
        self.assertEqual(counts, [[4, 4, 4, 4, 4]])
        self.assertEqual(
            feature_ids[0][1],
            [
                1153260154188202496,
                1153260154188202754,
                1153260154188210692,
                1153260154188215046,
            ],
        )

    def test_tie_selection_uses_raw_minimum_then_priority(self):
        sat = function_source("SAT sat", "void tangents")
        self.assertIn("depths[15]", sat)
        self.assertIn("md=fminf(md,dep)", sat)
        self.assertIn("depths[index]<=md+SAT_AXIS_TIE_EPSILON_M", sat)
        self.assertNotIn("axis_test", CUDA)
        self.assertIn("minimum_depth = min(record[0] for record in records)", REFERENCE)
        self.assertIn("record[0] <= minimum_depth + SAT_AXIS_TIE_EPSILON_M", REFERENCE)
        data = (
            None,
            None,
            [[0.001] * 5, [0.006, 0.001, 0.001, 0.001, 0.001], [0.001] * 5],
            None,
            None,
            [[4, 4, 4, 4, 4], [4, 4, 1, 4, 4], [4, 3, 4, 0, 4]],
        )
        gate = _timed_final_gate(data)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["minimum_final_contact_counts_per_pair"], [4, 3, 1, 0, 4])
        self.assertEqual(gate["final_multi_point_failure_world_count"], 2)
        self.assertEqual(gate["first_final_multi_point_failure_world_indices"], [1, 2])
        self.assertFalse(gate["final_penetration_within_threshold"])

    def test_cpu_control_step_chunking_is_invariant(self):
        state, mass, half, inertia, pairs, ids, impulses = make_manifold_stack_state(1, seed=41)
        batched = step_manifold_reference(
            state, mass, half, inertia, pairs, ids, impulses, SATConfig(), steps=40
        )
        current_state, current_ids, current_impulses = state, ids, impulses
        touched = [[False for _ in pairs]]
        for _ in range(40):
            current_state, contacts, penetration, current_ids, current_impulses, counts = (
                step_manifold_reference(
                    current_state,
                    mass,
                    half,
                    inertia,
                    pairs,
                    current_ids,
                    current_impulses,
                    SATConfig(),
                    steps=1,
                )
            )
            touched = [[left or right for left, right in zip(touched[0], contacts[0])]]
        repeated = (current_state, touched, penetration, current_ids, current_impulses, counts)
        self.assertEqual(batched, repeated)

    def test_solver_matches_patch_order_without_velocity_bias(self):
        kernel = function_source("void kernel", "std::vector<torch::Tensor>")
        self.assertLess(kernel.index("solve_normal("), kernel.index("solve_friction("))
        self.assertNotIn("bias=", CUDA)


if __name__ == "__main__":
    unittest.main()
