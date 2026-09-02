"""Remote/local preflight for the arm device policy path (ABI v8 reach kind).

    python -B arm/preflight_abi.py <variant> [library_path]

Replaces the inline `python -c` one-liner the GPU specs used (a `for` after a
`;` is a SyntaxError -- caught only on the sandbox, 2026-09-02). Exits 0 with
a one-line OK, non-zero on any assertion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from walk.env.arm_cuda_lane import CudaArmLane  # noqa: E402


def main(variant: str, library_path: str | None) -> int:
    kwargs = {"variant": variant, "fast_termination": True}
    if library_path:
        kwargs["library_path"] = library_path
    lane = CudaArmLane(4, **kwargs)
    try:
        assert lane._lib.dwc1_env_kind() == 1, "env kind must be REACH"
        assert lane._lib.dwc1_obs_width() == 27, "obs width must be 27"
        obs = lane.reset_policy(seed=1)
        assert obs.shape == (4, 27), obs.shape
        for _ in range(20):
            obs, _rew, _done, diag = lane.step_policy(np.zeros((4, 6), np.float32))
            assert (diag["status"] == 0).all(), diag["status"]
        rs = lane.reach_state()
        gp = lane.gate_proxy()
        assert (rs["valid"] == 1).all() and (rs["next_valid"] == 1).all()
        print("ABI v8 reach device path OK", obs.shape,
              "acq", rs["target_index"].tolist(),
              "viol", gp["alternation_violations"].tolist())
    finally:
        lane.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
