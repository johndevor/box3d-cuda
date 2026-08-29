from __future__ import annotations

from box3d_cuda.legacy_replay import gripper_replay, obb_plane_replay, sphere_replay


def test_stage_zero_replay_runs_the_sphere_oracle() -> None:
    replay = sphere_replay(steps=500, sample_every=10)
    assert replay["contract_id"] == "box3d.fixed-sphere-world/v0"
    assert replay["frames"][-1]["step"] == 500
    assert all(replay["contact_ever"])


def test_stage_one_replay_proves_bilateral_friction_grip_and_release() -> None:
    replay = gripper_replay(sample_every=4)
    assert replay["contract_id"] == "parallel-jaw-box-lift/v0"
    assert all(replay["contact_ever"])
    assert replay["maximum_cube_height_m"] > 0.09
    assert replay["frames"][-1]["cube"]["position_m"][1] < 0.04
    assert {frame["phase"] for frame in replay["frames"]} == {"settle", "close", "lift", "release", "fall"}


def test_stage_two_replay_runs_oriented_box_plane_oracle() -> None:
    replay = obb_plane_replay(steps=500, sample_every=10)
    assert replay["contract_id"] == "box3d.oriented-box-plane/v0"
    assert replay["frames"][-1]["step"] == 500
    assert all(replay["contact_ever"])
    assert any(
        abs(value) > 0.01
        for frame in replay["frames"]
        for body in frame["bodies"]
        for value in body["angular_velocity_rad_s"]
    )
