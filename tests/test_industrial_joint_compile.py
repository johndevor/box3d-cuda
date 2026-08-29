from __future__ import annotations

from box3d_cuda.industrial_joint_import import (
    IndustrialJointModel,
    JointDynamics,
    LinkDynamics,
    compile_industrial_joint_world,
)
from box3d_cuda.benchmark_industrial_joints import (
    CONTROLLER_DAMPING,
    DEFAULT_SEED,
    _limit_targets,
    _with_runtime_controller,
    target_positions,
)
from box3d_cuda.joint_reference import JointConfig, step_joint_reference


def _model() -> IndustrialJointModel:
    links = tuple(
        LinkDynamics(
            name=f"link_{index}",
            mass_kg=400.0 / (index + 1),
            center_of_mass_m_z_up=(0.03 * index, -0.01 * index, 0.08),
            inertia_diagonal_kg_m2=(10.0 + index, 12.0 + index, 14.0 + index),
        )
        for index in range(7)
    )
    joints = tuple(
        JointDynamics(
            name=f"joint_{index + 1}", kind="revolute", parent_index=index, child_index=index + 1,
            origin_m_z_up=(0.4 + 0.1 * index, 0.02 * index, 0.15),
            origin_rpy_rad=((3.1415927, 0.0, 3.1415927) if index < 2 else (0.0, 0.0, 0.0)),
            axis_z_up=((0.0, 0.0, 1.0) if index == 0 else (0.0, 1.0, 0.0)),
            lower=-1.5, upper=1.5, effort=1000.0, velocity=2.0, damping=20.0, friction=2.0,
        )
        for index in range(6)
    )
    return IndustrialJointModel(
        asset_id="test.arm", calibration_class="test", coordinate_system="URDF_Z_UP_RIGHT_HANDED",
        links=links, joints=joints,
        collision_filter_pairs=tuple((index, index + 1) for index in range(6)),
        source_urdf_sha256="0" * 64,
    )


def test_compiled_zero_pose_closes_every_anchor_and_reference_rotation() -> None:
    world = compile_industrial_joint_world(_model())
    result = step_joint_reference(
        [[list(body) for body in world.state_y_up]],
        [list(world.inverse_mass)],
        [[list(value) for value in world.inverse_inertia_local]],
        world.topology,
        [[0.0] * 6],
        [list(world.maximum_effort_nm)],
        JointConfig(gravity_y=0.0),
        motor_target_position=[[0.0] * 6],
        stiffness=(100.0,) * 6,
    )
    assert max(abs(value) for value in result.coordinate[0]) < 1.0e-6
    assert max(result.linear_error_m[0]) < 1.0e-6
    assert max(result.angular_error_rad[0]) < 1.0e-6
    assert max(result.limit_error[0]) == 0.0


def test_compiler_fixes_only_base_and_preserves_engineering_limits() -> None:
    world = compile_industrial_joint_world(_model())
    assert world.inverse_mass[0] == 0.0
    assert all(value > 0.0 for value in world.inverse_mass[1:])
    assert world.maximum_effort_nm == (1000.0,) * 6
    assert world.maximum_velocity_rad_s == (2.0,) * 6
    assert len(world.state_y_up) == 7


def test_industrial_control_schedule_is_held_then_seeded_per_world() -> None:
    assert target_positions(0, 0) == [0.0] * 6
    assert target_positions(59, 63) == [0.0] * 6
    assert target_positions(120, 0, DEFAULT_SEED) != target_positions(
        120, 63, DEFAULT_SEED
    )
    assert target_positions(120, 17, DEFAULT_SEED) == target_positions(
        120, 17, DEFAULT_SEED
    )


def test_representable_limit_probe_approaches_smoothly_then_recovers() -> None:
    world = compile_industrial_joint_world(_model())
    assert _limit_targets(0, world)[0][1] == 0.0
    assert _limit_targets(120, world)[0][1] < world.topology.lower_limit[1]
    assert _limit_targets(179, world)[1][1] > world.topology.upper_limit[1]
    assert _limit_targets(180, world)[0][1] > world.topology.lower_limit[1]
    assert _limit_targets(180, world)[1][1] < world.topology.upper_limit[1]


def test_runtime_controller_override_is_explicit_and_does_not_mutate_import() -> None:
    imported = compile_industrial_joint_world(_model())
    runtime = _with_runtime_controller(imported)
    assert imported.topology.damping == (20.0,) * 6
    assert runtime.topology.damping == CONTROLLER_DAMPING
