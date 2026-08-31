"""Separately named, source-bound Open Duck plain-14 adapter candidate.

Apache-2.0. Source interface/constants reviewed against Open_Duck_Playground
b9be205ac64488c23504ca42e5ec790337adeec3 (Apache-labelled task sources by
DeepMind Technologies Limited, Antoine Pirrone and Steve Nguyen).

Dependency-free float64 observation/controller fixtures ONLY. No policy,
simulator, importer, sensor synthesizer, torque integrator or native API.
"""

from dataclasses import dataclass
import math
from numbers import Real


SCHEMA = "box3d.open_duck_plain14.source_candidate/v1"
SOURCE_COMMIT = "b9be205ac64488c23504ca42e5ec790337adeec3"
FREE_JOINT_NAME = "floating_base"
JOINTS = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
)
HOME = (.002, .053, -.63, 1.368, -.784, 0., 0., 0., 0., -.003, -.065, .635, 1.379, -.796)
JOINT_LIMITS = (
    (-.5235987755982979, .5235987755982997),
    (-.4363323129985815, .43633231299858327),
    (-1.2217304763960306, .5235987755982988),
    (-1.5707963267948966, 1.5707963267948966),
    (-1.5707963267948957, 1.5707963267948974),
    (-.3490658503988437, 1.1344640137963364),
    (-.7853981633974483, .7853981633974483),
    (-2.792526803190927, 2.792526803190927),
    (-.523598775598218, .5235987755983796),
    (-.523598775598297, .5235987755983006),
    (-.4363323129985797, .43633231299858505),
    (-.5235987755982988, 1.2217304763960306),
    (-1.5707963267948966, 1.5707963267948966),
    (-1.5707963267948957, 1.5707963267948974),
)
CONTROL_DT = .02
SIMULATION_DT = .002
ACTION_SCALE = .25
TARGET_SPEED_LIMIT = 5.24
MAX_TARGET_INCREMENT = TARGET_SPEED_LIMIT * CONTROL_DT
LOADED_MOTOR_PROPERTIES = {
    "kp": 13.37, "kv": 0., "damping": .56,
    "frictionloss": .068, "armature": .027, "force_limit": 3.23,
}


def _vector(values, length, name, *, binary=False):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(name + " must be a vector") from exc
    if len(values) != length:
        raise ValueError(name + " has wrong width")
    converted = []
    for value in values:
        if not isinstance(value, Real) or (isinstance(value, bool) and not binary):
            raise ValueError(name + " must contain real scalars")
        try:
            item = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(name + " scalar out of range") from exc
        if not math.isfinite(item) or (binary and item not in (0., 1.)):
            raise ValueError(name + " contains nonfinite/nonbinary value")
        converted.append(item)
    return tuple(converted)


def _action(values):
    result = _vector(values, 14, "action")
    if any(abs(value) > 1 for value in result):
        raise ValueError("action must be in [-1,1]")
    return result


def _history(values):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("history must have three frames") from exc
    if len(values) != 3:
        raise ValueError("history must have three frames")
    return tuple(_action(frame) for frame in values)


@dataclass(frozen=True)
class ControllerState:
    raw_history: tuple
    motor_targets: tuple


@dataclass(frozen=True)
class StepResult:
    state: ControllerState
    observation_history: tuple
    motor_targets: tuple
    effective_controls: tuple


def reset():
    """Exact home targets and zero action/delay history; no model mutation."""
    return ControllerState(((0.,) * 14,) * 3, HOME)


def advance(state, action, delay_frames):
    """Prepare one 20 ms control frame; caller subsequently runs 10 physics steps.

    The observation after that frame uses PRE-shift raw action history, matching
    the pinned training source. Effective controls additionally apply the MJCF
    position actuator's inheritrange clamp. Stored/observed motor_targets retain
    the pre-actuator-clipping slew result, as in that training source.
    """
    if not isinstance(state, ControllerState):
        raise ValueError("state must be ControllerState")
    if type(delay_frames) is not int or delay_frames not in (0, 1, 2):
        raise ValueError("delay_frames must be integer 0, 1, or 2")
    history = _history(state.raw_history)
    previous = _vector(state.motor_targets, 14, "motor_targets")
    if any(abs(p - h) > ACTION_SCALE + 1e-12 for p, h in zip(previous, HOME)):
        raise ValueError("motor_targets outside reachable home +/- action-scale envelope")
    current = _action(action)
    selected = (current,) + history[:2]
    requested = tuple(h + ACTION_SCALE * a for h, a in zip(HOME, selected[delay_frames]))
    targets = tuple(max(p - MAX_TARGET_INCREMENT, min(p + MAX_TARGET_INCREMENT, r))
                    for p, r in zip(previous, requested))
    effective = tuple(max(lo, min(hi, target))
                      for target, (lo, hi) in zip(targets, JOINT_LIMITS))
    next_state = ControllerState((current,) + history[:2], targets)
    return StepResult(next_state, history, targets, effective)


def canonical_joints(names, values):
    """Only reorder an exact plain-14 permutation; never drop antenna/backlash DOFs."""
    try:
        names = tuple(names)
    except TypeError as exc:
        raise ValueError("joint names must be a vector") from exc
    if (len(names) != 14 or not all(isinstance(name, str) for name in names)
            or len(set(names)) != 14 or set(names) != set(JOINTS)):
        raise ValueError("joint names must be an exact plain-14 permutation")
    values = _vector(values, 14, "joint_values")
    by_name = dict(zip(names, values))
    return tuple(by_name[name] for name in JOINTS)


def observation(*, gyro, accelerometer, commands, q, qdot, contacts, phase,
                history, motor_targets):
    """Encode supplied physical site samples, not synthetic IMU/contact telemetry.

    All arrays are canonical-order. Input accelerometer is site-frame specific
    force from the simulator, with NO extra +1.3 X offset. Phase is caller-supplied
    because the released model/reference period has not been established. No
    noise/randomization, normalizer, tanh, clipping of measured q, or RNG is hidden.
    """
    gyro = _vector(gyro, 3, "gyro")
    accelerometer = _vector(accelerometer, 3, "accelerometer")
    commands = _vector(commands, 7, "commands")
    q = _vector(q, 14, "q")
    qdot = _vector(qdot, 14, "qdot")
    history = _history(history)
    motor_targets = _vector(motor_targets, 14, "motor_targets")
    contacts = _vector(contacts, 2, "contacts", binary=True)
    phase = _vector(phase, 2, "phase")
    return (gyro + accelerometer + commands
            + tuple(value - home for value, home in zip(q, HOME))
            + tuple(.05 * value for value in qdot)
            + tuple(value for frame in history for value in frame)
            + motor_targets + contacts + phase)
