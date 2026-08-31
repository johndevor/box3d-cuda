"""Narrow compiler-arithmetic metadata check for the pinned plain-14 robot.

MuJoCo 3.3.7 src/user/user_objects.cc:6844-6847, file SHA256
a1fedaace694c5b8ba364213cead4d7da4693698e4d8ce00a25f2df433fe3695.
No simulator/runtime import, model change or physical-health tolerance here.
"""
import math
from numbers import Real


def _pair(value):
    try:
        if (isinstance(value, (str, bytes, bytearray, memoryview)) or len(value) != 2
                or any(isinstance(x, bool) or not isinstance(x, Real) for x in value)):
            raise ValueError("expected two numeric endpoints")
        values = tuple(float(x) for x in value)
    except (TypeError, OverflowError) as exc:
        raise ValueError("expected two numeric endpoints") from exc
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite range")
    return values


def _outward(low, high):
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("range arithmetic overflow")
    bounds = (math.nextafter(low, -math.inf), math.nextafter(high, math.inf))
    if not all(math.isfinite(x) for x in bounds):
        raise ValueError("unbounded range arithmetic")
    return bounds


def inherited_control_range(actual, target, inheritrange=1.):
    """Return the fixed-one inheritance envelope, not a widened joint limit.

    Each source add/subtract and half-multiply is enclosed with one outward
    binary64 neighbour. Intervals propagate the operation error at cancellation
    scale rather than taking ULPs of a possibly near-zero final endpoint.
    Other inheritance factors are outside this pinned-model admission.
    """
    actual, target = _pair(actual), _pair(target)
    if isinstance(inheritrange, bool) or not isinstance(inheritrange, Real) or inheritrange != 1.:
        raise ValueError("only the pinned inheritrange=1 is supported")
    low, high = target
    if low >= high:
        raise ValueError("target range must be strictly ordered")
    summed = _outward(high+low, high+low)
    difference = _outward(high-low, high-low)
    mean = _outward(.5*summed[0], .5*summed[1])
    radius = _outward(.5*difference[0], .5*difference[1])
    lower = _outward(mean[0]-radius[1], mean[1]-radius[0])
    upper = _outward(mean[0]+radius[0], mean[1]+radius[1])
    predicted_mean = .5*(high+low)
    predicted_radius = .5*(high-low)*inheritrange
    predicted = (predicted_mean-predicted_radius, predicted_mean+predicted_radius)
    bounds_low, bounds_high = [lower[0], upper[0]], [lower[1], upper[1]]
    accepted = actual[0] < actual[1] and all(bounds_low[i] <= actual[i] <= bounds_high[i] for i in range(2))
    return {"schema": "box3d.mujoco337.inherited_control_range/r3",
            "accepted": accepted, "target": target, "actual": actual,
            "failure": None if accepted else "compiled range outside source arithmetic envelope",
            "inheritrange": inheritrange, "predicted": predicted,
            "lower_bounds": bounds_low, "upper_bounds": bounds_high,
            "maximum_absolute_envelope": max(max(abs(bounds_low[i]-target[i]), abs(bounds_high[i]-target[i])) for i in range(2)),
            "physical_joint_limits_modified": False}
