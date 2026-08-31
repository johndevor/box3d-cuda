"""Bounded scalar reference laws; no simulator import or native implementation.

SI units: q rad, v rad/s, inertia kg m^2, torque N m, dt s.
Friction-loss is a bounded soft constraint, not sign(v) damping.
"""
from dataclasses import asdict, dataclass
import math

DT = .002
STEPS_PER_CASE = 3
WALL_CAP_SECONDS = 60
BODY_INERTIA = .02
KP = 13.37
KV = 0.
EFFORT_CAP = 3.23
ARMATURE = .027
DAMPING = .56
FRICTION_LOSS = .068
TOLERANCES = {'inertia':1e-12, 'torque':1e-11, 'acceleration':1e-9,
              'state':1e-11, 'clock':1e-12, 'qp_diagnostic_torque':1e-9}


@dataclass(frozen=True)
class Case:
    name: str
    q: float = 0.
    velocity: float = 0.
    target: float = 0.
    armature: float = ARMATURE
    damping: float = DAMPING
    friction_loss: float = 0.
    purpose: str = ''


CASES = (
    Case('bare_positive',target=.1,armature=0.,damping=0.,purpose='bare inertia analytic'),
    Case('armature_positive',target=.1,damping=0.,purpose='rotor inertia analytic'),
    Case('bare_negative',target=-.1,armature=0.,damping=0.,purpose='bare sign symmetry'),
    Case('armature_negative',target=-.1,damping=0.,purpose='rotor sign symmetry'),
    Case('damping_positive',velocity=2.,purpose='separate passive damping'),
    Case('damping_negative',velocity=-2.,purpose='separate passive damping sign'),
    Case('positive_cap_and_damping',velocity=2.,target=1.,purpose='motor clamp before passive damping'),
    Case('negative_cap_and_damping',velocity=-2.,target=-1.,purpose='motor clamp sign'),
    Case('friction_exact_rest',friction_loss=FRICTION_LOSS,purpose='zero input and velocity; exact rest'),
    Case('friction_subthreshold_positive',target=.02/KP,friction_loss=FRICTION_LOSS,
         purpose='soft below-threshold rest, not ideal rigid stiction'),
    Case('friction_subthreshold_negative',target=-.02/KP,friction_loss=FRICTION_LOSS,
         purpose='soft below-threshold rest sign'),
    Case('friction_slip_positive',velocity=2.,friction_loss=FRICTION_LOSS,purpose='positive sliding'),
    Case('friction_slip_negative',velocity=-2.,friction_loss=FRICTION_LOSS,purpose='negative sliding'),
    Case('friction_above_threshold_positive',target=(2*FRICTION_LOSS)/KP,friction_loss=FRICTION_LOSS,
         purpose='positive breakaway request from rest'),
    Case('friction_above_threshold_negative',target=-(2*FRICTION_LOSS)/KP,friction_loss=FRICTION_LOSS,
         purpose='negative breakaway request from rest'),
    Case('position_error_negative',q=.1,friction_loss=FRICTION_LOSS,purpose='nonzero initial q, negative position effort'),
)


def finite(*values):
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values):
        raise ValueError('finite numeric scalar required')


def clamp_effort(value, cap):
    finite(value,cap)
    if cap < 0:
        raise ValueError('negative effort cap')
    return max(-cap,min(cap,value))


def forces(q, velocity, target, damping, kp=KP, kv=KV, cap=EFFORT_CAP):
    finite(q,velocity,target,damping,kp,kv,cap)
    if min(damping,kp,kv,cap)<0:
        raise ValueError('negative physical coefficient')
    raw=kp*(target-q)-kv*velocity
    actuator=clamp_effort(raw,cap)
    passive=-damping*velocity
    finite(passive,actuator+passive)
    return {'actuator_unclamped':raw,'actuator':actuator,'passive':passive,'smooth_total':actuator+passive}


def scalar_friction_optimum(mass, smooth_acceleration, reference_acceleration, regularizer, bound, jacobian=1.):
    """Exact scalar dual-QP solution for supplied pinned-solver R/aref.

    This does NOT derive R/aref from MuJoCo solref/solimp. It is an independent
    optimization calculation conditional on those captured solver coefficients.
    A one-iteration runtime solve is not assumed to attain this optimum.
    """
    finite(mass,smooth_acceleration,reference_acceleration,regularizer,bound,jacobian)
    if mass<=0 or regularizer<0 or bound<0 or jacobian==0:
        raise ValueError('invalid scalar constraint domain')
    inverse=jacobian*jacobian/mass+regularizer
    finite(inverse)
    if inverse<=0:
        raise ValueError('nonpositive constraint Hessian')
    return clamp_effort((reference_acceleration-jacobian*smooth_acceleration)/inverse,bound)


def semi_implicit_step(q, velocity, acceleration, dt=DT):
    finite(q,velocity,acceleration,dt)
    if dt<=0:
        raise ValueError('nonpositive step')
    next_velocity=velocity+dt*acceleration
    next_q=q+dt*next_velocity
    finite(next_velocity,next_q)
    return next_q,next_velocity


def case_manifest():
    return {'schema':'box3d.open_duck.native_compat.hinge_cases/v1',
            'cases':[asdict(c) for c in CASES], 'case_count':len(CASES),
            'steps_per_case':STEPS_PER_CASE, 'maximum_total_steps':len(CASES)*STEPS_PER_CASE,
            'dt':DT,'wall_cap_seconds':WALL_CAP_SECONDS,'body_inertia_kg_m2':BODY_INERTIA,
            'kp':KP,'kv':KV,'actuator_effort_cap_nm':EFFORT_CAP,'tolerances':TOLERANCES,
            'fixture':'synthetic 1kg rotor, COM at world-fixed +Z hinge, diagonal inertia .01/.02/.02, gravity zero, no contacts/limits',
            'solver':'MuJoCo3.3.7 Newton, iterations1, ls_iterations5, Euler with eulerdamp disabled',
            'reference_scope':'analytic M/actuator/passive/integration; conditional scalar-QP diagnostic for soft friction',
            'no_full_robot_steps':True,'native_execution':False,'retry_allowed':False}


def model_xml(case):
    if case not in CASES:
        raise ValueError('case outside frozen matrix')
    return f'''<mujoco model="native_compat_{case.name}">
  <compiler angle="radian"/>
  <option timestep="{DT}" gravity="0 0 0" integrator="Euler" solver="Newton" iterations="1" ls_iterations="5" jacobian="dense">
    <flag eulerdamp="disable"/>
  </option>
  <worldbody><body name="rotor">
    <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.02 0.02"/>
    <joint name="hinge" type="hinge" pos="0 0 0" axis="0 0 1" limited="false" armature="{case.armature}" damping="{case.damping}" frictionloss="{case.friction_loss}" solreffriction="0.02 1" solimpfriction="0.9 0.95 0.001 0.5 2"/>
  </body></worldbody>
  <actuator><position name="motor" joint="hinge" kp="{KP}" kv="{KV}" ctrllimited="false" forcelimited="true" forcerange="-{EFFORT_CAP} {EFFORT_CAP}"/></actuator>
</mujoco>'''
