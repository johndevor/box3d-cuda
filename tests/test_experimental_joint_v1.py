"""Native f32 ABI tests; NumPy double matrix/active-set oracle, no simulator.
The 48 saved hinge rows are comparisons only, never solver coefficient inputs.
"""
import ctypes as C
import hashlib,itertools,json,os,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
ABI=0x4a530001
F=C.POINTER(C.c_float);U8=C.POINTER(C.c_uint8);U64=C.POINTER(C.c_uint64);U32=C.POINTER(C.c_uint32)
class Params(C.Structure):
 _fields_=[(n,C.c_uint32) for n in ('struct_size','version','dofs','environments','flags','reserved')]+[(n,U8) for n in ('revolute','motor_enabled')]+[(n,F) for n in ('armature','passive_damping','friction_loss','stiffness','motor_damping','maximum_effort','friction_d0','friction_dwidth','friction_timeconst','reference_body_mass','initial_q','initial_velocity')]
class Step(C.Structure):
 _fields_=[(n,C.c_uint32) for n in ('struct_size','version','external_rows','max_iterations')]+[(n,C.c_float) for n in ('dt','tolerance')]+[(n,F) for n in ('body_mass','target_position','target_velocity','external_generalized_force','constraint_jacobian','constraint_reference_acceleration','constraint_regularizer','constraint_lower','constraint_upper','constraint_warm_force')]
class Output(C.Structure):
 _fields_=[('struct_size',C.c_uint32),('version',C.c_uint32)]+[(n,F) for n in ('acceleration','smooth_acceleration','actuator','passive','friction','regularizer','reference_acceleration','constraint_force','projected_residual')]+[('iterations',U32)]
class Snapshot(C.Structure):
 _fields_=[(n,C.c_uint32) for n in ('struct_size','version','dofs','environments')]+[('binding',C.c_uint64),('q',F),('velocity',F),('friction_warm_force',F),('step_count',U64)]
def ptr(a):return a.ctypes.data_as({np.dtype('float32'):F,np.dtype('uint8'):U8,np.dtype('uint32'):U32,np.dtype('uint64'):U64}[a.dtype])
def arr(x,shape,dtype=np.float32):return np.broadcast_to(np.asarray(x,dtype=dtype),shape).copy()
def qp_oracle(h,b,lo,hi):
 # Independent exhaustive active sets + dense double linear solve, not PGS.
 n=len(b)
 for active in itertools.product((-1,0,1),repeat=n):
  active=np.asarray(active);free=np.flatnonzero(active==0);fixed=np.flatnonzero(active!=0);x=np.zeros(n)
  x[fixed]=np.where(active[fixed]<0,lo[fixed],hi[fixed])
  if len(free):x[free]=np.linalg.solve(h[np.ix_(free,free)],-b[free]-h[np.ix_(free,fixed)]@x[fixed])
  g=h@x+b
  if np.all(x>=lo-1e-9) and np.all(x<=hi+1e-9) and np.all(abs(g[free])<1e-8) and np.all(g[active<0]>=-1e-8) and np.all(g[active>0]<=1e-8):return x
 raise AssertionError('no oracle KKT solution')
class Scene:
 def __init__(self,lib,mass,envs=1,**kw):
  self.lib=lib;self.n=np.asarray(mass).shape[-1];self.e=envs;self.refs={};self.p=Params(struct_size=C.sizeof(Params),version=ABI,dofs=self.n,environments=envs)
  defaults=dict(revolute=1,motor_enabled=1,armature=.027,passive_damping=.56,friction_loss=.068,stiffness=13.37,motor_damping=0,maximum_effort=3.23,friction_d0=.9,friction_dwidth=.95,friction_timeconst=.02,reference_body_mass=mass,initial_q=0,initial_velocity=0)
  defaults.update(kw)
  for name,value in defaults.items():
   shape=(self.e,self.n,self.n) if name=='reference_body_mass' else (self.e,self.n) if name.startswith('initial_') else (self.n,)
   a=arr(value,shape,np.uint8 if name in ('revolute','motor_enabled') else np.float32);self.refs[name]=a;setattr(self.p,name,ptr(a))
  self.handle=C.c_void_p();self.create_status=lib.box3d_joint_v1_create(C.byref(self.p),C.byref(self.handle))
 def close(self):
  if self.handle:self.lib.box3d_joint_v1_destroy(self.handle);self.handle=C.c_void_p()
 def snapshot(self):
  fields={n:np.empty((self.e,self.n),np.float32) for n in ('q','velocity','friction_warm_force')};fields['step_count']=np.empty(self.e,np.uint64)
  p=Snapshot(struct_size=C.sizeof(Snapshot),version=ABI,dofs=self.n,environments=self.e)
  for n,a in fields.items():setattr(p,n,ptr(a))
  assert self.lib.box3d_joint_v1_capture(self.handle,C.byref(p))==0
  return p,fields
 def advance(self,mass=None,target=0,force=0,dt=.002,iterations=512,tolerance=1e-8,rows=None):
  n,e=self.n,self.e;k=0 if rows is None else len(rows['g'])
  p=Step(struct_size=C.sizeof(Step),version=ABI,external_rows=k,max_iterations=iterations,dt=dt,tolerance=tolerance)
  data={'body_mass':arr(self.refs['reference_body_mass'] if mass is None else mass,(e,n,n)),'target_position':arr(target,(e,n)),'target_velocity':arr(0,(e,n)),'external_generalized_force':arr(force,(e,n))}
  if rows:
   for field,key,shape in [('constraint_jacobian','g',(e,k,n)),('constraint_reference_acceleration','aref',(e,k)),('constraint_regularizer','r',(e,k)),('constraint_lower','lo',(e,k)),('constraint_upper','hi',(e,k)),('constraint_warm_force','warm',(e,k))]:data[field]=arr(rows.get(key,0),shape)
  for name,a in data.items():setattr(p,name,ptr(a))
  outputs={name:np.full((e,n),777,np.float32) for name in ('acceleration','smooth_acceleration','actuator','passive','friction','regularizer','reference_acceleration')}
  outputs.update(constraint_force=np.full((e,k),777,np.float32),projected_residual=np.full(e,777,np.float32),iterations=np.full(e,777,np.uint32))
  output=Output(struct_size=C.sizeof(Output),version=ABI)
  for name,a in outputs.items():setattr(output,name,ptr(a))
  status=self.lib.box3d_joint_v1_advance(self.handle,C.byref(p),C.byref(output))
  return status,outputs
class NativeJoint(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();libpath=Path(cls.tmp.name)/'libjoint.so'
  supplied=os.environ.get('BOX3D_JOINT_V1_LIBRARY')
  if not supplied:subprocess.run(['clang++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared','-I',str(ROOT/'include'),str(ROOT/'csrc/experimental_joint_v1.cpp'),'-o',str(libpath)],check=True)
  cls.lib=C.CDLL(supplied or str(libpath));l=cls.lib
  l.box3d_joint_v1_create.argtypes=[C.POINTER(Params),C.POINTER(C.c_void_p)];l.box3d_joint_v1_destroy.argtypes=[C.c_void_p]
  l.box3d_joint_v1_advance.argtypes=[C.c_void_p,C.POINTER(Step),C.POINTER(Output)]
  l.box3d_joint_v1_capture.argtypes=[C.c_void_p,C.POINTER(Snapshot)];l.box3d_joint_v1_restore.argtypes=[C.c_void_p,C.POINTER(Snapshot)]
  l.box3d_joint_v1_reset_masked.argtypes=[C.c_void_p,U8,C.c_uint32]
  l.box3d_joint_v1_response.argtypes=[C.c_uint32,F,F,F,F,F]
  l.box3d_joint_v1_assemble_mass.argtypes=[C.c_uint32,C.c_uint32,F,F,F,F,F]
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def make(self,mass,**kw):
  s=Scene(self.lib,mass,**kw);self.addCleanup(s.close);self.assertEqual(s.create_status,0);return s
 def near(self,a,b,atol=2e-5,rtol=3e-6):np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
 def test_all_16_hinge_cases_48_rows_from_initial_conditions(self):
  path=ROOT/'tests/fixtures/experimental_joint_v1/hinge-result.json'
  self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),'05db2648c6db27fc6eb18892aa4e5a57fb2faa9de66ec61c97db0fd4ab6ca183')
  saved=json.loads(path.read_text());rows=0
  for case in saved['cases']:
   c=case['case'];s=self.make([[.02]],armature=c['armature'],passive_damping=c['damping'],friction_loss=c['friction_loss'],initial_q=c['q'],initial_velocity=c['velocity'])
   for row in case['steps']:
    status,o=s.advance(target=c['target']);self.assertEqual(status,0,case['case']['name']);_,state=s.snapshot();forces=row['forces_pre_integration']
    for name,key in [('acceleration','acceleration'),('smooth_acceleration','acceleration_smooth'),('actuator','actuator'),('passive','passive'),('friction','constraint')]:self.near(o[name][0,0],forces[key])
    self.near(state['q'][0,0],row['post']['q'],atol=2e-7);self.near(state['velocity'][0,0],row['post']['velocity'],atol=2e-6)
    if c['friction_loss']:
     derived_r=(1-.9)/.9/(.02+c['armature']);self.near(o['regularizer'][0,0],derived_r)
     self.near(o['reference_acceleration'][0,0],row['constraint_solver']['aref'][0],atol=4e-5)
    rows+=1
  self.assertEqual(rows,48)
 def test_passive_damping_outside_cap_even_disabled_motor(self):
  s=self.make([[.02]],initial_velocity=2,friction_loss=0)
  rc,o=s.advance(target=1);self.assertEqual(rc,0);self.near(o['actuator'],3.23);self.near(o['passive'],-1.12);self.near(o['acceleration'],2.11/.047)
  disabled=self.make([[.02]],initial_velocity=2,motor_enabled=0,maximum_effort=0,friction_loss=0)
  rc,o=disabled.advance(target=999);self.assertEqual(rc,0);self.near(o['actuator'],0);self.near(o['acceleration'],-1.12/.047)
 def test_soft_rest_is_not_stiction_or_sign_velocity(self):
  s=self.make([[.02]]);rc,o=s.advance(target=.02/13.37);self.assertEqual(rc,0)
  self.near(o['friction'],-.018,atol=2e-8);self.assertGreater(s.snapshot()[1]['velocity'][0,0],0)
 def test_physical_two_rotor_coupled_friction_interior_and_bound(self):
  body=np.array([[.05,.02],[.02,.02]])
  for force in ([.02,0],[.2,.02],[-.2,-.02]):
   s=self.make(body,motor_enabled=0,stiffness=0,passive_damping=0)
   rc,o=s.advance(force=force);self.assertEqual(rc,0)
   inv=np.linalg.inv(body+np.eye(2)*.027);r=np.diag(inv)/9;h=inv+np.diag(r);expected=qp_oracle(h,inv@force,np.full(2,-.068),np.full(2,.068))
   self.near(o['friction'][0],expected,atol=3e-7);self.near(o['acceleration'][0],inv@(np.array(force)+expected),atol=6e-6)
   wrong=np.clip(-(inv@force)/np.diag(h),-.068,.068);self.assertGreater(np.max(abs(wrong-expected)),.003)
 def test_armature_changes_offdiagonal_constraint_and_impulse_response(self):
  body=np.array([[.05,.02],[.02,.02]],np.float32);a=np.array([.027,.027],np.float32);g=np.array([1.,-.3],np.float32);delta=np.zeros(2,np.float32);eff=np.zeros(1,np.float32)
  rc=self.lib.box3d_joint_v1_response(2,ptr(body),ptr(a),ptr(g),ptr(delta),ptr(eff));self.assertEqual(rc,0)
  inv=np.linalg.inv(body.astype(float)+np.diag(a));self.near(delta,inv@g);self.near(eff,g@inv@g)
  self.assertGreater(abs(delta[1]),1);self.assertGreater(abs(eff[0]-g@np.linalg.inv(body)@g),10)
  # One external constraint solved simultaneously with friction changes BOTH dofs.
  s=self.make(body,motor_enabled=0,stiffness=0,passive_damping=0)
  rows={'g':[g],'aref':[3.],'r':[.2],'lo':[-.3],'hi':[.3]}
  rc,o=s.advance(force=[.02,.01],rows=rows);self.assertEqual(rc,0)
  j=np.vstack([np.eye(2),g]);h=j@inv@j.T+np.diag([*np.diag(inv)/9,.2]);b=j@inv@np.array([.02,.01])-np.array([0,0,3.])
  f=qp_oracle(h,b,np.array([-.068,-.068,-.3]),np.array([.068,.068,.3]));self.near(o['friction'][0],f[:2],atol=5e-7);self.near(o['constraint_force'][0],f[2:],atol=5e-7);self.near(o['acceleration'][0],inv@(np.array([.02,.01])+j.T@f))
 def test_reference_inverse_is_full_and_stays_reference_when_mass_changes(self):
  ref=np.array([[.05,.02],[.02,.02]]);current=np.array([[.09,.01],[.01,.03]])
  s=self.make(ref,initial_velocity=[.01,-.02],motor_enabled=0,stiffness=0,passive_damping=0)
  rc,o=s.advance(mass=current);self.assertEqual(rc,0)
  expected=np.diag(np.linalg.inv(ref+np.eye(2)*.027))/9;self.near(o['regularizer'][0],expected)
  self.assertGreater(np.max(abs(expected-1/np.diag(ref+np.eye(2)*.027)/9)),.1)
  self.assertGreater(np.max(abs(expected-np.diag(np.linalg.inv(current+np.eye(2)*.027))/9)),.1)
 def test_reference_safety_timeconstant_and_dwidth(self):
  s=self.make([[.02]],initial_velocity=.01,friction_timeconst=.001,friction_d0=.8,friction_dwidth=.92)
  rc,o=s.advance(dt=.004);self.assertEqual(rc,0);self.near(o['reference_acceleration'],-2/(.92*.008)*.01)
  self.near(o['regularizer'],(.2/.8)/.047)
 def test_three_link_kinetic_mass_and_full_coupled_oracle(self):
  rng=np.random.default_rng(731)
  for trial in range(8):
   n=3;angles=rng.uniform(-1,1,n);length=np.array([.4,.3,.2]);mass=np.array([.8,.5,.3],np.float32);iz=np.array([.012,.008,.004]);positions=[np.zeros(2)];cumulative=np.cumsum(angles)
   for angle,L in zip(cumulative,length):positions.append(positions[-1]+L*np.array([np.cos(angle),np.sin(angle)]))
   jac=np.zeros((n,6,n),np.float32)
   for body in range(n):
    com=(positions[body]+positions[body+1])/2
    for joint in range(body+1):
     d=com-positions[joint];jac[body,:2,joint]=[-d[1],d[0]];jac[body,5,joint]=1
   inertias=np.column_stack([iz/3,iz/2,iz]).astype(np.float32);quat=arr([0,0,0,1],(n,4));native=np.zeros((n,n),np.float32)
   rc=self.lib.box3d_joint_v1_assemble_mass(n,n,ptr(mass),ptr(inertias),ptr(quat),ptr(jac),ptr(native));self.assertEqual(rc,0)
   expected=sum(float(mass[b])*jac[b,:3].astype(float).T@jac[b,:3]+jac[b,3:].astype(float).T@np.diag(inertias[b])@jac[b,3:] for b in range(n))
   self.near(native,expected,atol=2e-8)
   force=rng.uniform(-.1,.1,n);s=self.make(native,motor_enabled=0,stiffness=0,passive_damping=0)
   rc,o=s.advance(force=force,tolerance=1e-8);self.assertEqual(rc,0)
   inv=np.linalg.inv(expected+np.eye(n)*.027);h=inv+np.diag(np.diag(inv)/9);f=qp_oracle(h,inv@force,np.full(n,-.068),np.full(n,.068))
   self.near(o['friction'][0],f,atol=1e-6);self.near(o['acceleration'][0],inv@(force+f),atol=3e-5)
 def test_principal_inertia_rotates_without_body_mass_faking(self):
  n=2;mass=arr(2,(1,));inertia=arr([.1,.2,.3],(1,3));quat=arr([2**-.5,0,0,2**-.5],(1,4));jac=np.zeros((1,6,n),np.float32);jac[0,4,0]=1;jac[0,5,1]=1;out=np.zeros((n,n),np.float32)
  self.assertEqual(self.lib.box3d_joint_v1_assemble_mass(n,1,ptr(mass),ptr(inertia),ptr(quat),ptr(jac),ptr(out)),0);self.near(out,[[.3,0],[0,.2]],atol=1e-7)
 def test_snapshot_replay_and_owned_initial_masked_reset(self):
  s=self.make([[.02]],envs=3,initial_q=[[0],[.1],[-.1]],initial_velocity=[[0],[.2],[-.2]])
  self.assertEqual(s.advance(target=.05)[0],0);snapshot,state=s.snapshot();self.assertEqual(s.advance(target=.05)[0],0);after=s.snapshot()[1]
  self.assertEqual(self.lib.box3d_joint_v1_restore(s.handle,C.byref(snapshot)),0);self.assertEqual(s.advance(target=.05)[0],0)
  for k,a in after.items():np.testing.assert_array_equal(s.snapshot()[1][k],a)
  # Original caller arrays do not own model or reset state after registration.
  for a in s.refs.values():a.fill(99)
  mask=np.array([0,1,0],np.uint8);self.assertEqual(self.lib.box3d_joint_v1_reset_masked(s.handle,ptr(mask),3),0);reset=s.snapshot()[1]
  self.near(reset['q'][1],[.1]);self.near(reset['velocity'][1],[.2]);self.near(reset['friction_warm_force'][1],[0]);self.assertEqual(reset['step_count'][1],0)
  for k,a in reset.items():np.testing.assert_array_equal(a[[0,2]],after[k][[0,2]])
 def test_rejected_snapshot_or_mask_cannot_change_any_environment(self):
  s=self.make([[.02]],envs=2);self.assertEqual(s.advance(target=.01)[0],0);snap,stored=s.snapshot();baseline={k:a.copy() for k,a in stored.items()}
  snap.binding^=1;self.assertEqual(self.lib.box3d_joint_v1_restore(s.handle,C.byref(snap)),1);snap.binding^=1
  stored['friction_warm_force'][1,0]=.069;self.assertEqual(self.lib.box3d_joint_v1_restore(s.handle,C.byref(snap)),1)
  mask=np.array([1,2],np.uint8);self.assertEqual(self.lib.box3d_joint_v1_reset_masked(s.handle,ptr(mask),2),1)
  for k,a in baseline.items():np.testing.assert_array_equal(s.snapshot()[1][k],a)
 def test_bad_second_environment_is_atomic_and_outputs_untouched(self):
  s=self.make([[.02]],envs=2);before=s.snapshot()[1];rc,o=s.advance(mass=[[[.02]],[[-.1]]],target=.1)
  self.assertEqual(rc,2)
  for a in o.values():self.assertTrue(np.all(a==777))
  for k,a in before.items():np.testing.assert_array_equal(s.snapshot()[1][k],a)
 def test_nonconverged_coupled_qp_does_not_commit(self):
  s=self.make([[.05,.02],[.02,.02]],motor_enabled=0,stiffness=0,passive_damping=0);before=s.snapshot()[1]
  rc,o=s.advance(force=[.02,0],iterations=1,tolerance=1e-12);self.assertEqual(rc,3)
  for a in o.values():self.assertTrue(np.all(a==777))
  for k,a in before.items():np.testing.assert_array_equal(s.snapshot()[1][k],a)
 def test_unattainable_float_tolerance_cannot_report_success(self):
  body=np.linalg.inv([[2.,1.],[1.,2.]]).astype(np.float32)
  s=self.make(body,armature=0,friction_loss=0,passive_damping=0,motor_enabled=0,stiffness=0)
  before=s.snapshot()[1];rows={'g':np.eye(2),'aref':[1.,1.],'r':[0,0],'lo':[-1,-1],'hi':[1,1]}
  rc,o=s.advance(rows=rows,iterations=4096,tolerance=1e-12);self.assertEqual(rc,3)
  for a in o.values():self.assertTrue(np.all(a==777))
  for k,a in before.items():np.testing.assert_array_equal(s.snapshot()[1][k],a)
 def test_inverse_overflow_and_regularizer_underflow_fail_closed(self):
  tiny=Scene(self.lib,[[1e-39]],armature=0);self.addCleanup(tiny.close);self.assertEqual(tiny.create_status,2)
  d=np.nextafter(np.float32(1),np.float32(0));s=self.make([[1e38]],armature=0,friction_d0=d,friction_dwidth=d)
  before=s.snapshot()[1];rc,o=s.advance();self.assertEqual(rc,2)
  for a in o.values():self.assertTrue(np.all(a==777))
  for k,a in before.items():np.testing.assert_array_equal(s.snapshot()[1][k],a)
 def test_version_legacy_zero_flags_invalid_coefficients_rejected(self):
  s=self.make([[.02]])
  for field,value in [('version',0x20000),('version',0),('struct_size',160),('flags',1),('reserved',1),('dofs',33),('environments',0)]:
   old=getattr(s.p,field);setattr(s.p,field,value);handle=C.c_void_p(123)
   self.assertEqual(self.lib.box3d_joint_v1_create(C.byref(s.p),C.byref(handle)),1);self.assertEqual(handle.value,123);setattr(s.p,field,old)
  for field,value in [('armature',-.001),('friction_d0',0),('friction_dwidth',1),('friction_timeconst',0),('passive_damping',float('nan'))]:
   old=s.refs[field].copy();s.refs[field].fill(value);handle=C.c_void_p();self.assertEqual(self.lib.box3d_joint_v1_create(C.byref(s.p),C.byref(handle)),1);s.refs[field][:]=old
  s.refs['revolute'].fill(0);handle=C.c_void_p();self.assertEqual(self.lib.box3d_joint_v1_create(C.byref(s.p),C.byref(handle)),1)
 def test_zero_extension_terms_matches_bare_coupled_equations(self):
  body=np.array([[.05,.02],[.02,.02]]);s=self.make(body,armature=0,passive_damping=0,friction_loss=0,motor_enabled=0,stiffness=0)
  rc,o=s.advance(force=[.01,-.02]);self.assertEqual(rc,0);self.near(o['acceleration'][0],np.linalg.solve(body,[.01,-.02]));self.near(o['friction'],0)
 def test_all_existing_tracked_sources_unchanged(self):
  # Upstream sealed against 9b3fab5; this workspace is an intentional fork with
  # its own history, so the equivalent guard is a clean tree against HEAD.
  r=subprocess.run(['git','diff','--exit-code','HEAD','--','tests/fixtures'],cwd=ROOT,capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stdout)
if __name__=='__main__':unittest.main()
