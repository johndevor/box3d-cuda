"""ctypes bindings and exact accepted principal-frame fixture; no simulator."""
import ctypes as C
import hashlib,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
U=C.c_uint32;U64=C.c_uint64;F=C.c_float;D=C.c_double;DP=C.POINTER(D);FP=C.POINTER(F)
ABI=0x41520001
class Body(C.Structure):_fields_=[('mass',D),('inertia',D*3)]
class Hinge(C.Structure):
 _fields_=[('parent',U),('motor_enabled',U),('ap',D*3),('ac',D*3),('axis',D*3),('reference',D*4)]+[(s,F) for s in ['armature','damping','loss','kp','kv','cap','d0','dw','tc']]
class Model(C.Structure):_fields_=[(s,U) for s in ['struct_size','version','bodies','joints','flags','reserved']]+[('body',C.POINTER(Body)),('hinge',C.POINTER(Hinge)),('root_inertia',D*7),('reference_qpos',DP)]
class Evaluation(C.Structure):_fields_=[('struct_size',U),('version',U)]+[(s,DP) for s in ['pose','velocity','jacobian','mass','bias','abias']]+[('kinetic',D),('potential',D)]
class Registration(C.Structure):_fields_=[(s,U) for s in ['struct_size','version','environments','reserved']]+[('model',C.POINTER(Model)),('qpos',DP),('velocity',FP),('gravity',DP)]
class Snapshot(C.Structure):_fields_=[(s,U) for s in ['struct_size','version','environments','joints']]+[('binding',U64),('qpos',DP),('velocity',FP),('warm',FP),('count',C.POINTER(U64))]
class Step(C.Structure):_fields_=[(s,U) for s in ['struct_size','version','rows','max_iterations']]+[('dt',F),('tolerance',F)]+[(s,FP) for s in ['target','targetv','force','jacobian','aref','regularizer','lower','upper','warm']]
def checked(value,dtype,shape,name):
 a=np.ascontiguousarray(value,dtype=dtype)
 if a.shape!=shape:raise ValueError(name+' requires shape '+str(shape)+', received '+str(a.shape))
 return a
def dp(x):return x.ctypes.data_as(DP)
def fp(x):return x.ctypes.data_as(FP)
def struct(t):x=t();x.struct_size=C.sizeof(t);x.version=ABI;return x
def rot(q):
 x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def mul(a,b):return np.r_[a[3]*b[:3]+b[3]*a[:3]+np.cross(a[:3],b[:3]),a[3]*b[3]-a[:3]@b[:3]]
def exp(w):
 t=np.linalg.norm(w)
 return np.r_[w*(.5 if t<1e-14 else np.sin(t/2)/t),np.cos(t/2)]
def tangent(q,v,dt):
 a=q.copy();a[:3]+=dt*v[:3];a[3:7]=mul(exp(dt*v[3:6]),a[3:7]);a[7:]+=dt*v[6:];return a
class Fixture:
 def __init__(self,J):
  self.J=J;self.N=6+J;self.B=2+J
  self.body=(Body*self.B)();self.hinge=(Hinge*J)();self.reference=np.r_[np.zeros(6),1.,np.zeros(J)].astype('d')
  self.model=struct(Model);self.model.bodies=self.B;self.model.joints=J;self.model.body=self.body;self.model.hinge=self.hinge;self.model.root_inertia[6]=1;self.model.reference_qpos=dp(self.reference)
  for b in range(1,self.B):self.body[b]=Body(1,(D*3)(.2,.3,.4))
  for j,h in enumerate(self.hinge):h.parent=j+1;h.axis[:]=[0,0,1];h.reference[:]=[0,0,0,1];h.d0=.9;h.dw=.95;h.tc=.02
 def copy(self):
  f=Fixture(self.J);C.memmove(f.body,self.body,C.sizeof(self.body));C.memmove(f.hinge,self.hinge,C.sizeof(self.hinge));f.reference[:]=self.reference;f.model.root_inertia[:]=self.model.root_inertia[:];return f
 def evaluate(self,lib,q,v,gravity=(0,0,-9.81),sentinel=0):
  q=checked(q,'d',(7+self.J,),'qpos');v=checked(v,'d',(self.N,),'velocity');g=checked(gravity,'d',(3,),'gravity');e=Eval(self,sentinel)
  rc=lib.av1_evaluate(C.byref(self.model),dp(q),dp(v),dp(g),C.byref(e.desc));return rc,e
class Eval:
 def __init__(self,f,sentinel=0):
  self.pose=np.full((f.B,7),sentinel,dtype='d');self.velocity=np.full((f.B,6),sentinel,dtype='d');self.jacobian=np.full((f.B,6,f.N),sentinel,dtype='d');self.mass=np.full((f.N,f.N),sentinel,dtype='d');self.bias=np.full(f.N,sentinel,dtype='d');self.abias=np.full((f.B,6),sentinel,dtype='d')
  self.desc=struct(Evaluation)
  for s in ['pose','velocity','jacobian','mass','bias','abias']:setattr(self.desc,s,dp(getattr(self,s)))
 def bytes(self):return b''.join(getattr(self,s).tobytes() for s in ['pose','velocity','jacobian','mass','bias','abias'])
def pinned(name,expected):
 raw=(ROOT/'fixtures'/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==expected;return json.loads(raw)
def duck():
 g=pinned('geometry-goldens.json','e52ba7d0f79434499d8fb6c2d611eb46ee12e2f32cb36258b38cd22959d0b08b')['mapping']
 r=pinned('static-reference.json','58535c1e36728ced2c69b87c504a11116757313a6fd556c2435f3515a9f6e5a1')
 f=Fixture(14)
 for b,x in zip(g['bodies'],f.body):
  x.mass=b['mass'];x.inertia[:]=b.get('principal_inertia',[0,0,0])
 for j,h in zip(g['joints'],f.hinge):
  assert j['child']==j['id']+2
  h.parent=j['parent'];h.ap[:]=j['parent_anchor'];h.ac[:]=j['child_anchor'];h.axis[:]=j['axis_parent'];h.reference[:]=j['reference_xyzw'];h.armature=.027;h.damping=.56;h.loss=.068;h.kp=13.37;h.cap=3.23;h.motor_enabled=1
 b=g['bodies'][1];f.model.root_inertia[:]=[*b['source_COM'],*b['inertial_quaternion_wxyz'][1:],b['inertial_quaternion_wxyz'][0]]
 f.reference[:]=[*r['reference_qpos'][:3],*r['reference_qpos'][4:7],r['reference_qpos'][3],*r['reference_qpos'][7:]]
 f.mapping=g;f.record=r;return f
def library(path=None):
 lib=C.CDLL(str(path or os.environ['AV1_LIBRARY']))
 specs={'av1_evaluate':[C.POINTER(Model),DP,DP,DP,C.POINTER(Evaluation)],'av1_integrate_root':[DP,DP,D,DP],
 'av1_create':[C.POINTER(Registration),C.POINTER(C.c_void_p)],'av1_destroy':[C.c_void_p],
 'av1_capture':[C.c_void_p,C.POINTER(Snapshot)],'av1_restore':[C.c_void_p,C.POINTER(Snapshot)],
 'av1_reset_masked':[C.c_void_p,C.POINTER(C.c_uint8),U],'av1_read':[C.c_void_p,U,C.POINTER(Evaluation)],
 'av1_prepare':[C.c_void_p,C.POINTER(Step),C.POINTER(C.c_void_p)],'av1_stage_capture':[C.c_void_p,C.POINTER(Snapshot)],
 'av1_stage_read':[C.c_void_p,U,C.POINTER(Evaluation)],'av1_stage_diagnostics':[C.c_void_p,FP,FP],
 'av1_validate_commit':[C.c_void_p,C.c_void_p],'av1_commit':[C.c_void_p,C.c_void_p],
 'av1_stage_destroy':[C.c_void_p],'av1_response':[C.c_void_p,U,FP,FP,DP,DP]}
 for name,args in specs.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=None if name.endswith('destroy') else C.c_int
 return lib
class Saved:
 def __init__(self,f,E):
  self.q=np.zeros((E,7+f.J),'d');self.v=np.zeros((E,f.N),'f');self.warm=np.zeros_like(self.v);self.count=np.zeros(E,'uint64')
  self.d=struct(Snapshot);self.d.environments=E;self.d.joints=f.J;self.d.qpos=dp(self.q);self.d.velocity=fp(self.v);self.d.warm=fp(self.warm);self.d.count=self.count.ctypes.data_as(C.POINTER(U64))
 def bytes(self):return self.q.tobytes()+self.v.tobytes()+self.warm.tobytes()+self.count.tobytes()+bytes(self.d.binding.to_bytes(8,'little'))
class Scene:
 def __init__(self,lib,f,q,v,gravity=None):
  self.lib=lib;self.f=f;self.q=np.ascontiguousarray(q,dtype='d')
  if self.q.ndim!=2 or self.q.shape[1]!=7+f.J or not 1<=self.q.shape[0]<=4096:raise ValueError('qpos requires [E,7+J] with 1<=E<=4096')
  self.E=len(self.q);self.v=checked(v,'f',(self.E,f.N),'velocity');self.g=checked(gravity if gravity is not None else [[0,0,-9.81]]*self.E,'d',(self.E,3),'gravity')
  self.desc=struct(Registration);self.desc.environments=self.E;self.desc.model=C.pointer(f.model);self.desc.qpos=dp(self.q);self.desc.velocity=fp(self.v);self.desc.gravity=dp(self.g);self.h=C.c_void_p()
  rc=lib.av1_create(C.byref(self.desc),C.byref(self.h));assert rc==0,rc
 def close(self):self.lib.av1_destroy(self.h);self.h=None
 def capture(self,stage=None):
  a=Saved(self.f,self.E);rc=(self.lib.av1_stage_capture(stage,C.byref(a.d)) if stage else self.lib.av1_capture(self.h,C.byref(a.d)));assert rc==0,rc;return a
 def read(self,env=0,stage=None):
  e=Eval(self.f);rc=(self.lib.av1_stage_read(stage,env,C.byref(e.desc)) if stage else self.lib.av1_read(self.h,env,C.byref(e.desc)));assert rc==0,rc;return e
 def prepare(self,dt=.002,force=None,targets=None):
  s=struct(Step);s.max_iterations=4096;s.dt=dt;s.tolerance=1e-6
  target=np.zeros((self.E,self.f.J),'f') if targets is None else checked(targets,'f',(self.E,self.f.J),'targets');targetv=np.zeros_like(target)
  s.target=fp(target);s.targetv=fp(targetv)
  if force is not None:force=checked(force,'f',(self.E,self.f.N),'force');s.force=fp(force)
  stage=C.c_void_p();rc=self.lib.av1_prepare(self.h,C.byref(s),C.byref(stage));return rc,stage
