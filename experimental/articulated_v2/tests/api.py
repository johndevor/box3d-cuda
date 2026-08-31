import ctypes as C
import os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'articulated_v1'))
from articulated_v1 import Fixture,duck,Model,rot,exp,mul,dp,checked
U=C.c_uint32;D=C.c_double;DP=C.POINTER(D);ABI=0x41520002
class Limit(C.Structure):_fields_=[('enabled',U),('reserved',U)]+[(n,D) for n in ['lower','upper','margin','timeconst','dampratio']]+[('solimp',D*5)]
class Registration(C.Structure):_fields_=[(n,U) for n in ['struct_size','version','environments','reserved']]+[('model',C.POINTER(Model)),('limits',C.POINTER(Limit))]+[(n,DP) for n in ['q','v','gravity']]
class Step(C.Structure):_fields_=[('struct_size',U),('version',U)]+[(n,D) for n in ['dt','mtol','jtol']]+[(n,DP) for n in ['target','targetv','force']]
PRE_FIELDS=['q','v','mass','inverse','bias','actuator','passive','smooth','pose','bodyv','J','G','target','R','lower','upper','warm','gap','aref']
class PreView(C.Structure):_fields_=[(n,U) for n in ['struct_size','version','E','B','N','Joints','rows','reserved']]+[('generation',C.c_uint64),('dt',D)]+[(n,DP) for n in PRE_FIELDS]+[('kind',C.POINTER(U)),('active',C.POINTER(C.c_uint8))]
class Solution(C.Structure):_fields_=[('struct_size',U),('version',U)]+[(n,DP) for n in ['v','impulse','contact']]
class Snapshot(C.Structure):_fields_=[(n,U) for n in ['struct_size','version','E','J']]+[('binding',C.c_uint64)]+[(n,DP) for n in ['q','v','warm','time']]+[('count',C.POINTER(C.c_uint64))]
class StateView(C.Structure):_fields_=[(n,U) for n in ['struct_size','version','E','B','J','reserved']]+[(n,DP) for n in ['q','v','warm','time','pose','bodyv']]+[('count',C.POINTER(C.c_uint64))]
def desc(t):x=t();x.struct_size=C.sizeof(t);x.version=ABI;return x
def library(path=None):
 l=C.CDLL(str(path or os.environ['AV2_LIBRARY']));p=C.c_void_p
 specs={'av2_create':[C.POINTER(Registration),C.POINTER(p)],'av2_destroy':[p],'av2_read':[p,C.POINTER(StateView)],'av2_capture':[p,C.POINTER(Snapshot)],'av2_prepare':[p,C.POINTER(Step),C.POINTER(p)],'av2_pre_read':[p,C.POINTER(PreView)],'av2_pre_destroy':[p],'av2_complete':[p,C.POINTER(Solution),C.POINTER(p)],'av2_stage_read':[p,C.POINTER(StateView)],'av2_stage_capture':[p,C.POINTER(Snapshot)],'av2_prepare_restore':[p,C.POINTER(Snapshot),C.POINTER(p)],'av2_prepare_reset':[p,C.POINTER(C.c_uint8),U,C.POINTER(p)],'av2_validate_commit':[p,p],'av2_commit':[p,p],'av2_stage_destroy':[p]}
 for n,a in specs.items():getattr(l,n).argtypes=a;getattr(l,n).restype=None if n.endswith('destroy') else C.c_int
 return l
def limits(f):
 a=(Limit*f.J)()
 for l in a:l.enabled=1;l.lower=-.4;l.upper=.4;l.timeconst=.02;l.dampratio=1;l.solimp[:]=[.9,.95,.001,.5,2]
 if hasattr(f,'mapping'):
  for l,j in zip(a,f.mapping['joints']):l.lower=j['lower'];l.upper=j['upper']
 return a
class Saved:
 def __init__(self,s):
  self.q=np.zeros((s.E,s.Q));self.v=np.zeros((s.E,s.N));self.warm=np.zeros((s.E,s.R));self.time=np.zeros(s.E);self.count=np.zeros(s.E,'uint64');self.d=desc(Snapshot);self.d.E=s.E;self.d.J=s.J
  for n in ['q','v','warm','time']:setattr(self.d,n,dp(getattr(self,n)))
  self.d.count=self.count.ctypes.data_as(C.POINTER(C.c_uint64))
 def bytes(self):return b''.join(getattr(self,n).tobytes() for n in ['q','v','warm','time','count'])+int(self.d.binding).to_bytes(8,'little')
class Scene:
 def __init__(self,lib,f,q,v,lim=None,gravity=None):
  self.lib=lib;self.f=f;self.J=f.J;self.N=f.N;self.Q=f.J+7;self.R=3*f.J;self.B=f.B;self.q=np.ascontiguousarray(q,'d');self.E=len(self.q)
  self.q=checked(self.q,'d',(self.E,self.Q),'q');self.v=checked(v,'d',(self.E,self.N),'v');self.gravity=checked(gravity if gravity is not None else [[0,0,0]]*self.E,'d',(self.E,3),'gravity');self.limits=limits(f) if lim is None else lim
  self.d=desc(Registration);self.d.environments=self.E;self.d.model=C.pointer(f.model);self.d.limits=self.limits;self.d.q=dp(self.q);self.d.v=dp(self.v);self.d.gravity=dp(self.gravity);self.h=C.c_void_p();rc=lib.av2_create(C.byref(self.d),C.byref(self.h));assert rc==0,rc;self.pres=[];self.stages=[]
 def close(self):
  for p in self.pres:self.lib.av2_pre_destroy(p)
  for p in self.stages:self.lib.av2_stage_destroy(p)
  self.lib.av2_destroy(self.h)
 def capture(self,stage=None):
  x=Saved(self);rc=(self.lib.av2_stage_capture(stage,C.byref(x.d)) if stage else self.lib.av2_capture(self.h,C.byref(x.d)));assert rc==0,rc;return x
 def pre(self,dt=.002,target=None,force=None):
  target=np.zeros((self.E,self.J)) if target is None else checked(target,'d',(self.E,self.J),'target');tv=np.zeros_like(target);d=desc(Step);d.dt=dt;d.mtol=d.jtol=1e-8;d.target=dp(target);d.targetv=dp(tv)
  if force is not None:force=checked(force,'d',(self.E,self.N),'force');d.force=dp(force)
  p=C.c_void_p();rc=self.lib.av2_prepare(self.h,C.byref(d),C.byref(p));assert rc==0,rc;self.pres.append(p);view=desc(PreView);assert self.lib.av2_pre_read(p,C.byref(view))==0
  shapes={'q':(self.E,self.Q),'v':(self.E,self.N),'mass':(self.E,self.N,self.N),'inverse':(self.E,self.N,self.N),'bias':(self.E,self.N),'actuator':(self.E,self.N),'passive':(self.E,self.N),'smooth':(self.E,self.N),'pose':(self.E,self.B,7),'bodyv':(self.E,self.B,6),'J':(self.E,self.B,6,self.N),'G':(self.E,self.R,self.N)}
  arrays={n:np.ctypeslib.as_array(getattr(view,n),shape=shapes.get(n,(self.E,self.R))).copy() if all(shapes.get(n,(self.E,self.R))) else np.zeros(shapes.get(n,(self.E,self.R))) for n in PRE_FIELDS};arrays['active']=np.ctypeslib.as_array(view.active,shape=(self.E,self.R)).copy() if self.R else np.zeros((self.E,0));arrays['generation']=view.generation;arrays['dt']=view.dt;return p,arrays
 def complete(self,p,v,impulse,contact=None):
  v=checked(v,'d',(self.E,self.N),'v');impulse=checked(impulse,'d',(self.E,self.R),'impulse');d=desc(Solution);d.v=dp(v);d.impulse=dp(impulse)
  if contact is not None:contact=checked(contact,'d',(self.E,self.N),'contact');d.contact=dp(contact)
  stage=C.c_void_p();rc=self.lib.av2_complete(p,C.byref(d),C.byref(stage));
  if stage.value:self.stages.append(stage)
  return rc,stage
 def restore(self,snapshot):
  stage=C.c_void_p();rc=self.lib.av2_prepare_restore(self.h,C.byref(snapshot.d),C.byref(stage))
  if stage.value:self.stages.append(stage)
  return rc,stage
 def reset(self,mask):
  a=(C.c_uint8*len(mask))(*mask);stage=C.c_void_p();rc=self.lib.av2_prepare_reset(self.h,a,len(mask),C.byref(stage))
  if stage.value:self.stages.append(stage)
  return rc,stage
 def commit(self,stage):return self.lib.av2_commit(self.h,stage)
def solve(pre,external=None):
 # Independent exhaustive active-set box QP for small synthetic cases only.
 import itertools
 E,N=pre['v'].shape;imp=np.zeros_like(pre['target']);post=np.zeros_like(pre['v'])
 for e in range(E):
  active=np.flatnonzero(pre['active'][e]);G=pre['G'][e,active];inv=pre['inverse'][e];base=pre['smooth'][e].copy()
  if external is not None:base+=inv@external[e]
  H=G@inv@G.T+np.diag(pre['R'][e,active]);b=G@base-pre['target'][e,active];lo=pre['lower'][e,active];hi=pre['upper'][e,active];found=None
  for modes in itertools.product([-1,0,1],repeat=len(active)):
   x=np.zeros(len(active));free=[];fixed=[];valid=True
   for i,m in enumerate(modes):
    if m==0:free.append(i)
    else:
     x[i]=lo[i] if m<0 else hi[i];fixed.append(i)
     if not np.isfinite(x[i]):valid=False;break
   if not valid:continue
   if free:x[free]=np.linalg.solve(H[np.ix_(free,free)],-b[free]-H[np.ix_(free,fixed)]@x[fixed])
   grad=H@x+b
   if np.any(x<lo-1e-12) or np.any(x>hi+1e-12):continue
   if any((m==0 and abs(grad[i])>1e-10) or (m<0 and grad[i]<-1e-10) or (m>0 and grad[i]>1e-10) for i,m in enumerate(modes)):continue
   found=x;break
  if found is None:raise RuntimeError('active set oracle failed')
  imp[e,active]=found;post[e]=base+inv@G.T@found
 return post,imp
