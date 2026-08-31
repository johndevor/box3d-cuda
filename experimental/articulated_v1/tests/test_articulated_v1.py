"""Native adapter checks. Only synthetic models advance; duck queries are static."""
import ctypes as C
import hashlib,itertools,os,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from articulated_v1 import *
class Adapter(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.lib=library();cls.duck=duck();cls.rng=np.random.default_rng(2071)
 def close(self,a,b,atol=2e-10,rtol=2e-10):np.testing.assert_allclose(a,b,atol=atol,rtol=rtol)
 def evaluate(self,f,q,v,g=(0,0,-9.81)):
  rc,e=f.evaluate(self.lib,q,v,g);self.assertEqual(rc,0);return e
 def random_state(self,f):
  q=f.reference.copy();q[:3]=[.2,-.4,1.1];q[3:7]=exp(self.rng.normal(size=3)*.8);q[7:]=self.rng.normal(size=f.J)*.7;v=self.rng.normal(size=f.N);return q,v
 def synthetic(self,J=2):
  f=Fixture(J);f.model.root_inertia[:]=[.23,-.11,.19,*exp(np.array([.4,-.2,.3]))]
  for j,h in enumerate(f.hinge):
   h.parent=1 if j==1 else j+1;h.ap[:]=[.17,-.09,.2];h.ac[:]=[-.12,.06,.03];a=np.array([.3,.7,-.2]);h.axis[:]=a/np.linalg.norm(a);h.reference[:]=exp(np.array([-.1,.2,.4]));h.armature=.027
  return f
 def test_three_retained_static_checkpoints_full_reference(self):
  f=self.duck;maxima=dict(pose=0.,rotation=0.,jacobian=0.,velocity=0.,mass=0.,bias=0.)
  self.assertEqual(f.record['indices'],[0,250,500]);self.assertEqual(f.record['mj_step_calls'],0)
  for row in f.record['frames']:
   qp=row['qpos'];q=np.r_[qp[:3],qp[4:7],qp[3],qp[7:]];v=np.array(row['qvel']);R=rot(q[3:7]);v[3:6]=R@v[3:6];T=np.eye(20);T[3:6,3:6]=R.T
   e=self.evaluate(f,q,v);referenceM=np.array(row['mass_including_armature_local_root_angular']);actualM=e.mass+np.diag(f.record['dof_armature']);referencebias=T.T@row['bias_local_root_angular']
   self.close(actualM,T.T@referenceM@T);self.close(e.bias,referencebias)
   maxima['mass']=max(maxima['mass'],float(np.max(abs(actualM-T.T@referenceM@T))));maxima['bias']=max(maxima['bias'],float(np.max(abs(e.bias-referencebias))))
   for b in row['bodies']:
    i=b['id'];j=np.array(b['jacobian_local_root_angular'])@T;vel=b['velocity_angular_then_linear'];rv=np.r_[vel[3:],vel[:3]]
    for key,a,z in [('pose',e.pose[i,:3],b['position']),('rotation',rot(e.pose[i,3:]),b['rotation']),('jacobian',e.jacobian[i],j),('velocity',e.velocity[i],rv)]:
     self.close(a,z);maxima[key]=max(maxima[key],float(np.max(abs(np.asarray(a)-z))))
   if row['frame']==500:self.assertGreater(np.max(abs(actualM-referenceM)),1e-5) # local/world basis negative control
  if os.environ.get('AV1_METRICS'):
   Path(os.environ['AV1_METRICS']).write_text(json.dumps({'static_reference_max_errors':maxima,'checkpoints':[0,250,500],'new_duck_steps':0},indent=2)+'\n')
 def test_spatial_jacobian_geometric_ancestor_oracle_and_pose_fd(self):
  f=self.duck
  for trial in range(3):
   q,v=self.random_state(f);e=self.evaluate(f,q,v)
   for b in range(1,f.B):
    expected=np.zeros((6,f.N));expected[:3,:3]=np.eye(3);expected[3:,3:6]=np.eye(3)
    for k in range(3):expected[:3,3+k]=np.cross(np.eye(3)[k],e.pose[b,:3]-q[:3])
    ancestor=b
    while ancestor>1:
     j=ancestor-2;h=f.hinge[j];parent=h.parent;axis=rot(e.pose[parent,3:])@h.axis;anchor=e.pose[parent,:3]+rot(e.pose[parent,3:])@h.ap
     expected[:3,6+j]=np.cross(axis,e.pose[b,:3]-anchor);expected[3:,6+j]=axis;ancestor=parent
    self.close(e.jacobian[b],expected)
   for k in range(f.N):
    direction=np.eye(f.N)[k];h=2e-6;plus=self.evaluate(f,tangent(q,direction,h),v);minus=self.evaluate(f,tangent(q,direction,-h),v)
    self.close((plus.pose[:,:3]-minus.pose[:,:3])/(2*h),e.jacobian[:,:3,k],atol=2e-9)
    for b in range(1,f.B):
     W=((rot(plus.pose[b,3:])-rot(minus.pose[b,3:]))/(2*h))@rot(e.pose[b,3:]).T
     self.close([W[2,1],W[0,2],W[1,0]],e.jacobian[b,3:,k],atol=2e-9)
 def test_jdot_velocity_bias_acceleration_finite_difference(self):
  f=self.duck
  for trial in range(8):
   q,v=self.random_state(f);e=self.evaluate(f,q,v);h=2e-6
   plus=self.evaluate(f,tangent(q,v,h),v);minus=self.evaluate(f,tangent(q,v,-h),v)
   self.close(e.velocity,np.einsum('bkn,n->bk',e.jacobian,v))
   self.close(e.abias,(plus.velocity-minus.velocity)/(2*h),atol=4e-9)
 def test_kinetic_mass_energy_power_and_galilean_bias(self):
  f=self.duck
  for trial in range(8):
   q,v=self.random_state(f);e=self.evaluate(f,q,v,g=(0,0,0));kinetic=0
   for b in range(1,f.B):
    worldI=rot(e.pose[b,3:])@np.diag(f.body[b].inertia)@rot(e.pose[b,3:]).T
    kinetic+=.5*(f.body[b].mass*(e.velocity[b,:3]@e.velocity[b,:3])+e.velocity[b,3:]@worldI@e.velocity[b,3:])
   self.close(kinetic,e.desc.kinetic);self.close(kinetic,.5*v@e.mass@v)
   h=2e-6;plus=self.evaluate(f,tangent(q,v,h),v,g=(0,0,0));minus=self.evaluate(f,tangent(q,v,-h),v,g=(0,0,0));self.close(v@e.bias,.5*v@((plus.mass-minus.mass)/(2*h))@v,atol=2e-8)
   boosted=v.copy();boosted[:3]+=[23.,-31.,7.];self.close(e.bias,self.evaluate(f,q,boosted,g=(0,0,0)).bias)
 def test_gravity_potential_gradient_and_free_fall_full_coupling(self):
  f=self.duck;q,v=self.random_state(f);v[:]=0;g=np.array([1.3,-2.7,-9.81]);e=self.evaluate(f,q,v,g)
  for k in range(f.N):
   h=2e-6;direction=np.eye(f.N)[k];plus=self.evaluate(f,tangent(q,direction,h),v,g);minus=self.evaluate(f,tangent(q,direction,-h),v,g)
   self.close(e.bias[k],(plus.desc.potential-minus.desc.potential)/(2*h),atol=8e-9)
  MA=e.mass+np.diag([0]*6+[.027]*14);self.close(np.linalg.solve(MA,-e.bias),np.r_[g,np.zeros(17)],atol=4e-12)
 def test_rigid_aggregate_centripetal_and_gyroscopic_terms(self):
  f=self.duck;q,v=self.random_state(f);v[6:]=0;e=self.evaluate(f,q,v,(0,0,0));I=np.zeros((3,3));force=np.zeros(3);w=v[3:6]
  for b in range(1,f.B):
   r=e.pose[b,:3]-q[:3];mass=f.body[b].mass;R=rot(e.pose[b,3:]);I+=R@np.diag(f.body[b].inertia)@R.T+mass*((r@r)*np.eye(3)-np.outer(r,r));force+=mass*np.cross(w,np.cross(w,r))
  self.close(e.bias[:3],force);self.close(e.bias[3:6],np.cross(w,I@w));self.assertGreater(np.linalg.norm(e.bias[3:6]),1e-5)
 def test_root_world_exponential_preserves_origin_and_norm(self):
  q=np.r_[[.4,-.2,.8],exp(np.array([.4,-.7,.2]))];v=np.array([1.,2.,-3.,.7,-.3,.6]);out=np.full(7,777.)
  self.assertEqual(self.lib.av1_integrate_root(dp(q),dp(v),.07,dp(out)),0)
  self.close(out[:3],q[:3]+.07*v[:3]);self.close(rot(out[3:]),rot(exp(.07*v[3:]))@rot(q[3:]));self.close(np.linalg.norm(out[3:]),1.)
  self.assertGreater(np.max(abs(rot(out[3:])-rot(q[3:])@rot(exp(.07*v[3:])))),.001)
  before=out.copy();q[3:]*=2;self.assertEqual(self.lib.av1_integrate_root(dp(q),dp(v),.07,dp(out)),1);self.close(out,before)
 def test_synthetic_native_step_bias_armature_and_root_integration(self):
  f=self.synthetic();q,v=self.random_state(f);scene=Scene(self.lib,f,[q],[v]);self.addCleanup(scene.close);before=scene.capture();ev=scene.read();force=np.array([[.8,-.4,.2,.3,.1,-.2,.7,-.9]],'f');dt=float(np.float32(.002))
  expected=np.linalg.solve((ev.mass.astype('f')+np.diag(np.array([0]*6+[.027]*2,'f'))).astype('d'),(force[0].astype('d')-ev.bias).astype('f').astype('d'))
  rc,stage=scene.prepare(force=force);self.assertEqual(rc,0);self.addCleanup(self.lib.av1_stage_destroy,stage);self.assertEqual(scene.capture().bytes(),before.bytes());after=scene.capture(stage)
  acc=np.zeros((1,f.N),'f');self.assertEqual(self.lib.av1_stage_diagnostics(stage,fp(acc),None),0);self.close(acc[0],expected,atol=3e-6,rtol=3e-6)
  self.close(after.v[0],before.v[0]+dt*acc[0],atol=2e-7);self.close(after.q[0,:3],before.q[0,:3]+dt*after.v[0,:3]);self.close(rot(after.q[0,3:7]),rot(exp(dt*after.v[0,3:6].astype('d')))@rot(before.q[0,3:7]));self.close(after.q[0,7:],(before.q[0,7:].astype('f')+np.float32(dt)*after.v[0,6:]).astype('f'))
  post=scene.read(stage=stage);self.close(post.velocity,np.einsum('bkn,n->bk',post.jacobian,after.v[0]));self.assertEqual(self.lib.av1_commit(scene.h,stage),0);self.assertEqual(scene.capture().bytes(),after.bytes());self.assertEqual(self.lib.av1_commit(scene.h,stage),5)
 def test_synthetic_joint_parameters_and_reference_mass_friction_mapping(self):
  f=self.synthetic(1);h=f.hinge[0];h.damping=.56;h.loss=.068;h.kp=13.37;h.cap=.05;h.motor_enabled=1
  q,v=self.random_state(f);v[-1]=.08;scene=Scene(self.lib,f,[q],[v]);self.addCleanup(scene.close);before=scene.capture();e=scene.read();ref=self.evaluate(f,f.reference,np.zeros(f.N));A=np.diag([0]*6+[float(h.armature)]);inv=np.linalg.inv(e.mass.astype('f').astype('d')+A);refinv=np.linalg.inv(ref.mass.astype('f').astype('d')+A)
  tau=-e.bias;tau[-1]+=.05-float(h.damping)*float(before.v[0,-1]);smooth=inv@tau;R=(1-float(h.d0))/float(h.d0)*refinv[-1,-1];aref=-2/(float(h.dw)*float(h.tc))*float(before.v[0,-1]);friction=np.clip((aref-smooth[-1])/(inv[-1,-1]+R),-float(h.loss),float(h.loss));tau[-1]+=friction;expected=inv@tau
  rc,stage=scene.prepare(targets=[[100.]]);self.assertEqual(rc,0);self.addCleanup(self.lib.av1_stage_destroy,stage);acc=np.zeros((1,f.N),'f');self.lib.av1_stage_diagnostics(stage,fp(acc),None);self.close(acc[0],expected,atol=5e-6,rtol=3e-6)
  after=scene.capture(stage);self.close(after.warm[0,-1],friction,atol=2e-7);self.close(after.warm[0,:6],0)
 def test_owned_copy_capture_restore_reset_and_stale_stage(self):
  f=self.synthetic();q,v=self.random_state(f);q2=q.copy();q2[:3]+=[1.,2.,3.];scene=Scene(self.lib,f,[q,q2],[v,-v]);self.addCleanup(scene.close);initial=scene.capture();initial_eval=scene.read();rc,stage=scene.prepare();self.assertEqual(rc,0);self.addCleanup(self.lib.av1_stage_destroy,stage)
  scene.q[:]=77;scene.v[:]=88;scene.g[:]=99;f.body[1].mass=99;f.hinge[0].ap[0]=99;f.reference[:]=99
  self.assertEqual(scene.capture().bytes(),initial.bytes());self.close(scene.read().mass,initial_eval.mass)
  candidate=scene.capture(stage);self.assertEqual(self.lib.av1_commit(scene.h,stage),0);self.assertEqual(self.lib.av1_restore(scene.h,C.byref(initial.d)),0);rc,replay=scene.prepare();self.assertEqual(rc,0);self.addCleanup(self.lib.av1_stage_destroy,replay);self.assertEqual(scene.capture(replay).bytes(),candidate.bytes());self.assertEqual(self.lib.av1_commit(scene.h,replay),0)
  peer_before=scene.capture();mask=(C.c_uint8*2)(1,0);self.assertEqual(self.lib.av1_reset_masked(scene.h,mask,2),0);after=scene.capture()
  for name in ['q','v','warm','count']:
   self.assertEqual(getattr(after,name)[0].tobytes(),getattr(initial,name)[0].tobytes());self.assertEqual(getattr(after,name)[1].tobytes(),getattr(peer_before,name)[1].tobytes())
  rc,stale=scene.prepare();self.assertEqual(rc,0);self.addCleanup(self.lib.av1_stage_destroy,stale);self.assertEqual(self.lib.av1_restore(scene.h,C.byref(after.d)),0);self.assertEqual(self.lib.av1_validate_commit(scene.h,stale),5);self.assertEqual(self.lib.av1_commit(scene.h,stale),5);self.assertEqual(scene.capture().bytes(),after.bytes())
 def test_invalid_second_environment_snapshot_mask_and_step_are_atomic(self):
  f=self.synthetic();q,v=self.random_state(f);scene=Scene(self.lib,f,[q,q],[v,v]);self.addCleanup(scene.close);initial=scene.capture()
  for change in ['quaternion','warm','binding','joint_precision']:
   bad=scene.capture()
   if change=='quaternion':bad.q[1,3:7]*=2
   if change=='warm':bad.warm[1,0]=.001
   if change=='binding':bad.d.binding^=1
   if change=='joint_precision':bad.q[1,7]+=.0000000001
   self.assertEqual(self.lib.av1_restore(scene.h,C.byref(bad.d)),1);self.assertEqual(scene.capture().bytes(),initial.bytes())
  self.assertEqual(self.lib.av1_reset_masked(scene.h,(C.c_uint8*2)(1,2),2),1);self.assertEqual(scene.capture().bytes(),initial.bytes())
  force=np.zeros((2,f.N),'f');force[1,3]=np.nan;rc,stage=scene.prepare(force=force);self.assertEqual(rc,1);self.assertFalse(stage.value);self.assertEqual(scene.capture().bytes(),initial.bytes())
  bad=scene.capture();bad.count[1]=np.iinfo('uint64').max;self.assertEqual(self.lib.av1_restore(scene.h,C.byref(bad.d)),0);before=scene.capture();rc,stage=scene.prepare();self.assertNotEqual(rc,0);self.assertFalse(stage.value);self.assertEqual(scene.capture().bytes(),before.bytes())
 def test_duck_full_impulse_response_all_bodies_without_time_advance(self):
  f=self.duck;q,v=self.random_state(f);scene=Scene(self.lib,f,[q],[v]);self.addCleanup(scene.close);before=scene.capture();e=scene.read();g=np.zeros(f.N,'f');g[-1]=1;dv=np.zeros(f.N,'f');bdv=np.zeros((f.B,6),'d');eff=np.zeros(1,'d')
  self.assertEqual(self.lib.av1_response(scene.h,0,fp(g),fp(dv),dp(bdv),dp(eff)),0);M=e.mass.astype('f').astype('d')+np.diag([0]*6+[float(f.hinge[0].armature)]*14);expected=np.linalg.solve(M,g)
  self.close(dv,expected,atol=3e-5,rtol=3e-6);self.close(bdv,np.einsum('bkn,n->bk',e.jacobian,dv));self.close(eff[0],g@dv,atol=1e-6)
  self.assertGreater(np.linalg.norm(dv[:6]),.01);self.assertTrue(np.all(np.linalg.norm(bdv[1:],axis=1)>1e-6));self.assertEqual(scene.capture().bytes(),before.bytes())
 def test_version_topology_mass_and_invalid_static_inputs_fail_closed(self):
  f=self.synthetic();q,v=self.random_state(f)
  for field,value in [('version',0x20000),('struct_size',0),('flags',1),('reserved',1)]:
   old=getattr(f.model,field);setattr(f.model,field,value);rc,e=f.evaluate(self.lib,q,v,sentinel=777);self.assertEqual(rc,1);self.assertTrue(np.all(e.pose==777));setattr(f.model,field,old)
  f.hinge[0].parent=2;self.assertEqual(f.evaluate(self.lib,q,v)[0],1);f.hinge[0].parent=1;f.body[0].mass=1;self.assertEqual(f.evaluate(self.lib,q,v)[0],1);f.body[0].mass=0
  q[3:7]*=2;rc,e=f.evaluate(self.lib,q,v,sentinel=777);self.assertEqual(rc,1);self.assertTrue(np.all(e.mass==777))
 def test_snapshot_cannot_transfer_between_different_models(self):
  f=self.synthetic();q,v=self.random_state(f);a=Scene(self.lib,f,[q],[v]);self.addCleanup(a.close);f2=f.copy();f2.body[1].mass+=.01;b=Scene(self.lib,f2,[q],[v]);self.addCleanup(b.close);saved=a.capture();before=b.capture();self.assertEqual(self.lib.av1_restore(b.h,C.byref(saved.d)),1);self.assertEqual(b.capture().bytes(),before.bytes())
 def test_f32_bridge_post_and_restore_overflow_fail_atomically(self):
  f=Fixture(0);q=f.reference.copy();v=np.zeros(6);scene=Scene(self.lib,f,[q],[v],gravity=[[0,0,0]]);self.addCleanup(scene.close);initial=scene.capture()
  force=np.array([[0,0,0,1e25,2e25,0]],'f');rc,stage=scene.prepare(force=force);self.assertEqual(rc,2);self.assertFalse(stage.value);self.assertEqual(scene.capture().bytes(),initial.bytes())
  bad=scene.capture();bad.v[0,3:6]=[1e25,2e25,0];self.assertEqual(self.lib.av1_restore(scene.h,C.byref(bad.d)),2);self.assertEqual(scene.capture().bytes(),initial.bytes())
 def test_python_wrapper_rejects_truncated_or_mismatched_arrays(self):
  f=Fixture(1);q=f.reference.copy();v=np.zeros(f.N)
  backing=np.r_[q,1.234]
  for badq,badv,badg in [(backing[:7],v,[0,0,-9.81]),(q,v[:-1],[0,0,-9.81]),(q,v,[0,0]),(q,v,[0,0,-9.81,0]),(q.reshape(1,-1),v,[0,0,-9.81])]:
   with self.assertRaises(ValueError):f.evaluate(self.lib,badq,badv,badg)
  for badq,badv,badg in [([q],[[0]*6],[[0,0,-9.81]]),([q],[v],[[0,0]]),([q],[v],[0,0,-9.81]),(q,[v],[[0,0,-9.81]])]:
   with self.assertRaises(ValueError):Scene(self.lib,f,badq,badv,badg)
  scene=Scene(self.lib,f,[q],[v]);self.addCleanup(scene.close);before=scene.capture()
  for targets,force in [([],None),([[0,0]],None),(None,[[0]*6]),(None,np.zeros((2,7)))]:
   with self.assertRaises(ValueError):scene.prepare(targets=targets,force=force)
  self.assertEqual(scene.capture().bytes(),before.bytes())
 def test_frozen_joint_sources_unchanged(self):
  repo=ROOT.parents[1]
  for rel,sha in {'csrc/experimental_joint_v1.cpp':'1c142a7789f478f4ae55e14c49eb0820f6f2313e44869b99517196b9cfe3e24e','csrc/experimental_joint_v1_shared.h':'6e58509d7e80f9a0867377f2cae45a532dc7a5dce323e033c27492431a2e0ad8','include/box3d_cuda/experimental_joint_v1.h':'6c5bc50e4777279cb4ae30817bb02177389eb6ef36d729b37f9909d68ef1eb6b'}.items():self.assertEqual(hashlib.sha256((repo/rel).read_bytes()).hexdigest(),sha)
if __name__=='__main__':unittest.main()
