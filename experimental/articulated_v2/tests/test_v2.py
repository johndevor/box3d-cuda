import unittest
from api import *
class Hooks(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.lib=library()
 def scene(self,qj=0.,vj=0.,J=1,E=1):
  f=Fixture(J)
  for h in f.hinge:h.ap[:]=[.2,.1,0];h.ac[:]=[-.1,.05,0];h.armature=.027
  q=np.tile(f.reference,(E,1));q[:,7:]=qj;v=np.zeros((E,f.N));v[:,6:]=vj;s=Scene(self.lib,f,q,v);self.addCleanup(s.close);return s
 def assertClose(self,a,b,atol=2e-10):np.testing.assert_allclose(a,b,atol=atol,rtol=2e-10)
 def test_strict_limit_boundary_and_both_sides(self):
  for q,v,expected in [(-.4,-1,[0,0,0]),(.4,1,[0,0,0]),(-.4005,-.1,[0,1,0]),(.4005,.1,[0,0,1]),(0,2,[0,0,0])]:
   s=self.scene(q,v);p,a=s.pre();self.assertEqual(a['active'][0].tolist(),expected)
   post,imp=solve(a);rc,t=s.complete(p,post,imp);self.assertEqual(rc,0);before=s.capture();self.assertEqual(s.capture().bytes(),before.bytes());st=s.capture(t)
   self.assertClose(st.q[0,7],q+.002*post[0,6]);self.assertClose(st.time,[.002]);self.assertEqual(st.count[0],1)
   if expected[1] or expected[2]:
    side=1 if expected[1] else 2;self.assertGreater(imp[0,side],0);self.assertClose(a['aref'][0,side],11.8074792243767,atol=1e-10);self.assertGreater(abs(st.q[0,7]),.4) # soft response, no clipping
 def test_limit_impedance_ramp_reference_weight_and_dt_safety(self):
  for penetration,d in [(.0001,.901),(.0005,.925),(.002,.95)]:
   s=self.scene(-.4-penetration,-.1);p,a=s.pre();ref=s.f.reference.copy();zero=np.zeros(s.N);rc,e=s.f.evaluate(self.lib,ref,zero,[0,0,0]);self.assertEqual(rc,0);M=e.mass+np.diag([0]*6+[float(s.f.hinge[0].armature)]);w=np.linalg.inv(M)[-1,-1]
   self.assertClose(a['R'][0,1],(1-d)/d*w);self.assertClose(a['target'][0,1],-.1+.002*a['aref'][0,1])
   p,b=s.pre(dt=.02);B=2/(.95*.04);K=1/(.95*.95*.04*.04);self.assertClose(b['aref'][0,1],B*.1+K*d*penetration)
 def test_retreating_limit_has_no_attractive_impulse(self):
  s=self.scene(-.4001,2);p,a=s.pre();post,imp=solve(a);self.assertClose(imp,0);self.assertEqual(s.complete(p,post,imp)[0],0)
 def test_command_range_clamp_then_motor_cap_passive_separate(self):
  f=Fixture(1);h=f.hinge[0];h.kp=2;h.cap=10;h.motor_enabled=1;h.damping=.56;h.loss=.068;h.armature=.027;q=f.reference.copy();v=np.zeros(f.N);v[-1]=.3;s=Scene(self.lib,f,[q],[v]);self.addCleanup(s.close);p,a=s.pre(target=[[99.]])
  self.assertClose(a['actuator'][0,-1],.8);self.assertClose(a['passive'][0,-1],-float(h.damping)*.3);self.assertClose(a['upper'][0,0],.002*float(h.loss));self.assertEqual(a['active'][0,0],1)
 def test_mixed_friction_limit_complete_coupled_oracle(self):
  f=Fixture(2)
  for h in f.hinge:h.ap[:]=[.2,0,0];h.ac[:]=[.1,.05,0];h.armature=.027;h.loss=.068
  q=f.reference.copy();q[7:]=[-.4005,.1];v=np.r_[np.zeros(6),-.08,.13];s=Scene(self.lib,f,[q],[v]);self.addCleanup(s.close);p,a=s.pre();post,imp=solve(a);self.assertGreater(imp[0,2],0);self.assertEqual(s.complete(p,post,imp)[0],0)
  bad=imp.copy();bad[0,0]=a['upper'][0,0];badpost=a['smooth']+np.einsum('eij,erj,er->ei',a['inverse'],a['G'],bad);self.assertEqual(s.complete(p,badpost,bad)[0],3) # momentum consistent but joint KKT fails
 def test_contact_impulse_enters_full_solve_before_exactly_once_root(self):
  s=self.scene(.1,.2);p,a=s.pre();contact=np.zeros((1,s.N));contact[0,0]=.2;contact[0,4]=.1;post,imp=solve(a,contact);before=s.capture();rc,t=s.complete(p,post,imp,contact);self.assertEqual(rc,0);st=s.capture(t);self.assertEqual(before.bytes(),s.capture().bytes());self.assertClose(st.q[0,:3],before.q[0,:3]+.002*post[0,:3]);self.assertClose(rot(st.q[0,3:7]),rot(exp(.002*post[0,3:6]))@rot(before.q[0,3:7]));self.assertEqual(s.commit(t),0);self.assertEqual(s.commit(t),5);self.assertEqual(s.capture().bytes(),st.bytes())
 def test_missing_contact_impulse_or_joint_solution_rejected(self):
  s=self.scene(-.4005,-.1);p,a=s.pre();contact=np.ones((1,s.N))*.001;post,imp=solve(a,contact);before=s.capture();self.assertEqual(s.complete(p,post,imp)[0],3);self.assertEqual(s.complete(p,post,imp,contact)[0],0);bad=post.copy();bad[0,0]+=.01;self.assertEqual(s.complete(p,bad,imp,contact)[0],3);self.assertEqual(before.bytes(),s.capture().bytes())
 def test_atomic_batch_failure_stale_generation_reset_restore(self):
  s=self.scene(-.4005,-.1,E=2);initial=s.capture();p,a=s.pre();post,imp=solve(a);bad=post.copy();bad[1,0]=np.nan;rc,t=s.complete(p,bad,imp);self.assertEqual(rc,1);self.assertFalse(t.value);self.assertEqual(s.capture().bytes(),initial.bytes());rc,t=s.complete(p,post,imp);self.assertEqual(rc,0);candidate=s.capture(t);self.assertEqual(s.commit(t),0);self.assertEqual(s.complete(p,post,imp)[0],5)
  rc,reset=s.reset([1,0]);self.assertEqual(rc,0);self.assertEqual(s.capture().bytes(),candidate.bytes());res=s.capture(reset)
  for name in ['q','v','warm','time','count']:self.assertEqual(getattr(res,name)[0].tobytes(),getattr(initial,name)[0].tobytes());self.assertEqual(getattr(res,name)[1].tobytes(),getattr(candidate,name)[1].tobytes())
  rc,restore=s.restore(initial);self.assertEqual(rc,0);self.assertEqual(s.commit(reset),0);self.assertEqual(s.commit(restore),5);rc,restore=s.restore(initial);self.assertEqual(rc,0);self.assertEqual(s.commit(restore),0);self.assertEqual(s.capture().bytes(),initial.bytes());p,a=s.pre();post,imp=solve(a);rc,t=s.complete(p,post,imp);self.assertEqual(rc,0);self.assertEqual(s.capture(t).bytes(),candidate.bytes())
 def test_two_participant_validation_prevents_partial_commit(self):
  a=self.scene();b=self.scene();pa,va=a.pre();pb,vb=b.pre();av,ai=solve(va);bv,bi=solve(vb);_,sa=a.complete(pa,av,ai);_,sb=b.complete(pb,bv,bi);_,reset=b.reset([1]);self.assertEqual(b.commit(reset),0);beforea=a.capture();beforeb=b.capture();codes=[self.lib.av2_validate_commit(a.h,sa),self.lib.av2_validate_commit(b.h,sb)];self.assertEqual(codes,[0,5]);self.assertEqual(a.capture().bytes(),beforea.bytes());self.assertEqual(b.capture().bytes(),beforeb.bytes()) # coordinator commits neither on any bad generation
 def test_snapshot_binding_invalid_warm_and_owned_parameters(self):
  s=self.scene();initial=s.capture();p,a=s.pre();s.f.hinge[0].armature=99;s.limits[0].lower=-99;s.q[:]=99;s.gravity[:]=99;p,b=s.pre();self.assertClose(a['mass'],b['mass']);self.assertClose(a['target'],b['target']);self.assertEqual(s.capture().bytes(),initial.bytes())
  for change in ['warm','quat','binding','time']:
   bad=s.capture()
   if change=='warm':bad.warm[0,1]=-1
   if change=='quat':bad.q[0,3:7]*=2
   if change=='binding':bad.d.binding^=1
   if change=='time':bad.time[0]=-1
   rc,t=s.restore(bad);self.assertEqual(rc,1);self.assertFalse(t.value);self.assertEqual(s.capture().bytes(),initial.bytes())
  self.assertEqual(s.reset([2])[0],1)
 def test_saturated_friction_warm_rounding_keeps_valid_solution(self):
  f=Fixture(1);f.hinge[0].loss=.068;q=f.reference.copy();v=np.zeros(f.N);v[-1]=1;s=Scene(self.lib,f,[q],[v]);self.addCleanup(s.close)
  p,a=s.pre(dt=.014846857006363103);post,imp=solve(a);rc,t=s.complete(p,post,imp);self.assertEqual(rc,0);st=s.capture(t);self.assertLessEqual(abs(st.warm[0,0]),float(f.hinge[0].loss))
 def test_large_impulse_cannot_hide_unattainable_kkt_tolerance(self):
  f=Fixture(1)
  for b in f.body[1:]:b.inertia[:]=[.4,.4,.4]
  q=f.reference.copy();q[-1]=-.4005;s=Scene(self.lib,f,[q],[np.zeros(f.N)]);self.addCleanup(s.close);target=np.zeros((1,1));d=desc(Step);d.dt=.002;d.mtol=1e-5;d.jtol=1e-12;d.target=dp(target);d.targetv=dp(target);p=C.c_void_p();self.assertEqual(self.lib.av2_prepare(s.h,C.byref(d),C.byref(p)),0);s.pres.append(p);view=desc(PreView);self.lib.av2_pre_read(p,C.byref(view))
  def arr(name,shape):return np.ctypeslib.as_array(getattr(view,name),shape=shape).copy()
  M=arr('mass',(1,s.N,s.N));G=arr('G',(1,s.R,s.N));R=arr('R',(1,s.R));tar=arr('target',(1,s.R));smooth=arr('smooth',(1,s.N));lam=np.zeros((1,s.R));lam[0,1]=1e8;v=np.zeros((1,s.N));v[0,-1]=tar[0,1]-R[0,1]*lam[0,1]+1e-8;contact=np.einsum('eij,ej->ei',M,v-smooth)-np.einsum('ern,er->en',G,lam);before=s.capture();rc,t=s.complete(p,v,lam,contact);self.assertEqual(rc,3);self.assertFalse(t.value);self.assertEqual(s.capture().bytes(),before.bytes())
 def test_supported_limit_parameter_edges_follow_source_guards(self):
  for width,ratio,solimp,dexpected in [(1e-20,1,[.9,.95,1e-20,.5,2],.925),(.001,1e-6,[.9,.95,.001,.5,2],.925),(.001,1,[1e-8,.95,.001,.5,2],(.0001+.95)/2)]:
   f=Fixture(1);q=f.reference.copy();q[-1]=-.4005;v=np.zeros(f.N);v[-1]=-.1;lim=limits(f);lim[0].solimp[:]=solimp;lim[0].dampratio=ratio;s=Scene(self.lib,f,[q],[v],lim=lim);self.addCleanup(s.close);p,a=s.pre();K=1/max(1e-15,.95**2*.02**2*ratio**2);expected=2/(.95*.02)*.1+K*dexpected*.0005;self.assertClose(a['aref'][0,1],expected,atol=1e-3 if ratio<1e-5 else 1e-10)
 def test_zero_hinges_empty_rows_and_free_root_step(self):
  s=self.scene(J=0);p,a=s.pre();self.assertEqual(a['G'].shape,(1,0,6));self.assertEqual(a['active'].shape,(1,0));post,imp=solve(a);self.assertEqual(imp.shape,(1,0));rc,t=s.complete(p,post,imp);self.assertEqual(rc,0);self.assertEqual(s.commit(t),0);self.assertEqual(s.capture().count[0],1)
 def test_exact_duck_pre_only_no_duck_step(self):
  f=duck();row=f.record['frames'][0];qp=row['qpos'];q=np.r_[qp[:3],qp[4:7],qp[3],qp[7:]];v=np.array(row['qvel']);s=Scene(self.lib,f,[q],[v],gravity=[[0,0,-9.81]]);self.addCleanup(s.close);p,a=s.pre(target=[q[7:]]);self.assertEqual(a['mass'].shape,(1,20,20));self.assertClose(a['actuator'],0);self.assertEqual(a['active'][0,:14].sum(),14);self.assertEqual(a['active'][0,14:].sum(),0);self.assertClose(a['mass']@a['inverse'],np.eye(20)[None],atol=2e-12);self.assertEqual(s.capture().count[0],0)
if __name__=='__main__':unittest.main()
