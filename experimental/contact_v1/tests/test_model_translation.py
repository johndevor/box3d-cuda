"""Native f32 geometry/velocity gates on frozen robot records; no robot step."""
import copy
import ctypes as C
import math
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import model_translation as m

LIBRARY=Path(os.environ.get('BOX3D_CONTACT_V1_LIBRARY','/tmp/box3d-contact-v1-local/libcontact_v1.dylib'))
UROUND=2**-24
METRICS={}
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.lib=m.library(LIBRARY);cls.model=m.Model(cls.lib)
    def test_all_recorded_native_frames_anchors_coordinates_supports(self):
        model=self.model;worst_pos=worst_q=worst_anchor=worst_support=worst_qdot=0.
        for frame,expected in zip(model.record['frames'],model.golden['frames']):
            bodies=model.bodies(frame)
            for body,pose in zip(bodies,expected['principal_poses_xyzw']):
                err=max(abs(a-b) for a,b in zip(body.state[:3],pose[:3]));worst_pos=max(worst_pos,err)
                # gamma128 absolute operand bound, not 1e-9 double tolerance.
                self.assertLessEqual(err,128*UROUND*(1+math.dist(pose[:3],[0]*3)))
                qerr=min(max(abs(a-s*b) for a,b in zip(body.state[3:7],pose[3:7])) for s in [-1,1]);self.assertLessEqual(qerr,128*UROUND)
            for j,truth,rate in zip(model.joints,frame['joint_q'],frame['joint_qdot']):
                out=(m.F*5)();m.ok(self.lib.bcv1_joint_geometry(bodies[j['parent']].state,bodies[j['child']].state,m.arr(j['parent_anchor']),m.arr(j['child_anchor']),m.arr(j['axis_parent']),m.arr(j['reference_xyzw']),out))
                qe=abs(out[0]-truth);ve=abs(out[1]-rate);ae=math.sqrt(sum(x*x for x in out[2:]));worst_q=max(worst_q,qe);worst_qdot=max(worst_qdot,ve);worst_anchor=max(worst_anchor,ae)
                self.assertLessEqual(qe,256*UROUND*(1+abs(truth)));self.assertLessEqual(ve,256*UROUND*(1+abs(rate)));self.assertLessEqual(ae,256*UROUND)
            for col,truth in zip(model.mapping['colliders'][:2],expected['foot_supports']):
                out=(m.F*4)();index=m.U();b=col['body'];m.ok(self.lib.bcv1_support(C.byref(model.shapes[b]),bodies[b].state,m.arr([0,0,-1]),out,C.byref(index)))
                err=abs(-out[3]-truth['captured_distance_m']);worst_support=max(worst_support,err)
                radius=max(math.dist(v,[0]*3) for v in col['vertices']);scale=math.dist(list(bodies[b].state[:3]),[0]*3)+radius+math.dist(col['local_pose_xyzw'][:3],[0]*3)
                self.assertLessEqual(err,128*UROUND*(scale+.001))
        METRICS.update(frames=501,body_poses=8016,joint_comparisons=7014,supports=1002,max_position_m=worst_pos,max_joint_rad=worst_q,max_joint_velocity=worst_qdot,max_anchor_m=worst_anchor,max_support_m=worst_support,robot_steps=0)
    def test_native_registration_actual_501_states_only_queries(self):
        handle=self.model.register(self.model.record['frames'])
        try:
            out=(m.Manifold*(501*3))();m.ok(self.lib.bcv1_query(handle,out));self.assertEqual(len(out),1503)
            self.assertTrue(all(x.count<=4 for x in out))
        finally:self.lib.bcv1_destroy(handle)
    def test_origin_to_com_velocity_against_kinematic_derivative(self):
        # Synthetic generalized velocities excite every axis. Differentiate
        # source FK positions in f64; analytic native COM velocity is independent
        # of this central-difference calculation. No simulator or time step.
        worst=0
        for frame_index in [0,111,250,500]:
            f=copy.deepcopy(self.model.record['frames'][frame_index]);f['qvel'][:6]=[.4,-.2,.3,.7,-.3,.5];f['joint_qdot']=[.1+.01*i for i in range(14)]
            h=1e-5;minus=copy.deepcopy(f);plus=copy.deepcopy(f)
            for sign,other in [(-1,minus),(1,plus)]:
                other['base_pose'][:3]=[x+sign*h*v for x,v in zip(f['base_pose'][:3],f['qvel'][:3])]
                omega=f['qvel'][3:6];speed=math.sqrt(m.dot(omega,omega));dq=(*m.scale(omega,math.sin(sign*h*speed/2)/speed),math.cos(sign*h*speed/2));q=m.mul(m.xyzw(f['base_pose'][3:]),dq);other['base_pose'][3:]=[q[3],*q[:3]]
                other['joint_q']=[q+sign*h*v for q,v in zip(f['joint_q'],f['joint_qdot'])]
            sm=self.model.source_states(minus);sp=self.model.source_states(plus);bodies=self.model.bodies(f)
            for b in self.model.body_map[1:]:
                def com(states):p,q,_,_=states[b['name']];return m.add(p,m.rot(q,b['source_COM']))
                fd=m.scale(m.sub(com(sp),com(sm)),1/(2*h));err=max(abs(x-y) for x,y in zip(fd,bodies[b['id']].state[7:10]));worst=max(worst,err);self.assertLessEqual(err,256*UROUND*(1+max(abs(v) for v in fd)))
        METRICS['max_velocity_fd_mps']=worst
    def test_mapping_negative_controls_discriminate(self):
        model=self.model;b=model.body_map[6];state=model.source_states(model.record['frames'][0])[b['name']];source=m.arr([x for item in state for x in item]);bad=(m.F*13)();good=model.bodies(model.record['frames'][0])[6]
        m.ok(self.lib.bcv1_to_principal(source,m.arr([0,0,0]),m.arr(m.xyzw(b['inertial_quaternion_wxyz'])),bad));self.assertGreater(math.dist(bad[:3],good.state[:3]),.01)
        c=model.mapping['colliders'][0];direction=model.golden['negative_controls'][4]['direction'];exact=max(m.dot(v,direction) for v in c['vertices']);obb=sum(max(v[k] for v in c['vertices'])*direction[k] if direction[k]>=0 else min(v[k] for v in c['vertices'])*direction[k] for k in range(3));self.assertGreater(obb-exact,.005)
        self.assertEqual(model.mapping['source_name_to_native_id']['base'],model.mapping['source_name_to_native_id']['trunk_assembly']);self.assertEqual(len(model.shapes),16);self.assertEqual(model.mu,[.6,.6,1.])
    def test_generalized_contact_row_virtual_power_and_velocity(self):
        f=self.model.record['frames'][123];J=self.model.spatial_jacobian(f);bodies=self.model.bodies(f);v=f['qvel'][:6]+f['joint_qdot'];dofs=20
        for b in range(16):
            for k in range(6):self.assertLessEqual(abs(sum(J[(b*6+k)*20+j]*v[j] for j in range(20))-bodies[b].state[7+k]),4e-6)
        point=[.04,.02,.001];direction=m.normalize([.2,-.3,1]);row=(m.F*20)();m.ok(self.lib.bcv1_contact_row(16,20,bodies,6,15,m.arr(point),m.arr(direction),m.arr(J),row))
        def pointv(b):s=list(bodies[b].state);return m.add(s[7:10],m.cross(s[10:13],m.sub(point,s[:3])))
        speed=m.dot(direction,m.sub(pointv(15),pointv(6)));self.assertLessEqual(abs(sum(row[j]*v[j] for j in range(dofs))-speed),4e-6)
        # Equal/opposite signed wrench virtual work, including COM lever arm.
        virtual=[.03*(j-7) for j in range(20)];work=0
        for b,sgn in [(6,-1),(15,1)]:
            bv=[sum(J[(b*6+k)*20+j]*virtual[j] for j in range(20)) for k in range(6)];torque=m.cross(m.sub(point,list(bodies[b].state[:3])),direction);work+=sgn*(m.dot(bv[:3],direction)+m.dot(bv[3:],torque))
        self.assertLessEqual(abs(sum(row[j]*virtual[j] for j in range(20))-work),4e-6)
    def test_near_unit_roundtrip_and_failed_output_canary(self):
        q=m.scale(m.normalize([.2,.3,.4,.5]),1+2e-6);pc=m.scale(m.normalize([-.3,.4,.1,.2]),1-2e-6);s=m.arr([1,2,3,*q,4,5,6,.7,.3,.2]);com=m.arr([.1,-.2,.3]);out=(m.F*13)();back=(m.F*13)();m.ok(self.lib.bcv1_to_principal(s,com,m.arr(pc),out));m.ok(self.lib.bcv1_from_principal(out,com,m.arr(pc),back))
        for i in [0,1,2,7,8,9,10,11,12]:self.assertLessEqual(abs(back[i]-s[i]),2e-6)
        out[:]=[42]*13;s[8]=float('nan');self.assertNotEqual(self.lib.bcv1_to_principal(s,com,m.arr(pc),out),0);self.assertEqual(list(out),[42]*13)
    def test_exact_convex_feet_isolated_contact_response(self):
        # One free foot / fixed plane, then two free feet. These are isolated
        # collision fixtures, not a stepped robot or altered robot evidence.
        model=self.model;record=model.bodies(model.record['frames'][0]);shapes=(m.Shape*2)();bodies=(m.Body*2)();shapes[0]=model.shapes[6];shapes[1]=model.shapes[0];bodies[0]=record[6];bodies[1]=record[0]
        support=(m.F*4)();idx=m.U();m.ok(self.lib.bcv1_support(C.byref(shapes[0]),bodies[0].state,m.arr([0,0,-1]),support,C.byref(idx)));bodies[0].state[2]+=support[3];bodies[0].state[7:]=[0,0,-.1,0,0,0]
        pairs=(m.Pair*1)(m.Pair(3,0,1));gravity=m.arr([0,0,0]);friction=m.arr([.6]);handle=C.c_void_p();desc=m.Registration(1,1,2,1,shapes,pairs,bodies,gravity,friction);m.ok(self.lib.bcv1_create(C.byref(desc),C.byref(handle)))
        try:
            out=(m.Manifold*1)();m.ok(self.lib.bcv1_query(handle,out));self.assertGreater(out[0].count,0);m.ok(self.lib.bcv1_step(handle,.0001,64));final=(m.Body*2)();cache=(m.Manifold*1)();m.ok(self.lib.bcv1_read(handle,final,cache,None));self.assertGreater(sum(p.normal_impulse for p in cache[0].points),0);self.assertGreater(final[0].state[9],bodies[0].state[9])
        finally:self.lib.bcv1_destroy(handle)
        shapes[1]=model.shapes[15];bodies[0]=record[6];bodies[1]=record[15]
        for b in bodies:b.state[:]=[0,0,0,0,0,0,1,0,0,0,0,0,0]
        upper=max(v[0] for v in shapes[0].vertices[:18]);lower=min(v[0] for v in shapes[1].vertices[:18]);bodies[1].state[0]=upper-lower-1e-5;bodies[0].state[7]=.1;bodies[1].state[7]=-.1;friction[0]=1;desc=m.Registration(1,1,2,1,shapes,pairs,bodies,gravity,friction);m.ok(self.lib.bcv1_create(C.byref(desc),C.byref(handle)))
        try:
            # Bounding X intervals overlap but exact tapered hulls do NOT.
            # Retained as an anti-OBB/false-contact negative control.
            out=(m.Manifold*1)();m.ok(self.lib.bcv1_query(handle,out));self.assertEqual(out[0].count,0)
        finally:self.lib.bcv1_destroy(handle)
        bodies[1].state[0]=.001  # deliberately overlapping isolated hull fixture
        m.ok(self.lib.bcv1_create(C.byref(desc),C.byref(handle)))
        try:
            out=(m.Manifold*1)();m.ok(self.lib.bcv1_query(handle,out));self.assertGreater(out[0].count,0);m.ok(self.lib.bcv1_step(handle,.0001,64));cache=(m.Manifold*1)();final=(m.Body*2)();m.ok(self.lib.bcv1_read(handle,final,cache,None));self.assertGreater(sum(p.normal_impulse for p in cache[0].points),0);self.assertLess(final[0].state[7],.1);self.assertGreater(final[1].state[7],-.1)
        finally:self.lib.bcv1_destroy(handle)

if __name__=='__main__':
    import json
    result=unittest.main(exit=False,verbosity=2).result
    print(json.dumps({'metrics':METRICS,'cuda':False,'full_robot_steps':0},sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
