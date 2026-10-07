"""Pinned OpenDuck lowerer into experimental native contact-v1. No simulation.

Reuses the accepted source-to-principal eigendecomposition, not a new mass fit.
XML FK and spatial velocity propagation remain in source body frames; C++
converts each to principal COM state. MuJoCo free-joint qvel has WORLD linear
and LOCAL angular components (3.3.7 overview, Floating objects).
"""
import ctypes as C
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

GOLDEN_SHA='e52ba7d0f79434499d8fb6c2d611eb46ee12e2f32cb36258b38cd22959d0b08b'
RECORD_SHA='a6d578064b433e730612d7144742b706471e63a37e3c81bcbc24acb7a7203a58'
XML_SHA='968b18de4e3f55b31252155f52779fa490989f5da92bc9b308e0bb4e81d6bb5c'
DEFAULT_REFERENCE=Path('/Users/john/Code/box3d-cuda-voxel-gate-c1/evidence')
F=C.c_float;U=C.c_uint32;P=C.POINTER(F)
class Body(C.Structure):_fields_=[('state',F*13),('inverse_mass',F),('inverse_inertia',F*3)]
class Shape(C.Structure):_fields_=[('caller_id',U),('kind',U),('vertex_count',U),('fixed',U),('vertices',(F*3)*32),('plane_normal',F*3),('plane_offset',F)]
class Pair(C.Structure):_fields_=[('caller_id',U),('body_a',U),('body_b',U)]
class Point(C.Structure):_fields_=[('feature',C.c_uint64),('point',F*3),('depth',F),('normal_impulse',F),('tangent_impulse',F*2)]
class Manifold(C.Structure):_fields_=[('count',U),('normal',F*3),('tangent1',F*3),('tangent2',F*3),('points',Point*4)]
class Registration(C.Structure):_fields_=[('version',U),('environments',U),('bodies',U),('pairs',U),('shapes',C.POINTER(Shape)),('contact_pairs',C.POINTER(Pair)),('initial',C.POINTER(Body)),('gravity_xyz',P),('pair_friction',P)]
def arr(values):return (F*len(values))(*values)
def ok(status):
    if status:raise RuntimeError('native status '+str(status))
def library(path):
    lib=C.CDLL(str(Path(path).resolve()))
    signatures={'bcv1_to_principal':[P,P,P,P],'bcv1_from_principal':[P,P,P,P],
      'bcv1_bake_convex':[P,P,U,P],'bcv1_support':[C.POINTER(Shape),P,P,P,C.POINTER(U)],
      'bcv1_joint_geometry':[P,P,P,P,P,P,P],
      'bcv1_contact_row':[U,U,C.POINTER(Body),U,U,P,P,P,P],
      'bcv1_create':[C.POINTER(Registration),C.POINTER(C.c_void_p)],
      'bcv1_destroy':[C.c_void_p], 'bcv1_query':[C.c_void_p,C.POINTER(Manifold)],
      'bcv1_read':[C.c_void_p,C.POINTER(Body),C.POINTER(Manifold),C.POINTER(C.c_double)],
      'bcv1_step':[C.c_void_p,F,U]}
    for name,args in signatures.items():getattr(lib,name).argtypes=args;getattr(lib,name).restype=None if name=='bcv1_destroy' else C.c_int
    return lib
def add(a,b):return tuple(x+y for x,y in zip(a,b))
def sub(a,b):return tuple(x-y for x,y in zip(a,b))
def scale(a,s):return tuple(x*s for x in a)
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def normalize(a):return scale(a,1/math.sqrt(dot(a,a)))
def mul(a,b):return (*add(add(scale(b[:3],a[3]),scale(a[:3],b[3])),cross(a[:3],b[:3])),a[3]*b[3]-dot(a[:3],b[:3]))
def rot(q,v):
    t=scale(cross(q[:3],v),2);return add(add(v,scale(t,q[3])),cross(q[:3],t))
def xyzw(q):return (*q[1:],q[0])
def numbers(s,default):return tuple(float(x) for x in (s or default).split())
def read_pin(path,sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('pin mismatch '+str(path))
    return raw
class Model:
    def __init__(self,lib,reference=DEFAULT_REFERENCE):
        self.lib=lib;self.reference=Path(reference)
        self.golden=json.loads(read_pin(self.reference/'open-duck-native-compat-v1/geometry-goldens.json',GOLDEN_SHA))
        self.record=json.loads(read_pin(self.reference/'open-duck-zero-hold-cpu-v1/cpu-result.json',RECORD_SHA))
        self.mapping=self.golden['mapping'];self.body_map=self.mapping['bodies'];self.joints=self.mapping['joints']
        self.xml=ET.fromstring(read_pin(self.reference/'open-duck-zero-hold-cpu-v1/model/open_duck_mini_v2.xml',XML_SHA))
        self.nodes=[]
        def visit(el,parent):
            name=el.attrib['name'];joint=el.find('joint');free=el.find('freejoint') is not None
            node={'name':name,'parent':parent,'p':numbers(el.get('pos'),'0 0 0'),'q':normalize(xyzw(numbers(el.get('quat'),'1 0 0 0'))),'free':free,'joint':None}
            if joint is not None:
                if joint.get('type','hinge')!='hinge':raise ValueError('non-hinge')
                node['joint']={'name':joint.attrib['name'],'axis':normalize(numbers(joint.get('axis'),'0 0 1')),'anchor':numbers(joint.get('pos'),'0 0 0'),'ref':float(joint.get('ref','0'))}
            self.nodes.append(node)
            for child in el.findall('body'):visit(child,name)
        for el in self.xml.find('worldbody').findall('body'):visit(el,None)
        if [n['joint']['name'] for n in self.nodes if n['joint']]!=[j['name'] for j in self.joints]:raise ValueError('joint order')
        if self.mapping['virtual_alias']!={'base':'trunk_assembly'} or len(self.body_map)!=16:raise ValueError('virtual root')
        trunk=next(n for n in self.nodes if n['name']=='trunk_assembly')
        if trunk['parent']!='base' or trunk['joint'] or trunk['p']!=(0,0,0) or trunk['q']!=(0,0,0,1):raise ValueError('nonidentity root weld')
        for b in self.body_map[1:]:
            el=self.xml.find(".//body[@name='"+b['name']+"']/inertial")
            if el is None or float(el.attrib['mass'])!=b['mass'] or tuple(b['source_COM'])!=numbers(el.get('pos'),'0 0 0'):raise ValueError('authored mass/COM')
        if abs(sum(b['mass'] for b in self.body_map)-2.1071407)>1e-12:raise ValueError('mass')
        self.shapes=(Shape*16)()
        for b,s in zip(self.body_map,self.shapes):s.caller_id=b['id'];s.fixed=int(b['motion']=='fixed')
        for c in self.mapping['colliders']:
            s=self.shapes[c['body']]
            if c['shape']=='infinite_plane':s.kind=2;s.plane_normal[:]=c['normal'];continue
            if len(c['vertices'])!=18:raise ValueError('exact convex count')
            s.kind=1;s.vertex_count=18;out=(F*54)();ok(lib.bcv1_bake_convex(arr(c['local_pose_xyzw']),arr([x for v in c['vertices'] for x in v]),18,out))
            for i in range(18):s.vertices[i][:]=out[i*3:i*3+3]
        self.pairs=(Pair*3)()
        for p,x in zip(self.mapping['pairs'],self.pairs):
            x.caller_id=p['id'];x.body_a=self.mapping['colliders'][p['colliders'][0]]['body'];x.body_b=self.mapping['colliders'][p['colliders'][1]]['body']
        self.mu=[p['sliding_friction'] for p in self.mapping['pairs']]
        if self.mu!=[.6,.6,1.]:raise ValueError('effective pair material')
    def source_states(self,frame):
        """Analytic FK + origin velocities, no time advancement or simulator."""
        qmap={j['name']:(frame['joint_q'][i],frame['joint_qdot'][i]) for i,j in enumerate(self.joints)};states={}
        for n in self.nodes:
            if n['free']:
                p=tuple(frame['base_pose'][:3]);q=normalize(xyzw(frame['base_pose'][3:]));v=tuple(frame['qvel'][:3]);w=rot(q,frame['qvel'][3:6])
            else:
                pp,pq,pv,pw=states[n['parent']];p=add(pp,rot(pq,n['p']));q=normalize(mul(pq,n['q']));v=add(pv,cross(pw,sub(p,pp)));w=pw
                if n['joint']:
                    j=n['joint'];angle,rate=qmap[j['name']];half=(angle-j['ref'])/2;qr=(*scale(j['axis'],math.sin(half)),math.cos(half));anchor=add(p,rot(q,j['anchor']));va=add(v,cross(w,sub(anchor,p)));axis=rot(q,j['axis']);q=normalize(mul(q,qr));p=sub(anchor,rot(q,j['anchor']));w=add(w,scale(axis,rate));v=sub(va,cross(w,sub(anchor,p)))
            states[n['name']]=(p,q,v,w)
        return states
    def bodies(self,frame):
        source=self.source_states(frame);out=(Body*16)()
        for b,r in zip(self.body_map,out):
            r.inverse_mass=b['inverse_mass'];r.inverse_inertia[:]=b['inverse_inertia_local']
            if b['id']==0:r.state[6]=1;continue
            p,q,v,w=source[b['name']];values=arr((*p,*q,*v,*w))
            ok(self.lib.bcv1_to_principal(values,arr(b['source_COM']),arr(xyzw(b['inertial_quaternion_wxyz'])),r.state))
        return out
    def register(self,frames):
        if not frames or len(frames)>4096:raise ValueError('environments')
        batch=(Body*(len(frames)*16))()
        for e,f in enumerate(frames):
            for b,r in enumerate(self.bodies(f)):batch[e*16+b]=r
        gravity=arr([x for _ in frames for x in [0,0,-9.81]]);mu=arr(self.mu*len(frames))
        desc=Registration(1,len(frames),16,3,self.shapes,self.pairs,batch,gravity,mu);handle=C.c_void_p()
        ok(self.lib.bcv1_create(C.byref(desc),C.byref(handle)));return handle
    def spatial_jacobian(self,frame):
        """[B,6,20], linear-first world Jacobian; free angular DOFs local.

        Build by analytic velocity propagation on unit generalized velocities;
        no finite differencing, force application or native physics stepping.
        """
        import copy
        result=[0.]*(16*6*20)
        for j in range(20):
            f=copy.deepcopy(frame);f['qvel']=[0.]*20;f['joint_qdot']=[0.]*14
            if j<6:f['qvel'][j]=1.
            else:f['joint_qdot'][j-6]=1.
            bodies=self.bodies(f)
            for b in range(16):
                for k in range(6):result[(b*6+k)*20+j]=bodies[b].state[7+k]
        return result
