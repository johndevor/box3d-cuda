// SPDX-License-Identifier: MIT
#include "contact_v1.h"
#include "contact_math.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace bcv1_math;
int checks=0;
constexpr float EPSILON=2e-6f;
void check(bool b,const char* what){++checks;if(!b)throw std::runtime_error(what);}
void near(float a,float b,float tol,const char* what){check(std::isfinite(a)&&std::fabs(a-b)<=tol,what);}
bcv1_shape box(uint32_t id,float h=.5f){bcv1_shape s{};s.caller_id=id;s.kind=BCV1_CONVEX;s.vertex_count=8;for(int i=0;i<8;i++)for(int k=0;k<3;k++)s.vertices[i][k]=(i&(1<<k))?h:-h;return s;}
bcv1_shape plane(uint32_t id){bcv1_shape s{};s.caller_id=id;s.kind=BCV1_PLANE;s.fixed=1;s.plane_normal[2]=1;return s;}
bcv1_body body(V p={},bool fixed=false){bcv1_body b{};save(p,b.state);b.state[6]=1;b.inverse_mass=fixed?0:1;for(float& v:b.inverse_inertia)v=fixed?0:1;return b;}
struct Fixture {
 std::vector<bcv1_shape> shapes{box(17),plane(99)};
 std::vector<bcv1_pair> pairs{{123,0,1}};
 std::vector<bcv1_body> bodies;
 std::vector<float> g,mu;
 bcv1_scene* s{};uint32_t E;
 explicit Fixture(uint32_t e=1):E(e){for(uint32_t i=0;i<E;i++){bodies.push_back(body({0,0,.5}));bodies.push_back(body({},true));g.insert(g.end(),{0,0,0});mu.push_back(.6f);}}
 bcv1_status create(){bcv1_registration r{1,E,2,1,shapes.data(),pairs.data(),bodies.data(),g.data(),mu.data()};return bcv1_create(&r,&s);}
 ~Fixture(){bcv1_destroy(s);}
};
void geometry(){
 {Fixture f;check(f.create()==BCV1_OK,"cube register");bcv1_manifold m;check(bcv1_query(f.s,&m)==0,"cube plane query");check(m.count==4,"coplanar box manifold four");near(m.normal[2],-1,1e-6,"A convex normal towardplane");for(uint32_t i=0;i<m.count;i++){near(m.points[i].depth,0,EPSILON,"zero depth");near(m.points[i].point[2],0,1e-6,"infinite plane location");}}
 {Fixture f;f.bodies[0].state[0]=100;check(f.create()==0,"far horizontal box");bcv1_manifold m;check(bcv1_query(f.s,&m)==0&&m.count==4,"plane infinite no finite box proxy");}
 {Fixture f;f.bodies[0].state[2]=-.75f;check(f.create()==0,"submerged register");bcv1_manifold m;check(bcv1_query(f.s,&m)==0&&m.count>0,"submerged responds");float max=0;for(uint32_t i=0;i<m.count;i++)max=std::fmax(max,m.points[i].depth);near(max,1.25,1e-6,"submerged depth");}
 {Fixture f;f.shapes[1]=box(99,2);f.shapes[1].fixed=1;f.bodies[0].state[2]=0;check(f.create()==0,"containment register");bcv1_manifold m;check(bcv1_query(f.s,&m)==0&&m.count>0,"containment manifold");near(m.points[0].depth,2.5,5e-6,"signed exit containment depth not intersectionwidth");}
 {Fixture f;f.shapes[1]=box(99);f.shapes[1].fixed=1;f.bodies[0]=body({0,0,0});f.bodies[1]=body({1.01f,0,0},true);check(f.create()==0,"separatedconvex register");bcv1_manifold m;check(bcv1_query(f.s,&m)==0&&m.count==0,"convex separation");}
 {Fixture f;f.shapes[1]=box(99);f.shapes[1].fixed=1;f.bodies[0]=body({0,0,0});f.bodies[1]=body({1,0,0},true);check(f.create()==0,"facecontact register");bcv1_manifold m;check(bcv1_query(f.s,&m)==0&&m.count==4,"faceclipping four");near(m.normal[0],1,1e-6,"face normal");for(uint32_t i=0;i<m.count;i++)near(m.points[i].point[0],.5,1e-6,"facepoints");}
}
void dynamics(){
 Fixture f;f.bodies[0].state[9]=-1;check(f.create()==0,"impact register");check(bcv1_step(f.s,.002f,64)==0,"impact step");bcv1_body b[2];bcv1_manifold m;double t;check(bcv1_read(f.s,b,&m,&t)==0,"impact read");near(b[0].state[9],0,2e-5,"inelastic impact");check(m.count==4,"impact fourpoints");float sum=0;for(uint32_t i=0;i<m.count;i++){sum+=m.points[i].normal_impulse;float jt=std::hypot(m.points[i].tangent_impulse[0],m.points[i].tangent_impulse[1]);check(jt<=.6f*m.points[i].normal_impulse+1e-6,"diskbound");}near(sum,1,2e-5,"normal impulse momentum");near(float(t),.002f,1e-8,"clock");
}
void lifecycle(){
 Fixture f(2);f.g={0,0,-9.81f,0,0,-3};f.mu={.6f,1};f.bodies[0].state[7]=2;f.bodies[2].state[7]=-1;check(f.create()==0,"E2 create");bcv1_snapshot* snap=nullptr;check(bcv1_capture(f.s,&snap)==0,"capture");
 check(bcv1_step(f.s,.002f,32)==0,"E2 step");bcv1_body before[4],after[4];bcv1_manifold cb[2],ca[2];double tb[2],ta[2];bcv1_read(f.s,before,cb,tb);uint8_t mask[2]={255,0};check(bcv1_restore(f.s,snap,mask)==0,"maskedrestore");bcv1_read(f.s,after,ca,ta);check(std::memcmp(&after[2],&before[2],2*sizeof(bcv1_body))==0,"peer body bitexact");check(std::memcmp(&ca[1],&cb[1],sizeof(bcv1_manifold))==0&&ta[1]==tb[1],"peer cacheclock exact");check(std::memcmp(after,f.bodies.data(),2*sizeof(bcv1_body))==0&&ta[0]==0&&ca[0].count==0,"selected initial exact");
 check(bcv1_restore(f.s,snap,nullptr)==0&&bcv1_step(f.s,.002f,32)==0,"replay");bcv1_read(f.s,after,ca,ta);check(std::memcmp(after,before,sizeof(after))==0&&std::memcmp(ca,cb,sizeof(ca))==0&&std::memcmp(ta,tb,sizeof(ta))==0,"full deterministicreplay");
 check(bcv1_step(f.s,std::numeric_limits<float>::quiet_NaN(),32)==BCV1_INVALID,"invalid dt");bcv1_read(f.s,after,ca,ta);check(std::memcmp(after,before,sizeof(after))==0&&std::memcmp(ca,cb,sizeof(ca))==0,"failedstep atomic");bcv1_snapshot_destroy(snap);
 Fixture invalid;invalid.bodies[0].state[7]=9999.9f;invalid.g={100,0,0};check(invalid.create()==0,"numeric boundary scene");bcv1_body old[2],now[2];bcv1_read(invalid.s,old,nullptr,nullptr);check(bcv1_step(invalid.s,.01,1)!=BCV1_OK,"poststep boundsfail");bcv1_read(invalid.s,now,nullptr,nullptr);check(std::memcmp(old,now,sizeof(old))==0,"numericfail rollback");
}
void snapshot_parameters_and_empty_pairs(){
 Fixture source(2),target(2);
 source.g={0,0,-2,0,0,-4};source.mu={.1f,.2f};source.bodies[0].inverse_mass=.5f;source.bodies[2].inverse_mass=.25f;
 target.g={0,0,-8,0,0,-9};target.mu={.7f,1};
 for(auto f:{&source,&target}){f->bodies[0].state[7]=1;f->bodies[2].state[7]=2;check(f->create()==0,"parameter scene");}
 bcv1_snapshot* source_snap=nullptr;check(bcv1_capture(source.s,&source_snap)==0,"parametercapture");
 uint8_t mask[2]={3,0};check(bcv1_restore(target.s,source_snap,mask)==0,"crossscene selected parameters");
 check(bcv1_step(source.s,.002f,32)==0&&bcv1_step(target.s,.002f,32)==0,"parameterstep");
 bcv1_body a[4],b[4];bcv1_manifold ac[2],bc[2];double at[2],bt[2];bcv1_read(source.s,a,ac,at);bcv1_read(target.s,b,bc,bt);
 check(std::memcmp(a,b,2*sizeof(bcv1_body))==0&&std::memcmp(ac,bc,sizeof(bcv1_manifold))==0,"selected mass/gravity/mu/cache restore behavior");
 check(std::memcmp(a+2,b+2,2*sizeof(bcv1_body))!=0,"peer parameters retained");bcv1_snapshot_destroy(source_snap);
 bcv1_snapshot* warm=nullptr;check(bcv1_capture(target.s,&warm)==0&&bc[0].points[0].normal_impulse>0,"nonzero warm capture");
 check(bcv1_step(target.s,.002f,32)==0,"warm advance");bcv1_body advanced[4];bcv1_manifold advcache[2];double advclock[2];bcv1_read(target.s,advanced,advcache,advclock);
 check(bcv1_restore(target.s,warm,mask)==0,"masked warm restore");bcv1_read(target.s,a,ac,at);
 check(std::memcmp(a,b,2*sizeof(bcv1_body))==0&&std::memcmp(ac,bc,sizeof(bcv1_manifold))==0,"selected nonzero cache exact");
 check(std::memcmp(a+2,advanced+2,2*sizeof(bcv1_body))==0&&std::memcmp(ac+1,advcache+1,sizeof(bcv1_manifold))==0,"unselected advanced cache exact");bcv1_snapshot_destroy(warm);
 bcv1_shape s{};s.caller_id=4;s.kind=BCV1_NONE;bcv1_body bb=body();float g[3]={0,0,-1};bcv1_registration r{1,1,1,0,&s,nullptr,&bb,g,nullptr};bcv1_scene* empty=nullptr;check(bcv1_create(&r,&empty)==0,"P0 register");check(bcv1_query(empty,nullptr)==0&&bcv1_step(empty,.002,1)==0,"P0 free motion");bcv1_destroy(empty);
}
void invalids(){
 {Fixture f;f.shapes[0].vertex_count=33;check(f.create()==BCV1_CAPACITY&&f.s==nullptr,"vertex cap");}
 {Fixture f;for(int i=0;i<8;i++)f.shapes[0].vertices[i][2]=0;check(f.create()!=0&&f.s==nullptr,"degenerate hull");}
 {Fixture f;f.shapes[0].vertices[0][0]=NAN;check(f.create()==BCV1_INVALID,"nan shape");}
 {Fixture f;f.shapes[1].plane_normal[2]=0;check(f.create()==BCV1_INVALID,"zero plane normal");}
 {Fixture f;f.pairs[0].body_b=0;check(f.create()==BCV1_INVALID,"selfpair");}
 {Fixture f;f.bodies[1].state[7]=1;check(f.create()==BCV1_INVALID,"kinematic unsupported");}
 {Fixture f;f.bodies[0].state[6]=.7;check(f.create()==BCV1_INVALID,"nonnormal quat");}
}
void transforms(){
 float s[13]={1,2,3,0,0,.70710678118f,.70710678118f,4,5,6,0,0,2},com[3]={.5,0,0},pc[4]={.3826834324f,0,0,.9238795325f},out[13],back[13];
 check(bcv1_to_principal(s,com,pc,out)==0,"principal convert");near(out[0],1,1e-6,"COM x");near(out[1],2.5,1e-6,"COM y");near(out[7],3,1e-6,"velocity leverarm correction");near(out[8],5,1e-6,"velocityy");check(bcv1_from_principal(out,com,pc,back)==0,"inverse transform");for(int i=0;i<13;i++)near(back[i],s[i],2e-6,"roundtrip");
 s[0]=NAN;for(float& x:out)x=42;check(bcv1_to_principal(s,com,pc,out)==BCV1_INVALID,"nonfinite transform");for(float x:out)near(x,42,0,"transform atomic");
}
int main(){try{geometry();dynamics();lifecycle();snapshot_parameters_and_empty_pairs();invalids();transforms();std::cout<<"{\"status\":\"passed\",\"native_cpu_checks\":"<<checks<<",\"cuda\":false,\"full_robot_steps\":0}\n";return 0;}catch(const std::exception& e){std::cerr<<"REJECTED after "<<checks<<" checks: "<<e.what()<<"\n";return 1;}}
