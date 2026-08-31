#include "articulated_v1.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);std::abort();} } while(0)
struct Snapshot {
 std::array<double,16> q{};std::array<float,14> v{},warm{};std::array<uint64_t,2> count{};
 av1_snapshot d{sizeof(d),AV1_ABI,2,1,0,q.data(),v.data(),warm.data(),count.data()};
};
int main(){
 // Synthetic free root plus one hinge only; never advances the duck fixture.
 av1_body bodies[3]={{0,{0,0,0}},{2,{.2,.3,.4}},{1,{.1,.15,.2}}};
 av1_hinge h{};h.parent=1;h.parent_anchor[0]=.2;h.child_anchor[1]=.1;h.axis_parent[2]=1;h.reference_xyzw[3]=1;h.armature=.027f;h.friction_d0=.9f;h.friction_dwidth=.95f;h.friction_timeconst=.02f;
 double ref[8]={0,0,0,0,0,0,1,0};av1_model model{sizeof(model),AV1_ABI,3,1,0,0,bodies,&h,{.1,.05,.2,0,0,0,1},ref};
 double q[16]={0,0,1,0,0,0,1,.2, 1,2,3,0,0,0,1,-.3};float v[14]={};double g[6]={0,0,-9.81,0,0,-9.81};
 av1_registration registration{sizeof(registration),AV1_ABI,2,0,&model,q,v,g};av1_scene* scene=nullptr;CHECK(av1_create(&registration,&scene)==0);
 Snapshot initial;CHECK(av1_capture(scene,&initial.d)==0);
 double pose[21],vel[18],J[126],M[49],bias[7],acc[18];av1_evaluation eval{sizeof(eval),AV1_ABI,pose,vel,J,M,bias,acc,0,0};CHECK(av1_read(scene,0,&eval)==0);CHECK(std::abs(M[0]-3)<1e-12);CHECK(std::abs(bias[2]-29.43)<1e-12);
 float direction[7]={0,0,0,0,0,0,1},dv[7];double bdv[18],effective;CHECK(av1_response(scene,0,direction,dv,bdv,&effective)==0);CHECK(effective>0);CHECK(std::abs(dv[5])>.001);CHECK(std::abs(bdv[11])>.001);
 float target[2]={0,0};av1_step step{};step.struct_size=sizeof(step);step.version=AV1_ABI;step.max_iterations=512;step.dt=.002f;step.tolerance=1e-6f;step.target_position=target;step.target_velocity=target;
 av1_stage *stage=nullptr,*stale=nullptr;CHECK(av1_prepare(scene,&step,&stage)==0);CHECK(av1_prepare(scene,&step,&stale)==0);
 Snapshot before;CHECK(av1_capture(scene,&before.d)==0);CHECK(before.q==initial.q&&before.v==initial.v&&before.count==initial.count);
 Snapshot candidate;CHECK(av1_stage_capture(stage,&candidate.d)==0);CHECK(candidate.count[0]==1);CHECK(candidate.q[2]<initial.q[2]);CHECK(std::abs(candidate.v[2]+.01962f)<1e-6);CHECK(std::abs(candidate.v[6])<1e-6);
 // Caller storage mutation cannot alter owned topology, reset state or gravity.
 q[2]=99;g[2]=99;bodies[1].mass=99;h.parent_anchor[0]=99;
 CHECK(av1_validate_commit(scene,stage)==0);CHECK(av1_commit(scene,stage)==0);CHECK(av1_validate_commit(scene,stale)==AV1_STALE);CHECK(av1_commit(scene,stale)==AV1_STALE);
 av1_stage_destroy(stale);av1_stage_destroy(stage);
 Snapshot bad;CHECK(av1_capture(scene,&bad.d)==0);bad.q[3]=2;CHECK(av1_restore(scene,&bad.d)==AV1_INVALID);Snapshot after_bad;CHECK(av1_capture(scene,&after_bad.d)==0);CHECK(after_bad.q==candidate.q);
 uint8_t mask[2]={1,0};CHECK(av1_reset_masked(scene,mask,2)==0);Snapshot reset;CHECK(av1_capture(scene,&reset.d)==0);CHECK(reset.q[2]==initial.q[2]);CHECK(reset.q[10]==candidate.q[10]);CHECK(reset.count[0]==0&&reset.count[1]==1);
 CHECK(av1_restore(scene,&initial.d)==0);Snapshot replay;CHECK(av1_capture(scene,&replay.d)==0);CHECK(replay.q==initial.q&&replay.v==initial.v&&replay.warm==initial.warm&&replay.count==initial.count);
 av1_destroy(scene);std::puts("PASS synthetic native articulated mass/bias/fullresponse/root integration/ownership/staging/reset");
}
