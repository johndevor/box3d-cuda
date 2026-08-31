#include "articulated_v2.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"failed line %d: %s\n",__LINE__,#x);std::abort();} } while(0)
struct Snapshot {
 std::array<double,16> q{};std::array<double,14> v{};std::array<double,6> warm{};std::array<double,2> time{};std::array<uint64_t,2> count{};
 av2_snapshot d{sizeof(d),AV2_ABI,2,1,0,q.data(),v.data(),warm.data(),time.data(),count.data()};
 bool same(const Snapshot& b)const{return q==b.q&&v==b.v&&warm==b.warm&&time==b.time&&count==b.count&&d.binding==b.d.binding;}
};
int main(){
 // Synthetic two-environment free-root/one-hinge fixture. No robot trajectory.
 av1_body bodies[3]={{0,{0,0,0}},{2,{.2,.3,.4}},{1,{.1,.15,.2}}};
 av1_hinge h{};h.parent=1;h.parent_anchor[0]=.2;h.child_anchor[1]=.1;h.axis_parent[2]=1;h.reference_xyzw[3]=1;h.armature=.027f;h.friction_d0=.9f;h.friction_dwidth=.95f;h.friction_timeconst=.02f;
 double ref[8]={0,0,0,0,0,0,1,0};av1_model model{sizeof(model),AV1_ABI,3,1,0,0,bodies,&h,{.1,.05,.2,0,0,0,1},ref};
 av2_limit limit{1,0,-.4,.4,0,.02,1,{.9,.95,.001,.5,2}};
 double q[16]={0,0,1,0,0,0,1,-.4005, 1,2,3,0,0,0,1,.4005};double v[14]={0,0,0,0,0,0,-.1, 0,0,0,0,0,0,.1};double gravity[6]={0,0,-9.81,0,0,-9.81};
 av2_registration registration{sizeof(registration),AV2_ABI,2,0,&model,&limit,q,v,gravity};av2_scene* scene=nullptr;CHECK(av2_create(&registration,&scene)==0);
 Snapshot initial;CHECK(av2_capture(scene,&initial.d)==0);
 double target[2]={0,0};av2_step step{sizeof(step),AV2_ABI,.002,1e-9,1e-9,target,target,nullptr};av2_pre* pre=nullptr;CHECK(av2_prepare(scene,&step,&pre)==0);
 av2_pre_view a{};a.struct_size=sizeof(a);a.version=AV2_ABI;CHECK(av2_pre_read(pre,&a)==0);CHECK(a.dofs==7&&a.rows==3&&a.environments==2);CHECK(a.row_active[1]&&a.row_active[5]);
 Snapshot unchanged;CHECK(av2_capture(scene,&unchanged.d)==0);CHECK(unchanged.same(initial));
 std::array<double,14> contact{},post{};std::array<double,6> impulse{};
 for(unsigned e=0;e<2;e++){
  contact[e*7]=.003;contact[e*7+4]=.02;
  for(unsigned i=0;i<7;i++){post[e*7+i]=a.smooth_velocity[e*7+i];for(unsigned k=0;k<7;k++)post[e*7+i]+=a.inverse_mass[e*49+i*7+k]*contact[e*7+k];}
  unsigned row=e==0?1:2,k=e*3+row;double sign=e==0?1:-1;
  impulse[k]=std::max(0.,(a.row_target[k]-sign*post[e*7+6])/(a.inverse_mass[e*49+48]+a.row_regularizer[k]));CHECK(impulse[k]>0);
  for(unsigned i=0;i<7;i++)post[e*7+i]+=a.inverse_mass[e*49+i*7+6]*sign*impulse[k];
 }
 av2_solution solution{sizeof(solution),AV2_ABI,post.data(),impulse.data(),contact.data()};av2_stage* stage=nullptr;
 auto bad=post;bad[7]=std::numeric_limits<double>::quiet_NaN();solution.velocity=bad.data();CHECK(av2_complete(pre,&solution,&stage)==AV2_INVALID&&stage==nullptr);CHECK(av2_capture(scene,&unchanged.d)==0&&unchanged.same(initial));
 bad=post;bad[7]+=.1;CHECK(av2_complete(pre,&solution,&stage)==AV2_NO_CONVERGENCE&&stage==nullptr);
 solution.velocity=post.data();CHECK(av2_complete(pre,&solution,&stage)==0);Snapshot candidate;CHECK(av2_stage_capture(stage,&candidate.d)==0);CHECK(av2_capture(scene,&unchanged.d)==0&&unchanged.same(initial));
 for(unsigned e=0;e<2;e++){
  CHECK(candidate.time[e]==step.dt&&candidate.count[e]==1);CHECK(std::abs(candidate.q[e*8+7])>.4);
  for(unsigned i=0;i<3;i++)CHECK(std::abs(candidate.q[e*8+i]-initial.q[e*8+i]-step.dt*post[e*7+i])<1e-14);
  double w=std::hypot(post[e*7+3],post[e*7+4],post[e*7+5]);CHECK(w>0);double angle=step.dt*w;
  for(unsigned i=0;i<3;i++)CHECK(std::abs(candidate.q[e*8+3+i]-std::sin(angle/2)*post[e*7+3+i]/w)<1e-14);
  CHECK(std::abs(candidate.q[e*8+6]-std::cos(angle/2))<1e-14);
 }
 av2_stage* stale=nullptr;CHECK(av2_complete(pre,&solution,&stale)==0);CHECK(av2_validate_commit(scene,stage)==0);CHECK(av2_commit(scene,stage)==0);CHECK(av2_commit(scene,stage)==AV2_STALE);CHECK(av2_validate_commit(scene,stale)==AV2_STALE);av2_stage_destroy(stage);av2_stage_destroy(stale);
 av2_stage* rejected=nullptr;CHECK(av2_complete(pre,&solution,&rejected)==AV2_STALE&&rejected==nullptr);av2_pre_destroy(pre);
 uint8_t mask[2]={1,0};av2_stage* reset=nullptr;CHECK(av2_prepare_reset(scene,mask,2,&reset)==0);Snapshot reset_state;CHECK(av2_stage_capture(reset,&reset_state.d)==0);CHECK(reset_state.time[0]==0&&reset_state.time[1]==candidate.time[1]);CHECK(reset_state.q[7]==initial.q[7]&&reset_state.q[15]==candidate.q[15]);CHECK(av2_commit(scene,reset)==0);av2_stage_destroy(reset);
 // Registration storage is no longer authoritative; restore uses owned binding.
 q[2]=99;gravity[2]=99;h.armature=99;limit.lower=-99;
 av2_stage* restore=nullptr;CHECK(av2_prepare_restore(scene,&initial.d,&restore)==0);CHECK(av2_commit(scene,restore)==0);av2_stage_destroy(restore);CHECK(av2_capture(scene,&unchanged.d)==0&&unchanged.same(initial));
 av2_destroy(scene);std::puts("PASS native full solve hook, soft lower/upper limits, exactly-once integration, private failure/reset/restore/stale transaction");
}
