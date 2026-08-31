// SPDX-License-Identifier: MIT
#include "integrated_duck_v1.h"
#include "contact_transaction_v1.h"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>
namespace {
struct Error { int status; };
void need(bool x,int code=IDV1_INVALID){if(!x)throw Error{code};}
template<class T> T desc(){T x{};x.struct_size=sizeof(T);x.version=AV2_ABI;return x;}
template<class F> int guard(F f){try{f();return 0;}catch(Error e){return e.status;}catch(const std::bad_alloc&){return IDV1_ALLOCATION;}catch(...){return IDV1_INVALID;}}
using AP=std::unique_ptr<av2_pre,decltype(&av2_pre_destroy)>;
using AS=std::unique_ptr<av2_stage,decltype(&av2_stage_destroy)>;
using CS=std::unique_ptr<bcx1_stage,decltype(&bcx1_stage_destroy)>;
struct Saved {
 av2_snapshot d{};std::vector<double> q,v,w,t;std::vector<uint64_t> count;
 Saved(uint32_t E,uint32_t J):q(E*(7+J)),v(E*(6+J)),w(E*3*J),t(E),count(E){d=desc<av2_snapshot>();d.environments=E;d.joints=J;bind();}
 void bind(){d.qpos=q.data();d.velocity=v.data();d.joint_warm_force=w.data();d.time=t.data();d.step_count=count.data();}
 Saved(const Saved& x):d(x.d),q(x.q),v(x.v),w(x.w),t(x.t),count(x.count){bind();}
};
struct Rows {
 size_t N;std::vector<double> g,target,r,lo,hi,warm;std::vector<civ1_contact> contacts;
 explicit Rows(size_t n):N(n){}
 void add(const double* row,double tar,double reg,double lower,double upper,double w){g.insert(g.end(),row,row+N);target.push_back(tar);r.push_back(reg);lo.push_back(lower);hi.push_back(upper);warm.push_back(w);}
};
void row(const av2_pre_view& pre,uint32_t e,const bcv1_pair& pair,const float* point,const float* direction,double* out){
 const size_t B=pre.bodies,N=pre.dofs;std::fill_n(out,N,0.);
 for(uint32_t body:{pair.body_a,pair.body_b}){
  const double* p=pre.body_pose+(e*B+body)*7;double arm[3]={point[0]-p[0],point[1]-p[1],point[2]-p[2]};
  double torque[3]={arm[1]*direction[2]-arm[2]*direction[1],arm[2]*direction[0]-arm[0]*direction[2],arm[0]*direction[1]-arm[1]*direction[0]};double sign=body==pair.body_a?-1.:1.;
  for(size_t n=0;n<N;n++)for(size_t k=0;k<3;k++)out[n]+=sign*(direction[k]*pre.spatial_jacobian[((e*B+body)*6+k)*N+n]+torque[k]*pre.spatial_jacobian[((e*B+body)*6+3+k)*N+n]);
 }
}
double dot3(const float* a,const float* b){return double(a[0])*b[0]+double(a[1])*b[1]+double(a[2])*b[2];}
}
struct idv1_scene {
 std::mutex lock;av2_scene* art=nullptr;bcv1_scene* contact=nullptr;
 uint32_t E{},B{},J{},N{},P{};std::vector<av1_body> body;std::vector<bcv1_pair> pair;std::vector<float> mu;
 bcv1_snapshot* initial_contact=nullptr;
 ~idv1_scene(){bcv1_snapshot_destroy(initial_contact);bcv1_destroy(contact);av2_destroy(art);}
};
struct idv1_snapshot {
 const idv1_scene* owner;Saved art;bcv1_snapshot* contact=nullptr;
 explicit idv1_snapshot(const idv1_scene& s):owner(&s),art(s.E,s.J){}
 ~idv1_snapshot(){bcv1_snapshot_destroy(contact);}
};
namespace {
std::vector<bcv1_body> bodies(const idv1_scene& s,const av2_state_view& view){
 std::vector<bcv1_body> out(size_t(s.E)*s.B);
 for(uint32_t e=0;e<s.E;e++)for(uint32_t b=0;b<s.B;b++){
  size_t i=e*s.B+b;auto& x=out[i];for(int k=0;k<7;k++)x.state[k]=float(view.body_pose[i*7+k]);for(int k=0;k<6;k++)x.state[7+k]=float(view.body_velocity[i*6+k]);
  if(b){x.inverse_mass=float(1/s.body[b].mass);for(int k=0;k<3;k++)x.inverse_inertia[k]=float(1/s.body[b].principal_inertia[k]);}
 }
 return out;
}
void commit(idv1_scene& s,av2_stage* a,bcx1_stage* c){
 need(av2_validate_commit(s.art,a)==0&&bcx1_validate_commit(s.contact,c)==0,IDV1_TRANSACTION);
 // Lock held; both validated. Both functions only swap preallocated payloads.
 // Any impossible post-validation failure is a programming invariant breach,
 // never an ordinary return claiming that a half-committed scene is usable.
 if(av2_commit(s.art,a)!=0)std::terminate();if(bcx1_commit(s.contact,c)!=0)std::terminate();
}
void reset(idv1_scene& s,const idv1_snapshot* snap,const uint8_t* mask){
 if(snap)need(snap->owner==&s);
 av2_stage* a=nullptr;bcx1_stage* c=nullptr;
 AS astage(nullptr,av2_stage_destroy);CS cstage(nullptr,bcx1_stage_destroy);
 if(snap){
  Saved merged(s.E,s.J);need(av2_capture(s.art,&merged.d)==0,IDV1_ARTICULATED);need(merged.d.binding==snap->art.d.binding,IDV1_TRANSACTION);
  for(uint32_t e=0;e<s.E;e++)if(!mask||mask[e]){std::copy_n(snap->art.q.data()+e*(7+s.J),7+s.J,merged.q.data()+e*(7+s.J));std::copy_n(snap->art.v.data()+e*s.N,s.N,merged.v.data()+e*s.N);std::copy_n(snap->art.w.data()+e*3*s.J,3*s.J,merged.w.data()+e*3*s.J);merged.t[e]=snap->art.t[e];merged.count[e]=snap->art.count[e];}
  need(av2_prepare_restore(s.art,&merged.d,&a)==0,IDV1_ARTICULATED);astage.reset(a);
 }else{std::vector<uint8_t> normalized(s.E);for(uint32_t e=0;e<s.E;e++)normalized[e]=!mask||mask[e]?1:0;need(av2_prepare_reset(s.art,normalized.data(),s.E,&a)==0,IDV1_ARTICULATED);astage.reset(a);}
 need(bcx1_prepare_restore(s.contact,snap?snap->contact:s.initial_contact,mask,&c)==0,IDV1_CONTACT);cstage.reset(c);
 auto view=desc<av2_state_view>();need(av2_stage_read(a,&view)==0,IDV1_ARTICULATED);std::vector<double> clocks(s.E);need(bcx1_stage_read(c,nullptr,nullptr,clocks.data())==0,IDV1_CONTACT);
 for(uint32_t e=0;e<s.E;e++)need(clocks[e]==view.time[e],IDV1_TRANSACTION);
 commit(s,a,c);
}
}
extern "C" {
int idv1_create(const idv1_registration* r,idv1_scene** out){if(!out)return IDV1_INVALID;*out=nullptr;return guard([&]{
 need(r&&r->articulation&&r->articulation->model&&r->shapes&&r->reserved==0&&r->pairs<=16&&(!r->pairs||(r->contact_pairs&&r->friction)));
 auto s=std::make_unique<idv1_scene>();const auto& a=*r->articulation;const auto& m=*a.model;
 need(m.bodies<=32&&m.joints<=26&&a.environments>0&&a.environments<=4096&&m.body);
 s->E=a.environments;s->B=m.bodies;s->J=m.joints;s->N=6+m.joints;s->P=r->pairs;s->body.assign(m.body,m.body+m.bodies);
 if(s->P){s->pair.assign(r->contact_pairs,r->contact_pairs+s->P);s->mu.assign(r->friction,r->friction+s->E*s->P);}
 for(uint32_t b=0;b<s->B;b++)need(r->shapes[b].fixed==(b==0));
 need(av2_create(&a,&s->art)==0,IDV1_ARTICULATED);auto state=desc<av2_state_view>();need(av2_read(s->art,&state)==0,IDV1_ARTICULATED);auto bs=bodies(*s,state);
 std::vector<float> gravity(s->E*3);for(size_t i=0;i<gravity.size();i++)gravity[i]=float(a.gravity[i]);
 bcv1_registration cr{1,s->E,s->B,s->P,r->shapes,s->pair.data(),bs.data(),gravity.data(),s->mu.data()};need(bcv1_create(&cr,&s->contact)==0,IDV1_CONTACT);need(bcv1_capture(s->contact,&s->initial_contact)==0,IDV1_CONTACT);*out=s.release();
 });}
void idv1_destroy(idv1_scene* s){delete s;}
int idv1_step(idv1_scene* s,const av2_step* step,uint32_t max_iterations,double tolerance,idv1_diagnostic* diagnostic){return guard([&]{
 need(s&&step&&diagnostic);std::lock_guard<std::mutex> lock(s->lock);std::fill_n(diagnostic,s->E,idv1_diagnostic{});
 for(uint32_t e=0;e<s->E;e++){diagnostic[e].environment=e;diagnostic[e].phase=1;}
 av2_pre* rawpre=nullptr;int rc=av2_prepare(s->art,step,&rawpre);if(rc){diagnostic[0].native_status=rc;throw Error{IDV1_ARTICULATED};}AP pre(rawpre,av2_pre_destroy);auto p=desc<av2_pre_view>();need(av2_pre_read(pre.get(),&p)==0,IDV1_ARTICULATED);
 const size_t N=s->N,JRows=3*s->J;std::vector<bcv1_manifold> manifolds(s->E*s->P),old(s->E*s->P);
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=2;
 need(bcv1_query(s->contact,manifolds.data())==0&&bcv1_read(s->contact,nullptr,old.data(),nullptr)==0,IDV1_CONTACT);
 std::vector<double> v(s->E*N),ji(s->E*JRows),ci(s->E*N,0),g(N);
 for(uint32_t e=0;e<s->E;e++){
  auto& d=diagnostic[e];d.phase=3;Rows rows(N);
  for(size_t r=0;r<JRows;r++){size_t k=e*JRows+r;rows.add(p.row_jacobian+k*N,p.row_target[k],p.row_regularizer[k],p.row_lower[k],p.row_upper[k],p.row_warm_impulse[k]);if(r>=s->J&&p.row_active[k])d.active_limits++;}
  for(uint32_t pair=0;pair<s->P;pair++){
   auto& m=manifolds[e*s->P+pair];const auto& previous=old[e*s->P+pair];double mu=s->mu[e*s->P+pair];
   for(uint32_t k=0;k<m.count;k++){
    auto& x=m.points[k];double warm[3]{};
    if(dot3(m.normal,previous.normal)>.98)for(uint32_t q=0;q<previous.count;q++){
     const auto& y=previous.points[q];double dist=0;for(int a=0;a<3;a++)dist+=(double(x.point[a])-y.point[a])*(double(x.point[a])-y.point[a]);
     if(x.feature==y.feature&&dist<.0004){warm[0]=y.normal_impulse;float t[3];for(int a=0;a<3;a++)t[a]=previous.tangent1[a]*y.tangent_impulse[0]+previous.tangent2[a]*y.tangent_impulse[1];warm[1]=dot3(t,m.tangent1);warm[2]=dot3(t,m.tangent2);break;}
    }
    rows.contacts.push_back({uint32_t(rows.target.size()),mu});
    const float* direction[3]={m.normal,m.tangent1,m.tangent2};
    for(int a=0;a<3;a++){row(p,e,s->pair[pair],x.point,direction[a],g.data());double target=a==0?std::min(1.,.2*std::max(0.,double(x.depth)-2e-6)/step->dt):0;rows.add(g.data(),target,0,a==0?0:-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity(),warm[a]);}
    d.contact_points++;d.maximum_penetration=std::max(d.maximum_penetration,double(x.depth));
   }
  }
  std::vector<double> impulse(rows.target.size());civ1_problem problem{uint32_t(N),uint32_t(impulse.size()),uint32_t(rows.contacts.size()),max_iterations,tolerance,p.mass+e*N*N,p.smooth_velocity+e*N,rows.g.data(),rows.target.data(),rows.r.data(),rows.lo.data(),rows.hi.data(),rows.warm.data(),rows.contacts.data()};
  civ1_result result{};result.velocity=v.data()+e*N;result.impulse=impulse.data();rc=civ1_solve(&problem,&result);
  if(rc){d.native_status=rc;throw Error{IDV1_SOLVER};}
  d.iterations=result.iterations;d.joint_residual=result.joint_residual;d.normal_residual=result.normal_residual;d.tangent_residual=result.tangent_residual;d.momentum_residual=result.momentum_residual;
  std::copy_n(impulse.data(),JRows,ji.data()+e*JRows);for(size_t r=JRows;r<impulse.size();r++)for(size_t n=0;n<N;n++)ci[e*N+n]+=rows.g[r*N+n]*impulse[r];
  size_t r=JRows;for(uint32_t pair=0;pair<s->P;pair++){auto& m=manifolds[e*s->P+pair];for(uint32_t k=0;k<m.count;k++){auto& x=m.points[k];x.normal_impulse=float(impulse[r]);x.tangent_impulse[0]=float(impulse[r+1]);x.tangent_impulse[1]=float(impulse[r+2]);r+=3;d.maximum_normal_impulse=std::max(d.maximum_normal_impulse,double(x.normal_impulse));}}
 }
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=4;
 auto solution=desc<av2_solution>();solution.velocity=v.data();solution.joint_impulse=ji.data();solution.contact_generalized_impulse=ci.data();av2_stage* araw=nullptr;rc=av2_complete(pre.get(),&solution,&araw);
 if(rc){diagnostic[0].native_status=rc;throw Error{IDV1_ARTICULATED};}AS astage(araw,av2_stage_destroy);auto state=desc<av2_state_view>();need(av2_stage_read(astage.get(),&state)==0,IDV1_ARTICULATED);auto bs=bodies(*s,state);
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=5;
 bcx1_stage* craw=nullptr;rc=bcx1_prepare_solved(s->contact,bs.data(),manifolds.data(),step->dt,&craw);if(rc){diagnostic[0].native_status=rc;throw Error{IDV1_CONTACT};}CS cstage(craw,bcx1_stage_destroy);
 std::vector<double> clocks(s->E);need(bcx1_stage_read(cstage.get(),nullptr,nullptr,clocks.data())==0,IDV1_CONTACT);for(uint32_t e=0;e<s->E;e++)need(clocks[e]==state.time[e],IDV1_TRANSACTION);
 for(uint32_t e=0;e<s->E;e++)diagnostic[e].phase=6;
 commit(*s,astage.get(),cstage.get());
 });}
int idv1_read(idv1_scene* s,double* q,double* v,double* w,double* t,uint64_t* count,bcv1_body* b,bcv1_manifold* cache,bcv1_manifold* geometry){return guard([&]{
 need(s);std::lock_guard<std::mutex> lock(s->lock);auto view=desc<av2_state_view>();need(av2_read(s->art,&view)==0,IDV1_ARTICULATED);
 // Query can allocate/validate; complete it before touching any output.
 std::vector<bcv1_manifold> ms(size_t(s->E)*s->P);if(geometry)need(bcv1_query(s->contact,ms.data())==0,IDV1_CONTACT);
 if(q)std::copy_n(view.qpos,s->E*(7+s->J),q);if(v)std::copy_n(view.velocity,s->E*s->N,v);if(w)std::copy_n(view.joint_warm_force,s->E*3*s->J,w);if(t)std::copy_n(view.time,s->E,t);if(count)std::copy_n(view.step_count,s->E,count);
 need(bcv1_read(s->contact,b,cache,nullptr)==0,IDV1_CONTACT);if(geometry)std::copy(ms.begin(),ms.end(),geometry);
 });}
int idv1_capture(idv1_scene* s,idv1_snapshot** out){if(!out)return IDV1_INVALID;return guard([&]{need(s);std::lock_guard<std::mutex> lock(s->lock);auto snap=std::make_unique<idv1_snapshot>(*s);need(av2_capture(s->art,&snap->art.d)==0,IDV1_ARTICULATED);need(bcv1_capture(s->contact,&snap->contact)==0,IDV1_CONTACT);*out=snap.release();});}
void idv1_snapshot_destroy(idv1_snapshot* s){delete s;}
int idv1_restore(idv1_scene* s,const idv1_snapshot* snap,const uint8_t* mask){return guard([&]{need(s&&snap);std::lock_guard<std::mutex> lock(s->lock);reset(*s,snap,mask);});}
int idv1_reset(idv1_scene* s,const uint8_t* mask){return guard([&]{need(s);std::lock_guard<std::mutex> lock(s->lock);reset(*s,nullptr,mask);});}
}
